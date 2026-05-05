"""Sphinx process launcher / supervisor.

Each drone is a pair of subprocesses:
  * ``sphinx <descriptor>::firmware="..."``  — drone simulation core
  * ``parrot-ue4-<world>``                    — the UE renderer

We track both PIDs and bring both down on stop. Logs go to per-drone
files under ``logs/drone-<id>/{sphinx,ue4}.log``.

This module is deliberately tolerant of missing dependencies:
  * If ``/usr/bin/sphinx`` is absent (e.g. macOS dev box), ``dry_run``
    is auto-enabled and each "drone" is replaced by a long ``sleep``,
    so the management UI works locally for development.
  * If the configured netns mode fails (no CAP_NET_ADMIN, no bridge),
    we fall back to ports mode automatically and surface the reason in
    the dashboard.
"""
from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from network import (
    DroneEndpoint,
    NetnsMode,
    PortsMode,
    established_peers_to_port,
    listening_pids_on_port,
)
from session import SessionInfo, detect_active_session, parse_explicit
from state import (
    DroneRecord,
    EnvironmentRecord,
    FlightControllerRecord,
    StateStore,
)
from worlds import Registry

log = logging.getLogger("sphinx-control.launcher")


def _port_in_use(port: int) -> bool:
    """True iff anything is already listening on the given TCP port on
    any local interface. Used as a pre-flight check before spawning a
    subprocess that would bind that port."""
    import socket as _socket
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", int(port)))
        return False
    except OSError:
        return True
    finally:
        s.close()


def _wait_for_port_release(port: int | None, timeout_s: float = 8.0,
                           poll_interval_s: float = 0.2) -> bool:
    """Poll until ``port`` is free (i.e. nothing is bound to it on the
    loopback interface). Returns True if released within ``timeout_s``,
    False on timeout. Used after killing a drone process to make sure
    the kernel has released the netns / port BEFORE we try to bind it
    again — earlier code used a fixed 1.0 s sleep which was sometimes
    too short on a busy host, leaving the next spawn to fail with
    EADDRINUSE."""
    if port is None or int(port) <= 0:
        return True
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if not _port_in_use(int(port)):
            return True
        time.sleep(poll_interval_s)
    return not _port_in_use(int(port))


def _find_port_holders(port: int) -> list[int]:
    """Return the list of PIDs that are LISTEN-bound to ``port`` on this
    host. Uses /proc/net/tcp directly (no external commands) so it works
    even on minimal containers without ss/lsof/fuser installed.

    Used to recover from cases where a stop_fc() PID-kill missed a
    forked child or the recorded PID was stale — the port holder is the
    ground truth of "what's actually running"."""
    import struct
    pids: set[int] = set()
    # Walk /proc/net/tcp + tcp6 and extract inodes for sockets in
    # LISTEN state (state hex 0x0A) on the requested port.
    target_inodes: set[str] = set()
    for proc_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(proc_file) as f:
                next(f)  # skip header
                for line in f:
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    local = parts[1]   # e.g. "0100007F:1F40"
                    state = parts[3]   # "0A" = LISTEN
                    inode = parts[9]
                    if state != "0A":
                        continue
                    try:
                        local_port = int(local.split(":")[1], 16)
                    except (IndexError, ValueError):
                        continue
                    if local_port == int(port):
                        target_inodes.add(inode)
        except FileNotFoundError:
            continue
    if not target_inodes:
        return []
    # Now find which PIDs own these inodes by scanning /proc/<pid>/fd/
    # symlinks for "socket:[<inode>]". Skip processes we can't read.
    try:
        proc_pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return []
    for pid in proc_pids:
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    target = os.readlink(f"{fd_dir}/{fd}")
                except (FileNotFoundError, PermissionError, OSError):
                    continue
                if target.startswith("socket:["):
                    inode = target[8:-1]
                    if inode in target_inodes:
                        pids.add(pid)
                        break
        except (FileNotFoundError, PermissionError):
            continue
    return sorted(pids)


def _proc_alive_running(pid: int) -> bool:
    """True iff pid exists AND is in a running/sleeping state.
    Returns False for zombie ('Z') or dead processes — which
    ``os.kill(pid, 0)`` would otherwise report as alive."""
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("State:"):
                    state = line.split()[1]
                    return state not in ("Z", "X")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    return False


def _read_tail(path: "Path", n_bytes: int) -> str:
    try:
        from pathlib import Path as _P
        p = _P(path)
        if not p.is_file():
            return "(log file not yet flushed)"
        size = p.stat().st_size
        with p.open("rb") as f:
            if size > n_bytes:
                f.seek(size - n_bytes)
            return f.read().decode(errors="replace")
    except Exception as e:
        return f"(could not read log: {e})"


@dataclass
class LaunchRequest:
    drone_profile: str
    world_name: str | None = None  # ignored — kept for back-compat with old API
    instance_id: int | None = None  # auto-allocated if None
    firmware_url: str | None = None  # falls back to config default


@dataclass
class EnvironmentRequest:
    """Start an environment (UE4 only) — drone-agnostic."""

    world_name: str
    # Optional UE window settings; None falls back to launcher.config
    # defaults (which themselves fall back to 1280x1024).
    res_x: int | None = None
    res_y: int | None = None
    hide_panels: bool | None = None  # send F10 once UE window is up


@dataclass
class FCRequest:
    """Start the flight controller (controller_unified/unified_api_server.py).
    All fields optional; defaults come from launcher config."""

    cwd: str | None = None
    python: str | None = None
    script: str | None = None
    http_port: int | None = None
    anafi_ip: str | None = None


class Launcher:
    """Supervisor for Sphinx drone subprocesses. Single instance per
    service; thread-safe via an internal lock around mutating ops."""

    def __init__(
        self,
        config: dict[str, Any],
        registry: Registry,
        state: StateStore,
        log_dir: Path,
    ) -> None:
        self.config = config
        self.registry = registry
        self.state = state
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        # Auto-detect dry-run if Sphinx isn't installed. Spares
        # developers on Mac from having to flip a config flag.
        binary = (config.get("sphinx", {}) or {}).get("binary", "/usr/bin/sphinx")
        self.dry_run = bool(config.get("dry_run", False)) or not Path(binary).is_file()
        if self.dry_run and not config.get("dry_run"):
            log.warning(
                "Sphinx binary not found at %s — auto-enabling dry-run mode. "
                "Spawned drones will be `sleep` placeholders.",
                binary,
            )

        # Network mode setup. NetnsMode is only attempted if explicitly
        # configured AND we have CAP_NET_ADMIN. Otherwise we drop to
        # PortsMode and surface the reason via ``self.network_warning``.
        net = config.get("network", {}) or {}
        mode = net.get("mode", "ports")
        self.network_warning: str | None = None
        if mode == "netns":
            if os.geteuid() != 0 and shutil.which("sudo") is None:
                self.network_warning = (
                    "netns mode requested but no root/sudo available — "
                    "falling back to ports mode."
                )
                mode = "ports"
        if mode == "netns":
            self.network = NetnsMode(
                subnet=net.get("subnet", "10.202.0.0/24"),
                bridge_name=net.get("bridge_name", "sphinx-br0"),
                ip_offset=int(net.get("ip_offset", 10)),
            )
            try:
                self.network.ensure_bridge()
            except RuntimeError as e:
                self.network_warning = (
                    f"netns bridge setup failed ({e}); falling back to ports mode."
                )
                mode = "ports"
        if mode == "ports":
            self.network = PortsMode(
                base_port=int(net.get("base_port", 9080)),
                bind_ip=net.get("bind_ip"),
            )
        self.mode = mode

        self.max_drones = int(config.get("max_drones", 10))

        # ── Active-session attachment ────────────────────────────
        # Decide once at startup which graphical session UE4 should
        # render into. If sphinx-control runs as a system service the
        # parent env has no DISPLAY/WAYLAND_DISPLAY, so we have to
        # discover the user's session via loginctl. Re-detect on every
        # spawn instead if the operator suspects session changes between
        # spawns (rare; not worth the loginctl latency on the hot path).
        attach = str(config.get("gnome_session_attach", "auto")).lower()
        self.session_attach_setting = attach
        self.session: SessionInfo | None = None
        self.session_warning: str | None = None
        if attach == "off":
            log.info("session-attach: disabled by config")
        elif attach == "auto":
            self.session = detect_active_session()
            if self.session is None:
                self.session_warning = (
                    "no active graphical session detected — UE4 windows will "
                    "open with the parent process's environment "
                    "(may render headless)."
                )
                log.warning(self.session_warning)
            else:
                log.info(
                    "session-attach: %s for %s on %s",
                    self.session.session_type, self.session.user, self.session.display,
                )
        else:
            # Explicit override like "x11::0" or "wayland:wayland-0".
            override_env = parse_explicit(attach)
            if override_env is None:
                self.session_warning = (
                    f"gnome_session_attach value {attach!r} unrecognized "
                    f"— expected 'auto', 'off', 'x11::N', or 'wayland:wayland-N'. "
                    f"Falling back to inherited environment."
                )
                log.warning(self.session_warning)
            else:
                # Synthesize a minimal SessionInfo so /api/system surfaces it.
                self.session = SessionInfo(
                    session_id="manual",
                    user=os.environ.get("USER", "unknown"),
                    uid=os.getuid(),
                    session_type=override_env.get("XDG_SESSION_TYPE", "x11"),
                    display=override_env.get("DISPLAY") or override_env.get("WAYLAND_DISPLAY", ""),
                    seat="manual",
                    env=override_env,
                )

    # ─── public API ──────────────────────────────────────────────

    # ─── environment (UE4-only) ─────────────────────────────────

    def start_environment(self, req: EnvironmentRequest) -> EnvironmentRecord:
        """Start a UE4 environment with no drone bound to it. The
        operator can fly the spectator camera, inspect the arena
        geometry, and later attach a drone via ``spawn(drone)``.

        Singleton: only one environment runs at a time on this host.
        Stop the existing environment before starting a new one.
        """
        with self._lock:
            existing = self.state.current_env()
            if existing is not None and self._is_env_alive(existing):
                raise RuntimeError(
                    "an environment is already running; stop it before "
                    "starting a new one (Sphinx 2 supports one drone per host)."
                )

            world = self.registry.world(req.world_name)
            if not self.dry_run and not world.available:
                raise RuntimeError(f"world unavailable: {world.unavailable_reason}")

            env_id = f"env-{uuid.uuid4().hex[:6]}"
            rec = EnvironmentRecord(
                env_id=env_id,
                world_name=req.world_name,
                world_app=world.name,
                config_file=world.config_file,
                status="starting",
            )
            self.state.upsert_env(rec)

            try:
                ue4_pid = self._launch_ue4(
                    env_id=env_id, world=world,
                    res_x=req.res_x, res_y=req.res_y,
                )
            except Exception as e:
                self.state.update_env_status(env_id, "error", last_error=str(e))
                raise

            self.state.update_env_status(env_id, "running", ue4_pid=ue4_pid)
            updated = self.state.get_env(env_id)
            assert updated is not None
            log.info(
                "started env %s world=%s ue4_pid=%s res=%sx%s",
                env_id, req.world_name, ue4_pid,
                req.res_x or "default", req.res_y or "default",
            )
            # If requested, send F10 to hide the HMI panels so only the
            # 3D viewport is visible. Default is enabled for sdc_arena.
            ue4_cfg = self.config.get("ue4", {}) or {}
            hide = (req.hide_panels if req.hide_panels is not None
                    else ue4_cfg.get("hide_panels_default", True))
            if hide:
                self._hide_ue4_panels(ue4_pid=ue4_pid)
            # NOTE: sphinx-cli world params (e.g. sky preset) can ONLY
            # be applied AFTER a drone (sphinx process) is spawned —
            # without sphinx, sphinx-cli reports "An instance of Sphinx
            # is not running" and bails. We defer the post-start params
            # until the first successful drone spawn (see spawn()).
            return updated

    def _apply_post_start_world_params(
        self, world_name: str, world: Any
    ) -> None:
        """Best-effort runtime configuration of the freshly-started world.
        Currently sets the sky preset via sphinx-cli; failures are
        logged but don't abort the env startup (the env still works,
        just with the default sky)."""
        if self.dry_run:
            return
        # Resolve sky preset: per-world override > global default > "indoor".
        # The Registry's World object exposes config attributes via
        # ``world.extras`` if present, plus we look at the launcher
        # config under worlds.sky_preset.
        sky_preset = None
        worlds_cfg = self.config.get("worlds", {}) or {}
        sky_preset = (
            getattr(world, "sky_preset", None)
            or worlds_cfg.get("sky_preset")
            or "indoor"
        )
        if not sky_preset:
            return
        # Poll sphinx-cli up to 30 s. After spawn() returns, sphinx is
        # still initializing and connecting to UE4 — calling sphinx-cli
        # too early gets 'An instance of Sphinx is not running'. Retry
        # on a 1.5s cadence until it succeeds or we give up.
        deadline = time.time() + 30.0
        attempt = 0
        last_err = ""
        sky_ok = False
        while time.time() < deadline and not sky_ok:
            attempt += 1
            try:
                result = subprocess.run(
                    ["sphinx-cli", "param", "-m", "world",
                     "sky/sky", "preset", sky_preset],
                    capture_output=True, text=True, timeout=4,
                )
                if result.returncode == 0:
                    log.info(
                        "world '%s' sky preset → %s (attempt %d)",
                        world_name, sky_preset, attempt,
                    )
                    sky_ok = True
                    break
                last_err = (result.stdout + result.stderr).strip()
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                last_err = str(e)
            time.sleep(1.5)
        if not sky_ok:
            log.warning(
                "sphinx-cli sky preset never succeeded after 30s: %s",
                last_err,
            )
        # Also raise the drone's spawn position to clear the raised
        # arena floor (top face at z=+0.40 m). Sphinx supports moving
        # the drone via the omniscient component's `pose` action.
        # Best-effort — failures are logged but not fatal.
        spawn_z_m = float(
            self.config.get("ue4", {}).get("drone_spawn_z_m", 0.5)
        )
        try:
            r = subprocess.run(
                ["sphinx-cli", "action", "-m", "omniscient", "pose",
                 "0", "0", str(spawn_z_m), "0", "0", "0"],
                capture_output=True, text=True, timeout=4,
            )
            if r.returncode == 0:
                log.info("drone teleported to z=%s m via omniscient/pose",
                         spawn_z_m)
            else:
                log.warning("omniscient/pose failed (rc=%d): %s%s",
                            r.returncode, r.stdout, r.stderr)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            log.warning("omniscient/pose call failed: %s", e)
        # AMS actors spawn paused by default per Sphinx docs. Unpause
        # them so spectators actually walk/drive along PerimeterPath.
        try:
            r = subprocess.run(
                ["sphinx-cli", "param", "-m", "world",
                 "actors", "pause", "false"],
                capture_output=True, text=True, timeout=4,
            )
            if r.returncode == 0:
                log.info("AMS actors unpaused (walking)")
            else:
                log.warning("actors/pause unpause failed (rc=%d): %s%s",
                            r.returncode, r.stdout, r.stderr)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            log.warning("actors/pause call failed: %s", e)

    def stop_environment(self) -> None:
        """Stop the current environment AND any drones attached to it.
        UE4 stays the source of truth; without it sphinx has nothing
        to render into, so a stop-environment cascades."""
        with self._lock:
            # First stop any running drones so they don't leak.
            for d in self.state.list_all(status="running"):
                try:
                    self._stop_drone_locked(d.drone_id)
                except Exception as e:
                    log.warning("cascade stop of drone %s failed: %s", d.drone_id, e)

            env = self.state.current_env()
            if env is None:
                return
            if env.ue4_pid:
                self._terminate_pid(env.ue4_pid)
            self.state.update_env_status(env.env_id, "stopped")
            log.info("stopped env %s", env.env_id)

    def restart_environment(self) -> EnvironmentRecord:
        with self._lock:
            env = self.state.current_env()
            if env is None:
                raise KeyError("no environment running")
            world_name = env.world_name
            self.stop_environment()
            self.state.delete_env(env.env_id)
            time.sleep(1.0)
            return self.start_environment(EnvironmentRequest(world_name=world_name))

    def current_environment(self) -> EnvironmentRecord | None:
        env = self.state.current_env()
        if env is None:
            return None
        # Auto-reconcile: if the recorded ue4_pid is no longer alive,
        # mark the env stopped so the dashboard reflects reality. Without
        # this the DB row stays "running" forever after a crash, the
        # restart endpoint tries to "stop" a dead pid (no-op) and the
        # operator sees "Stop does nothing" because there's nothing to
        # stop in the first place. Caught live: ue4 process 8657 had
        # died but the dashboard was still showing running uptime 9m.
        if not self._is_env_alive(env):
            log.warning(
                "env %s ue4_pid %s no longer alive — auto-marking stopped",
                env.env_id, env.ue4_pid,
            )
            self.state.update_env_status(env.env_id, "stopped")
            return None
        env.extras["alive"] = True
        return env

    # ─── flight controller (unified_api_server.py) ──────────────

    def start_fc(self, req: FCRequest) -> FlightControllerRecord:
        """Start the controller_unified/unified_api_server.py process.

        This is the FC ("flight controller") — the Olympe-driven HTTP
        service that talks to a real or simulated Anafi at ANAFI_IP.
        Singleton: it binds a fixed HTTP port (default 8080).

        Doesn't require a running drone — Olympe will retry until the
        Anafi answers, so the operator can start the FC before, after,
        or instead of a sphinx drone. Useful for talking to a real
        Anafi from this same management console.
        """
        with self._lock:
            existing = self.state.current_fc()
            if existing is not None and self._is_fc_alive(existing):
                raise RuntimeError(
                    "the flight controller is already running; "
                    "stop it before starting a new one."
                )

            fc_cfg = self.config.get("flight_controller", {}) or {}
            cwd = req.cwd or fc_cfg.get("cwd")
            python = req.python or fc_cfg.get("python") or "python3"
            script = req.script or fc_cfg.get("script") or "unified_api_server.py"
            http_port = req.http_port or fc_cfg.get("http_port") or 8080
            anafi_ip = req.anafi_ip or fc_cfg.get("anafi_ip")

            if not cwd:
                raise ValueError(
                    "flight_controller.cwd is not configured; set it in "
                    "config.yaml (e.g. /home/sdc/sdc-tobe/controller_unified)."
                )
            cwd_path = Path(cwd)
            if not self.dry_run and not cwd_path.is_dir():
                raise RuntimeError(f"FC cwd does not exist: {cwd}")
            if not self.dry_run and not (cwd_path / script).is_file():
                raise RuntimeError(f"FC script not found: {cwd_path / script}")

            fc_id = f"fc-{uuid.uuid4().hex[:6]}"
            rec = FlightControllerRecord(
                fc_id=fc_id,
                cwd=str(cwd_path),
                script=script,
                http_port=int(http_port),
                anafi_ip=anafi_ip,
                status="starting",
            )
            self.state.upsert_fc(rec)

            # Pre-flight: if the FC's HTTP port is already bound, try to
            # auto-recover before giving up. Common case: previous FC
            # crashed (or stop_fc() raced a child fork) and left a
            # process holding the port that we never cleaned up.
            if not self.dry_run and _port_in_use(int(http_port)):
                holders = _find_port_holders(int(http_port))
                if holders:
                    log.warning(
                        "port %d in use by PIDs %s on FC startup — "
                        "auto-cleaning before retry", http_port, holders,
                    )
                    for h_pid in holders:
                        try:
                            self._terminate_pid(h_pid)
                        except Exception as e:
                            log.warning(
                                "auto-cleanup kill pid=%d failed: %s",
                                h_pid, e,
                            )
                    _wait_for_port_release(int(http_port), timeout_s=5.0)
                if _port_in_use(int(http_port)):
                    # Still bound after our cleanup — give up with a
                    # message that explains where to look.
                    raise RuntimeError(
                        f"port {http_port} already in use and "
                        f"sphinx-control's auto-cleanup couldn't free it. "
                        f"Inspect with `sudo ss -tlnp 'sport = :{http_port}'` "
                        f"and kill the holder manually."
                    )

            try:
                pid = self._launch_fc(
                    fc_id=fc_id, cwd=cwd_path, python=python,
                    script=script, http_port=int(http_port),
                    anafi_ip=anafi_ip,
                    extra_env=dict(fc_cfg.get("extra_env", {}) or {}),
                )
            except Exception as e:
                self.state.update_fc_status(fc_id, "error", last_error=str(e))
                raise

            # Liveness check: poll the process once after a short delay
            # to catch the common "ModuleNotFoundError on import" /
            # "wrong python interpreter" failures. We check against
            # /proc/<pid>/status because plain `os.kill(pid, 0)` returns
            # success for zombies — and a Python script that errored on
            # `import flask` becomes a zombie until we wait() on it,
            # which would have made the launcher falsely report
            # status=running. Verified live on the SDC host: a system-
            # python spawn that hit ModuleNotFoundError used to slip
            # past the kill-0 check; the State='Z' read here catches it.
            if not self.dry_run:
                time.sleep(1.5)
                if not _proc_alive_running(pid):
                    log_path = self.log_dir / fc_id / "fc.log"
                    tail = _read_tail(log_path, 1500)
                    self.state.update_fc_status(
                        fc_id, "error",
                        last_error=f"FC died on startup. Tail of fc.log:\n{tail}",
                    )
                    raise RuntimeError(
                        f"FC died on startup. See {log_path}\nTail:\n{tail}"
                    )

            self.state.update_fc_status(fc_id, "running", pid=pid)
            updated = self.state.get_fc(fc_id)
            assert updated is not None
            log.info(
                "started fc %s pid=%s cwd=%s script=%s port=%d anafi_ip=%s",
                fc_id, pid, cwd, script, http_port, anafi_ip or "(default)",
            )
            return updated

    def stop_fc(self) -> None:
        """Stop the flight controller. Brain-dead simple now after
        repeated user reports that the previous implementation didn't
        work:

          1. kill -9 fc.pid (SIGKILL, direct, no process-group dance,
             no SIGTERM grace).
          2. kill -9 any other PIDs holding the FC's HTTP port (catches
             children that escaped the recorded PID, etc.).
          3. Mark state stopped.

        SIGKILL is unconditional — Olympe atexit hangs that previously
        forced us into the SIGKILL fallback after a 1.5–5s SIGTERM grace
        are now skipped entirely. The user pointed out 'all you need
        is to send kill <pid>'. Done.
        """
        with self._lock:
            fc = self.state.current_fc()
            if fc is None:
                return
            fc_id = fc.fc_id
            port = fc.http_port
            killed_pids = []
            if fc.pid:
                try:
                    os.kill(fc.pid, signal.SIGKILL)
                    killed_pids.append(fc.pid)
                except ProcessLookupError:
                    pass
                except PermissionError as e:
                    log.warning("SIGKILL fc.pid=%d denied: %s", fc.pid, e)
            # Catch anything still on the port (forked children, orphan
            # processes, etc.) — the port is the ground truth.
            if port:
                for h_pid in _find_port_holders(int(port)):
                    if h_pid in killed_pids:
                        continue
                    try:
                        os.kill(h_pid, signal.SIGKILL)
                        killed_pids.append(h_pid)
                        log.info("SIGKILL'd port-%d holder pid=%d",
                                 port, h_pid)
                    except (ProcessLookupError, PermissionError) as e:
                        log.warning("SIGKILL pid=%d failed: %s", h_pid, e)
            # Final safety net — pkill anything matching the FC's
            # script name, just in case the recorded PID + port-holder
            # scan both missed something. This covers the case where
            # the user manually started an FC outside sphinx-control
            # (different session, different terminal) and our state
            # has no clue what PID it has.
            try:
                script_name = fc.script or "unified_api_server.py"
                # -9 = SIGKILL, -f = match against full command line
                # so "unified_api_server.py" matches even though the
                # python process's own argv[0] is just "python3".
                r = subprocess.run(
                    ["pkill", "-9", "-f", script_name],
                    capture_output=True, text=True, timeout=4,
                )
                # rc=0 means at least one killed; rc=1 means none —
                # both are fine, only worth logging the kill.
                if r.returncode == 0:
                    log.info("pkill -9 -f %s swept additional FC procs",
                             script_name)
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                log.warning("pkill sweep failed: %s", e)
            self.state.update_fc_status(fc_id, "stopped")
            log.info("stopped fc %s — killed pids=%s", fc_id, killed_pids)

    def restart_fc(self) -> FlightControllerRecord:
        """Stop + start the FC reliably.

        Drops the stopped state row so any subsequent sweep doesn't
        misattribute it, then waits for the port to be free before
        re-launching."""
        with self._lock:
            fc = self.state.current_fc()
            if fc is None:
                raise KeyError("no flight controller running")
            req = FCRequest(
                cwd=fc.cwd, script=fc.script,
                http_port=fc.http_port, anafi_ip=fc.anafi_ip,
            )
            port = fc.http_port
            self.stop_fc()
            self.state.delete_fc(fc.fc_id)
            # Wait for full port release before starting. stop_fc() did
            # this internally with a 4-second budget; the second wait
            # below is a safety net for slower hosts.
            _wait_for_port_release(port, timeout_s=4.0)
            time.sleep(0.3)
            return self.start_fc(req)

    def current_flight_controller(self) -> FlightControllerRecord | None:
        fc = self.state.current_fc()
        if fc is None:
            # ORPHAN ADOPTION — common scenario: sphinx-control was
            # restarted while a previously-launched FC kept running. The
            # state DB row for that FC is gone (or marked stopped), but
            # the python process is still bound to port 8080. Without
            # this adoption step, the user's 'stop FC' button does
            # nothing because there's no record to act on. We scan the
            # configured FC port for a unified_api_server-like process
            # and re-create a state row pointing at it.
            adopted = self._adopt_orphan_fc()
            if adopted is not None:
                log.warning(
                    "adopted orphan FC pid=%d on port %d — sphinx-control "
                    "had no state row for it, likely from a previous run "
                    "before this server restart",
                    adopted.pid, adopted.http_port,
                )
                return adopted
            return None
        # Auto-reconcile (see current_environment for rationale).
        if not self._is_fc_alive(fc):
            log.warning(
                "fc %s pid %s no longer alive — auto-marking stopped",
                fc.fc_id, fc.pid,
            )
            self.state.update_fc_status(fc.fc_id, "stopped")
            return None
        fc.extras["alive"] = True
        return fc

    def _adopt_orphan_fc(self) -> FlightControllerRecord | None:
        """If sphinx-control's state DB has no current FC but something
        is listening on the configured FC port, look up its PID and
        adopt it into the state DB so the UI's stop/restart buttons can
        act on it.

        Returns the new state row if an orphan was adopted, else None.

        Doesn't try to be subtle about ownership — if anything is bound
        to the configured FC port on this host, we treat it as 'the FC'
        and let the user stop/restart it. That's exactly the recovery
        path users want when sphinx-control has been bounced and they
        want to take control of the still-running FC.
        """
        if self.dry_run:
            return None
        fc_cfg = self.config.get("flight_controller", {}) or {}
        http_port = int(fc_cfg.get("http_port") or 8080)
        if not _port_in_use(http_port):
            return None
        holders = _find_port_holders(http_port)
        if not holders:
            return None
        pid = holders[0]
        # Sanity check the holder: must be alive and look like the FC.
        # We don't enforce a name match (Anafi/Olympe may rename the
        # process via setproctitle), just the alive + listening test.
        if not _proc_alive_running(pid):
            return None
        # Try to read the FC's working directory from /proc — useful
        # context for the UI even though we don't strictly need it.
        try:
            cwd_link = os.readlink(f"/proc/{pid}/cwd")
        except (FileNotFoundError, PermissionError, OSError):
            cwd_link = fc_cfg.get("cwd", "")
        fc_id = f"fc-orphan-{pid}"
        rec = FlightControllerRecord(
            fc_id=fc_id,
            cwd=str(cwd_link or fc_cfg.get("cwd") or ""),
            script=fc_cfg.get("script") or "unified_api_server.py",
            http_port=http_port,
            anafi_ip=fc_cfg.get("anafi_ip"),
            pid=pid,
            status="running",
            last_error="adopted by sphinx-control after server restart",
        )
        self.state.upsert_fc(rec)
        self.state.update_fc_status(fc_id, "running", pid=pid)
        adopted = self.state.get_fc(fc_id)
        if adopted is not None:
            adopted.extras["alive"] = True
            adopted.extras["adopted"] = True
        return adopted

    def _launch_fc(
        self,
        fc_id: str,
        cwd: Path,
        python: str,
        script: str,
        http_port: int,
        anafi_ip: str | None,
        extra_env: dict[str, str],
    ) -> int:
        log_dir = self.log_dir / fc_id
        log_dir.mkdir(parents=True, exist_ok=True)
        fc_log = open(log_dir / "fc.log", "ab")

        if self.dry_run:
            sleeper = ["sleep", "999999"]
            proc = subprocess.Popen(
                sleeper, stdout=fc_log, stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            return proc.pid

        # Resolve python path. If it's relative, treat as relative to cwd.
        py_path = Path(python)
        if not py_path.is_absolute():
            candidate = cwd / python
            if candidate.is_file():
                py_path = candidate
        # If still not a real file, fall back to the literal name
        # (let PATH resolution try) — useful when python is just
        # "python3".
        argv = [str(py_path), script]

        env = dict(os.environ)
        env["HTTP_PORT"] = str(http_port)
        if anafi_ip:
            env["ANAFI_IP"] = anafi_ip
        env.update(extra_env)

        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=fc_log,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
        return proc.pid

    def _is_fc_alive(self, fc: FlightControllerRecord) -> bool:
        if not fc.pid:
            return False
        # Zombie-aware check (a python that crashed on import is a
        # zombie until we wait() on it; plain os.kill(pid, 0) would
        # call it alive). See _proc_alive_running.
        return _proc_alive_running(fc.pid) and fc.status == "running"

    # ─── drone (sphinx, attached to running env) ────────────────

    def spawn(self, req: LaunchRequest) -> DroneRecord:
        """Start a sphinx drone process attached to the running
        environment. Requires an environment to be running first —
        sphinx and UE4 rendezvous via firmwared, and without UE4
        already up the drone has nowhere to render.
        """
        with self._lock:
            env = self.state.current_env()
            if env is None or not self._is_env_alive(env):
                raise RuntimeError(
                    "no environment running — start one first via "
                    "POST /api/environment, then spawn the drone."
                )

            profile = self.registry.profile(req.drone_profile)
            if not self.dry_run and not profile.available:
                raise RuntimeError(f"profile unavailable: {profile.unavailable_reason}")

            instance_id = req.instance_id or self._allocate_instance_id()
            if instance_id < 1 or instance_id > self.max_drones:
                raise ValueError(
                    f"instance_id {instance_id} outside 1..{self.max_drones}"
                )
            if instance_id in self.state.used_instance_ids():
                if self.max_drones == 1:
                    raise RuntimeError(
                        "a drone is already running in this environment; "
                        "stop it before starting a new one."
                    )
                raise RuntimeError(
                    f"instance_id {instance_id} already in use; stop the existing "
                    f"drone first."
                )

            endpoint = self.network.endpoint_for(instance_id)
            try:
                self.network.setup(instance_id)
            except RuntimeError as e:
                raise RuntimeError(f"network setup failed: {e}") from e

            drone_id = f"drone-{instance_id}-{uuid.uuid4().hex[:6]}"
            firmware = (
                req.firmware_url
                or (self.config.get("sphinx", {}) or {}).get("firmware_url")
            )

            rec = DroneRecord(
                drone_id=drone_id,
                instance_id=instance_id,
                drone_type=profile.name,
                descriptor=profile.descriptor_path,
                world_app=env.world_app,
                config_file=env.config_file,
                firmware_url=firmware,
                drone_ip=endpoint.ip,
                drone_port=endpoint.port,
                ue4_pid=env.ue4_pid,  # share env's UE4
                status="spawning",
            )
            removed = self.state.delete_stopped_at_instance_id(instance_id)
            if removed:
                log.info(
                    "released %d stopped slot(s) at instance_id=%d before spawn",
                    removed, instance_id,
                )
            self.state.upsert(rec)

            try:
                sphinx_pid = self._launch_sphinx_only(
                    drone_id=drone_id,
                    descriptor_path=profile.descriptor_path,
                    endpoint=endpoint,
                    firmware_url=firmware,
                )
            except Exception as e:
                self.state.update_status(drone_id, "error", last_error=str(e))
                self._safe_teardown(instance_id)
                raise

            self.state.update_status(
                drone_id,
                status="running",
                sphinx_pid=sphinx_pid,
                ue4_pid=env.ue4_pid,
            )
            updated = self.state.get(drone_id)
            assert updated is not None
            log.info(
                "spawned drone %s instance=%d sphinx_pid=%s endpoint=%s "
                "(attached to env %s, ue4_pid=%s)",
                drone_id, instance_id, sphinx_pid, endpoint.display(),
                env.env_id, env.ue4_pid,
            )
            # Now that sphinx is running, sphinx-cli can talk to it —
            # apply the deferred world preset (sky etc.) the env asked
            # for. Idempotent, so running it on every spawn is fine
            # (cheap, and self-corrects if the world drifted).
            try:
                world = self.registry.world(env.world_name)
                self._apply_post_start_world_params(env.world_name, world)
            except Exception as e:
                log.warning("post-spawn world params failed: %s", e)
            return updated

    def stop(self, drone_id: str) -> None:
        """Stop a drone (sphinx) but keep the environment (UE4) running."""
        with self._lock:
            self._stop_drone_locked(drone_id)

    def _stop_drone_locked(self, drone_id: str) -> None:
        rec = self.state.get(drone_id)
        if rec is None:
            raise KeyError(drone_id)
        # Only stop the sphinx process; the ue4_pid in the row
        # belongs to the environment and is owned by it.
        if rec.sphinx_pid:
            self._terminate_pid(rec.sphinx_pid)
        self._safe_teardown(rec.instance_id)
        self.state.update_status(drone_id, status="stopped")
        log.info("stopped drone %s (instance=%d) — env unchanged",
                 drone_id, rec.instance_id)

    def delete(self, drone_id: str) -> None:
        """Stop the drone, tear down its network namespace, and remove
        its row from state. After this returns the drone's port is
        guaranteed-or-best-effort released so a subsequent ``spawn()``
        with the same instance_id won't race the kernel's port
        teardown.

        Hardening over the previous naive version:
          - Always run network.teardown(), even if stop() failed (the
            previous code skipped teardown on stop errors which left
            stale network namespaces around).
          - Wait for the drone port to be released before returning.
          - Return idempotently for unknown drone_ids (lookup misses
            after a crash were silently doing nothing — same here, but
            documented).
        """
        with self._lock:
            rec = self.state.get(drone_id)
            if rec is None:
                log.info("delete: drone %s not in state — nothing to do",
                         drone_id)
                return
            instance_id = rec.instance_id
            port = rec.drone_port
            try:
                self.stop(drone_id)
            except KeyError:
                pass
            except Exception as e:
                log.warning("delete: stop(%s) failed: %s — continuing "
                            "with teardown anyway", drone_id, e)
            # Always tear down the network namespace, even if stop()
            # raised. Stale netns are the #1 cause of the next spawn
            # failing.
            self._safe_teardown(instance_id)
            self.state.delete(drone_id)
            # Wait for port to be released so an immediately-following
            # spawn at the same instance_id doesn't race the kernel.
            _wait_for_port_release(port, timeout_s=5.0)
            log.info("deleted drone %s (instance=%d, port=%s)",
                     drone_id, instance_id, port)

    def restart(self, drone_id: str) -> DroneRecord:
        """Stop and respawn with the same parameters and instance_id.

        The drone_id changes (a fresh UUID is minted) but the instance_id
        stays the same, so external clients tracking ``host:port``
        keep working seamlessly. We delete the old stopped row before
        respawning so the unique instance_id constraint doesn't fire.

        Reliability improvements over the naive 'stop, sleep 1s, spawn'
        approach (added because users reported the second spawn
        intermittently failed when the first drone had crashed):
          1. WAIT for the drone's port to actually be released by the
             kernel before re-binding (poll up to 8s instead of a
             fixed 1.0 s sleep). On a loaded host the netns teardown
             can easily take 2-3 s.
          2. RETRY the spawn once on failure, with a fresh teardown
             between attempts. This catches the case where the first
             spawn raced with stale state.
        """
        with self._lock:
            rec = self.state.get(drone_id)
            if rec is None:
                raise KeyError(drone_id)
            old_port = rec.drone_port
            old_instance = rec.instance_id
            self.stop(drone_id)
            # Drop the stopped row first; otherwise the new row's
            # instance_id collides with it on INSERT.
            self.state.delete(drone_id)
            # Wait for the kernel to actually free the port before we
            # try to rebind it. Falls back to a fixed sleep if the
            # port wasn't released within the timeout — better than
            # giving up entirely.
            released = _wait_for_port_release(old_port, timeout_s=8.0)
            if not released:
                log.warning(
                    "port %s still in use after 8s; sleeping a further 2s "
                    "and trying spawn anyway", old_port,
                )
                time.sleep(2.0)
            else:
                # Brief settling pause even after the port released.
                time.sleep(0.2)

            launch_req = LaunchRequest(
                drone_profile=rec.drone_type,
                world_name=rec.world_app,
                instance_id=old_instance,
                firmware_url=rec.firmware_url,
            )
            try:
                return self.spawn(launch_req)
            except Exception as first_err:
                log.warning(
                    "first respawn of drone %s (instance=%d) failed: %s — "
                    "tearing down and retrying once", drone_id,
                    old_instance, first_err,
                )
                self._safe_teardown(old_instance)
                # Sweep any zombies/leftover rows the failed spawn left.
                self.state.delete_stopped_at_instance_id(old_instance)
                _wait_for_port_release(old_port, timeout_s=5.0)
                time.sleep(0.5)
                return self.spawn(launch_req)

    def restart_all(self) -> list[DroneRecord]:
        with self._lock:
            running = [r for r in self.state.list_all() if r.status == "running"]
            return [self.restart(r.drone_id) for r in running]

    def stop_all(self) -> int:
        with self._lock:
            running = [r for r in self.state.list_all() if r.status == "running"]
            for r in running:
                try:
                    self.stop(r.drone_id)
                except Exception as e:
                    log.warning("stop_all: failed to stop %s: %s", r.drone_id, e)
            return len(running)

    def list_drones(self) -> list[DroneRecord]:
        recs = self.state.list_all()
        for r in recs:
            alive = self._is_alive(r)
            # Auto-reconcile (see current_environment for rationale).
            if r.status == "running" and not alive:
                log.warning(
                    "drone %s sphinx_pid %s no longer alive — "
                    "auto-marking stopped",
                    r.drone_id, r.sphinx_pid,
                )
                self.state.update_status(r.drone_id, "stopped")
                r.status = "stopped"
            r.extras["alive"] = alive
        return recs

    def connections_for(self, drone_id: str) -> dict[str, Any]:
        rec = self.state.get(drone_id)
        if rec is None:
            raise KeyError(drone_id)
        port = rec.drone_port
        if not port:
            return {"drone_id": drone_id, "note": "no port (netns mode); use ss/ip on host"}
        return {
            "drone_id": drone_id,
            "endpoint": f"{rec.drone_ip}:{port}",
            "listeners_pids": listening_pids_on_port(port),
            "established_peers": established_peers_to_port(port),
        }

    def _refresh_session(self) -> None:
        """Re-run session detection. No-op on platforms without
        loginctl. Updates ``self.session`` and clears
        ``self.session_warning`` if a session is found this time."""
        fresh = detect_active_session()
        if fresh is None:
            # Don't clobber an existing session with None — a transient
            # loginctl hiccup shouldn't drop the operator's display
            # mid-flight.
            if self.session is None and not self.session_warning:
                self.session_warning = (
                    "no active graphical session detected — UE4 windows will "
                    "open with the parent process's environment "
                    "(may render headless)."
                )
            return
        if self.session is None:
            log.info(
                "session-attach: discovered active session %s for %s on %s",
                fresh.session_type, fresh.user, fresh.display or "(default)",
            )
            self.session_warning = None
        elif (fresh.session_id != self.session.session_id
              or fresh.user != self.session.user):
            log.info(
                "session-attach: switched to %s for %s on %s "
                "(was %s for %s)",
                fresh.session_type, fresh.user, fresh.display or "(default)",
                self.session.session_type, self.session.user,
            )
        self.session = fresh

    def reconcile_on_startup(self) -> None:
        """Find rows whose pids are dead and mark them stopped. Called
        once when the management service boots so stale rows from a
        previous run don't lie about being live."""
        with self._lock:
            for r in self.state.list_all(status="running"):
                alive = self._is_alive(r)
                if not alive:
                    log.info(
                        "reconcile: drone %s pids %s/%s dead, marking stopped",
                        r.drone_id, r.sphinx_pid, r.ue4_pid,
                    )
                    self.state.update_status(r.drone_id, "stopped")
            for env in self.state.list_envs():
                if env.status != "running":
                    continue
                if not self._is_env_alive(env):
                    log.info(
                        "reconcile: env %s ue4_pid %s dead, marking stopped",
                        env.env_id, env.ue4_pid,
                    )
                    self.state.update_env_status(env.env_id, "stopped")
            fc = self.state.current_fc()
            if fc is not None and fc.status == "running":
                if not self._is_fc_alive(fc):
                    log.info(
                        "reconcile: fc %s pid %s dead, marking stopped",
                        fc.fc_id, fc.pid,
                    )
                    self.state.update_fc_status(fc.fc_id, "stopped")

    def _is_env_alive(self, env: EnvironmentRecord) -> bool:
        if not env.ue4_pid:
            return False
        try:
            os.kill(env.ue4_pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return env.status == "running"

    # ─── internals ───────────────────────────────────────────────

    def _allocate_instance_id(self) -> int:
        used = self.state.used_instance_ids()
        for i in range(1, self.max_drones + 1):
            if i not in used:
                return i
        raise RuntimeError(f"all {self.max_drones} instance slots in use")

    def _build_session_env(self) -> dict[str, str]:
        """Refresh & return the session env to splice into UE4 / sphinx
        children. Re-detects on every call when in auto mode."""
        if self.session_attach_setting == "auto":
            self._refresh_session()
        return self.session.env if self.session else {}

    def _launch_ue4(self, env_id: str, world,
                     res_x: int | None = None,
                     res_y: int | None = None) -> int:
        """Start the UE4 application for the environment. Returns
        ue4_pid. Drone-agnostic: no sphinx is started here."""
        env_log_dir = self.log_dir / env_id
        env_log_dir.mkdir(parents=True, exist_ok=True)
        ue4_log = open(env_log_dir / "ue4.log", "ab")

        session_env = self._build_session_env()

        if self.dry_run:
            sleeper = ["sleep", "999999"]
            ue4_proc = subprocess.Popen(
                sleeper, stdout=ue4_log, stderr=subprocess.STDOUT,
                env={**os.environ, **session_env}, preexec_fn=os.setsid,
            )
            return ue4_proc.pid

        ue4_argv, ue4_env = self._ue4_argv(
            world_binary=world.binary,
            instance_id=0,
            endpoint=None,
            config_file=world.config_file,
            res_x=res_x,
            res_y=res_y,
            extra_args=getattr(world, "extra_ue4_args", None) or [],
        )
        ue4_proc = subprocess.Popen(
            ue4_argv, stdout=ue4_log, stderr=subprocess.STDOUT,
            env={**os.environ, **session_env, **ue4_env},
            preexec_fn=os.setsid,
        )
        return ue4_proc.pid

    def _launch_sphinx_only(
        self,
        drone_id: str,
        descriptor_path: str,
        endpoint: DroneEndpoint,
        firmware_url: str | None,
    ) -> int:
        """Start ONLY the sphinx process. Assumes UE4 is already
        running (started via ``start_environment``). Returns sphinx_pid.

        Dry-run: a sleep placeholder.
        """
        drone_log_dir = self.log_dir / drone_id
        drone_log_dir.mkdir(parents=True, exist_ok=True)
        sphinx_log = open(drone_log_dir / "sphinx.log", "ab")

        session_env = self._build_session_env()

        if self.dry_run:
            sleeper = ["sleep", "999999"]
            sphinx_proc = subprocess.Popen(
                sleeper, stdout=sphinx_log, stderr=subprocess.STDOUT,
                env={**os.environ, **session_env}, preexec_fn=os.setsid,
            )
            return sphinx_proc.pid

        sphinx_argv, sphinx_env = self._sphinx_argv(
            instance_id=endpoint.instance_id,
            descriptor_path=descriptor_path,
            firmware_url=firmware_url,
            config_file=None,  # config_file went to UE4 at env-start time
            endpoint=endpoint,
        )
        if endpoint.netns:
            sphinx_argv = ["sudo", "-n", "ip", "netns", "exec",
                           endpoint.netns, *sphinx_argv]

        sphinx_proc = subprocess.Popen(
            sphinx_argv, stdout=sphinx_log, stderr=subprocess.STDOUT,
            env={**os.environ, **session_env, **sphinx_env},
            preexec_fn=os.setsid,
        )
        return sphinx_proc.pid

    def _sphinx_argv(
        self,
        instance_id: int,
        descriptor_path: str,
        firmware_url: str | None,
        config_file: str | None,
        endpoint: DroneEndpoint,
    ) -> tuple[list[str], dict[str, str]]:
        """Build the `sphinx` argv. Verified against Sphinx 2.15.1.

        Sphinx 2.15 supports VERY few flags — really just the descriptor
        with optional ``::param=value`` suffixes and ``--config-file``.
        There is no ``--instance`` / ``--port`` / ``-i`` flag. Multi-
        instance support relies on isolating Sphinx's network state
        (typically via netns); the flag-based approach attempted here
        in earlier revisions silently broke every spawn because Sphinx
        rejected the unknown flag. Don't reintroduce flags without
        verifying with ``sphinx --help`` first.
        """
        sphinx_bin = (self.config.get("sphinx", {}) or {}).get(
            "binary", "/usr/bin/sphinx"
        )
        argv: list[str] = [sphinx_bin]
        # Sphinx accepts the drone descriptor with "::firmware=..." appended.
        descriptor_arg = descriptor_path
        if firmware_url:
            descriptor_arg = f"{descriptor_path}::firmware={firmware_url}"
        # Initial pose so the drone spawns ABOVE the raised arena floor
        # (top at z=+0.40 m). Default lifts to 0.5 m. Configurable via
        # config.yaml: ue4.drone_spawn_pose ("x y z roll pitch yaw"
        # space-separated; matches Sphinx's <pose> XML element format).
        ue4_cfg = self.config.get("ue4", {}) or {}
        spawn_pose = ue4_cfg.get("drone_spawn_pose", "0 0 0.5 0 0 0")
        if spawn_pose:
            descriptor_arg = f"{descriptor_arg}::pose={spawn_pose}"
        argv.append(descriptor_arg)
        # NOTE: -config-file is a UE4 application flag, NOT a Sphinx
        # flag. Pass it to parrot-ue4-* in _ue4_argv. Verified live on
        # the SDC host: when given to sphinx, it gets forwarded to
        # Gazebo, which expects an XML file and emits "Failed to load
        # XML file: <path>" before continuing without the meshes.
        return argv, {}

    def _ue4_argv(
        self,
        world_binary: str,
        instance_id: int,
        endpoint: DroneEndpoint | None,
        config_file: str | None = None,
        res_x: int | None = None,
        res_y: int | None = None,
        extra_args: list[str] | None = None,
    ) -> tuple[list[str], dict[str, str]]:
        """Build the parrot-ue4-<world> argv.

        ``-config-file`` is a UE4 application command-line option (per
        Parrot's docs). When supplied, the UE4 binary loads the YAML
        and instantiates ``Meshes:`` entries before the world starts —
        this is how custom geometry like the SDC arena gets injected
        without building a custom UE app.
        """
        argv: list[str] = [world_binary]
        # Parrot's UE4 binaries accept -ResX/-ResY/-Windowed for size.
        # Resolution is configurable per env-start request; defaults
        # come from launcher.config.ue4.res_x/res_y, falling back to
        # 1280x1024 (operator's preferred default for the SDC arena).
        ue4_cfg = self.config.get("ue4", {}) or {}
        rx = int(res_x or ue4_cfg.get("res_x") or 1280)
        ry = int(res_y or ue4_cfg.get("res_y") or 1024)
        argv.extend([f"-ResX={rx}", f"-ResY={ry}", "-Windowed"])
        # `-Port` only relevant in multi-instance mode. When started
        # by `start_environment` we have no endpoint (env-only spawn),
        # and the single-instance default port is fine.
        if endpoint is not None and endpoint.port is not None:
            argv.append(f"-Port={endpoint.port}")
        if config_file:
            argv.append(f"-config-file={config_file}")
        if extra_args:
            argv.extend(extra_args)
        return argv, {}

    def _hide_ue4_panels(self, ue4_pid: int | None = None) -> None:
        """Send F10 to the UE4 window after launch so only the 3D
        viewport is visible (Sphinx's HMI overlay panels — cameras list,
        details panel, sub-windows — are toggled off via F10 per the
        UE4 app HMI docs). Best-effort: requires xdotool to be
        installed and the X DISPLAY environment to be reachable.

        Reads DISPLAY + XAUTHORITY directly from the UE4 process's
        /proc/<pid>/environ instead of relying on the launcher's own
        session-detect — guarantees we connect to the same X server
        the UE4 window is on, which is the only reliable way under
        gdm/Wayland-X11 mixed sessions where session-detect produces
        an empty DISPLAY."""
        if self.dry_run:
            return
        if not shutil.which("xdotool"):
            log.warning("xdotool not installed — can't auto-hide UE panels")
            return
        env = dict(os.environ)
        # Override DISPLAY/XAUTHORITY from the UE4 process if available.
        if ue4_pid:
            try:
                with open(f"/proc/{ue4_pid}/environ", "rb") as f:
                    raw = f.read()
                for kv in raw.split(b"\0"):
                    if kv.startswith(b"DISPLAY="):
                        env["DISPLAY"] = kv[len(b"DISPLAY="):].decode()
                    elif kv.startswith(b"XAUTHORITY="):
                        env["XAUTHORITY"] = kv[len(b"XAUTHORITY="):].decode()
            except (FileNotFoundError, PermissionError) as e:
                log.warning("can't read /proc/%d/environ for DISPLAY: %s",
                            ue4_pid, e)
        # Poll for the UE4 window — Vulkan + assets loading can take
        # 5–15 s on first start so a fixed-sleep approach was racy. Try
        # for up to 30s, sending F10 the moment the window appears.
        deadline = time.time() + 30.0
        last_search_rc = None
        while time.time() < deadline:
            try:
                # First, just search — don't chain the key send. That
                # way we know exactly when the window is up before we
                # try to send the key.
                find = subprocess.run(
                    ["xdotool", "search", "--name", "Development"],
                    capture_output=True, text=True, timeout=3, env=env,
                )
                last_search_rc = find.returncode
                if find.returncode == 0 and find.stdout.strip():
                    wids = find.stdout.strip().splitlines()
                    # Now send F10 to each matched window.
                    send_ok = False
                    for wid in wids:
                        s = subprocess.run(
                            ["xdotool", "key", "--window", wid,
                             "--clearmodifiers", "F10"],
                            capture_output=True, text=True, timeout=3,
                            env=env,
                        )
                        if s.returncode == 0:
                            send_ok = True
                    if send_ok:
                        log.info(
                            "sent F10 to UE4 window(s) %s — panels hidden "
                            "(DISPLAY=%s)", wids, env.get("DISPLAY", "?"),
                        )
                        return
                time.sleep(1.0)
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                log.warning("xdotool poll failed: %s", e)
                return
        log.warning(
            "UE4 window not found by xdotool within 30s "
            "(last search rc=%s, DISPLAY=%s) — panels NOT hidden, "
            "press F10 manually in the UE4 window if needed.",
            last_search_rc, env.get("DISPLAY", "?"),
        )

    def _terminate_pid(self, pid: int, sigterm_grace_s: float = 1.5) -> None:
        """Kill the process group containing ``pid``: send SIGTERM, wait
        ``sigterm_grace_s`` seconds for graceful exit, then SIGKILL.

        Default grace was 5 seconds — too long for UI feedback when
        users click 'stop FC' or 'restart drone' (the unified_api_server
        Olympe-based FC sometimes hangs in atexit handlers and always
        needed the SIGKILL fallback). 1.5 s catches the common quick
        exit while still letting Flask flush logs; anything stuck
        longer than that is unlikely to clean up by waiting more.
        """
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.time() + float(sigterm_grace_s)
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return

    def _safe_teardown(self, instance_id: int) -> None:
        try:
            self.network.teardown(instance_id)
        except Exception as e:
            log.warning("teardown of instance %d failed: %s", instance_id, e)

    def _is_alive(self, rec: DroneRecord) -> bool:
        for pid in (rec.sphinx_pid, rec.ue4_pid):
            if not pid:
                continue
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                # The pid exists, we just don't own it (rare in our
                # context). Treat as alive.
                return True
        return rec.status == "running" and rec.sphinx_pid is not None
