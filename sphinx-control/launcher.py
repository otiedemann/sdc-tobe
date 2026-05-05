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
                ue4_pid = self._launch_ue4(env_id=env_id, world=world)
            except Exception as e:
                self.state.update_env_status(env_id, "error", last_error=str(e))
                raise

            self.state.update_env_status(env_id, "running", ue4_pid=ue4_pid)
            updated = self.state.get_env(env_id)
            assert updated is not None
            log.info(
                "started env %s world=%s ue4_pid=%s",
                env_id, req.world_name, ue4_pid,
            )
            return updated

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

            # Pre-flight: refuse to start if something's already on the
            # FC's HTTP port. Without this the subprocess would launch
            # and silently fail to bind, leaving the launcher reporting
            # "running" while the actual python died 100 ms later.
            if not self.dry_run and _port_in_use(int(http_port)):
                raise RuntimeError(
                    f"port {http_port} already in use — another FC may be "
                    f"running outside sphinx-control. `sudo ss -tlnp 'sport "
                    f"= :{http_port}'` to find it."
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
        with self._lock:
            fc = self.state.current_fc()
            if fc is None:
                return
            if fc.pid:
                self._terminate_pid(fc.pid)
            self.state.update_fc_status(fc.fc_id, "stopped")
            log.info("stopped fc %s", fc.fc_id)

    def restart_fc(self) -> FlightControllerRecord:
        with self._lock:
            fc = self.state.current_fc()
            if fc is None:
                raise KeyError("no flight controller running")
            req = FCRequest(
                cwd=fc.cwd, script=fc.script,
                http_port=fc.http_port, anafi_ip=fc.anafi_ip,
            )
            self.stop_fc()
            self.state.delete_fc(fc.fc_id)
            time.sleep(0.5)
            return self.start_fc(req)

    def current_flight_controller(self) -> FlightControllerRecord | None:
        fc = self.state.current_fc()
        if fc is None:
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
        """Stop the drone and remove its row from state."""
        with self._lock:
            rec = self.state.get(drone_id)
            if rec is None:
                return
            try:
                self.stop(drone_id)
            except KeyError:
                pass
            self.state.delete(drone_id)
            log.info("deleted drone %s", drone_id)

    def restart(self, drone_id: str) -> DroneRecord:
        """Stop and respawn with the same parameters and instance_id.

        The drone_id changes (a fresh UUID is minted) but the instance_id
        stays the same, so external clients tracking ``host:port``
        keep working seamlessly. We delete the old stopped row before
        respawning so the unique instance_id constraint doesn't fire.
        """
        with self._lock:
            rec = self.state.get(drone_id)
            if rec is None:
                raise KeyError(drone_id)
            self.stop(drone_id)
            # Drop the stopped row first; otherwise the new row's
            # instance_id collides with it on INSERT.
            self.state.delete(drone_id)
            # Tiny pause to let the kernel release the port and netns
            # before we try to bind again.
            time.sleep(1.0)
            return self.spawn(LaunchRequest(
                drone_profile=rec.drone_type,
                world_name=rec.world_app,
                instance_id=rec.instance_id,
                firmware_url=rec.firmware_url,
            ))

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

    def _launch_ue4(self, env_id: str, world) -> int:
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
        # Resolution is intentionally generous so the operator can see
        # the arena clearly via Sunshine — UE4 defaults to a small
        # window otherwise.
        argv.extend(["-ResX=1280", "-ResY=720", "-Windowed"])
        # `-Port` only relevant in multi-instance mode. When started
        # by `start_environment` we have no endpoint (env-only spawn),
        # and the single-instance default port is fine.
        if endpoint is not None and endpoint.port is not None:
            argv.append(f"-Port={endpoint.port}")
        if config_file:
            argv.append(f"-config-file={config_file}")
        return argv, {}

    def _terminate_pid(self, pid: int) -> None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
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
