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
from state import DroneRecord, StateStore
from worlds import Registry

log = logging.getLogger("sphinx-control.launcher")


@dataclass
class LaunchRequest:
    drone_profile: str
    world_name: str
    instance_id: int | None = None  # auto-allocated if None
    firmware_url: str | None = None  # falls back to config default


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

    def spawn(self, req: LaunchRequest) -> DroneRecord:
        with self._lock:
            profile = self.registry.profile(req.drone_profile)
            world = self.registry.world(req.world_name)
            # In dry-run mode the descriptor / UE4 binary don't have to
            # exist — we'll spawn `sleep` placeholders. Outside dry-run
            # they're real CLI args to subprocess and must be present.
            if not self.dry_run and not profile.available:
                raise RuntimeError(f"profile unavailable: {profile.unavailable_reason}")
            if not self.dry_run and not world.available:
                raise RuntimeError(f"world unavailable: {world.unavailable_reason}")

            instance_id = req.instance_id or self._allocate_instance_id()
            if instance_id < 1 or instance_id > self.max_drones:
                raise ValueError(
                    f"instance_id {instance_id} outside 1..{self.max_drones}"
                )
            if instance_id in self.state.used_instance_ids():
                # Tailored message for the one-slot host (the common case
                # for Sphinx 2 on a single PC). Helps the dashboard alert
                # reflect reality instead of suggesting "use a different
                # instance_id" which has no meaning here.
                if self.max_drones == 1:
                    raise RuntimeError(
                        "another environment is already running on this host; "
                        "stop it before starting a new one. Sphinx 2 only "
                        "supports one drone per host."
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
                world_app=world.name,
                config_file=world.config_file,
                firmware_url=firmware,
                drone_ip=endpoint.ip,
                drone_port=endpoint.port,
                status="spawning",
            )
            # Clean up any STOPPED row holding this instance_id from a
            # previous run. UNIQUE(instance_id) would otherwise reject
            # the INSERT below. Restart() already does this for the
            # running-then-stopped case; this catches the case where
            # the management service was restarted (reconcile marks
            # all running rows stopped) and the user spawns again.
            removed = self.state.delete_stopped_at_instance_id(instance_id)
            if removed:
                log.info(
                    "released %d stopped slot(s) at instance_id=%d before spawn",
                    removed, instance_id,
                )
            self.state.upsert(rec)

            try:
                sphinx_pid, ue4_pid = self._launch_subprocesses(
                    drone_id=drone_id,
                    instance_id=instance_id,
                    descriptor_path=profile.descriptor_path,
                    world=world,
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
                ue4_pid=ue4_pid,
            )
            updated = self.state.get(drone_id)
            assert updated is not None
            log.info(
                "spawned drone %s instance=%d sphinx_pid=%s ue4_pid=%s endpoint=%s",
                drone_id, instance_id, sphinx_pid, ue4_pid, endpoint.display(),
            )
            return updated

    def stop(self, drone_id: str) -> None:
        with self._lock:
            rec = self.state.get(drone_id)
            if rec is None:
                raise KeyError(drone_id)
            for pid in (rec.sphinx_pid, rec.ue4_pid):
                if pid:
                    self._terminate_pid(pid)
            self._safe_teardown(rec.instance_id)
            self.state.update_status(drone_id, status="stopped")
            log.info("stopped drone %s (instance=%d)", drone_id, rec.instance_id)

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
            r.extras["alive"] = self._is_alive(r)
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

    # ─── internals ───────────────────────────────────────────────

    def _allocate_instance_id(self) -> int:
        used = self.state.used_instance_ids()
        for i in range(1, self.max_drones + 1):
            if i not in used:
                return i
        raise RuntimeError(f"all {self.max_drones} instance slots in use")

    def _launch_subprocesses(
        self,
        drone_id: str,
        instance_id: int,
        descriptor_path: str,
        world,  # WorldEntry
        endpoint: DroneEndpoint,
        firmware_url: str | None,
    ) -> tuple[int, int]:
        """Start the sphinx + UE pair. Returns (sphinx_pid, ue4_pid).

        DRY-RUN MODE replaces both processes with a long-running
        ``sleep`` so the rest of the supervisor can be exercised
        locally.

        REAL MODE: this is the part most likely to need iteration on
        your specific Sphinx version. The exact CLI flags for "spawn
        on a specific port" / "spawn in a specific netns" change
        between Sphinx releases. The default below is a best-effort
        construction; verify it against your installed Sphinx and
        adjust ``_sphinx_argv`` if needed.
        """
        drone_log_dir = self.log_dir / drone_id
        drone_log_dir.mkdir(parents=True, exist_ok=True)
        sphinx_log = open(drone_log_dir / "sphinx.log", "ab")
        ue4_log = open(drone_log_dir / "ue4.log", "ab")

        # Session env (DISPLAY/XAUTHORITY/etc) gets spliced into the UE4
        # child whether we're in dry-run or not — this way the feature
        # is actually testable via dry-run on the real host (operator
        # can `cat /proc/<pid>/environ` to verify the injection works
        # without spawning a heavy real Sphinx).
        #
        # Re-detect on every spawn (auto mode) so a service that booted
        # before the user logged in to GNOME picks up their session
        # once it appears. Cheap: ~1 ms per spawn for a couple of
        # loginctl calls. Manual / "off" / explicit-display modes are
        # respected — only "auto" gets the live re-detect.
        if self.session_attach_setting == "auto":
            self._refresh_session()
        session_env: dict[str, str] = self.session.env if self.session else {}

        if self.dry_run:
            sleeper = ["sleep", "999999"]
            # IMPORTANT: setsid detaches the child into its own process
            # group. Without this, killpg(getpgid(child)) on stop kills
            # the management service itself, since both processes share
            # the parent's group. This applies in dry-run too — the
            # "real" branch already does it.
            sphinx_proc = subprocess.Popen(
                sleeper,
                stdout=sphinx_log,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            ue4_proc = subprocess.Popen(
                sleeper,
                stdout=ue4_log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **session_env},
                preexec_fn=os.setsid,
            )
            return sphinx_proc.pid, ue4_proc.pid

        sphinx_argv, sphinx_env = self._sphinx_argv(
            instance_id=instance_id,
            descriptor_path=descriptor_path,
            firmware_url=firmware_url,
            config_file=world.config_file,
            endpoint=endpoint,
        )
        ue4_argv, ue4_env = self._ue4_argv(
            world_binary=world.binary,
            instance_id=instance_id,
            endpoint=endpoint,
            config_file=world.config_file,
        )

        # If netns, prefix with ``ip netns exec <ns>`` so the child runs
        # inside the namespace.
        if endpoint.netns:
            sphinx_argv = ["sudo", "-n", "ip", "netns", "exec", endpoint.netns, *sphinx_argv]
            ue4_argv = ["sudo", "-n", "ip", "netns", "exec", endpoint.netns, *ue4_argv]

        # Sphinx's CLI wrapper at /usr/bin/sphinx is a bash script with
        # `set -u`, and it references $DISPLAY internally to decide
        # whether to fork a GUI helper. When this service runs under
        # systemd (no inherited DISPLAY) the sphinx script aborts with
        # "DISPLAY: unbound variable" before it ever invokes the
        # firmware sim. Splicing session_env into sphinx's env too
        # gives it the DISPLAY (and matching XAUTHORITY) so the script
        # passes its own sanity checks.
        sphinx_proc = subprocess.Popen(
            sphinx_argv,
            stdout=sphinx_log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **session_env, **sphinx_env},
            preexec_fn=os.setsid,  # own process group → killpg works
        )
        # Tiny wait so Sphinx claims its IPC sockets before UE attaches.
        time.sleep(0.5)
        # The UE4 child needs DISPLAY/WAYLAND_DISPLAY/XAUTHORITY pointing
        # at the active graphical session, otherwise it has nothing to
        # render into when sphinx-control runs as a system service. The
        # `session_env` dict (computed above) overrides whatever the
        # parent process inherited; if no session was detected, we just
        # use the parent env unchanged (existing behavior).
        ue4_proc = subprocess.Popen(
            ue4_argv,
            stdout=ue4_log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **session_env, **ue4_env},
            preexec_fn=os.setsid,
        )
        return sphinx_proc.pid, ue4_proc.pid

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
        endpoint: DroneEndpoint,
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
        # Parrot's UE4 binaries accept -ResX/-ResY/-Windowed for size,
        # and may accept a non-default port via -Port. Verify on your
        # version.
        argv.extend(["-ResX=1280", "-ResY=720", "-Windowed"])
        if endpoint.port is not None:
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
