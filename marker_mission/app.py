"""Combined entry point: one Flask app serving the unified drone REST
API AND the marker_mission operator UI from a single process.

Usage:
    python -m marker_mission.app [mission-args...]

What this does that ``python -m marker_mission.mission`` (legacy) and
``python controller_unified/unified_api_server.py`` (legacy) do not:

* Imports ``controller_unified.unified_api_server`` so its Flask
  ``app`` and 81 routes load into this process.
* Calls ``unified_api_server.init_backend_and_threads()`` to bring up
  the Olympe/Tello backend + the 5 supporting threads (telemetry,
  reconnect, RC, watchdog, drone-ping, positioning).
* Hands the same Flask ``app`` to :class:`marker_mission.ui.UiServer`
  via ``external_app=...`` so the UI's 55+ routes register on it too.
* Wires :mod:`marker_mission.mission` to use :class:`DroneApiInProc`
  instead of HTTP, so RC ticks / telemetry / takeoff / etc go through
  direct function calls into ``drone_core`` — no JSON, no sockets.
* Binds the combined app to ``--ui-port`` (default 8080, the mission's
  current port). Externally, the combined app is just "the mission
  UI" — but it also serves every endpoint of the old unified server.

The legacy entry points are kept and mutually exclusive:

* ``python controller_unified/unified_api_server.py`` — unified only,
  HTTP on 5050. Use when marker_mission is on a different host.
* ``python -m marker_mission.app`` — combined, HTTP on 8080. Use when
  marker_mission and the drone backend share a host (the production
  flightctrl deployment).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make ``import unified_api_server`` and ``import drone_core`` work as
# sibling top-level modules. The launcher used to do this by setting
# cwd; doing it here means the combined entry point can be launched
# from anywhere (e.g. `python -m marker_mission.app` from repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CTL_DIR = _REPO_ROOT / "controller_unified"
if str(_CTL_DIR) not in sys.path:
    sys.path.insert(0, str(_CTL_DIR))

import unified_api_server  # noqa: E402 — sys.path tweak above must precede the import


def _serve_combined_app(host: str, port: int) -> None:
    """Run the combined Flask app on the mission's port. Spawned on a
    daemon thread by :func:`main` so mission setup + flight loop can
    proceed on the main thread; the process exits when the mission
    returns and the daemon is torn down.

    Routes registered after this thread starts (e.g. UiServer's
    ``_register_routes`` running inside mission setup) still serve
    correctly because Flask's url_map mutation under ``debug=False``
    is lookup-time, not register-time."""
    import logging as _log
    _log.getLogger("werkzeug").setLevel(_log.WARNING)
    unified_api_server.app.run(host=host, port=port,
                               threaded=True, use_reloader=False)


def main() -> int:
    """Combined-mode entry point. Brings up the drone backend, runs
    one Flask app serving both unified routes and the marker_mission
    UI on the mission's port, then runs the mission flight loop.

    Mission-specific CLI args (``--ui-port``, ``--api-url``, etc.) are
    parsed by :func:`marker_mission.mission.main`. We force the
    ``--in-process`` flag through to mission so DroneApiInProc and
    InProcMjpegReader are used and UiServer attaches its routes to the
    combined app. Standalone HTTP-only deployments should keep using
    ``python controller_unified/unified_api_server.py`` instead.
    """
    import threading as _t

    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    # Step 1: bring up the drone backend + its background threads.
    # If this fails (e.g. neither djitellopy nor olympe installed),
    # bail out before mission tries to talk to anything.
    if not unified_api_server.init_backend_and_threads():
        print("[app] unified_api_server init failed — aborting.",
              file=sys.stderr)
        return 1

    # Step 2: import mission late so the unified backend is fully
    # initialised before MissionConfig.load() triggers anything that
    # might look it up. mission.main parses sys.argv; we splice
    # --in-process in so cmd_fly picks DroneApiInProc + external_app
    # without the operator having to remember the flag.
    from marker_mission.config import MissionConfig
    from marker_mission.mission import main as mission_main

    cfg = MissionConfig.load()
    argv = list(sys.argv[1:])
    if not argv or argv[0] not in ("fly", "calibrate"):
        argv = ["fly"] + argv
    if argv[0] == "fly" and "--in-process" not in argv:
        argv = [argv[0], "--in-process"] + argv[1:]

    # Step 3: spawn the combined Flask app on a daemon thread so the
    # mission setup (which registers UI routes on the same app) and
    # the flight loop run on the main thread. We resolve port now
    # rather than after argv parsing — for the rare case the operator
    # passes --ui-port, the daemon may have started on the default
    # port. Acceptable: --ui-port from CLI gets through to mission
    # but the Flask app stays on cfg.ui_port. If you need a custom
    # port, set MM_UI_PORT in the environment.
    host = cfg.ui_host
    port = cfg.ui_port
    print(f"[app] Combined Flask app on {host}:{port} "
          f"(unified routes + marker_mission UI)")
    _t.Thread(target=_serve_combined_app, args=(host, port),
              daemon=True, name="combined-flask").start()

    return mission_main(argv)


if __name__ == "__main__":
    sys.exit(main())
