"""Server-side heartbeat loop. Pings every reachable Pi's /api/heartbeat
at a steady interval so the Pi's watchdog (which auto-lands after
``REMOTE_TIMEOUT_S`` of silence) stays satisfied — without forcing the
browser to spend any of its connection pool on heartbeats."""
from __future__ import annotations

import threading
import time


_HEARTBEAT_INTERVAL_S: float = 1.0


def _loop(drones: dict, drone_ws: dict, http_session) -> None:
    """Daemon-thread loop. Per-drone failures are swallowed — the Pi
    watchdog only needs SOME successful heartbeat in its window."""
    while True:
        try:
            for did, info in drones.items():
                base = (info or {}).get("base")
                if not base:
                    continue
                cli = drone_ws.get(str(did))
                if cli is not None:
                    all_down = (not cli._ws_connected.get("telemetry")
                                and not cli._ws_connected.get("position")
                                and not cli._ws_connected.get("rc"))
                    if all_down:
                        continue
                try:
                    http_session.get(f"{base.rstrip('/')}/api/heartbeat",
                                     timeout=0.4)
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(_HEARTBEAT_INTERVAL_S)


def start(drones: dict, drone_ws: dict, http_session) -> None:
    """Spawn the daemon thread once. Takes its dependencies as arguments
    so the entrypoint stays the single owner of the fleet/session
    state."""
    threading.Thread(target=_loop, args=(drones, drone_ws, http_session),
                     daemon=True, name="heartbeat-loop").start()
    print(f"[HEARTBEAT] background loop started "
          f"({_HEARTBEAT_INTERVAL_S:.1f}s interval)")
