"""Simulator launcher.

One process that:
  * builds the shared World from a sim config,
  * runs a physics thread ticking the world at ``sim_hz``,
  * serves each simulated drone's marker_mission FC API on its own port
    (so the C2 config lists them like real flight controllers),
  * serves the 3D web view on ``ui_port``.

Run: ``python -m marker_mission_sim --config marker_mission_sim/sim_config.example.json``
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from typing import Optional

from werkzeug.serving import make_server

from .config import SimConfig
from .world import World
from .fc_api import make_fc_app
from .ui import make_ui_app

logger = logging.getLogger("marker_mission_sim")


class _ServerThread(threading.Thread):
    """Run a WSGI app on its own port in a daemon thread."""

    def __init__(self, app, host: str, port: int, label: str) -> None:
        super().__init__(name=f"sim-{label}", daemon=True)
        self._srv = make_server(host, port, app, threaded=True)
        self.host = host
        self.port = port
        self.label = label

    def run(self) -> None:
        self._srv.serve_forever()

    def shutdown(self) -> None:
        try:
            self._srv.shutdown()
        except Exception:
            pass


class PhysicsThread(threading.Thread):
    """Ticks the world at a fixed rate."""

    def __init__(self, world: World, hz: float) -> None:
        super().__init__(name="sim-physics", daemon=True)
        self.world = world
        self.dt = 1.0 / max(1.0, hz)
        self._stop = threading.Event()

    def run(self) -> None:
        last = time.time()
        while not self._stop.is_set():
            now = time.time()
            dt = max(0.0, now - last)
            last = now
            try:
                self.world.tick(dt, now)
            except Exception:
                logger.exception("physics tick crashed (continuing)")
            elapsed = time.time() - now
            time.sleep(max(0.0, self.dt - elapsed))

    def stop(self) -> None:
        self._stop.set()


def run(cfg: SimConfig) -> int:
    world = World(cfg)

    physics = PhysicsThread(world, cfg.sim_hz)
    physics.start()

    servers: list[_ServerThread] = []
    # One FC API per drone, each on its own port.
    for dc in cfg.drones:
        app = make_fc_app(world, dc.id, cfg)
        st = _ServerThread(app, cfg.fc_host, dc.fc_port, f"fc-{dc.id}")
        st.start()
        servers.append(st)
        logger.info("drone %-8s (%s) FC API → http://%s:%d  spawn=(%.1f,%.1f) hdg=%.0f",
                    dc.id, dc.team, cfg.fc_host, dc.fc_port,
                    dc.spawn_x, dc.spawn_y, dc.spawn_heading_deg)

    # The 3D UI.
    ui_app = make_ui_app(world, cfg)
    ui = _ServerThread(ui_app, cfg.ui_host, cfg.ui_port, "ui")
    ui.start()
    servers.append(ui)
    logger.info("3D arena view → http://%s:%d", cfg.ui_host, cfg.ui_port)

    stop_evt = threading.Event()

    def _shutdown(signum, frame):  # noqa: ARG001
        logger.info("shutting down (signal %s)", signum)
        stop_evt.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("simulator up: %d drone(s), %.0f Hz physics. Ctrl-C to stop.",
                len(cfg.drones), cfg.sim_hz)
    try:
        while not stop_evt.is_set():
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        physics.stop()
        for st in servers:
            st.shutdown()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="marker_mission_sim",
                                 description="Local marker_mission drone+arena simulator.")
    ap.add_argument("--config", "-c", default=None,
                    help="Path to sim config JSON (see sim_config.example.json). "
                         "If omitted, a single default drone is used.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = SimConfig.load(args.config)
    if not cfg.drones:
        from .config import SimDroneConfig
        cfg.drones = [SimDroneConfig("sim1", "red", 9101, spawn_x=0.0, spawn_y=-9.0)]
        logger.warning("no drones in config — using one default drone sim1 on :9101")
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
