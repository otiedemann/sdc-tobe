"""Strategy app entrypoint.

Spins up:
  - An asyncio loop in a background thread (owns the SwarmRunner + C2Client).
  - A Flask app on port 8091 by default (configurable via CLI).

Run via ``python -m marker_mission_c2.strategy`` (see ``__main__``).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
from typing import Optional

from .c2_client import C2Client
from .markers import MarkerTracker
from .runner import SwarmRunner
from .settings import SettingsStore
from .web import build_app

logger = logging.getLogger(__name__)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="marker_mission_c2.strategy")
    p.add_argument("--host", default="0.0.0.0",
                   help="Flask bind host (default 0.0.0.0)")
    p.add_argument("--port", type=int, default=8091,
                   help="Flask bind port (default 8091)")
    p.add_argument("--c2", default=os.environ.get("STRATEGY_C2", "http://127.0.0.1:8090"),
                   help="C2 server base URL (default http://127.0.0.1:8090)")
    p.add_argument("--settings", default=None,
                   help="Path to settings.json (default: alongside this package)")
    p.add_argument("--fc-names", default=None,
                   help="Comma-separated FC names for bootstrap "
                        "(e.g. 'flightctrl1,flightctrl2,flightctrl3')")
    p.add_argument("--config", default=None,
                   help="Path to marker_mission_c2 config JSON to extract FC "
                        "names from (used only when --fc-names is not set)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _fc_names_from_config(path: str) -> list[str]:
    """Extract the list of fc names from a marker_mission_c2 config JSON.

    Quiet on error: the strategy can still bootstrap once the C2 overview is
    reachable; this is just a nice-to-have for fresh installs.
    """
    import json
    try:
        with open(path, "r") as f:
            raw = json.load(f)
        out: list[str] = []
        for spec in raw.get("fcs") or []:
            name = (spec or {}).get("name")
            if name:
                out.append(str(name))
        return out
    except (OSError, ValueError) as e:
        logger.warning("strategy: could not read FC names from %s: %s", path, e)
        return []


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class _AsyncLoopThread(threading.Thread):
    """Owns an asyncio loop running on a background thread."""

    def __init__(self) -> None:
        super().__init__(name="strategy-asyncio", daemon=True)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()

    def run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        try:
            self.loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(loop=self.loop)
                for t in pending:
                    t.cancel()
                self.loop.run_until_complete(asyncio.sleep(0.1))
            except Exception:
                pass
            self.loop.close()

    def wait_ready(self, timeout: float = 5.0) -> None:
        self._ready.wait(timeout)

    def stop(self) -> None:
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log_level)

    fc_names: Optional[list[str]] = None
    if args.fc_names:
        fc_names = [s.strip() for s in args.fc_names.split(",") if s.strip()]
    elif args.config:
        fc_names = _fc_names_from_config(args.config) or None
        if fc_names:
            logger.info("strategy: bootstrapped FC names from %s: %s",
                        args.config, fc_names)

    settings = SettingsStore(path=args.settings, fc_names=fc_names)
    markers = MarkerTracker(
        red_live_ids=settings.snapshot().markers.red_live_ids,
        blue_live_ids=settings.snapshot().markers.blue_live_ids,
    )

    # Start the async loop on a background thread; build runner + client on it.
    th = _AsyncLoopThread()
    th.start()
    th.wait_ready()
    assert th.loop is not None

    async def _bootstrap():
        c2 = C2Client(base_url=args.c2)
        runner = SwarmRunner(settings=settings, c2=c2, markers=markers)
        await runner.start()
        return c2, runner

    fut = asyncio.run_coroutine_threadsafe(_bootstrap(), th.loop)
    c2, runner = fut.result()

    flask_app = build_app(
        settings=settings, runner=runner, markers=markers, loop=th.loop
    )

    # Graceful shutdown on SIGTERM/SIGINT.
    def _shutdown(signum, frame):  # noqa: ARG001
        logger.info("strategy: shutting down (signal %s)", signum)
        try:
            asyncio.run_coroutine_threadsafe(runner.stop(), th.loop).result(2.0)
        except Exception:
            pass
        try:
            asyncio.run_coroutine_threadsafe(c2.aclose(), th.loop).result(2.0)
        except Exception:
            pass
        th.stop()
        # Hard exit; Flask's dev server otherwise hangs on reload threads.
        os._exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info(
        "strategy: serving on http://%s:%d  (C2: %s)",
        args.host, args.port, args.c2,
    )
    flask_app.run(host=args.host, port=args.port,
                  threaded=True, use_reloader=False, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
