"""C2 V2 entry point — Phase 1.

Starts an asyncio loop (background thread) that owns the FC pool polling the five
real flight controllers, and a Flask dashboard (main thread) for the read-only
live view.

Run:
    python -m c2_v2                       # connect to flightctrl1-5:8080
    python -m c2_v2 --fcs flightctrl2     # subset for bench testing
    python -m c2_v2 --port 8100
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

from .config import default_fcs, fcs_from_hosts
from .pool import FCPool
from .web import build_app

logger = logging.getLogger("c2_v2")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="c2_v2")
    p.add_argument("--host", default="0.0.0.0", help="dashboard bind host")
    p.add_argument("--port", type=int, default=8100, help="dashboard bind port")
    p.add_argument("--fcs", default=None,
                   help="comma-separated FC hosts (default flightctrl1-5); "
                        "each may carry :port, default 8080")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


class _AsyncLoopThread(threading.Thread):
    """Owns an asyncio loop on a background thread (the pool lives here)."""

    def __init__(self) -> None:
        super().__init__(name="c2v2-asyncio", daemon=True)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()

    def run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        self.loop.run_forever()

    def wait_ready(self, timeout: float = 5.0) -> None:
        self._ready.wait(timeout)

    def stop(self) -> None:
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    specs = fcs_from_hosts(args.fcs.split(",")) if args.fcs else default_fcs()
    pool = FCPool(specs)

    th = _AsyncLoopThread()
    th.start()
    th.wait_ready()
    assert th.loop is not None
    asyncio.run_coroutine_threadsafe(pool.start(), th.loop).result(5.0)

    flask_app = build_app(pool)

    def _shutdown(signum, frame):  # noqa: ARG001
        logger.info("c2_v2: shutting down (signal %s)", signum)
        try:
            asyncio.run_coroutine_threadsafe(pool.stop(), th.loop).result(2.0)
        except Exception:
            pass
        th.stop()
        os._exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info("c2_v2: dashboard on http://%s:%d  (FCs: %s)",
                args.host, args.port, ", ".join(s.name for s in specs))
    flask_app.run(host=args.host, port=args.port,
                  threaded=True, use_reloader=False, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
