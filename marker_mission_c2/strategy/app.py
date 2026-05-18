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
from typing import Any, Dict, Optional

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
    p.add_argument("--c2", default=os.environ.get("STRATEGY_C2"),
                   help="C2 server base URL. If omitted, derived from "
                        "the --config bind block (falls back to "
                        "http://127.0.0.1:8090)")
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


def _read_c2_config(path: str) -> Dict[str, Any]:
    """Load the marker_mission_c2 config JSON. Quiet on error."""
    import json
    try:
        with open(path, "r") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError) as e:
        logger.warning("strategy: could not read C2 config %s: %s", path, e)
        return {}


def _fc_names_from_config(path: str) -> list[str]:
    """Extract the list of fc names from a marker_mission_c2 config JSON."""
    raw = _read_c2_config(path)
    out: list[str] = []
    for spec in raw.get("fcs") or []:
        name = (spec or {}).get("name")
        if name:
            out.append(str(name))
    return out


def _c2_url_from_config(path: str) -> Optional[str]:
    """Compose the C2 base URL from its config's ``bind`` block.

    The C2 server may bind to a specific IP (e.g. the Tailscale address)
    rather than 0.0.0.0/127.0.0.1, so naive defaults like ``--c2 http://127.0.0.1``
    can fail in production. Pull the host:port the C2 actually listens on
    and use that.
    """
    raw = _read_c2_config(path)
    bind = raw.get("bind") or {}
    host = bind.get("host")
    port = bind.get("port")
    if not host or not port:
        return None
    # If the C2 is bound to 0.0.0.0 we still want to talk to ourselves —
    # rewrite to loopback so we don't depend on the public route.
    if str(host) == "0.0.0.0":
        host = "127.0.0.1"
    try:
        return f"http://{host}:{int(port)}"
    except (TypeError, ValueError):
        return None


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

    # Resolve --c2: explicit flag wins, otherwise derive from --config,
    # otherwise fall back to localhost. This matters because the C2 may
    # bind to a specific IP (e.g. its Tailscale address) rather than
    # 0.0.0.0, in which case 127.0.0.1 would never connect.
    c2_url = args.c2
    if not c2_url and args.config:
        c2_url = _c2_url_from_config(args.config)
        if c2_url:
            logger.info("strategy: C2 URL bootstrapped from %s: %s",
                        args.config, c2_url)
    if not c2_url:
        c2_url = "http://127.0.0.1:8090"

    settings = SettingsStore(path=args.settings, fc_names=fc_names)
    markers = MarkerTracker(
        active_slots=settings.snapshot().markers.active_slots,
    )

    # Start the async loop on a background thread; build runner + client on it.
    th = _AsyncLoopThread()
    th.start()
    th.wait_ready()
    assert th.loop is not None

    async def _bootstrap():
        c2 = C2Client(base_url=c2_url)
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
        args.host, args.port, c2_url,
    )
    flask_app.run(host=args.host, port=args.port,
                  threaded=True, use_reloader=False, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
