"""HTTP server + web dashboard for the SDC26 C2 strategy server.

Exposes the live :class:`~c2.models.WorldSnapshot` and a manual command
channel over a small FastAPI app, and serves a dependency-free operator
console from ``c2/web/``.

The orchestration lives in :class:`c2.commander.Commander`; this module only
reads its snapshots and forwards operator commands. The server owns the
Commander lifecycle via FastAPI startup/shutdown events.

Endpoints
---------
- ``GET  /``                 -> web/index.html (the operator console)
- ``GET  /web/*``            -> static dashboard assets
- ``GET  /api/state``        -> latest snapshot as JSON
- ``GET  /api/state/stream`` -> Server-Sent Events, ~4 Hz snapshot pushes
- ``POST /api/command``      -> Command.from_dict(body) -> commander.submit(cmd)
- ``POST /api/mode``         -> {"mode": "manual"|"auto"} -> commander.set_mode(...)

Run with ``python -m c2 --config config.yaml`` (see ``c2/__main__.py``)
or ``c2.server.main()``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from .config import C2Config
from .commander import Commander
from .models import Command, GameMode

log = logging.getLogger("c2.server")

WEB_DIR = Path(__file__).resolve().parent / "web"

# Snapshot stream cadence (seconds). The task asks for ~0.25-0.5 s; the
# commander loop runs at tick_hz (default 4 Hz) so 0.25 s keeps the UI in
# step without hammering the event loop.
STREAM_INTERVAL_S = 0.25
# Emit a comment heartbeat if no data has gone out for this long so proxies
# and the browser keep the connection alive even when snapshots stall.
HEARTBEAT_S = 10.0


def _snapshot_json(commander: Commander) -> str:
    """Serialise the current world snapshot to a compact JSON string."""
    return json.dumps(commander.snapshot().to_dict(), separators=(",", ":"))


def create_app(cfg: C2Config, commander: Commander) -> FastAPI:
    """Build the FastAPI app bound to *cfg* and *commander*.

    The Commander is started/stopped with the app's lifecycle so a single
    ``uvicorn.run(app, ...)`` brings the whole C2 up and tears it down
    cleanly on SIGINT.
    """
    app = FastAPI(title="SDC26 C2", version="0.1.0")

    # Keep references on the app so handlers (and tests) can reach them.
    app.state.cfg = cfg
    app.state.commander = commander

    # ------------------------------------------------------------------
    # Lifecycle: own the commander loop.
    # ------------------------------------------------------------------
    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - exercised at runtime
        log.info("C2 server starting; launching commander")
        await commander.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - exercised at runtime
        log.info("C2 server stopping; halting commander")
        await commander.stop()

    # ------------------------------------------------------------------
    # Static dashboard.
    # ------------------------------------------------------------------
    if WEB_DIR.is_dir():
        app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> JSONResponse:
        return JSONResponse({"ok": True})

    # ------------------------------------------------------------------
    # State.
    # ------------------------------------------------------------------
    @app.get("/api/state")
    async def api_state() -> JSONResponse:
        return JSONResponse(commander.snapshot().to_dict())

    @app.get("/api/state/stream")
    async def api_state_stream(request: Request) -> StreamingResponse:
        async def gen():
            # Prime the stream so the client renders immediately instead of
            # waiting a full tick for the first push.
            last = _snapshot_json(commander)
            yield f"data: {last}\n\n"
            idle = 0.0
            while True:
                if await request.is_disconnected():
                    break
                await asyncio.sleep(STREAM_INTERVAL_S)
                try:
                    payload = _snapshot_json(commander)
                except Exception as exc:  # snapshot should never raise, but be safe
                    log.warning("snapshot serialise failed: %s", exc)
                    payload = None

                if payload is not None and payload != last:
                    last = payload
                    idle = 0.0
                    yield f"data: {payload}\n\n"
                else:
                    # Force a refresh roughly every second even if the
                    # snapshot is byte-identical so age_s counters keep
                    # ticking client-side, and heartbeat if truly idle.
                    idle += STREAM_INTERVAL_S
                    if payload is not None and idle >= 1.0:
                        idle = 0.0
                        yield f"data: {payload}\n\n"
                    elif idle >= HEARTBEAT_S:
                        idle = 0.0
                        yield ": keep-alive\n\n"

        headers = {
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
        }
        return StreamingResponse(
            gen(), media_type="text/event-stream", headers=headers
        )

    # ------------------------------------------------------------------
    # Commands.
    # ------------------------------------------------------------------
    @app.post("/api/command")
    async def api_command(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"ok": False, "error": "invalid JSON body"}, status_code=400
            )
        if not isinstance(body, dict):
            return JSONResponse(
                {"ok": False, "error": "command body must be an object"},
                status_code=400,
            )
        try:
            cmd = Command.from_dict(body)
        except (KeyError, ValueError, TypeError) as exc:
            return JSONResponse(
                {"ok": False, "error": f"bad command: {exc}"}, status_code=400
            )
        try:
            result = commander.submit(cmd)
        except Exception as exc:  # commander rejected / blew up
            log.exception("commander.submit failed")
            return JSONResponse(
                {"ok": False, "error": f"submit failed: {exc}"}, status_code=400
            )
        if not isinstance(result, dict):
            result = {"ok": bool(result)}
        return JSONResponse(result)

    @app.post("/api/mode")
    async def api_mode(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"ok": False, "error": "invalid JSON body"}, status_code=400
            )
        mode_val = body.get("mode") if isinstance(body, dict) else None
        try:
            mode = GameMode(mode_val)
        except ValueError:
            return JSONResponse(
                {"ok": False, "error": f"unknown mode: {mode_val!r}"},
                status_code=400,
            )
        commander.set_mode(mode)
        return JSONResponse({"ok": True, "mode": mode.value})

    return app


# ---------------------------------------------------------------------------
# Entry point (used by c2/__main__.py).
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="c2",
        description="SDC26 C2 strategy server + web dashboard",
    )
    p.add_argument(
        "--config",
        default=None,
        help="path to a C2 YAML config (see c2/config.example.yaml); "
        "omit to use built-in defaults",
    )
    p.add_argument("--host", default=None, help="override the bind host")
    p.add_argument("--port", type=int, default=None, help="override the bind port")
    p.add_argument(
        "--log-level",
        default="info",
        help="uvicorn/log level (debug, info, warning, error)",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    """Console entry: load config, build the app, run uvicorn."""
    import uvicorn

    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    cfg = C2Config.load(args.config)
    if args.host is not None:
        cfg.host = args.host
    if args.port is not None:
        cfg.port = args.port

    commander = Commander(cfg)
    app = create_app(cfg, commander)

    log.info(
        "Serving C2 dashboard on http://%s:%d  (team=%s, mode=%s, drones=%d)",
        cfg.host,
        cfg.port,
        cfg.our_team.value,
        "auto" if cfg.mode_start_auto else "manual",
        len(cfg.drones),
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=args.log_level)


if __name__ == "__main__":  # pragma: no cover
    main()
