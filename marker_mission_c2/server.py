"""Flask app + lifecycle.

Architecture:
  - One Flask app, served via Werkzeug threaded=True, bound to
    127.0.0.1.
  - A dedicated background event-loop thread (``c2-loop``) owns the
    ``httpx.AsyncClient``, the per-FC poll tasks (``FCPool``), and the
    calibration sync worker (``CalibrationLibrary``).
  - Flask handlers bridge into the loop via
    ``asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=X)``
    so handlers stay simple sync code.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Awaitable, Callable

from flask import Flask, jsonify, render_template_string, request, send_file

from .calibration_sync import CalibrationLibrary, is_valid_calibration_name
from .config import C2Config
from .fc_pool import FCPool
from .ui_pages import (
    PAGE_ARENA,
    PAGE_CALIBRATE,
    PAGE_OVERVIEW,
    PAGE_SCRIPTS,
    PAGE_TUNE,
)
from .ui_shared import common_script, head_block, tail_block, topbar


class C2Loop:
    """A background daemon thread that owns one asyncio event loop.

    Use :meth:`submit` to dispatch a coroutine and block for its result.
    The loop is kept alive for the entire process lifetime; it is shut
    down in :meth:`stop` only on Ctrl-C / process exit.
    """

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="c2-loop", daemon=True)
        self._log = logging.getLogger("c2.loop")

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            try:
                self.loop.close()
            except Exception:
                pass

    def submit(self, coro: Awaitable[Any], timeout: float = 30.0) -> Any:
        fut: Future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=timeout)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=3.0)


def _render(template: str, *, cfg: C2Config, active: str, **extra) -> str:
    fc_specs = [{"name": f.name, "host": f.host, "port": f.port}
                for f in cfg.fcs]
    ctx = dict(
        active=active,
        fc_specs=fc_specs,
        fc_names=[f.name for f in cfg.fcs],
        emergency_land_key=cfg.emergency_land_key,
        ui_refresh_ms=int(1000.0 / max(0.1, cfg.ui_refresh_hz)),
    )
    ctx.update(extra)
    return render_template_string(
        head_block() + topbar() + template + common_script() + tail_block(),
        **ctx,
    )


def _build_app(cfg: C2Config, pool: FCPool, library: CalibrationLibrary,
               loop: C2Loop) -> Flask:
    app = Flask(__name__)
    log = logging.getLogger("c2.http")

    def _await(coro: Awaitable[Any], timeout: float = 10.0) -> Any:
        return loop.submit(coro, timeout=timeout)

    # ------------------------------------------------------------ logo
    # Mirrors marker_mission's /team_logo.png — same source asset so
    # both surfaces stay visually consistent. Cached for an hour;
    # browsers only fetch the 2.8 MiB transparent PNG once per session.
    @app.get("/team_logo.png")
    def team_logo():
        repo_root = Path(__file__).resolve().parent.parent
        p = repo_root / "1_Doc" / "team_logo_transparent.png"
        if not p.is_file():
            return ("logo not found", 404)
        resp = send_file(str(p), mimetype="image/png")
        resp.headers["Cache-Control"] = "public, max-age=3600"
        return resp

    # ------------------------------------------------------------ pages
    @app.get("/")
    def page_overview():
        return _render(PAGE_OVERVIEW, cfg=cfg, active="overview")

    @app.get("/arena")
    def page_arena():
        return _render(PAGE_ARENA, cfg=cfg, active="arena")

    @app.get("/tune")
    def page_tune():
        return _render(PAGE_TUNE, cfg=cfg, active="tune")

    @app.get("/calibrate")
    def page_calibrate():
        return _render(PAGE_CALIBRATE, cfg=cfg, active="calibrate")

    @app.get("/scripts")
    def page_scripts():
        return _render(PAGE_SCRIPTS, cfg=cfg, active="scripts")

    # ------------------------------------------------------------ overview API
    @app.get("/api/c2/overview")
    def api_overview():
        return jsonify(pool.snapshot())

    @app.get("/api/c2/config")
    def api_config():
        return jsonify({
            "fcs": [{"name": f.name, "host": f.host, "port": f.port}
                    for f in cfg.fcs],
            "emergency_land_key": cfg.emergency_land_key,
            "state_poll_hz": cfg.state_poll_hz,
            "ui_refresh_hz": cfg.ui_refresh_hz,
            "calibration_sync_seconds": cfg.calibration_sync_seconds,
        })

    # ------------------------------------------------------------ per-FC ctrl
    def _fc_or_404(name: str):
        client = pool.client(name)
        if client is None:
            return None, (jsonify({"ok": False,
                                   "error": f"unknown FC: {name}"}), 404)
        return client, None

    @app.post("/api/c2/<fc>/start")
    def api_fc_start(fc: str):
        client, err = _fc_or_404(fc)
        if err:
            return err
        body = request.get_json(silent=True) or {}
        script = body.get("script")
        ok, payload = _await(client.start_mission(script), timeout=10.0)
        return jsonify({"ok": ok, "payload": payload,
                        "error": None if ok else _err(payload)})

    @app.post("/api/c2/<fc>/stop")
    def api_fc_stop(fc: str):
        client, err = _fc_or_404(fc)
        if err:
            return err
        ok, payload = _await(client.stop_mission(), timeout=10.0)
        # 409 from marker_mission means "mission was not running" — we
        # already mapped that to ok=True in the client, but if a future
        # error comes back as 409 the body will tell us via payload.
        noop = bool(isinstance(payload, dict) and payload.get("error")
                    and "not running" in str(payload.get("error", "")))
        return jsonify({"ok": ok, "payload": payload, "noop": noop,
                        "error": None if ok else _err(payload)})

    # ----------------------------------------------------------- start all
    # Fan out POST /api/start to every configured FC. No script body —
    # each FC uses its own active draft (set per-FC from the overview
    # card editor or fleet-broadcast via /scripts). 409 "mission
    # already running" is mapped to noop=True so the UI shows green for
    # FCs already in flight.
    @app.post("/api/c2/start-all")
    def api_start_all():
        async def fan_out():
            clients = pool.all_clients()
            coros = [c.start_mission(None) for c in clients]
            results = await asyncio.gather(*coros, return_exceptions=True)
            return {c.spec.name: r for c, r in zip(clients, results)}

        raw = _await(fan_out(), timeout=30.0)
        out: dict[str, dict] = {}
        for name, r in raw.items():
            if isinstance(r, Exception):
                log.error("start-all %s failed: %s", name, r)
                out[name] = {"ok": False, "error": str(r)}
                continue
            ok, payload = r
            entry: dict[str, Any] = {"ok": bool(ok), "payload": payload}
            if not ok:
                err = _err(payload)
                # Map "already running" / 409 to a soft success so the
                # UI shows green for FCs that didn't need to be started.
                if "already running" in err or "already started" in err:
                    entry = {"ok": True, "noop": True, "reason": err}
                else:
                    entry["error"] = err
            out[name] = entry
        return jsonify(out)

    # --------------------------------------------------------- emergency land
    @app.post("/api/c2/emergency-land-all")
    def api_emergency_land_all():
        async def fan_out():
            clients = pool.all_clients()
            coros = [c.stop_mission() for c in clients]
            results = await asyncio.gather(*coros, return_exceptions=True)
            return {c.spec.name: r for c, r in zip(clients, results)}

        raw = _await(fan_out(), timeout=20.0)
        out: dict[str, dict] = {}
        for name, r in raw.items():
            if isinstance(r, Exception):
                log.error("emergency-land %s failed: %s", name, r)
                out[name] = {"ok": False, "error": str(r)}
                continue
            ok, payload = r
            entry: dict[str, Any] = {"ok": bool(ok)}
            # Map "not running" responses to noop=True so the UI shows
            # green instead of red for FCs that were idle.
            if ok and isinstance(payload, dict) and "noop" not in entry:
                err = payload.get("error") if isinstance(payload, dict) else None
                if err and "not running" in str(err):
                    entry["noop"] = True
                    entry["reason"] = err
            if not ok:
                entry["error"] = _err(payload)
            out[name] = entry
        # Always 200 — partial failures are reported per-FC in the body.
        return jsonify(out)

    # ----------------------------------------------------------- per-FC script
    @app.get("/api/c2/<fc>/script")
    def api_fc_script_get(fc: str):
        client, err = _fc_or_404(fc)
        if err:
            return err
        ok, payload = _await(client.get_mission_script(), timeout=5.0)
        return jsonify({"ok": ok, "payload": payload,
                        "error": None if ok else _err(payload)})

    @app.post("/api/c2/<fc>/script")
    def api_fc_script_set(fc: str):
        client, err = _fc_or_404(fc)
        if err:
            return err
        body = request.get_json(silent=True) or {}
        text = str(body.get("text") or "")
        ok, payload = _await(client.set_mission_script(text), timeout=5.0)
        return jsonify({"ok": ok, "payload": payload,
                        "error": None if ok else _err(payload)})

    # ------------------------------------------------------------- arena
    @app.get("/api/c2/arena/load-from/<fc>")
    def api_arena_load(fc: str):
        client, err = _fc_or_404(fc)
        if err:
            return err
        ok, payload = _await(client.get_active_arena(), timeout=5.0)
        return jsonify({"ok": ok, "payload": payload,
                        "error": None if ok else _err(payload)})

    @app.post("/api/c2/arena/push")
    def api_arena_push():
        body = request.get_json(silent=True) or {}
        fcs = body.get("fcs") or []
        payload = body.get("payload") or {}
        if not isinstance(fcs, list) or not fcs:
            return jsonify({"ok": False, "error": "fcs[] required"}), 400
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be object"}), 400

        async def fan_out():
            targets = [pool.client(n) for n in fcs if pool.client(n)]
            coros = [c.set_active_arena(payload) for c in targets]
            results = await asyncio.gather(*coros, return_exceptions=True)
            return {c.spec.name: r for c, r in zip(targets, results)}

        raw = _await(fan_out(), timeout=20.0)
        return jsonify({"ok": True,
                        "results": _summarize_fan_out(raw)})

    # -------------------------------------------------------------- tune
    @app.get("/api/c2/tune/load-from/<fc>")
    def api_tune_load(fc: str):
        client, err = _fc_or_404(fc)
        if err:
            return err
        ok, payload = _await(client.get_tune(), timeout=5.0)
        if not ok or not isinstance(payload, dict):
            return jsonify({"ok": False, "error": _err(payload)}), 200
        # marker_mission's /api/tune returns
        # {groups: [{name, items: [{name, value, kind, label, ...}]}]}.
        # We flatten that to a flat {field_name: value} dict for the
        # C2 editor — schema (groups / labels / hints) lives on each
        # FC and we don't try to reconcile cross-FC version differences
        # in v1. Older shapes (current_values / values) are kept as
        # graceful fallbacks for forward compatibility, but the live
        # marker_mission ships only `groups`.
        values: dict[str, Any] = {}
        groups = payload.get("groups")
        if isinstance(groups, list):
            for g in groups:
                if not isinstance(g, dict):
                    continue
                items = g.get("items") or []
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    if isinstance(name, str) and name:
                        values[name] = item.get("value")
        if not values:
            # Legacy shapes — kept for FCs running older marker_mission
            # ahead of an upgrade.
            values = (payload.get("current_values")
                      or payload.get("values")
                      or {})
        return jsonify({"ok": True, "values": values})

    @app.post("/api/c2/tune/apply")
    def api_tune_apply():
        body = request.get_json(silent=True) or {}
        fcs = body.get("fcs") or []
        updates = body.get("updates") or {}
        also_save = bool(body.get("also_save"))
        if not isinstance(fcs, list) or not fcs:
            return jsonify({"ok": False, "error": "fcs[] required"}), 400
        if not isinstance(updates, dict) or not updates:
            return jsonify({"ok": False, "error": "updates required"}), 400

        async def fan_out():
            targets = [pool.client(n) for n in fcs if pool.client(n)]

            async def one(c):
                ok, payload = await c.apply_tune(updates)
                if ok and also_save:
                    ok2, payload2 = await c.save_tune()
                    if not ok2:
                        return False, {"apply": payload, "save": payload2}
                    return True, {"apply": payload, "save": payload2}
                return ok, payload
            results = await asyncio.gather(*(one(c) for c in targets),
                                            return_exceptions=True)
            return {c.spec.name: r for c, r in zip(targets, results)}

        raw = _await(fan_out(), timeout=30.0)
        return jsonify({"ok": True,
                        "results": _summarize_fan_out(raw)})

    # ----------------------------------------------------------- scripts page
    @app.post("/api/c2/scripts/push")
    def api_scripts_push():
        body = request.get_json(silent=True) or {}
        fcs = body.get("fcs") or []
        text = str(body.get("text") or "")
        if not isinstance(fcs, list) or not fcs:
            return jsonify({"ok": False, "error": "fcs[] required"}), 400

        async def fan_out():
            targets = [pool.client(n) for n in fcs if pool.client(n)]
            coros = [c.set_mission_script(text) for c in targets]
            results = await asyncio.gather(*coros, return_exceptions=True)
            return {c.spec.name: r for c, r in zip(targets, results)}

        raw = _await(fan_out(), timeout=15.0)
        return jsonify({"ok": True,
                        "results": _summarize_fan_out(raw)})

    @app.post("/api/c2/scripts/start")
    def api_scripts_start():
        body = request.get_json(silent=True) or {}
        fcs = body.get("fcs") or []
        text = str(body.get("text") or "")
        if not isinstance(fcs, list) or not fcs:
            return jsonify({"ok": False, "error": "fcs[] required"}), 400

        async def fan_out():
            targets = [pool.client(n) for n in fcs if pool.client(n)]

            async def one(c):
                # Push the script first so the FC's persisted draft
                # matches the operator's intent even if the start
                # response gets lost mid-flight.
                ok1, payload1 = await c.set_mission_script(text)
                if not ok1:
                    return False, {"set_script": payload1}
                ok2, payload2 = await c.start_mission(text)
                return ok2, {"set_script": payload1, "start": payload2}
            results = await asyncio.gather(*(one(c) for c in targets),
                                            return_exceptions=True)
            return {c.spec.name: r for c, r in zip(targets, results)}

        raw = _await(fan_out(), timeout=20.0)
        return jsonify({"ok": True,
                        "results": _summarize_fan_out(raw)})

    @app.post("/api/c2/scripts/stop")
    def api_scripts_stop():
        body = request.get_json(silent=True) or {}
        fcs = body.get("fcs") or []
        if not isinstance(fcs, list) or not fcs:
            return jsonify({"ok": False, "error": "fcs[] required"}), 400

        async def fan_out():
            targets = [pool.client(n) for n in fcs if pool.client(n)]
            coros = [c.stop_mission() for c in targets]
            results = await asyncio.gather(*coros, return_exceptions=True)
            return {c.spec.name: r for c, r in zip(targets, results)}

        raw = _await(fan_out(), timeout=15.0)
        return jsonify({"ok": True,
                        "results": _summarize_fan_out(raw)})

    # ----------------------------------------------------- calibration library
    @app.get("/api/c2/calibrations")
    def api_calibrations():
        return jsonify({
            "entries": library.list_entries(),
            "last_sync_age_s": library.last_sync_age_s,
            "sync_count": library.sync_count,
            "added_since_start": library.added_since_start,
            "library_dir": str(library.library_dir),
        })

    @app.post("/api/c2/calibrations/sync")
    def api_calibrations_sync():
        result = _await(library.sweep_once(), timeout=30.0)
        return jsonify({"ok": True, **result})

    @app.post("/api/c2/calibrations/<name>/push")
    def api_calibrations_push(name: str):
        if not is_valid_calibration_name(name):
            return jsonify({"ok": False,
                            "error": f"invalid filename {name!r}"}), 400
        body = request.get_json(silent=True) or {}
        fc = str(body.get("fc") or "")
        if not fc:
            return jsonify({"ok": False, "error": "fc required"}), 400
        result = _await(library.push(name, fc), timeout=15.0)
        status = 200 if result.get("ok") else 502
        return jsonify(result), status

    @app.delete("/api/c2/calibrations/<name>")
    def api_calibrations_delete(name: str):
        if not is_valid_calibration_name(name):
            return jsonify({"ok": False,
                            "error": f"invalid filename {name!r}"}), 400
        ok = library.delete(name)
        if not ok:
            return jsonify({"ok": False,
                            "error": "not found in library"}), 404
        return jsonify({"ok": True})

    return app


def _err(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        body = payload.get("body")
        if isinstance(body, dict) and body.get("error"):
            return str(body.get("error"))
        return str(payload)
    return str(payload)


def _summarize_fan_out(raw: dict[str, Any]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, r in raw.items():
        if isinstance(r, Exception):
            out[name] = {"ok": False, "error": str(r)}
            continue
        ok, payload = r
        entry: dict[str, Any] = {"ok": bool(ok), "payload": payload}
        if not ok:
            entry["error"] = _err(payload)
        out[name] = entry
    return out


def run_server(cfg: C2Config) -> int:
    log = logging.getLogger("c2")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    loop = C2Loop()
    loop.start()

    pool = FCPool(cfg)
    library = CalibrationLibrary(cfg, pool)

    loop.submit(pool.start(), timeout=5.0)
    loop.submit(library.start(), timeout=5.0)

    app = _build_app(cfg, pool, library, loop)

    # Best-effort graceful shutdown on Ctrl-C.
    def _on_sigint(_sig, _frame):
        log.info("shutting down (SIGINT)")
        try:
            loop.submit(library.stop(), timeout=5.0)
        except Exception:
            pass
        try:
            loop.submit(pool.stop(), timeout=10.0)
        except Exception:
            pass
        loop.stop()
        # Re-raise so Flask's run_simple sees it and exits.
        raise KeyboardInterrupt()

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except ValueError:
        # signal.signal can only run on the main thread; under pytest
        # or embedded use we just skip the handler.
        pass

    log.info("C2 listening on http://%s:%d (FCs: %s)",
             cfg.bind.host, cfg.bind.port,
             ", ".join(f.name for f in cfg.fcs))
    try:
        app.run(host=cfg.bind.host, port=cfg.bind.port,
                threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            loop.submit(library.stop(), timeout=3.0)
        except Exception:
            pass
        try:
            loop.submit(pool.stop(), timeout=5.0)
        except Exception:
            pass
        loop.stop()
    return 0
