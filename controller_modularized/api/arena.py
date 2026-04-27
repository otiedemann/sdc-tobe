"""Routes for the **arena** blueprint.

Moved out of ``remote_web_controller.py`` in PR 4. The route bodies are
unchanged; only their location moved. Helpers used by these routes are
imported from the entrypoint module (which still owns the C2-wide
state)."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

from flask import (
    Response, jsonify, render_template, request, send_file,
)

from controller_modularized import state
from controller_modularized.api import bp_arena
from controller_modularized.config import (
    C2_CODE_VERSION, FLIGHT_LOG_DIR, FLIGHT_LOG_HZ,
    HTTP_HOST, HTTP_PORT,
    TIMEOUT_CMD, TIMEOUT_FAST, TIMEOUT_SLOW, TIMEOUT_STATUS,
    VIDEO_FPS, VIDEO_JPEG_QUALITY, VIDEO_UDP_FORWARD_PORT,
)
from controller_modularized.http_client import (
    _http_session, _heartbeat_pool,
    _record_pi_call, _slow_calls, _slow_calls_lock,
    command_log_enabled, command_log_last, command_log_path,
    log_command, pi_get, pi_post,
)
from controller_modularized.controller.pause import (
    is_paused as _is_paused,
    pause_guard_response as _pause_guard_response,
)

from controller_modularized.state import DRONES, save_drones_config
from controller_modularized.model.drone_ws import drone_ws
from controller_modularized.remote_web_controller import (  # noqa: E402
    aruco_fleet, mission_manager, flight_logger, _GIT_REVISION,
)

# Helpers that still live in the entrypoint. The import resolves at
# api/<bp>.py load time, which the entrypoint does at the END of its
# own initialization — so by the time this runs every name below is
# already defined in remote_web_controller.__dict__.
from controller_modularized.remote_web_controller import (
    POSITION_PRESETS_PATH, _calib_target_base,
    _js_arena_to_pi, _load_position_presets,
    _pi_arena_to_js, _save_position_presets,)

@bp_arena.get("/proxy/position/config")
def proxy_position_config_get():
    try:
        r = pi_get("/api/position/config", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_arena.post("/proxy/position/config")
def proxy_position_config_set():
    data = request.get_json(silent=True) or {}
    r = pi_post("/api/position/config", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_arena.post("/proxy/calibration/start")
def proxy_calibration_start():
    did, base = _calib_target_base(request.args.get("drone_id"))
    if not base:
        return jsonify(ok=False, error=f"drone {did} not configured"), 404
    try:
        r = _http_session.post(f"{base.rstrip('/')}/api/calibration/start",
                               json={}, timeout=TIMEOUT_CMD)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_arena.get("/proxy/calibration/status")
def proxy_calibration_status():
    did, base = _calib_target_base(request.args.get("drone_id"))
    if not base:
        return jsonify(ok=False, error=f"drone {did} not configured"), 404
    try:
        r = _http_session.get(f"{base.rstrip('/')}/api/calibration/status",
                              timeout=TIMEOUT_FAST)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_arena.post("/proxy/calibration/abort")
def proxy_calibration_abort():
    did, base = _calib_target_base(request.args.get("drone_id"))
    if not base:
        return jsonify(ok=False, error=f"drone {did} not configured"), 404
    try:
        r = _http_session.post(f"{base.rstrip('/')}/api/calibration/abort",
                               json={}, timeout=TIMEOUT_CMD)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_arena.post("/proxy/position/calibration")
def proxy_position_calibration():
    """Proxy NPZ calibration file upload to Pi."""
    if "file" not in request.files:
        return jsonify(ok=False, error="no file"), 400
    f = request.files["file"]
    try:
        resp = _http_session.post(
            state.PI_BASE + "/api/position/calibration",
            files={"file": (f.filename, f.read(), "application/octet-stream")},
            timeout=15,
        )
        return Response(resp.content, status=resp.status_code,
                        content_type=resp.headers.get("Content-Type", "application/json"))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@bp_arena.get("/proxy/arena/config")
def proxy_arena_config_get():
    try:
        r = pi_get("/api/arena/config", timeout=TIMEOUT_STATUS)
        d = r.json()
        return jsonify(**_pi_arena_to_js(d))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_arena.post("/proxy/arena/config")
def proxy_arena_config_set():
    data = request.get_json(silent=True) or {}
    try:
        pi_payload = _js_arena_to_pi(data)
        r = pi_post("/api/arena/config", pi_payload)
        d = r.json()
        return jsonify(**_pi_arena_to_js(d))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_arena.post("/proxy/arena/config/reset")
def proxy_arena_config_reset():
    try:
        r = pi_post("/api/arena/config/reset", {})
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_arena.get("/proxy/position/presets")
def proxy_position_presets_list():
    return jsonify(ok=True, presets=_load_position_presets(),
                   path=str(POSITION_PRESETS_PATH))


@bp_arena.post("/proxy/position/presets")
def proxy_position_presets_save():
    """Save a named preset. Body: {"name": "X", "params": {...}}"""
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    params = data.get("params")
    if not name or not isinstance(params, dict):
        return jsonify(ok=False, error="name + params required"), 400
    presets = _load_position_presets()
    presets[name] = params
    try:
        _save_position_presets(presets)
    except Exception as e:
        return jsonify(ok=False, error=f"save failed: {e}"), 500
    log_command("position_preset_save", {"name": name})
    return jsonify(ok=True, name=name)


@bp_arena.delete("/proxy/position/presets")
def proxy_position_presets_delete():
    name = str(request.args.get("name", "")).strip()
    if not name:
        return jsonify(ok=False, error="name required"), 400
    presets = _load_position_presets()
    if name not in presets:
        return jsonify(ok=False, error="preset not found"), 404
    del presets[name]
    try:
        _save_position_presets(presets)
    except Exception as e:
        return jsonify(ok=False, error=f"save failed: {e}"), 500
    log_command("position_preset_delete", {"name": name})
    return jsonify(ok=True)


@bp_arena.post("/proxy/position/presets/apply")
def proxy_position_presets_apply():
    """Load a preset and fan it out to every drone via /proxy/position/config.
    Body: {"name": "..."} — any unknown keys in the preset are accepted
    by the position config endpoint (it silently ignores unknown fields)."""
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    if not name:
        return jsonify(ok=False, error="name required"), 400
    presets = _load_position_presets()
    params = presets.get(name)
    if not isinstance(params, dict):
        return jsonify(ok=False, error="preset not found"), 404
    # Fan out to every drone's /api/position/config
    results = {}
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            results[str(did)] = {"ok": False, "error": "no base url"}
            continue
        try:
            r = _http_session.post(f"{base.rstrip('/')}/api/position/config",
                                   json=params, timeout=2.0)
            results[str(did)] = r.json() if r.ok else {"ok": False, "status": r.status_code}
        except Exception as e:
            results[str(did)] = {"ok": False, "error": str(e)[:120]}
    log_command("position_preset_apply", {"name": name})
    ok_count = sum(1 for v in results.values() if v.get("ok"))
    return jsonify(ok=True, name=name, applied_to=ok_count,
                   total=len(results), results=results)
