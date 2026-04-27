"""Routes for the **telemetry** blueprint.

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
from controller_modularized.api import bp_telemetry
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
    _transport_mode, drone_ws,)

@bp_telemetry.get("/proxy/heartbeat")
def proxy_heartbeat():
    """Send heartbeat to all REACHABLE drones in parallel.

    Skips drones whose WS client reports every channel down — no point
    spending 300 ms × N offline drones every 750 ms. That was the main
    reason the heartbeat poll stalled the HTTP pool when most of the
    fleet was offline.
    """
    def _ping(did_info):
        did, info = did_info
        try:
            r = _http_session.get(f"{info['base']}/api/heartbeat", timeout=0.25)
            return did, r.status_code
        except Exception:
            return did, "timeout"

    # Filter out offline drones using the WS client's health flag.
    live_items = []
    skipped = {}
    for did, info in DRONES.items():
        did_s = str(did)
        cli = drone_ws.get(did_s) if 'drone_ws' in globals() else None
        if cli is not None:
            all_down = (not cli._ws_connected.get("telemetry") and
                        not cli._ws_connected.get("position") and
                        not cli._ws_connected.get("rc"))
            if all_down:
                skipped[did_s] = "offline"
                continue
        live_items.append((did, info))

    results = dict(_heartbeat_pool.map(_ping, live_items)) if live_items else {}
    results.update(skipped)
    return jsonify(ok=True, drones=results)


@bp_telemetry.get("/proxy/telemetry")
def proxy_telemetry():
    mode = _transport_mode("telemetry")
    if mode in ("auto", "ws"):
        ws = drone_ws.get(str(state.active_drone_id))
        if ws:
            tel, age = ws.latest_telemetry()
            if tel is not None and age < 1.5:
                tel = dict(tel)
                tel["_source"] = "ws"
                tel["_age_ms"] = int(age * 1000)
                headers = {
                    "Content-Type": "application/json",
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                }
                return (json.dumps(tel, default=str), 200, headers)
        if mode == "ws":
            return jsonify(ok=False, error="ws forced but not connected", via="ws"), 503
    try:
        r = pi_get("/api/telemetry", timeout=TIMEOUT_STATUS)
        headers = {
            "Content-Type": r.headers.get("Content-Type", "application/json"),
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        return (r.text, r.status_code, headers)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_telemetry.get("/proxy/position")
def proxy_position_get():
    try:
        r = pi_get("/api/position", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_telemetry.get("/proxy/position/events")
def proxy_position_events():
    """SSE proxy — streams ArUco position events from Pi to browser."""
    pi_url = state.PI_BASE + "/api/position/events"

    def generate():
        try:
            with _http_session.get(pi_url, stream=True, timeout=(3, 300)) as resp:
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
        except Exception as e:
            import json as _json
            yield f"data: {_json.dumps({'error': str(e)})}\n\n".encode()

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@bp_telemetry.get("/proxy/telemetry/stream")
def proxy_telemetry_stream():
    """SSE proxy — streams telemetry events from Pi to browser (replaces polling)."""
    pi_url = state.PI_BASE + "/api/telemetry/stream"

    def generate():
        try:
            with _http_session.get(pi_url, stream=True, timeout=(3, 300)) as resp:
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
        except Exception as e:
            import json as _json
            yield f"data: {_json.dumps({'error': str(e)})}\n\n".encode()

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@bp_telemetry.get("/proxy/position/video")
def proxy_position_video():
    """MJPEG proxy — streams ArUco-annotated frames from Pi."""
    pi_url = state.PI_BASE + "/api/position/video"

    def generate():
        try:
            with _http_session.get(pi_url, stream=True, timeout=(3, 300)) as resp:
                for chunk in resp.iter_content(chunk_size=16384):
                    if chunk:
                        yield chunk
        except Exception:
            pass

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})
