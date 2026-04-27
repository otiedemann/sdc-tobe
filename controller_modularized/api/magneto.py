"""Routes for the **magneto** blueprint.

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
from controller_modularized.api import bp_magneto
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
    _magneto_cycle,)

@bp_magneto.get("/proxy/magneto")
def proxy_magneto_status():
    """Current magnetometer calibration status for the active drone."""
    try:
        r = pi_get("/api/magneto", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_magneto.post("/proxy/magneto/calibrate")
def proxy_magneto_calibrate():
    """One-shot: send the StartMagnetoCalibration command and return.
    The caller is then responsible for the figure-8 dance and for polling
    /proxy/magneto until all axes report ok."""
    log_command("magneto_calibrate")
    try:
        r = pi_post("/api/magneto/calibrate", timeout=TIMEOUT_CMD)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_magneto.post("/proxy/magneto/recalibrate")
def proxy_magneto_recalibrate():
    """Blocking orchestrator: runs the full recalibration cycle and returns a
    single summary JSON. See /proxy/magneto/recalibrate/stream for live
    per-step progress.

    Body (optional):
      { "timeout_s": 60, "poll_s": 1.0 }"""
    data = request.get_json(silent=True) or {}
    try:
        timeout_s = max(5.0, float(data.get("timeout_s", 60)))
        poll_s = max(0.25, float(data.get("poll_s", 1.0)))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="timeout_s/poll_s must be numeric"), 400

    log_command("magneto_recalibrate", {"timeout_s": timeout_s, "poll_s": poll_s})

    final = {"ok": False, "error": "no final event"}
    for ev in _magneto_cycle(timeout_s, poll_s):
        if ev.get("kind") == "final":
            final = {k: v for k, v in ev.items() if k != "kind"}
    return jsonify(final), 200


@bp_magneto.get("/proxy/magneto/recalibrate/stream")
def proxy_magneto_recalibrate_stream():
    """SSE stream of the recalibration cycle. Used by the wizard GUI to light
    up each step + per-axis indicator as it happens.

    Query params: timeout_s (default 60), poll_s (default 1.0)."""
    try:
        timeout_s = max(5.0, float(request.args.get("timeout_s", 60)))
        poll_s = max(0.25, float(request.args.get("poll_s", 1.0)))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="timeout_s/poll_s must be numeric"), 400

    log_command("magneto_recalibrate_stream",
                {"timeout_s": timeout_s, "poll_s": poll_s})

    def generate():
        for ev in _magneto_cycle(timeout_s, poll_s):
            yield f"data: {json.dumps(ev)}\n\n".encode()

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})
