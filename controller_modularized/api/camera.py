"""Routes for the **camera** blueprint.

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
from controller_modularized.api import bp_camera
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
    _start_udp_receiver, _stop_udp_receiver,
    _udp_last_frame_lock, _udp_receiver_running,)

@bp_camera.post("/proxy/stream")
def proxy_stream():
    data = request.get_json(silent=True) or {}
    log_command("stream", data)
    r = pi_post("/api/stream", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_camera.post("/proxy/camera/photo")
def proxy_camera_photo():
    log_command("camera_photo")
    r = pi_post("/api/camera/photo")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_camera.post("/proxy/camera/record/start")
def proxy_camera_record_start():
    log_command("camera_record_start")
    r = pi_post("/api/camera/record/start")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_camera.post("/proxy/camera/record/stop")
def proxy_camera_record_stop():
    log_command("camera_record_stop")
    r = pi_post("/api/camera/record/stop")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_camera.post("/proxy/gimbal")
def proxy_gimbal():
    data = request.get_json(silent=True) or {}
    log_command("gimbal", data)
    r = pi_post("/api/gimbal", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_camera.get("/proxy/video")
def proxy_video_feed():
    """Proxy the MJPEG video stream from the Pi API server."""
    try:
        r = _http_session.get(f"{state.PI_BASE}/api/video", stream=True, timeout=30)
        return Response(
            r.iter_content(chunk_size=32768),
            mimetype=r.headers.get("Content-Type", "multipart/x-mixed-replace; boundary=frame"),
        )
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_camera.post("/proxy/video/start")
def proxy_video_start():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "mjpeg")
    # For forward mode, auto-detect C2 IP (this machine) so the Pi sends UDP here
    if mode == "forward" and not data.get("target_host"):
        # Use the host part of request.host (what the browser connected to)
        c2_host = request.host.split(":")[0]
        if c2_host in ("127.0.0.1", "localhost"):
            # Try to get our real IP
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                c2_host = s.getsockname()[0]
                s.close()
            except Exception:
                pass
        data["target_host"] = c2_host
        data["target_port"] = data.get("target_port", VIDEO_UDP_FORWARD_PORT)
    log_command("video_start", data)
    # For forward mode, start the UDP receiver BEFORE telling the Pi to forward
    # so ffmpeg is already listening when packets arrive
    if mode == "forward":
        _start_udp_receiver()
        time.sleep(0.5)  # Give ffmpeg time to bind the UDP port
    r = pi_post("/api/video/start", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_camera.post("/proxy/video/stop")
def proxy_video_stop():
    log_command("video_stop")
    _stop_udp_receiver()
    r = pi_post("/api/video/stop")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_camera.get("/proxy/video/status")
def proxy_video_status():
    try:
        r = pi_get("/api/video/status", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": "application/json"})
    except Exception as e:
        return jsonify(ok=False, error=str(e), mode="off"), 502


@bp_camera.get("/proxy/video/forward_stream")
def proxy_video_forward_stream():
    """Serve decoded UDP forward frames as MJPEG stream."""
    def gen():
        while _udp_receiver_running:
            with _udp_last_frame_lock:
                jpg = _udp_last_jpeg
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(1.0 / max(1, VIDEO_FPS))
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@bp_camera.post("/proxy/camera/zoom")
def proxy_camera_zoom():
    data = request.get_json(silent=True) or {}
    try:
        r = _http_session.post(f"{state.PI_BASE}/api/camera/zoom", json=data, timeout=1.5)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_camera.get("/proxy/video/record/status")
def proxy_rec_status():
    try:
        r = pi_get("/api/video/record/status", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_camera.post("/proxy/video/record/start")
def proxy_rec_start():
    try:
        data = request.get_json(silent=True) or {}
        r = pi_post("/api/video/record/start", data)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_camera.post("/proxy/video/record/stop")
def proxy_rec_stop():
    try:
        r = pi_post("/api/video/record/stop", {})
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502
