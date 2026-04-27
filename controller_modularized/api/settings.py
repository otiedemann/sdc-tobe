"""Routes for the **settings** blueprint.

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
from controller_modularized.api import bp_settings
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
    _apply_camera_face_to_fleet, _camera_face_center_lock,
    _transport, _transport_lock,)

@bp_settings.get("/proxy/settings")
def proxy_settings_get():
    try:
        r = pi_get("/api/settings", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_settings.post("/proxy/settings")
def proxy_settings_set():
    data = request.get_json(silent=True) or {}
    log_command("settings", data)
    r = pi_post("/api/settings", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_settings.get("/proxy/config/transport")
def proxy_transport_get():
    """Return the current per-subsystem transport preference."""
    with _transport_lock:
        snap = dict(_transport)
    return jsonify(ok=True, transport=snap, ws_available=HAS_WSCLIENT)


@bp_settings.post("/proxy/config/transport")
def proxy_transport_set():
    """Update the transport preference for one or more subsystems.
    Body: {"rc":"http", "telemetry":"auto", "position":"ws"}

    Valid values: "auto" (WS if up, HTTP fallback), "ws" (force WS),
    "http" (force HTTP). Unknown subsystems / values are ignored.
    """
    global WS_RC_ENABLED
    data = request.get_json(silent=True) or {}
    allowed = {"auto", "ws", "http"}
    changed = {}
    with _transport_lock:
        for subsys in ("rc", "telemetry", "position"):
            v = data.get(subsys)
            if v is None:
                continue
            v = str(v).lower()
            if v in allowed:
                _transport[subsys] = v
                changed[subsys] = v
        snap = dict(_transport)
    WS_RC_ENABLED = snap["rc"] in ("auto", "ws")
    log_command("transport_set", {"changed": changed, "active": snap})
    print(f"[TRANSPORT] updated: {changed} → now {snap}")
    return jsonify(ok=True, transport=snap, changed=changed)


@bp_settings.get("/proxy/config/camera_face_center")
def proxy_camera_face_center_get():
    with _camera_face_center_lock:
        return jsonify(
            ok=True,
            enabled=_camera_face_center_enabled,
            xy=list(_camera_face_center_xy),
        )


@bp_settings.post("/proxy/config/camera_face_center")
def proxy_camera_face_center_set():
    """Enable/disable the toggle, optionally override the centre XY.
    Body: {"enabled": true, "xy": [0, 5.4]}"""
    global _camera_face_center_enabled, _camera_face_center_xy
    data = request.get_json(silent=True) or {}
    changed = {}
    with _camera_face_center_lock:
        if "enabled" in data:
            _camera_face_center_enabled = bool(data["enabled"])
            changed["enabled"] = _camera_face_center_enabled
        if "xy" in data and isinstance(data["xy"], (list, tuple)) and len(data["xy"]) >= 2:
            try:
                _camera_face_center_xy = (float(data["xy"][0]), float(data["xy"][1]))
                changed["xy"] = list(_camera_face_center_xy)
            except (TypeError, ValueError):
                pass
    _apply_camera_face_to_fleet()
    log_command("camera_face_center_set", changed)
    print(f"[CAMERA_FACE] enabled={_camera_face_center_enabled} xy={_camera_face_center_xy}")
    return jsonify(ok=True, changed=changed,
                   enabled=_camera_face_center_enabled,
                   xy=list(_camera_face_center_xy))


@bp_settings.get("/proxy/environment")
def proxy_environment_get():
    try:
        r = _http_session.get(f"{state.PI_BASE}/api/environment", timeout=3)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_settings.post("/proxy/environment")
def proxy_environment_set():
    data = request.get_json(silent=True) or {}
    log_command("environment_set", data)
    try:
        r = _http_session.post(f"{state.PI_BASE}/api/environment", json=data, timeout=4)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_settings.get("/proxy/wifi/status")
def proxy_wifi_status():
    try:
        r = _http_session.get(f"{state.PI_BASE}/api/wifi/status", timeout=3)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_settings.post("/proxy/wifi/channel")
def proxy_wifi_channel():
    data = request.get_json(silent=True) or {}
    log_command("wifi_channel_set", data)
    try:
        r = _http_session.post(f"{state.PI_BASE}/api/wifi/channel", json=data, timeout=8)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_settings.post("/proxy/wifi/scan")
def proxy_wifi_scan():
    data = request.get_json(silent=True) or {}
    try:
        r = _http_session.post(f"{state.PI_BASE}/api/wifi/scan", json=data, timeout=10)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502
