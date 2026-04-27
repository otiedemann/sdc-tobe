"""Routes for the **drones** blueprint.

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
from controller_modularized.api import bp_drones
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
    aruco_fleet,)

@bp_drones.get("/proxy/drones")
def proxy_drones():
    return jsonify(drones=DRONES, active=state.active_drone_id)


@bp_drones.post("/proxy/switch")
def proxy_switch():
    data = request.get_json(silent=True) or {}
    drone_id = str(data.get("id", ""))
    if drone_id not in DRONES:
        return jsonify(ok=False, error="unknown drone id"), 400
    state.active_drone_id = drone_id
    state.PI_BASE = DRONES[drone_id]["base"]
    log_command("switch_drone", {"id": drone_id, "name": DRONES[drone_id]["name"]})
    print(f"[REMOTE UI] Switched to {DRONES[drone_id]['name']} @ {state.PI_BASE}")
    return jsonify(ok=True, active=drone_id, name=DRONES[drone_id]["name"], base=state.PI_BASE)


@bp_drones.get("/proxy/drones/config")
def proxy_drones_config():
    """Return full drone config for editing."""
    return jsonify(drones=DRONES, config_path=str(DRONES_CONFIG_PATH))


@bp_drones.post("/proxy/drones/config")
def proxy_drones_config_save():
    """Save updated drone config. Expects {drones: {id: {name, type, base}, ...}}"""
    global DRONES
    data = request.get_json(silent=True) or {}
    new_drones = data.get("drones")
    if not new_drones or not isinstance(new_drones, dict):
        return jsonify(ok=False, error="drones dict required"), 400
    for did, info in new_drones.items():
        if not all(k in info for k in ("name", "type", "base")):
            return jsonify(ok=False, error=f"drone {did} missing name/type/base"), 400
        if info["type"] not in ("tello", "anafi"):
            return jsonify(ok=False, error=f"drone {did} type must be tello or anafi"), 400
    DRONES.clear()
    DRONES.update(new_drones)
    if state.active_drone_id in DRONES:
        state.PI_BASE = DRONES[state.active_drone_id]["base"]
    save_drones_config(DRONES)
    # Keep ArUco fleet in sync with drone config
    try:
        aruco_fleet.configure(DRONES)
    except Exception as e:
        print(f"[ARUCO] fleet reconfigure failed: {e}")
    print(f"[CONFIG] Saved {len(DRONES)} drones to {DRONES_CONFIG_PATH}")
    return jsonify(ok=True, drones=DRONES)
