"""Routes for the **logs** blueprint.

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
from controller_modularized.api import bp_logs
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
    _collect_fleet_videos, flight_logger,)

@bp_logs.get("/proxy/flight_logs")
def proxy_flight_logs_list():
    """List archived per-flight log files (newest first), with video
    metadata attached when a matching .mp4 lives on any Pi."""
    files = flight_logger.list_files()
    vids = _collect_fleet_videos()
    for f in files:
        stem = f["name"][:-len(".jsonl")] if f["name"].endswith(".jsonl") else f["name"]
        video_name = stem + ".mp4"
        v = vids.get(video_name)
        if v:
            f["video"] = {
                "name":     video_name,
                "drone_id": v["drone_id"],
                "size":     v["size"],
                "mtime":    v["mtime"],
            }
    return jsonify(ok=True, files=files)


@bp_logs.get("/proxy/flight_video/<path:name>")
def proxy_flight_video_download(name):
    """Stream a flight video from whichever Pi has it. Filename is the
    mp4 basename produced by FlightLogger (flight_YYYY-..._drone-N_X.mp4);
    we look up the drone id from the filename and proxy straight through."""
    # Locate the video across the fleet
    vids = _collect_fleet_videos()
    entry = vids.get(name)
    if not entry:
        return jsonify(ok=False, error="video not found on any drone"), 404
    did = entry["drone_id"]
    base = (DRONES.get(did, {}) or {}).get("base")
    if not base:
        return jsonify(ok=False, error="drone has no base url"), 500
    try:
        upstream = _http_session.get(
            f"{base.rstrip('/')}/api/video/recordings/{name}",
            stream=True, timeout=(3, 60))
        if not upstream.ok:
            return jsonify(ok=False, error=f"drone returned {upstream.status_code}"), 502
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502
    def gen():
        for chunk in upstream.iter_content(chunk_size=64 * 1024):
            if chunk:
                yield chunk
    headers = {
        "Content-Type": "video/mp4",
        "Content-Disposition": f'attachment; filename="{name}"',
    }
    size = upstream.headers.get("Content-Length")
    if size:
        headers["Content-Length"] = size
    return Response(gen(), headers=headers)


@bp_logs.get("/proxy/flight_logs/<path:name>")
def proxy_flight_logs_download(name):
    p = flight_logger.file_path(name)
    if p is None:
        return jsonify(ok=False, error="file not found"), 404
    # Inline when "as_attachment=false" so the viewer can fetch-and-parse
    # rather than trigger a download.
    as_att = request.args.get("dl", "1") != "0"
    return send_file(str(p), mimetype="application/jsonlines",
                     as_attachment=as_att, download_name=p.name)


@bp_logs.get("/proxy/logging/commands")
def proxy_command_log_status():
    return jsonify(enabled=command_log_enabled, path=str(command_log_path))


@bp_logs.post("/proxy/logging/commands")
def proxy_command_log_config():
    global command_log_enabled, command_log_path
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled")
    path = data.get("path")
    if enabled is not None and not isinstance(enabled, bool):
        return jsonify(ok=False, error="enabled must be boolean"), 400
    if path is not None and (not isinstance(path, str) or not path.strip()):
        return jsonify(ok=False, error="path must be non-empty string"), 400
    if isinstance(enabled, bool):
        command_log_enabled = enabled
    if isinstance(path, str) and path.strip():
        command_log_path = Path(path.strip())
    print(f"[REMOTE CMD] command logging {'enabled' if command_log_enabled else 'disabled'}")
    return jsonify(ok=True, enabled=command_log_enabled, path=str(command_log_path))


@bp_logs.get("/proxy/logging/commands/download")
def proxy_command_log_download():
    p = command_log_path
    if not p.exists():
        return jsonify(ok=False, error="command log file not found", path=str(p)), 404
    return send_file(p, as_attachment=True, download_name=p.name, mimetype="application/x-ndjson")


@bp_logs.post("/proxy/logging/commands/clear")
def proxy_command_log_clear():
    p = command_log_path
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        return jsonify(ok=True, cleared=True, path=str(p))
    except Exception as e:
        return jsonify(ok=False, error=str(e), path=str(p)), 500


@bp_logs.get("/proxy/logging/telemetry")
def proxy_log_status():
    try:
        r = pi_get("/api/logging/telemetry", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@bp_logs.post("/proxy/logging/telemetry")
def proxy_log_config():
    data = request.get_json(silent=True) or {}
    log_command("telemetry_log_set", data)
    r = pi_post("/api/logging/telemetry", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@bp_logs.get("/proxy/logging/telemetry/download")
def proxy_log_download():
    r = pi_get("/api/logging/telemetry/download")
    headers = {
        "Content-Type": r.headers.get("Content-Type", "application/octet-stream"),
        "Content-Disposition": r.headers.get("Content-Disposition", "attachment; filename=telemetry_log.jsonl"),
    }
    return (r.content, r.status_code, headers)


@bp_logs.post("/proxy/logging/telemetry/clear")
def proxy_log_clear():
    log_command("telemetry_log_clear")
    r = pi_post("/api/logging/telemetry/clear")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
