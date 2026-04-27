"""Routes for the **missions** blueprint.

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
from controller_modularized.api import bp_missions
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
    MISSION_PRESETS_PATH, _load_mission_presets,
    _parse_marker_list, _save_mission_presets,
    _scan_and_capture_thread, _scan_cap_abort,
    _scan_cap_lock, mission_manager,)

@bp_missions.get("/proxy/missions/status")
def proxy_missions_status():
    return jsonify(mission_manager.status())


@bp_missions.get("/proxy/missions/presets")
def proxy_mission_presets_list():
    """Return all presets grouped by mission type."""
    return jsonify(ok=True, presets=_load_mission_presets(),
                   path=str(MISSION_PRESETS_PATH))


@bp_missions.post("/proxy/missions/presets")
def proxy_mission_presets_save():
    """Create or overwrite a named preset.
    Body: {"mission_type": "scan_all", "name": "tight-margin",
           "params": {...}}"""
    data = request.get_json(silent=True) or {}
    mtype = str(data.get("mission_type", "")).strip()
    name  = str(data.get("name", "")).strip()
    params = data.get("params")
    if not mtype or not name or not isinstance(params, dict):
        return jsonify(ok=False, error="mission_type, name, params required"), 400
    if mtype not in ("scan_all", "capture_targets"):
        return jsonify(ok=False, error=f"unknown mission_type: {mtype}"), 400
    # Minimal sanity checks per mission type — we don't want a broken
    # preset to silently save.
    if mtype == "scan_all":
        if "target_markers" not in params:
            return jsonify(ok=False, error="target_markers required for scan_all"), 400
    if mtype == "capture_targets":
        if not isinstance(params.get("target_boxes"), list):
            return jsonify(ok=False, error="target_boxes (list) required for capture_targets"), 400
    presets = _load_mission_presets()
    presets.setdefault(mtype, {})[name] = params
    try:
        _save_mission_presets(presets)
    except Exception as e:
        return jsonify(ok=False, error=f"save failed: {e}"), 500
    log_command("mission_preset_save", {"mission_type": mtype, "name": name})
    return jsonify(ok=True, mission_type=mtype, name=name)


@bp_missions.delete("/proxy/missions/presets")
def proxy_mission_presets_delete():
    """Delete a named preset. Query: ?mission_type=X&name=Y"""
    mtype = str(request.args.get("mission_type", "")).strip()
    name  = str(request.args.get("name", "")).strip()
    if not mtype or not name:
        return jsonify(ok=False, error="mission_type, name required"), 400
    presets = _load_mission_presets()
    bucket  = presets.get(mtype, {})
    if name not in bucket:
        return jsonify(ok=False, error="preset not found"), 404
    del bucket[name]
    try:
        _save_mission_presets(presets)
    except Exception as e:
        return jsonify(ok=False, error=f"save failed: {e}"), 500
    log_command("mission_preset_delete", {"mission_type": mtype, "name": name})
    return jsonify(ok=True)


@bp_missions.post("/proxy/missions/scan_all/start")
def proxy_missions_scan_all_start():
    guarded = _pause_guard_response()
    if guarded is not None:
        return guarded
    data = request.get_json(silent=True) or {}
    drone_ids = data.get("drone_ids") or []
    if not isinstance(drone_ids, list) or not drone_ids:
        return jsonify(ok=False, error="drone_ids (non-empty list) required"), 400
    target_markers = _parse_marker_list(data.get("target_markers", "1-12"))
    if not target_markers:
        return jsonify(ok=False, error="target_markers must parse to at least one id"), 400
    hover_seconds = float(data.get("hover_seconds", 3.0))
    approach_tolerance_m = float(data.get("approach_tolerance_m", 0.30))
    approach_skew_tol   = float(data.get("approach_skew_tol", 0.12))
    approach_err_x_tol  = float(data.get("approach_err_x_tol", 0.15))
    auto_takeoff = bool(data.get("auto_takeoff", False))
    ok, msg = mission_manager.start_scan_all(
        drone_ids=[str(d) for d in drone_ids],
        target_markers=target_markers,
        hover_seconds=hover_seconds,
        approach_tolerance_m=approach_tolerance_m,
        approach_skew_tol=approach_skew_tol,
        approach_err_x_tol=approach_err_x_tol,
        auto_takeoff=auto_takeoff,
    )
    log_command("mission_scan_all_start", {
        "drone_ids": drone_ids, "target_markers": target_markers,
        "hover_seconds": hover_seconds, "ok": ok, "msg": msg,
    })
    status = 200 if ok else 409
    return jsonify(ok=ok, message=msg, status=mission_manager.status()), status


@bp_missions.post("/proxy/missions/capture_targets/start")
def proxy_missions_capture_targets_start():
    """Launch the SDC26 capture-all-targets mission. Body:
    {
      "drone_ids":    ["1", "2", ...],
      "target_boxes": [{"id":1,"x":-5.0,"y":2.0,"z":0.0}, ...],
                                                  # z optional — marker
                                                  # height above floor
                                                  # (e.g. 0.5 for a stand)
      "home_xy":      [x, y],                    # team home coord
      "arena_face_xy":[x, y],                    # where the camera aims
      "hover_above_m": 1.5,                      # HEIGHT above the target
                                                  # marker; waypoint Z =
                                                  # box.z + hover_above_m
      "hover_seconds": 4.0,
      "auto_takeoff":  false
    }
    """
    guarded = _pause_guard_response()
    if guarded is not None:
        return guarded
    data = request.get_json(silent=True) or {}
    drone_ids = data.get("drone_ids") or []
    if not isinstance(drone_ids, list) or not drone_ids:
        return jsonify(ok=False, error="drone_ids (non-empty list) required"), 400
    target_boxes = data.get("target_boxes") or []
    if not isinstance(target_boxes, list) or not target_boxes:
        return jsonify(ok=False, error="target_boxes (non-empty list) required"), 400
    home_xy = data.get("home_xy", [0.0, 1.5])
    arena_face_xy = data.get("arena_face_xy", [0.0, 5.4])
    hover_above_m = float(data.get("hover_above_m", 1.5))
    hover_seconds = float(data.get("hover_seconds", 4.0))
    nav_tol_xy_m  = float(data.get("nav_tol_xy_m", 0.3))
    auto_takeoff  = bool(data.get("auto_takeoff", False))
    ok, msg = mission_manager.start_capture_all_targets(
        drone_ids=[str(d) for d in drone_ids],
        target_boxes=target_boxes,
        home_xy=tuple(home_xy),
        arena_face_xy=tuple(arena_face_xy),
        hover_above_m=hover_above_m,
        hover_seconds=hover_seconds,
        nav_tol_xy_m=nav_tol_xy_m,
        auto_takeoff=auto_takeoff,
    )
    log_command("mission_capture_targets_start", {
        "drone_ids": drone_ids,
        "target_boxes": target_boxes,
        "ok": ok, "msg": msg,
    })
    status = 200 if ok else 409
    return jsonify(ok=ok, message=msg, status=mission_manager.status()), status


@bp_missions.post("/proxy/missions/stop")
def proxy_missions_stop():
    data = request.get_json(silent=True) or {}
    land = bool(data.get("land", False))
    ok = mission_manager.stop(land=land)
    log_command("mission_stop", {"land": land, "ok": ok})
    return jsonify(ok=ok, status=mission_manager.status())


@bp_missions.post("/proxy/missions/scan_and_capture/start")
def proxy_scan_and_capture_start():
    """Scan-and-capture: the first selected drone rotates to discover
    SDC26 target boxes (IDs 31-36 Blue, 41-46 Red) via ArUco. All
    selected drones then visit boxes concurrently at stacked altitudes
    using the standard CaptureAllTargetsMission (target-claim prevents
    two drones from approaching the same box; per-drone altitude offset
    keeps 3-D paths separated). Body:
      {
        "drone_ids":      ["2", "1"],       # multi-drone list; first scans
        "drone_id":       "2",              # (legacy single-drone fallback)
        "rotation_deg":   60,               # degrees per rotate step
        "rotation_steps": 6,                # full 360° by default
        "dwell_s":        2.0,              # observation window per step
        "hover_seconds":  3.0,              # dwell over each target box
        "hover_above_m":  1.5,              # HEIGHT above the target marker
        "home_xy":        [0.0, 1.5],       # landing / return point
        "arena_face_xy":  [0.0, 5.4],       # where camera points during capture
        "nav_tol_xy_m":   0.3,              # arrival tolerance
        "altitude_stack_m": 1.0,            # per-drone Z-offset step
        "min_separation_m": 1.2             # inter-drone 3-D proximity pause
      }
    """
    with _scan_cap_lock:
        if _scan_cap_state["active"]:
            return jsonify(ok=False, error="scan-and-capture already active",
                           state=dict(_scan_cap_state)), 409
    guarded = _pause_guard_response()
    if guarded is not None:
        return guarded
    data = request.get_json(silent=True) or {}
    # Accept either `drone_ids: [list]` (preferred, multi-drone) or the
    # legacy `drone_id: str` (single-drone). Falls back to active drone.
    drone_ids_raw = data.get("drone_ids")
    if isinstance(drone_ids_raw, list) and drone_ids_raw:
        drone_ids = [str(d) for d in drone_ids_raw if str(d) in DRONES]
    else:
        did = str(data.get("drone_id") or state.active_drone_id)
        drone_ids = [did] if did in DRONES else []
    if not drone_ids:
        return jsonify(ok=False, error="no valid drones selected"), 400
    rotation_deg   = max(10, min(180, int(data.get("rotation_deg", 60))))
    rotation_steps = max(1, min(24, int(data.get("rotation_steps", 6))))
    dwell_s        = max(0.5, min(10.0, float(data.get("dwell_s", 2.0))))
    hover_seconds  = max(1.0, min(20.0, float(data.get("hover_seconds", 3.0))))
    hover_above_m  = max(0.3, min(5.0, float(data.get("hover_above_m", 1.5))))
    home_xy        = data.get("home_xy") or [0.0, 1.5]
    arena_face_xy  = data.get("arena_face_xy") or [0.0, 5.4]
    nav_tol_xy_m   = max(0.1, min(2.0, float(data.get("nav_tol_xy_m", 0.3))))
    altitude_stack_m = max(0.0, min(3.0, float(data.get("altitude_stack_m", 1.0))))
    min_separation_m = max(0.0, min(5.0, float(data.get("min_separation_m", 1.2))))
    with _scan_cap_lock:
        _scan_cap_state.update({
            "active":        True,
            "phase":         "starting",
            "drone_id":      drone_ids[0],    # legacy single-id field — first drone
            "drone_ids":     drone_ids,
            "step_name":     "starting",
            "n_detected":    0,
            "targets_found": {},
            "target_boxes":  [],
            "started_at":    time.time(),
            "ended_at":      0.0,
            "elapsed_s":     0.0,
            "last_error":    None,
            "result":        None,
        })
    _scan_cap_abort.clear()
    threading.Thread(
        target=_scan_and_capture_thread,
        args=(drone_ids, rotation_deg, rotation_steps, dwell_s,
              hover_seconds, hover_above_m,
              tuple(home_xy), tuple(arena_face_xy), nav_tol_xy_m,
              altitude_stack_m, min_separation_m),
        daemon=True, name="scan-and-capture",
    ).start()
    log_command("scan_and_capture_start", {
        "drone_ids": drone_ids, "rotation_steps": rotation_steps,
        "hover_seconds": hover_seconds,
        "altitude_stack_m": altitude_stack_m,
        "min_separation_m": min_separation_m,
    })
    return jsonify(ok=True, message="scan-and-capture started",
                   drone_ids=drone_ids)


@bp_missions.get("/proxy/missions/scan_and_capture/status")
def proxy_scan_and_capture_status():
    with _scan_cap_lock:
        snap = dict(_scan_cap_state)
    return jsonify(ok=True, **snap)


@bp_missions.post("/proxy/missions/scan_and_capture/abort")
def proxy_scan_and_capture_abort():
    """Abort the scan-and-capture. Sets a flag that the background thread
    checks; the thread exits at the next dwell boundary. If the capture
    mission has already handed off to mission_manager, also stop that."""
    _scan_cap_abort.set()
    # Also try to stop any running mission (best-effort — might not be ours)
    try:
        if _scan_cap_state.get("phase") in {"capture", "done"}:
            mission_manager.stop(land=False)
    except Exception:
        pass
    with _scan_cap_lock:
        active = _scan_cap_state["active"]
    return jsonify(ok=True, message="abort requested", active=active)


@bp_missions.get("/proxy/missions/trace")
def proxy_missions_trace():
    """Download the trace log for the most recent mission. The mission
    class writes a JSONL file per run; we return the current one (or the
    most recent if no mission is active)."""
    from pathlib import Path as _P
    import glob as _glob
    path = None
    try:
        cur = mission_manager.current
        if cur is not None and getattr(cur, "trace_path", None):
            path = cur.trace_path
    except Exception:
        pass
    if not path:
        # Fall back to newest file in the logs dir
        try:
            from aruco_seek_multi import MISSION_LOG_DIR
            files = sorted(_glob.glob(str(MISSION_LOG_DIR / "mission_*.jsonl")))
            if files:
                path = files[-1]
        except Exception:
            pass
    if not path or not _P(path).exists():
        return jsonify(ok=False, error="no trace available yet"), 404
    return send_file(path, mimetype="application/x-ndjson",
                     as_attachment=True,
                     download_name=_P(path).name)


@bp_missions.get("/proxy/missions/traces")
def proxy_missions_traces():
    """List all mission trace files on disk with size and mtime."""
    import glob as _glob
    from aruco_seek_multi import MISSION_LOG_DIR
    files = []
    try:
        for f in sorted(_glob.glob(str(MISSION_LOG_DIR / "mission_*.jsonl"))):
            p = Path(f)
            st = p.stat()
            files.append({
                "name": p.name,
                "size": st.st_size,
                "mtime": st.st_mtime,
            })
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500
    return jsonify(ok=True, files=files)
