"""
Simulator API Server — drop-in replacement for tello_pi_api_server / olympe_pi_api_server.

Instead of controlling a real drone, it bridges commands to the browser-based 3D simulator
via WebSocket.  The remote-controller web UI (or any client) calls the same REST endpoints
(/api/takeoff, /api/land, /api/rc, /api/key_down, etc.) and this server forwards them to
whichever browser tab is running the sim_swarm_API 3D viewer.
"""

import base64
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Set

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_sock import Sock

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HTTP_HOST = "0.0.0.0"
HTTP_PORT = int(os.getenv("SIM_API_PORT", "8080"))
TELEMETRY_HZ = float(os.getenv("TELEMETRY_HZ", "2.0"))
KEY_STALE_S = float(os.getenv("KEY_STALE_S", "1.0"))
SAFE_TAKEOFF_S = float(os.getenv("SAFE_TAKEOFF_S", "3.0"))
SAFE_TAKEOFF_DEFAULT = os.getenv("SAFE_TAKEOFF_DEFAULT", "0") in {"1", "true", "True"}
COMMAND_LOG_ENABLED = os.getenv("API_COMMAND_LOG", "1") in {"1", "true", "True"}
COMMAND_LOG_PATH = Path(
    os.getenv("API_COMMAND_LOG_PATH", str(Path(__file__).with_name("api_command_log.jsonl")))
)

WEB3D_DIR = Path(__file__).parent / "web3d"

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=None)
sock = Sock(app)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sim_api")


@app.after_request
def add_cors_headers(response):
    """Allow cross-origin requests from the C2 dashboard."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
running = True
flying = False

pressed_web: Set[str] = set()
key_last_seen: Dict[str, float] = {}
pressed_lock = threading.Lock()

safe_takeoff_enabled = SAFE_TAKEOFF_DEFAULT
command_log_lock = threading.Lock()

# ArUco processor for virtual camera frames
_aruco_processor = None
_aruco_lock = threading.Lock()
_aruco_result: Dict = {}   # latest ArUco result per drone
_aruco_enabled = True       # can be toggled

def _get_aruco_processor():
    global _aruco_processor
    if _aruco_processor is None:
        try:
            from sim_aruco_processor import SimArucoProcessor
            _aruco_processor = SimArucoProcessor(
                frame_width=320, frame_height=240, fov_deg=75.0,
            )
            log.info("ArUco processor initialized")
        except ImportError as e:
            log.warning(f"ArUco processor not available (missing opencv?): {e}")
            _aruco_processor = False  # sentinel: don't retry
    return _aruco_processor if _aruco_processor is not False else None

# Telemetry received from the browser sim via WebSocket
telemetry: Dict = {
    "battery": 100,
    "temperature": 25,
    "height_cm": 0,
    "tof_cm": 0,
    "barometer_cm": 0,
    "flight_time_s": 0,
    "pitch": 0,
    "roll": 0,
    "yaw": 0,
    "vgx": 0,
    "vgy": 0,
    "vgz": 0,
    "agx": 0,
    "agy": 0,
    "agz": 0,
    "speed": 0,
    "flying": False,
    "connected": False,
    "updated_at": 0.0,
}
telemetry_lock = threading.Lock()

# Commands queued for the browser sim (consumed by WebSocket poll)
_cmd_queue: list[dict] = []
_cmd_lock = threading.Lock()

# Connected browser sim WebSocket clients
_ws_clients: list = []
_ws_lock = threading.Lock()

# Selected drone id in the simulator (set via /api/select_drone)
selected_drone_id: str = "B1"
selected_drone_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Key helpers (same interface as tello/olympe servers)
# ---------------------------------------------------------------------------

def normalize_key(k: str) -> str:
    return (k or "").lower()


def add_key(k: str):
    k = normalize_key(k)
    if not k:
        return
    with pressed_lock:
        pressed_web.add(k)
        key_last_seen[k] = time.time()


def remove_key(k: str):
    k = normalize_key(k)
    if not k:
        return
    with pressed_lock:
        pressed_web.discard(k)
        key_last_seen.pop(k, None)


def has_key(k: str) -> bool:
    k = normalize_key(k)
    timeout = KEY_STALE_S if KEY_STALE_S > 0 else 1.0
    with pressed_lock:
        ts = key_last_seen.get(k)
        if ts is not None and (time.time() - ts) > timeout:
            pressed_web.discard(k)
            key_last_seen.pop(k, None)
            return False
        return k in pressed_web


def reap_stale_keys(now: float):
    timeout = KEY_STALE_S if KEY_STALE_S > 0 else 1.0
    with pressed_lock:
        stale = [k for k, ts in key_last_seen.items() if (now - ts) > timeout]
        for k in stale:
            pressed_web.discard(k)
            key_last_seen.pop(k, None)


# ---------------------------------------------------------------------------
# Command queue helpers
# ---------------------------------------------------------------------------

def enqueue_cmd(cmd: dict):
    """Queue a command to be forwarded to the browser sim via WebSocket."""
    with _cmd_lock:
        _cmd_queue.append(cmd)
    _broadcast_cmd(cmd)


def _broadcast_cmd(cmd: dict):
    """Push command immediately to all connected WebSocket clients."""
    msg = json.dumps(cmd)
    with _ws_lock:
        dead = []
        for ws in _ws_clients:
            try:
                ws.send(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def append_command_log(event: str, payload: dict | None = None):
    if not COMMAND_LOG_ENABLED:
        return
    try:
        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            "payload": payload or {},
        }
        with command_log_lock:
            COMMAND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with COMMAND_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# WebSocket endpoint — browser sim connects here
# ---------------------------------------------------------------------------

@sock.route("/ws/sim")
def ws_sim(ws):
    """WebSocket bridge to the browser-based 3D simulator."""
    with _ws_lock:
        _ws_clients.append(ws)
    log.info("Simulator browser connected via WebSocket")

    # Send current selected drone
    with selected_drone_lock:
        drone_id = selected_drone_id
    try:
        ws.send(json.dumps({"type": "select_drone", "drone_id": drone_id}))
    except Exception:
        pass

    # Mark as connected
    with telemetry_lock:
        telemetry["connected"] = True
        telemetry["updated_at"] = time.time()

    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            try:
                data = json.loads(msg)
            except (json.JSONDecodeError, TypeError):
                continue

            msg_type = data.get("type")

            if msg_type == "telemetry":
                with telemetry_lock:
                    for k in telemetry:
                        if k in data:
                            telemetry[k] = data[k]
                    telemetry["connected"] = True
                    telemetry["updated_at"] = time.time()
                    # Merge ArUco position if available
                    with _aruco_lock:
                        if _aruco_result:
                            for k, v in _aruco_result.items():
                                telemetry[k] = v

            elif msg_type == "camera_frame":
                # Virtual camera frame for ArUco processing
                if _aruco_enabled:
                    proc = _get_aruco_processor()
                    if proc:
                        frame_b64 = data.get("frame")
                        drone = data.get("drone_id", "unknown")
                        if frame_b64:
                            try:
                                jpeg_bytes = base64.b64decode(frame_b64)
                                result = proc.process_frame(jpeg_bytes, drone)
                                arena_pos = proc.get_arena_position(result)
                                if arena_pos:
                                    with _aruco_lock:
                                        _aruco_result.update(arena_pos)
                                # Send result back to browser for debug overlay
                                try:
                                    ws.send(json.dumps({
                                        "type": "aruco_result",
                                        "markers_detected": result.get("markers_detected", 0),
                                        "ref_markers": result.get("ref_markers", []),
                                        "position": result.get("cam"),
                                        "arena": arena_pos,
                                    }))
                                except Exception:
                                    pass
                                # Also send as UDP (pi_position format) for C2 locator
                                proc.send_udp(result)
                            except Exception as e:
                                log.debug(f"ArUco frame error: {e}")
    except Exception:
        pass
    finally:
        with _ws_lock:
            if ws in _ws_clients:
                _ws_clients.remove(ws)
        with telemetry_lock:
            if not _ws_clients:
                telemetry["connected"] = False
        log.info("Simulator browser disconnected")


# ---------------------------------------------------------------------------
# Static file serving for the 3D simulator
# ---------------------------------------------------------------------------

@app.get("/sim/")
@app.get("/sim/<path:subpath>")
def serve_sim(subpath="index.html"):
    return send_from_directory(str(WEB3D_DIR), subpath)


# ---------------------------------------------------------------------------
# REST API — same interface as tello/olympe servers
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return jsonify(ok=True, service="sim_api_server", simulator=True)


@app.post("/api/key_down")
def api_key_down():
    data = request.get_json(silent=True) or {}
    k = data.get("key", "")
    add_key(k)
    enqueue_cmd({"type": "key_down", "key": k})
    append_command_log("key_down", {"key": k})
    return jsonify(ok=True)


@app.post("/api/key_up")
def api_key_up():
    data = request.get_json(silent=True) or {}
    k = data.get("key", "")
    remove_key(k)
    enqueue_cmd({"type": "key_up", "key": k})
    append_command_log("key_up", {"key": k})
    return jsonify(ok=True)


@app.post("/api/takeoff")
def api_takeoff():
    global flying
    flying = True
    enqueue_cmd({"type": "takeoff"})
    append_command_log("takeoff")
    with telemetry_lock:
        telemetry["flying"] = True
    return jsonify(ok=True)


@app.post("/api/land")
def api_land():
    global flying
    flying = False
    enqueue_cmd({"type": "land"})
    append_command_log("land")
    with telemetry_lock:
        telemetry["flying"] = False
    return jsonify(ok=True)


@app.post("/api/flip")
def api_flip():
    data = request.get_json(silent=True) or {}
    direction = data.get("direction", "f")
    enqueue_cmd({"type": "flip", "direction": direction})
    append_command_log("flip", {"direction": direction})
    return jsonify(ok=True)


@app.post("/api/emergency")
def api_emergency():
    global flying
    flying = False
    enqueue_cmd({"type": "emergency"})
    append_command_log("emergency")
    with telemetry_lock:
        telemetry["flying"] = False
    return jsonify(ok=True)


@app.post("/api/speed")
def api_speed():
    data = request.get_json(silent=True) or {}
    speed = data.get("speed", 50)
    enqueue_cmd({"type": "speed", "speed": speed})
    append_command_log("speed", {"speed": speed})
    return jsonify(ok=True)


@app.post("/api/move")
def api_move():
    data = request.get_json(silent=True) or {}
    direction = data.get("direction", "forward")
    distance = data.get("distance", 30)
    enqueue_cmd({"type": "move", "direction": direction, "distance": distance})
    append_command_log("move", {"direction": direction, "distance": distance})
    return jsonify(ok=True)


@app.post("/api/rotate")
def api_rotate():
    data = request.get_json(silent=True) or {}
    direction = data.get("direction", "cw")
    degrees = data.get("degrees", 90)
    enqueue_cmd({"type": "rotate", "direction": direction, "degrees": degrees})
    append_command_log("rotate", {"direction": direction, "degrees": degrees})
    return jsonify(ok=True)


@app.post("/api/go")
def api_go():
    data = request.get_json(silent=True) or {}
    enqueue_cmd({"type": "go", **data})
    append_command_log("go", data)
    return jsonify(ok=True)


@app.post("/api/curve")
def api_curve():
    data = request.get_json(silent=True) or {}
    enqueue_cmd({"type": "curve", **data})
    append_command_log("curve", data)
    return jsonify(ok=True)


@app.post("/api/rc")
def api_rc():
    data = request.get_json(silent=True) or {}
    lr = data.get("left_right", data.get("roll", 0))
    fb = data.get("forward_back", data.get("pitch", 0))
    ud = data.get("up_down", data.get("throttle", 0))
    yaw = data.get("yaw", 0)
    enqueue_cmd({"type": "rc", "roll": lr, "pitch": fb, "throttle": ud, "yaw": yaw})
    return jsonify(ok=True)


@app.post("/api/stream")
def api_stream():
    data = request.get_json(silent=True) or {}
    action = data.get("action", "on")
    return jsonify(ok=True, action=action)


@app.post("/api/sdk")
def api_sdk():
    data = request.get_json(silent=True) or {}
    return jsonify(ok=True, raw="ok simulator")


@app.post("/api/recover")
def api_recover():
    global flying
    flying = False
    enqueue_cmd({"type": "recover"})
    append_command_log("recover")
    return jsonify(ok=True)


@app.get("/api/safety/takeoff")
def api_safety_takeoff_get():
    return jsonify(ok=True, safe_takeoff=safe_takeoff_enabled, delay_s=SAFE_TAKEOFF_S)


@app.post("/api/safety/takeoff")
def api_safety_takeoff_post():
    global safe_takeoff_enabled
    data = request.get_json(silent=True) or {}
    safe_takeoff_enabled = bool(data.get("enabled", not safe_takeoff_enabled))
    return jsonify(ok=True, safe_takeoff=safe_takeoff_enabled)


@app.get("/api/telemetry")
def api_telemetry():
    with telemetry_lock:
        t = dict(telemetry)
    t["flying"] = flying
    age = time.time() - t.get("updated_at", 0)
    t["state_age_s"] = round(age, 3)
    t["state_fresh"] = age <= 3.0 and t.get("connected", False)
    resp = jsonify(t)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.get("/api/telemetry/stream")
def api_telemetry_stream():
    def gen():
        while True:
            with telemetry_lock:
                t = dict(telemetry)
            t["flying"] = flying
            age = time.time() - t.get("updated_at", 0)
            t["state_age_s"] = round(age, 3)
            t["state_fresh"] = age <= 3.0 and t.get("connected", False)
            yield f"data: {json.dumps(t)}\n\n"
            time.sleep(1.0 / max(0.1, TELEMETRY_HZ))
    return Response(gen(), mimetype="text/event-stream")


@app.get("/api/logging/commands")
def api_logging_commands():
    return jsonify(ok=True, enabled=COMMAND_LOG_ENABLED, path=str(COMMAND_LOG_PATH))


@app.get("/api/logging/commands/download")
def api_logging_commands_download():
    p = COMMAND_LOG_PATH
    if not p.exists():
        return jsonify(ok=False, error="no log file"), 404
    return Response(p.read_text(encoding="utf-8"), mimetype="application/x-ndjson",
                    headers={"Content-Disposition": f"attachment; filename={p.name}"})


@app.post("/api/logging/commands/clear")
def api_logging_commands_clear():
    with command_log_lock:
        if COMMAND_LOG_PATH.exists():
            COMMAND_LOG_PATH.write_text("", encoding="utf-8")
    return jsonify(ok=True)


@app.get("/api/logging/telemetry")
def api_logging_telemetry():
    return jsonify(ok=True, enabled=False)


@app.post("/api/logging/telemetry")
def api_logging_telemetry_post():
    return jsonify(ok=True, enabled=False)


@app.get("/api/logging/telemetry/download")
def api_logging_telemetry_download():
    return jsonify(ok=False, error="telemetry logging not available in simulator"), 404


@app.post("/api/logging/telemetry/clear")
def api_logging_telemetry_clear():
    return jsonify(ok=True)


# Simulator-specific: select which drone the API controls
@app.post("/api/select_drone")
def api_select_drone():
    global selected_drone_id
    data = request.get_json(silent=True) or {}
    drone_id = data.get("drone_id", "B1")
    with selected_drone_lock:
        selected_drone_id = drone_id
    enqueue_cmd({"type": "select_drone", "drone_id": drone_id})
    append_command_log("select_drone", {"drone_id": drone_id})
    return jsonify(ok=True, drone_id=drone_id)


@app.get("/api/select_drone")
def api_select_drone_get():
    with selected_drone_lock:
        drone_id = selected_drone_id
    return jsonify(ok=True, drone_id=drone_id)


# Camera / gimbal / RTH stubs (match olympe interface)
@app.post("/api/camera/photo")
def api_camera_photo():
    enqueue_cmd({"type": "camera_photo"})
    return jsonify(ok=True)


@app.post("/api/camera/record/start")
def api_camera_record_start():
    enqueue_cmd({"type": "camera_record_start"})
    return jsonify(ok=True)


@app.post("/api/camera/record/stop")
def api_camera_record_stop():
    enqueue_cmd({"type": "camera_record_stop"})
    return jsonify(ok=True)


@app.post("/api/gimbal")
def api_gimbal():
    data = request.get_json(silent=True) or {}
    enqueue_cmd({"type": "gimbal", **data})
    return jsonify(ok=True)


@app.post("/api/rth")
def api_rth():
    enqueue_cmd({"type": "rth"})
    append_command_log("rth")
    return jsonify(ok=True)


@app.post("/api/moveto")
def api_moveto():
    data = request.get_json(silent=True) or {}
    enqueue_cmd({"type": "moveto", **data})
    append_command_log("moveto", data)
    return jsonify(ok=True)


@app.get("/api/settings")
def api_settings_get():
    return jsonify(ok=True, settings={
        "max_altitude_m": 6.0,
        "max_vertical_speed": 1.0,
        "max_tilt": 15,
        "max_yaw_speed": 150,
        "max_horizontal_speed": 2.0,
        "geofence_radius_m": 30.0,
    })


@app.post("/api/settings")
def api_settings_post():
    data = request.get_json(silent=True) or {}
    enqueue_cmd({"type": "settings", **data})
    return jsonify(ok=True)


# ArUco processing endpoints
@app.get("/api/aruco")
def api_aruco_status():
    """Get ArUco processing status and latest result."""
    with _aruco_lock:
        result = dict(_aruco_result)
    proc = _get_aruco_processor()
    return jsonify(
        ok=True,
        enabled=_aruco_enabled,
        available=proc is not None,
        position=result,
        ref_markers=proc.last_ref_markers if proc else [],
        markers_detected=result.get("aruco_markers_detected", 0),
    )


@app.post("/api/aruco/toggle")
def api_aruco_toggle():
    """Enable/disable ArUco processing."""
    global _aruco_enabled
    data = request.get_json(silent=True) or {}
    _aruco_enabled = data.get("enabled", not _aruco_enabled)
    return jsonify(ok=True, enabled=_aruco_enabled)


@app.get("/api/aruco/frame")
def api_aruco_frame():
    """Get the latest annotated camera frame as JPEG."""
    proc = _get_aruco_processor()
    if proc and proc.last_annotated_frame:
        return Response(
            proc.last_annotated_frame,
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store"},
        )
    # Return a 1x1 black pixel JPEG if no frame available
    return Response(b"", status=204)


@app.get("/api/aruco/stream")
def api_aruco_stream():
    """MJPEG stream of annotated ArUco camera frames (for <img> tags)."""
    def generate():
        proc = _get_aruco_processor()
        last_frame = None
        while True:
            if proc is None:
                proc = _get_aruco_processor()
            if proc and proc.last_annotated_frame and proc.last_annotated_frame is not last_frame:
                last_frame = proc.last_annotated_frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + last_frame
                    + b"\r\n"
                )
            time.sleep(0.1)  # ~10fps max for MJPEG stream

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info(f"Simulator API server starting on {HTTP_HOST}:{HTTP_PORT}")
    log.info(f"3D Simulator UI: http://localhost:{HTTP_PORT}/sim/")
    log.info(f"WebSocket bridge: ws://localhost:{HTTP_PORT}/ws/sim")
    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True)
