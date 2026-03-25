import atexit
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import olympe
from flask import Flask, Response, jsonify, request, send_file
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveBy, PCMD
from olympe.messages.ardrone3.PilotingState import (
    FlyingStateChanged,
    AttitudeChanged,
    SpeedChanged,
    AltitudeChanged,
)
from olympe.messages.common.CommonState import BatteryStateChanged
from olympe.messages.ardrone3.Animations import Flip

try:
    from olympe.messages.ardrone3.Piloting import Emergency as EmergencyCmd
    HAS_EMERGENCY = True
except ImportError:
    HAS_EMERGENCY = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8080
DRONE_IP = os.getenv("ANAFI_IP", "192.168.42.1")
RC_HZ = 20
STICK = 60
CONNECT_RETRY_S = 3.0
TELEMETRY_HZ = float(os.getenv("TELEMETRY_HZ", "2.0"))
KEY_STALE_S = float(os.getenv("KEY_STALE_S", "1.0"))
SAFE_TAKEOFF_S = float(os.getenv("SAFE_TAKEOFF_S", "3.0"))
SAFE_TAKEOFF_DEFAULT = os.getenv("SAFE_TAKEOFF_DEFAULT", "0") in {"1", "true", "True"}
TELEMETRY_LOG_DEFAULT = False
TELEMETRY_LOG_PATH_DEFAULT = Path(__file__).with_name("telemetry_log.jsonl")
COMMAND_LOG_ENABLED = os.getenv("API_COMMAND_LOG", "1") in {"1", "true", "True"}
COMMAND_LOG_PATH = Path(
    os.getenv("API_COMMAND_LOG_PATH", str(Path(__file__).with_name("api_command_log.jsonl")))
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
app = Flask(__name__)
running = True
flying = False
drone: Optional[olympe.Drone] = None

pressed_web: Set[str] = set()
key_last_seen: Dict[str, float] = {}
pressed_lock = threading.Lock()

conn_state = {"connected": False, "last_reconnect": 0.0}
conn_lock = threading.Lock()
last_conn_print = None

rc_override: Optional[Tuple[int, int, int, int]] = None
rc_override_until = 0.0
rc_lock = threading.Lock()

telemetry: Dict = {
    "battery": None,
    "temperature": None,
    "height_cm": None,
    "tof_cm": None,
    "barometer_cm": None,
    "flight_time_s": None,
    "pitch": None,
    "roll": None,
    "yaw": None,
    "vgx": None,
    "vgy": None,
    "vgz": None,
    "agx": None,
    "agy": None,
    "agz": None,
    "speed": None,
    "flying": False,
    "connected": False,
    "updated_at": 0.0,
}
telemetry_lock = threading.Lock()
telemetry_log_enabled = TELEMETRY_LOG_DEFAULT
telemetry_log_path = TELEMETRY_LOG_PATH_DEFAULT
telemetry_log_lock = threading.Lock()
command_log_lock = threading.Lock()

command_lock = threading.Lock()
discrete_until = 0.0
takeoff_cooldown_until = 0.0
safe_takeoff_enabled = SAFE_TAKEOFF_DEFAULT

# Sequence number counter for PCMD
_pcmd_seq = 0
_pcmd_seq_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Key helpers
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


def reap_stale_keys(now: float):
    timeout = KEY_STALE_S if KEY_STALE_S > 0 else 1.0
    with pressed_lock:
        stale = [k for k, ts in key_last_seen.items() if (now - ts) > timeout]
        for k in stale:
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


def axis(pos: bool, neg: bool) -> int:
    return (1 if pos else 0) + (-1 if neg else 0)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def append_telemetry_log(payload: dict):
    global telemetry_log_enabled, telemetry_log_path
    with telemetry_log_lock:
        if not telemetry_log_enabled:
            return
        p = telemetry_log_path
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


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
# Discrete command window (blocks RC during/after discrete commands)
# ---------------------------------------------------------------------------

def start_discrete_window(seconds: float):
    global discrete_until
    with command_lock:
        discrete_until = max(discrete_until, time.time() + max(0.0, seconds))


# ---------------------------------------------------------------------------
# PCMD helper
# ---------------------------------------------------------------------------

def _send_pcmd(roll: int, pitch: int, yaw: int, gaz: int):
    """Send a single PCMD frame. Values -100..100. Roll=lr, Pitch=fb, Gaz=ud, Yaw=yaw."""
    global _pcmd_seq
    d = drone
    if d is None:
        return
    with _pcmd_seq_lock:
        _pcmd_seq = (_pcmd_seq + 1) & 0x7FFFFFFF
        seq = _pcmd_seq
    try:
        d(PCMD(1, roll, pitch, yaw, gaz, seq))
    except Exception:
        pass


def _stop_pcmd():
    """Send a zero PCMD to stop movement."""
    _send_pcmd(0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Telemetry collection
# ---------------------------------------------------------------------------

def _get_drone_state(msg_type):
    """Safely get the last state for a message type from Olympe."""
    d = drone
    if d is None:
        return None
    try:
        return d.get_state(msg_type)
    except Exception:
        return None


def _is_flying_state(state_dict) -> bool:
    """Return True if the Olympe FlyingStateChanged state indicates airborne."""
    if not state_dict:
        return False
    try:
        s = state_dict.get("state")
        if s is None:
            return False
        # state values: landed=0, takingoff=1, hovering=2, flying=3, landing=4, emergency=5
        name = str(s).lower()
        return any(x in name for x in ("hovering", "flying", "takingoff"))
    except Exception:
        return False


def telemetry_loop():
    global flying
    while running:
        with conn_lock:
            connected_now = conn_state["connected"]
        state = drone.get_state(olympe.messages.ardrone3.PilotingState.FlyingStateChanged())
        if state:
            telemetry.update(state)
        telemetry.update(state)
        time.sleep(0.5)

        # --- Battery ---
        bat = None
        bat_state = _get_drone_state(BatteryStateChanged)
        if bat_state:
            try:
                bat = int(bat_state["percent"])
            except Exception:
                pass

        # --- Attitude (pitch/roll/yaw in radians → degrees) ---
        pitch_deg = roll_deg = yaw_deg = None
        att_state = _get_drone_state(AttitudeChanged)
        if att_state:
            try:
                pitch_deg = round(math.degrees(float(att_state["pitch"])), 2)
                roll_deg = round(math.degrees(float(att_state["roll"])), 2)
                yaw_deg = round(math.degrees(float(att_state["yaw"])), 2)
            except Exception:
                pass

        # --- Speed (m/s → cm/s to match Tello convention) ---
        vgx = vgy = vgz = None
        spd_state = _get_drone_state(SpeedChanged)
        if spd_state:
            try:
                # Olympe SpeedChanged: speedX=forward, speedY=right, speedZ=down (NED)
                vgx = round(float(spd_state["speedX"]) * 100, 1)
                vgy = round(float(spd_state["speedY"]) * 100, 1)
                vgz = round(float(spd_state["speedZ"]) * 100, 1)
            except Exception:
                pass

        # --- Altitude (m → cm) ---
        height_cm = None
        alt_state = _get_drone_state(AltitudeChanged)
        if alt_state:
            try:
                height_cm = round(float(alt_state["altitude"]) * 100, 1)
            except Exception:
                pass

        # --- Flying state ---
        fly_state = _get_drone_state(FlyingStateChanged)
        sdk_flying = _is_flying_state(fly_state)
        # Update global flying flag from SDK state when not in a discrete command window.
        with command_lock:
            in_discrete = time.time() < discrete_until
        if not in_discrete:
            flying = sdk_flying

        # Compute aggregate speed
        speed = None
        if vgx is not None and vgy is not None and vgz is not None:
            speed = round((vgx ** 2 + vgy ** 2 + vgz ** 2) ** 0.5, 1)

        with telemetry_lock:
            if bat is not None:
                telemetry["battery"] = bat
            if pitch_deg is not None:
                telemetry["pitch"] = pitch_deg
            if roll_deg is not None:
                telemetry["roll"] = roll_deg
            if yaw_deg is not None:
                telemetry["yaw"] = yaw_deg
            if vgx is not None:
                telemetry["vgx"] = vgx
            if vgy is not None:
                telemetry["vgy"] = vgy
            if vgz is not None:
                telemetry["vgz"] = vgz
            if height_cm is not None:
                telemetry["height_cm"] = height_cm
            if speed is not None:
                telemetry["speed"] = speed
            telemetry["flying"] = flying
            telemetry["connected"] = connected_now
            telemetry["updated_at"] = time.time()
            snapshot = dict(telemetry)

        append_telemetry_log(snapshot)
        hz = TELEMETRY_HZ if TELEMETRY_HZ > 0 else 2.0
        time.sleep(max(0.05, 1.0 / hz))


# ---------------------------------------------------------------------------
# Reconnect loop
# ---------------------------------------------------------------------------

def reconnect_loop():
    global drone, last_conn_print
    while running:
        with conn_lock:
            last_try = conn_state["last_reconnect"]
            connected_now = conn_state["connected"]

        if connected_now != last_conn_print:
            print("[ANAFI API] Drone connected" if connected_now else "[ANAFI API] Drone disconnected (retrying...)")
            last_conn_print = connected_now

        now = time.time()
        should_retry = (not connected_now) and (now - last_try) >= CONNECT_RETRY_S

        if should_retry:
            with conn_lock:
                conn_state["last_reconnect"] = now
            d = drone
            if d is None:
                d = olympe.Drone(DRONE_IP)
                drone = d
            try:
                connected = d.connect()
                with conn_lock:
                    conn_state["connected"] = bool(connected)
            except Exception as e:
                print(f"[ANAFI API] Connect failed: {e}")
                with conn_lock:
                    conn_state["connected"] = False

        time.sleep(1.0)


# ---------------------------------------------------------------------------
# RC/PCMD loop
# ---------------------------------------------------------------------------

def rc_loop():
    global running, flying, rc_override, rc_override_until, takeoff_cooldown_until
    period = 1.0 / RC_HZ
    while running:
        t0 = time.time()
        reap_stale_keys(t0)

        with conn_lock:
            connected = conn_state["connected"]

        if not connected:
            time.sleep(period)
            continue

        # Takeoff via key
        if has_key("t") and not flying:
            try:
                start_discrete_window(SAFE_TAKEOFF_S if safe_takeoff_enabled else 3.0)
                with command_lock:
                    _stop_pcmd()
                    result = drone(TakeOff()).wait(_timeout=5)
                if result and result.success():
                    flying = True
                    takeoff_cooldown_until = time.time() + (SAFE_TAKEOFF_S if safe_takeoff_enabled else 3.0)
            except Exception as e:
                print(f"[ANAFI API] Takeoff error: {e}")
            remove_key("t")

        # Land via key
        if has_key("l") and flying:
            try:
                start_discrete_window(3.0)
                with pressed_lock:
                    pressed_web.clear()
                    key_last_seen.clear()
                with rc_lock:
                    rc_override = None
                    rc_override_until = 0.0
                _stop_pcmd()
                with command_lock:
                    drone(Landing()).wait(_timeout=5)
                flying = False
            except Exception as e:
                print(f"[ANAFI API] Land error: {e}")
            remove_key("l")

        # Build RC axes from held keys
        # Tello mapping: lr=left/right, fb=fwd/back, ud=up/down, yaw=cw/ccw
        # Olympe PCMD: roll=lr, pitch=fb, gaz=ud, yaw=yaw
        lr = axis(has_key("d"), has_key("a")) * STICK
        fb = axis(has_key("w"), has_key("s")) * STICK
        ud = axis(has_key("r"), has_key("f")) * STICK
        yaw = axis(has_key("e"), has_key("q")) * STICK

        if has_key("space") or has_key("x"):
            lr = fb = ud = yaw = 0

        now = time.time()
        with rc_lock:
            if rc_override is not None and now < rc_override_until:
                lr, fb, ud, yaw = rc_override
            elif rc_override is not None and now >= rc_override_until:
                rc_override = None

        with command_lock:
            in_discrete = now < discrete_until

        if in_discrete:
            _stop_pcmd()
        else:
            _send_pcmd(lr, fb, yaw, ud)

        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def shutdown():
    global running
    running = False
    d = drone
    if d is None:
        return
    try:
        _stop_pcmd()
    except Exception:
        pass
    if flying:
        try:
            d(Landing()).wait(_timeout=5)
        except Exception:
            pass
    try:
        d.disconnect()
        print("[ANAFI API] Drone disconnected.")
    except Exception as e:
        print(f"[ANAFI API] Shutdown error: {e}")


# ---------------------------------------------------------------------------
# Flask middleware
# ---------------------------------------------------------------------------

@app.before_request
def _log_incoming_api_command():
    try:
        if request.method != "POST":
            return
        if not request.path.startswith("/api/"):
            return
        if request.path.startswith("/api/logging/"):
            return
        payload = request.get_json(silent=True) or {}
        append_command_log(request.path, payload)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    with conn_lock:
        connected = conn_state["connected"]
    return jsonify(ok=True, service="olympe_pi_api_server", connected=connected)


@app.post("/api/key_down")
def api_key_down():
    data = request.get_json(silent=True) or {}
    add_key(data.get("key", ""))
    return jsonify(ok=True)


@app.post("/api/key_up")
def api_key_up():
    data = request.get_json(silent=True) or {}
    remove_key(data.get("key", ""))
    return jsonify(ok=True)
    dx = float(data.get("dx", 0))
    dy = float(data.get("dy", 0))
    dz = float(data.get("dz", 0))
    try:
        if drone(moveBy(dx, dy, dz, 0)).wait().success():
            return jsonify(ok=True)
    except Exception as e:
        print(f"Move command failed: {e}")
    return jsonify(ok=False, error="move_failed"), 500


@app.post("/api/takeoff")
def api_takeoff():
    global flying, takeoff_cooldown_until
    d = drone
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or d is None:
        return jsonify(ok=False, error="controller not ready"), 503
    try:
        if not flying:
            hold_s = SAFE_TAKEOFF_S if safe_takeoff_enabled else 3.0
            start_discrete_window(hold_s)
            with command_lock:
                _stop_pcmd()
                result = d(TakeOff()).wait(_timeout=8)
            if result and result.success():
                flying = True
                takeoff_cooldown_until = time.time() + hold_s
            else:
                return jsonify(ok=False, error="takeoff_failed"), 500
        return jsonify(ok=True, flying=flying, safe_takeoff=safe_takeoff_enabled)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/land")
def api_land():
    global flying, rc_override, rc_override_until
    d = drone
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or d is None:
        return jsonify(ok=False, error="controller not ready"), 503
    try:
        if not flying:
            return jsonify(ok=True, flying=False)
        start_discrete_window(3.0)
        with pressed_lock:
            pressed_web.clear()
            key_last_seen.clear()
        with rc_lock:
            rc_override = None
            rc_override_until = 0.0
        last_err = None
        for _ in range(3):
            try:
                _stop_pcmd()
                with command_lock:
                    result = d(Landing()).wait(_timeout=8)
                if result and result.success():
                    flying = False
                    return jsonify(ok=True, flying=False)
            except Exception as e:
                last_err = e
                time.sleep(0.25)
        raise last_err if last_err else RuntimeError("land_failed")
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/flip")
def api_flip():
    d = drone
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or d is None:
        return jsonify(ok=False, error="controller not ready"), 503

    data = request.get_json(silent=True) or {}
    direction = str(data.get("dir", data.get("direction", ""))).lower()

    # Accept both l/r/f/b and full names (front/back/left/right)
    dir_map = {
        "f": "front", "front": "front",
        "b": "back",  "back": "back",
        "l": "left",  "left": "left",
        "r": "right", "right": "right",
    }
    if direction not in dir_map:
        return jsonify(ok=False, error="dir must be one of l|r|f|b (or front|back|left|right)"), 400

    now = time.time()
    if now < takeoff_cooldown_until:
        return jsonify(ok=False, error="flip_blocked_takeoff_cooldown"), 409

    with telemetry_lock:
        bat = telemetry.get("battery")
        is_flying = bool(telemetry.get("flying"))

    if not is_flying:
        return jsonify(ok=False, error="flip_requires_flying"), 409
    if bat is not None and bat < 50:
        return jsonify(ok=False, error="flip_requires_battery_50_plus", battery=bat), 409

    flip_dir_map = {
        "front": Flip.direction.front,
        "back": Flip.direction.back,
        "left": Flip.direction.left,
        "right": Flip.direction.right,
    }
    olympe_dir = flip_dir_map[dir_map[direction]]

    try:
        start_discrete_window(2.0)
        with command_lock:
            _stop_pcmd()
            result = d(Flip(olympe_dir)).wait(_timeout=5)
        if result and result.success():
            return jsonify(ok=True, dir=direction)
        return jsonify(ok=False, error="flip_failed"), 500
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/emergency")
def api_emergency():
    global flying
    d = drone
    if d is None:
        return jsonify(ok=False, error="controller not ready"), 503
    try:
        if HAS_EMERGENCY:
            with command_lock:
                d(EmergencyCmd()).wait(_timeout=3)
        else:
            # Fallback: cut PCMD and land
            _stop_pcmd()
            with command_lock:
                d(Landing()).wait(_timeout=5)
        flying = False
        return jsonify(ok=True)
    except Exception as e:
        flying = False
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/move")
def api_move():
    """Move in a cardinal direction by a distance in cm (matches Tello API)."""
    d = drone
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or d is None:
        return jsonify(ok=False, error="controller not ready"), 503

    data = request.get_json(silent=True) or {}
    direction = str(data.get("dir", "")).lower()
    if direction not in {"up", "down", "left", "right", "forward", "back"}:
        return jsonify(ok=False, error="dir must be one of up|down|left|right|forward|back"), 400

    try:
        dist_cm = int(data.get("cm", 20))
        dist_cm = max(20, min(500, dist_cm))
        dist_m = dist_cm / 100.0

        # Olympe moveBy: dX=forward(+), dY=right(+), dZ=down(+), dPsi=yaw(rad)
        move_args = {
            "forward": (dist_m, 0, 0, 0),
            "back":    (-dist_m, 0, 0, 0),
            "right":   (0, dist_m, 0, 0),
            "left":    (0, -dist_m, 0, 0),
            "up":      (0, 0, -dist_m, 0),
            "down":    (0, 0, dist_m, 0),
        }[direction]

        start_discrete_window(max(1.0, dist_m * 2))
        with command_lock:
            _stop_pcmd()
            result = d(moveBy(*move_args)).wait(_timeout=30)
        if result and result.success():
            return jsonify(ok=True, dir=direction, cm=dist_cm)
        return jsonify(ok=False, error="move_failed"), 500
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/rotate")
def api_rotate():
    """Rotate clockwise or counter-clockwise by degrees (matches Tello API)."""
    d = drone
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or d is None:
        return jsonify(ok=False, error="controller not ready"), 503

    data = request.get_json(silent=True) or {}
    direction = str(data.get("dir", "")).lower()
    if direction not in {"cw", "ccw"}:
        return jsonify(ok=False, error="dir must be one of cw|ccw"), 400

    try:
        degrees = int(data.get("deg", 45))
        degrees = max(1, min(360, degrees))
        rad = math.radians(degrees)
        d_psi = rad if direction == "cw" else -rad

        start_discrete_window(max(1.0, degrees / 90.0))
        with command_lock:
            _stop_pcmd()
            result = d(moveBy(0, 0, 0, d_psi)).wait(_timeout=15)
        if result and result.success():
            return jsonify(ok=True, dir=direction, deg=degrees)
        return jsonify(ok=False, error="rotate_failed"), 500
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/go")
def api_go():
    """Move to relative xyz position in cm (matches Tello API; speed param ignored)."""
    d = drone
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or d is None:
        return jsonify(ok=False, error="controller not ready"), 503

    data = request.get_json(silent=True) or {}
    try:
        x = int(data.get("x", 0))  # forward(+)/back(-) in cm
        y = int(data.get("y", 0))  # right(+)/left(-) in cm
        z = int(data.get("z", 0))  # up(+)/down(-) in cm  [Tello convention]

        # Olympe moveBy: dX=fwd, dY=right, dZ=down(+)=up(-)
        dx = x / 100.0
        dy = y / 100.0
        dz = -z / 100.0  # invert: Tello z+ is up, Olympe dZ+ is down

        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        start_discrete_window(max(1.5, dist * 2))
        with command_lock:
            _stop_pcmd()
            result = d(moveBy(dx, dy, dz, 0)).wait(_timeout=30)
        if result and result.success():
            return jsonify(ok=True, x=x, y=y, z=z)
        return jsonify(ok=False, error="go_failed"), 500
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/rc")
def api_rc():
    """RC override: set PCMD values for a duration. Matches Tello API (lr/fb/ud/yaw, -100..100)."""
    global rc_override, rc_override_until

    data = request.get_json(silent=True) or {}

    def clamp(v):
        try:
            return max(-100, min(100, int(v)))
        except Exception:
            return 0

    lr = clamp(data.get("lr", 0))
    fb = clamp(data.get("fb", 0))
    ud = clamp(data.get("ud", 0))
    yaw = clamp(data.get("yaw", 0))
    dur_ms = max(50, min(2000, int(data.get("duration_ms", 250))))

    with rc_lock:
        rc_override = (lr, fb, ud, yaw)
        rc_override_until = time.time() + (dur_ms / 1000.0)

    return jsonify(ok=True, rc={"lr": lr, "fb": fb, "ud": ud, "yaw": yaw}, duration_ms=dur_ms)


@app.post("/api/recover")
def api_recover():
    """Attempt to reconnect drone. Resets flying state and clears inputs."""
    global flying, drone, rc_override, rc_override_until
    d = drone
    try:
        if d is not None:
            try:
                d.disconnect()
            except Exception:
                pass
        d = olympe.Drone(DRONE_IP)
        drone = d
        with pressed_lock:
            pressed_web.clear()
            key_last_seen.clear()
        with rc_lock:
            rc_override = None
            rc_override_until = 0.0
        connected = d.connect()
        flying = False
        with conn_lock:
            conn_state["connected"] = bool(connected)
            conn_state["last_reconnect"] = time.time()
        return jsonify(ok=bool(connected), message="recovered" if connected else "reconnect_failed")
    except Exception as e:
        with conn_lock:
            conn_state["connected"] = False
        return jsonify(ok=False, message=str(e)), 500


@app.get("/api/telemetry")
def api_telemetry():
    with telemetry_lock:
        payload = dict(telemetry)
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.get("/api/telemetry/stream")
def api_telemetry_stream():
    def gen():
        while running:
            with telemetry_lock:
                payload = dict(telemetry)
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.4)

    return Response(gen(), mimetype="text/event-stream")


@app.get("/api/safety/takeoff")
def api_safe_takeoff_get():
    return jsonify(enabled=safe_takeoff_enabled, hold_s=SAFE_TAKEOFF_S)


@app.post("/api/safety/takeoff")
def api_safe_takeoff_set():
    global safe_takeoff_enabled
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled")
    if not isinstance(enabled, bool):
        return jsonify(ok=False, error="enabled must be boolean"), 400
    safe_takeoff_enabled = enabled
    return jsonify(ok=True, enabled=safe_takeoff_enabled, hold_s=SAFE_TAKEOFF_S)


@app.get("/api/logging/commands")
def api_command_log_status():
    return jsonify(enabled=COMMAND_LOG_ENABLED, path=str(COMMAND_LOG_PATH))


@app.get("/api/logging/commands/download")
def api_command_log_download():
    p = COMMAND_LOG_PATH
    if not p.exists():
        return jsonify(ok=False, error="command log file not found", path=str(p)), 404
    return send_file(p, as_attachment=True, download_name=p.name, mimetype="application/x-ndjson")


@app.post("/api/logging/commands/clear")
def api_command_log_clear():
    p = COMMAND_LOG_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        return jsonify(ok=True, cleared=True, path=str(p))
    except Exception as e:
        return jsonify(ok=False, error=str(e), path=str(p)), 500


@app.get("/api/logging/telemetry")
def api_telemetry_log_status():
    with telemetry_log_lock:
        return jsonify(enabled=telemetry_log_enabled, path=str(telemetry_log_path))


@app.post("/api/logging/telemetry")
def api_telemetry_log_config():
    global telemetry_log_enabled, telemetry_log_path
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled")
    path = data.get("path")
    with telemetry_log_lock:
        if isinstance(enabled, bool):
            telemetry_log_enabled = enabled
        if isinstance(path, str) and path.strip():
            telemetry_log_path = Path(path.strip())
        return jsonify(enabled=telemetry_log_enabled, path=str(telemetry_log_path))


@app.get("/api/logging/telemetry/download")
def api_telemetry_log_download():
    with telemetry_log_lock:
        p = telemetry_log_path
    if not p.exists():
        return jsonify(ok=False, error="telemetry log file not found", path=str(p)), 404
    return send_file(p, as_attachment=True, download_name=p.name, mimetype="application/x-ndjson")


@app.post("/api/logging/telemetry/clear")
def api_telemetry_log_clear():
    with telemetry_log_lock:
        p = telemetry_log_path
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        return jsonify(ok=True, cleared=True, path=str(p))
    except Exception as e:
        return jsonify(ok=False, error=str(e), path=str(p)), 500


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def main():
    global drone
    logging.getLogger("olympe").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    drone = olympe.Drone(DRONE_IP)

    atexit.register(shutdown)

    threading.Thread(target=reconnect_loop, daemon=True).start()
    threading.Thread(target=telemetry_loop, daemon=True).start()
    threading.Thread(target=rc_loop, daemon=True).start()

    print(f"[ANAFI API] http://{HTTP_HOST}:{HTTP_PORT} (waiting for Anafi at {DRONE_IP}; auto-reconnect enabled)")
    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
