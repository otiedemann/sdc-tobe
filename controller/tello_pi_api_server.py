import atexit
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Set

from djitellopy import Tello
from flask import Flask, Response, jsonify, request, send_file

# Pi API server (single drone)
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8080
RC_HZ = 20
STICK = 60
RECONNECT_AFTER_S = 3.0
RECONNECT_RETRY_S = 2.0
CONNECT_RETRY_S = 2.0
WIFI_RETRY_S = 3.0
TELEMETRY_HZ = float(os.getenv("TELEMETRY_HZ", "2.0"))
KEY_STALE_S = float(os.getenv("KEY_STALE_S", "1.0"))
SAFE_TAKEOFF_S = float(os.getenv("SAFE_TAKEOFF_S", "3.0"))
SAFE_TAKEOFF_DEFAULT = os.getenv("SAFE_TAKEOFF_DEFAULT", "0") in {"1", "true", "True"}
WIFI_CFG_PATH = Path(__file__).with_name("tello_wifi_config.json")
TELLO_HOST = "192.168.10.1"
TELEMETRY_LOG_DEFAULT = False
TELEMETRY_LOG_PATH_DEFAULT = Path(__file__).with_name("telemetry_log.jsonl")

app = Flask(__name__)

running = True
flying = False
TELLO = None

pressed_web: Set[str] = set()
key_last_seen: Dict[str, float] = {}
pressed_lock = threading.Lock()

last_state_seen = 0.0
conn_state = {"connected": False, "last_reconnect": 0.0}
conn_lock = threading.Lock()
last_conn_print = None

rc_override = None
rc_override_until = 0.0
rc_lock = threading.Lock()

telemetry = {
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
    "flying": False,
    "connected": False,
    "updated_at": 0.0,
}
telemetry_lock = threading.Lock()
telemetry_log_enabled = TELEMETRY_LOG_DEFAULT
telemetry_log_path = TELEMETRY_LOG_PATH_DEFAULT
telemetry_log_lock = threading.Lock()

command_lock = threading.Lock()
discrete_until = 0.0
takeoff_cooldown_until = 0.0
safe_takeoff_enabled = SAFE_TAKEOFF_DEFAULT
sdk_version_value = None
serial_number_value = None


def load_wifi_config():
    try:
        data = json.loads(WIFI_CFG_PATH.read_text())
        ssid = str(data.get("ssid", "")).strip()
        password = str(data.get("password", "")).strip()
        if not ssid or not password:
            return None
        return {"ssid": ssid, "password": password}
    except Exception:
        return None


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def host_reachable(host: str) -> bool:
    # Lightweight connectivity check to avoid stale SDK state showing false positives.
    r = _run(["ping", "-c", "1", "-W", "1", host])
    return r.returncode == 0


def wifi_connected_to(ssid: str) -> bool:
    r = _run(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"])
    if r.returncode != 0:
        return False
    for line in r.stdout.splitlines():
        if line.startswith("yes:") and line[4:] == ssid:
            return True
    return False


def wifi_connect_loop():
    while running:
        cfg = load_wifi_config()
        if not cfg:
            time.sleep(WIFI_RETRY_S)
            continue

        ssid = cfg["ssid"]
        password = cfg["password"]
        if wifi_connected_to(ssid):
            time.sleep(WIFI_RETRY_S)
            continue

        _run(["nmcli", "dev", "wifi", "connect", ssid, "password", password])
        time.sleep(WIFI_RETRY_S)


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


def _as_int(v):
    try:
        return int(float(v))
    except Exception:
        return None


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


def start_discrete_window(seconds: float):
    global discrete_until
    with command_lock:
        discrete_until = max(discrete_until, time.time() + max(0.0, seconds))


def recover_drone():
    global flying, last_state_seen, TELLO, rc_override, rc_override_until
    old = TELLO
    if old is None:
        return False, "controller not ready"
    start_discrete_window(2.0)
    try:
        with command_lock:
            try:
                old.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass
            try:
                old.land()
            except Exception:
                pass
            try:
                old.streamoff()
            except Exception:
                pass
            try:
                old.end()
            except Exception:
                pass

            # Recreate SDK object (critical after crash/disconnect states)
            t = Tello(host=TELLO_HOST)
            TELLO = t
            time.sleep(1.0)
            t.connect()
            t.streamon()
            refresh_drone_info_cache(t)

        # Clear any latent control inputs after crash/recover.
        with pressed_lock:
            pressed_web.clear()
            key_last_seen.clear()
        with rc_lock:
            rc_override = None
            rc_override_until = 0.0

        flying = False
        last_state_seen = 0.0
        with conn_lock:
            # Mark disconnected until state packets are actually received.
            conn_state["connected"] = False
            conn_state["last_reconnect"] = time.time()
        return True, "recovered_waiting_state"
    except Exception as e:
        with conn_lock:
            conn_state["connected"] = False
        return False, str(e)


def refresh_drone_info_cache(t):
    global sdk_version_value, serial_number_value
    if t is None:
        return
    try:
        if sdk_version_value is None:
            sdk_version_value = str(t.send_command_with_return("sdk?")).strip()
    except Exception:
        pass
    try:
        if serial_number_value is None:
            serial_number_value = str(t.send_command_with_return("sn?")).strip()
    except Exception:
        pass


def telemetry_loop():
    global last_state_seen
    while running:
        t = TELLO
        st = {}
        if t is not None:
            try:
                st = t.get_current_state() or {}
            except Exception:
                st = {}

        if st:
            last_state_seen = time.time()
            with conn_lock:
                conn_state["connected"] = True

        tl = _as_int(st.get("templ"))
        th = _as_int(st.get("temph"))
        temp = int((tl + th) / 2) if tl is not None and th is not None else (tl if tl is not None else th)

        with conn_lock:
            connected_now = conn_state["connected"]

        with telemetry_lock:
            # Keep last known values when no fresh state arrives, instead of wiping to null.
            for k, src in (
                ("battery", "bat"),
                ("height_cm", "h"),
                ("tof_cm", "tof"),
                ("barometer_cm", "baro"),
                ("flight_time_s", "time"),
                ("wifi_snr", "wifi"),
                ("pitch", "pitch"),
                ("roll", "roll"),
                ("yaw", "yaw"),
                ("vgx", "vgx"),
                ("vgy", "vgy"),
                ("vgz", "vgz"),
                ("agx", "agx"),
                ("agy", "agy"),
                ("agz", "agz"),
                ("mid", "mid"),
                ("pad_x", "x"),
                ("pad_y", "y"),
                ("pad_z", "z"),
            ):
                v = _as_int(st.get(src))
                if v is not None:
                    telemetry[k] = v
            mpry = st.get("mpry")
            if mpry is not None:
                telemetry["pad_mpry"] = str(mpry)
            if temp is not None:
                telemetry["temperature"] = temp

            vgx = telemetry.get("vgx") or 0
            vgy = telemetry.get("vgy") or 0
            vgz = telemetry.get("vgz") or 0
            telemetry["speed"] = int((vgx * vgx + vgy * vgy + vgz * vgz) ** 0.5)
            telemetry["sdk_version"] = sdk_version_value
            telemetry["serial_number"] = serial_number_value
            telemetry["flying"] = flying
            telemetry["connected"] = connected_now
            telemetry["updated_at"] = time.time()
            snapshot = dict(telemetry)

        append_telemetry_log(snapshot)
        hz = TELEMETRY_HZ if TELEMETRY_HZ > 0 else 2.0
        time.sleep(max(0.02, 1.0 / hz))


def reconnect_loop():
    global last_state_seen, last_conn_print, TELLO
    while running:
        now = time.time()
        stale = (now - last_state_seen) if last_state_seen else 9999
        reachable = host_reachable(TELLO_HOST)

        with conn_lock:
            last_try = conn_state["last_reconnect"]
            connected_now = conn_state["connected"]

        # Force disconnected when host is unreachable (prevents stale "connected" state).
        if not reachable and connected_now:
            with conn_lock:
                conn_state["connected"] = False
            connected_now = False

        if connected_now != last_conn_print:
            print("[PI API] Drone connected" if connected_now else "[PI API] Drone disconnected (retrying...)")
            last_conn_print = connected_now

        should_retry = reachable and (
            (not connected_now and (now - last_try) >= CONNECT_RETRY_S) or
            (stale > RECONNECT_AFTER_S and (now - last_try) >= RECONNECT_RETRY_S)
        )

        if should_retry:
            with conn_lock:
                conn_state["last_reconnect"] = now
                conn_state["connected"] = False
            try:
                if TELLO is None:
                    TELLO = Tello(host=TELLO_HOST)
                TELLO.connect()
                TELLO.streamon()  # keep stream alive for external UDP forward
                refresh_drone_info_cache(TELLO)
                # Wait for telemetry loop to confirm real state packets before marking connected.
                last_state_seen = 0.0
                with conn_lock:
                    conn_state["connected"] = False
            except Exception:
                pass
        time.sleep(0.8)


def rc_loop():
    global running, flying, rc_override, takeoff_cooldown_until
    period = 1.0 / RC_HZ
    while running:
        t0 = time.time()
        reap_stale_keys(t0)
        t = TELLO

        if has_key("t") and not flying:
            try:
                if t is not None:
                    hold_s = SAFE_TAKEOFF_S if safe_takeoff_enabled else 3.0
                    start_discrete_window(hold_s)
                    with command_lock:
                        t.send_rc_control(0, 0, 0, 0)
                        t.takeoff()
                    flying = True
                    takeoff_cooldown_until = time.time() + hold_s
            except Exception:
                recover_drone()
            remove_key("t")

        if has_key("l") and flying:
            try:
                if t is not None:
                    start_discrete_window(1.5)
                    with command_lock:
                        t.send_rc_control(0, 0, 0, 0)
                        t.land()
            except Exception:
                pass
            flying = False
            remove_key("l")

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

        try:
            if t is not None:
                if in_discrete:
                    t.send_rc_control(0, 0, 0, 0)
                else:
                    t.send_rc_control(lr, fb, ud, yaw)
        except Exception:
            with conn_lock:
                conn_state["connected"] = False

        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def shutdown():
    global running
    running = False
    t = TELLO
    if t is None:
        return
    try:
        t.send_rc_control(0, 0, 0, 0)
    except Exception:
        pass
    if flying:
        try:
            t.land()
        except Exception:
            pass
    try:
        t.end()
    except Exception:
        pass


@app.get("/")
def root():
    return jsonify(ok=True, service="tello_pi_api_server")


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


@app.post("/api/takeoff")
def api_takeoff():
    global flying, takeoff_cooldown_until
    if TELLO is None:
        return jsonify(ok=False, error="controller not ready"), 503
    try:
        if not flying:
            hold_s = SAFE_TAKEOFF_S if safe_takeoff_enabled else 3.0
            start_discrete_window(hold_s)
            with command_lock:
                TELLO.send_rc_control(0, 0, 0, 0)
                TELLO.takeoff()
            flying = True
            takeoff_cooldown_until = time.time() + hold_s
        return jsonify(ok=True, flying=flying, safe_takeoff=safe_takeoff_enabled)
    except Exception:
        ok, msg = recover_drone()
        return jsonify(ok=False, error="takeoff_failed", recovered=ok, message=msg), 500


@app.post("/api/land")
def api_land():
    global flying
    if TELLO is None:
        return jsonify(ok=False, error="controller not ready"), 503
    try:
        if flying:
            start_discrete_window(1.5)
            with command_lock:
                TELLO.send_rc_control(0, 0, 0, 0)
                TELLO.land()
            flying = False
        return jsonify(ok=True, flying=flying)
    except Exception:
        ok, msg = recover_drone()
        return jsonify(ok=False, error="land_failed", recovered=ok, message=msg), 500


@app.post("/api/flip")
def api_flip():
    if TELLO is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    direction = str(data.get("dir", "")).lower()
    if direction not in {"l", "r", "f", "b"}:
        return jsonify(ok=False, error="dir must be one of l|r|f|b"), 400

    now = time.time()
    if now < takeoff_cooldown_until:
        return jsonify(ok=False, error="flip_blocked_takeoff_cooldown"), 409

    with telemetry_lock:
        bat = telemetry.get("battery")
        vgx = abs(int(telemetry.get("vgx") or 0))
        vgy = abs(int(telemetry.get("vgy") or 0))
        is_flying = bool(telemetry.get("flying"))

    if not is_flying:
        return jsonify(ok=False, error="flip_requires_flying"), 409
    if bat is not None and bat < 50:
        return jsonify(ok=False, error="flip_requires_battery_50_plus", battery=bat), 409
    if vgx > 20 or vgy > 20:
        return jsonify(ok=False, error="flip_requires_low_horizontal_speed", vgx=vgx, vgy=vgy), 409

    try:
        start_discrete_window(1.2)
        with command_lock:
            TELLO.send_rc_control(0, 0, 0, 0)
            time.sleep(0.25)
            try:
                TELLO.flip(direction)
            except Exception:
                time.sleep(0.25)
                TELLO.flip(direction)
        return jsonify(ok=True, dir=direction)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/emergency")
def api_emergency():
    global flying
    if TELLO is None:
        return jsonify(ok=False, error="controller not ready"), 503
    try:
        with command_lock:
            TELLO.emergency()
        flying = False
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/speed")
def api_speed():
    if TELLO is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    try:
        speed = int(data.get("speed", 30))
        speed = max(10, min(100, speed))
        with command_lock:
            TELLO.set_speed(speed)
        return jsonify(ok=True, speed=speed)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/move")
def api_move():
    if TELLO is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    direction = str(data.get("dir", "")).lower()
    if direction not in {"up", "down", "left", "right", "forward", "back"}:
        return jsonify(ok=False, error="dir must be one of up|down|left|right|forward|back"), 400
    try:
        dist_cm = int(data.get("cm", 20))
        dist_cm = max(20, min(500, dist_cm))
        start_discrete_window(1.0)
        with command_lock:
            TELLO.send_rc_control(0, 0, 0, 0)
            fn = {
                "up": TELLO.move_up,
                "down": TELLO.move_down,
                "left": TELLO.move_left,
                "right": TELLO.move_right,
                "forward": TELLO.move_forward,
                "back": TELLO.move_back,
            }[direction]
            fn(dist_cm)
        return jsonify(ok=True, dir=direction, cm=dist_cm)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/rotate")
def api_rotate():
    if TELLO is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    direction = str(data.get("dir", "")).lower()
    if direction not in {"cw", "ccw"}:
        return jsonify(ok=False, error="dir must be one of cw|ccw"), 400
    try:
        degrees = int(data.get("deg", 45))
        degrees = max(1, min(360, degrees))
        start_discrete_window(1.0)
        with command_lock:
            TELLO.send_rc_control(0, 0, 0, 0)
            if direction == "cw":
                TELLO.rotate_clockwise(degrees)
            else:
                TELLO.rotate_counter_clockwise(degrees)
        return jsonify(ok=True, dir=direction, deg=degrees)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/go")
def api_go():
    if TELLO is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    try:
        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        z = int(data.get("z", 0))
        speed = int(data.get("speed", 20))
        speed = max(10, min(100, speed))
        start_discrete_window(1.5)
        with command_lock:
            TELLO.send_rc_control(0, 0, 0, 0)
            TELLO.go_xyz_speed(x, y, z, speed)
        return jsonify(ok=True, x=x, y=y, z=z, speed=speed)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/curve")
def api_curve():
    if TELLO is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    try:
        x1 = int(data.get("x1", 20))
        y1 = int(data.get("y1", 0))
        z1 = int(data.get("z1", 0))
        x2 = int(data.get("x2", 40))
        y2 = int(data.get("y2", 0))
        z2 = int(data.get("z2", 0))
        speed = int(data.get("speed", 20))
        speed = max(10, min(60, speed))
        start_discrete_window(1.5)
        with command_lock:
            TELLO.send_rc_control(0, 0, 0, 0)
            TELLO.curve_xyz_speed(x1, y1, z1, x2, y2, z2, speed)
        return jsonify(ok=True, x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2, speed=speed)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/stream")
def api_stream():
    if TELLO is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    action = str(data.get("action", "on")).lower()
    try:
        with command_lock:
            if action == "on":
                TELLO.streamon()
            elif action == "off":
                TELLO.streamoff()
            else:
                return jsonify(ok=False, error="action must be on|off"), 400
        return jsonify(ok=True, action=action)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/sdk")
def api_sdk_passthrough():
    """Raw SDK passthrough for advanced commands not explicitly mapped above."""
    if TELLO is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    command = str(data.get("command", "")).strip()
    if not command:
        return jsonify(ok=False, error="command required"), 400
    try:
        start_discrete_window(0.8)
        with command_lock:
            TELLO.send_rc_control(0, 0, 0, 0)
            resp = TELLO.send_command_with_return(command)
        return jsonify(ok=True, command=command, response=resp)
    except Exception as e:
        return jsonify(ok=False, error=str(e), command=command), 500


@app.post("/api/rc")
def api_rc():
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
    ok, msg = recover_drone()
    return jsonify(ok=ok, message=msg)


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


@app.get("/api/telemetry")
def api_telemetry():
    now = time.time()
    age = (now - last_state_seen) if last_state_seen else 9999.0
    with telemetry_lock:
        payload = dict(telemetry)
    payload["state_age_s"] = round(age, 3)
    payload["state_fresh"] = age <= 2.0
    return jsonify(payload)


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


@app.get("/api/telemetry/stream")
def api_telemetry_stream():
    def gen():
        while running:
            with telemetry_lock:
                payload = dict(telemetry)
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.4)

    return Response(gen(), mimetype="text/event-stream")


def main():
    global TELLO, last_state_seen
    logging.getLogger("djitellopy").setLevel(logging.CRITICAL)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    TELLO = Tello(host=TELLO_HOST)
    # Do not fail startup when drone is absent. Background reconnect loop will keep trying.
    last_state_seen = 0.0
    with conn_lock:
        conn_state["connected"] = False
        conn_state["last_reconnect"] = 0.0

    atexit.register(shutdown)

    threading.Thread(target=wifi_connect_loop, daemon=True).start()
    threading.Thread(target=telemetry_loop, daemon=True).start()
    threading.Thread(target=reconnect_loop, daemon=True).start()
    threading.Thread(target=rc_loop, daemon=True).start()

    print(f"http://{HTTP_HOST}:{HTTP_PORT} (waiting for Tello; auto-reconnect enabled)")
    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
