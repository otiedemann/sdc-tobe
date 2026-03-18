import atexit
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Set

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
    "wifi_snr": None,
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


def remove_key(k: str):
    k = normalize_key(k)
    if not k:
        return
    with pressed_lock:
        pressed_web.discard(k)


def has_key(k: str) -> bool:
    with pressed_lock:
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
    global flying, last_state_seen, TELLO
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

        flying = False
        last_state_seen = time.time()
        with conn_lock:
            conn_state["connected"] = True
            conn_state["last_reconnect"] = time.time()
        return True, "recovered"
    except Exception as e:
        with conn_lock:
            conn_state["connected"] = False
        return False, str(e)


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
            telemetry["battery"] = _as_int(st.get("bat"))
            telemetry["temperature"] = temp
            telemetry["height_cm"] = _as_int(st.get("h"))
            telemetry["tof_cm"] = _as_int(st.get("tof"))
            telemetry["barometer_cm"] = _as_int(st.get("baro"))
            telemetry["flight_time_s"] = _as_int(st.get("time"))
            telemetry["wifi_snr"] = _as_int(st.get("wifi"))
            telemetry["pitch"] = _as_int(st.get("pitch"))
            telemetry["roll"] = _as_int(st.get("roll"))
            telemetry["yaw"] = _as_int(st.get("yaw"))
            telemetry["vgx"] = _as_int(st.get("vgx"))
            telemetry["vgy"] = _as_int(st.get("vgy"))
            telemetry["vgz"] = _as_int(st.get("vgz"))
            telemetry["agx"] = _as_int(st.get("agx"))
            telemetry["agy"] = _as_int(st.get("agy"))
            telemetry["agz"] = _as_int(st.get("agz"))
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
                last_state_seen = time.time()
                with conn_lock:
                    conn_state["connected"] = True
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
                    start_discrete_window(3.0)
                    with command_lock:
                        t.send_rc_control(0, 0, 0, 0)
                        t.takeoff()
                    flying = True
                    takeoff_cooldown_until = time.time() + 3.0
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
            start_discrete_window(3.0)
            with command_lock:
                TELLO.send_rc_control(0, 0, 0, 0)
                TELLO.takeoff()
            flying = True
            takeoff_cooldown_until = time.time() + 3.0
        return jsonify(ok=True, flying=flying)
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


@app.get("/api/telemetry")
def api_telemetry():
    with telemetry_lock:
        return jsonify(telemetry)


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
