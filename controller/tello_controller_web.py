import atexit
import logging
import select
import sys
import termios
import threading
import time
import tty
from typing import Set

import cv2
from djitellopy import Tello
from flask import Flask, Response, jsonify, request

# ----- Tuning -----
RC_HZ = 20
STICK = 60
JPEG_QUALITY = 80
SWAP_CHANNELS = True  # set True if colors look wrong
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8080
RECONNECT_AFTER_S = 3.0
RECONNECT_RETRY_S = 2.0

# ----- Global state -----
app = Flask(__name__)

pressed_web: Set[str] = set()
pressed_term: Set[str] = set()
pressed_lock = threading.Lock()

running = True
flying = False

last_jpeg = b""
last_jpeg_lock = threading.Lock()

frame_reader = None
frame_lock = threading.Lock()

last_state_seen = 0.0
conn_state = {"connected": False, "last_reconnect": 0.0}
conn_lock = threading.Lock()

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

TELLO = None


# ----- Key handling -----
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


def consume_term_key(k: str) -> bool:
    with pressed_lock:
        if k in pressed_term:
            pressed_term.discard(k)
            return True
        return False


def has_key(k: str) -> bool:
    with pressed_lock:
        return (k in pressed_web) or (k in pressed_term)


def axis(pos: bool, neg: bool) -> int:
    return (1 if pos else 0) + (-1 if neg else 0)


# ----- Terminal input thread -----
def terminal_key_reader():
    global running
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while running:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not r:
                continue
            ch = sys.stdin.read(1)
            with pressed_lock:
                if ch == "\x1b":
                    pressed_term.add("esc")
                elif ch:
                    pressed_term.add(ch.lower())
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ----- Flask HTTP API/UI -----
HTML = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Tello Controller</title>
  <style>
    body { background:#0f172a; color:#e2e8f0; font-family:Arial,sans-serif; margin:0; padding:16px; }
    .row { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }
    .panel { background:#111827; border:1px solid #334155; border-radius:8px; padding:12px; }
    img { width: 640px; max-width: 100%; aspect-ratio:4/3; border-radius:8px; border:1px solid #334155; background:#000; }
    .grid { display:grid; grid-template-columns:repeat(3,70px); gap:8px; }
    button { height:52px; border-radius:8px; border:1px solid #475569; background:#1e293b; color:#e2e8f0; font-weight:700; cursor:pointer; }
    button:active, .active { background:#0ea5e9; color:#001018; }
    .small { color:#94a3b8; font-size:12px; }
  </style>
</head>
<body>
  <h2>Tello Controller (Web + Terminal)</h2>
  <div class=\"row\">
    <div class=\"panel\">
      <img src=\"/video\" alt=\"video\" />
      <div class=\"small\">Stream: /video</div>
    </div>
    <div class=\"panel\">
      <div class=\"grid\" id=\"grid\">
        <button data-k=\"q\">Q</button><button data-k=\"w\">W</button><button data-k=\"e\">E</button>
        <button data-k=\"a\">A</button><button data-k=\"x\">STOP</button><button data-k=\"d\">D</button>
        <button data-k=\"r\">R</button><button data-k=\"s\">S</button><button data-k=\"f\">F</button>
      </div>
      <div style=\"margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;\">
        <button id=\"takeoff\">Takeoff (T)</button>
        <button id=\"land\">Land (L)</button>
        <button id=\"recover\">Recover</button>
      </div>
      <div style=\"margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;\">
        <button id=\"flip_l\">Flip L</button>
        <button id=\"flip_r\">Flip R</button>
        <button id=\"flip_f\">Flip F</button>
        <button id=\"flip_b\">Flip B</button>
      </div>
      <div class=\"small\" style=\"margin-top:8px;\">Keyboard in browser: W/A/S/D R/F Q/E, T, L, Space stop</div>
      <div class=\"panel\" style=\"margin-top:10px;\">
        <div><b>Telemetry</b></div>
        <div id=\"telemetry\" class=\"small\" style=\"white-space:pre-wrap; margin-top:6px;\">loading...</div>
      </div>
    </div>
  </div>
<script>
async function post(url, body){ await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}); }

function keyDown(k){ post('/api/key_down',{key:k}); }
function keyUp(k){ post('/api/key_up',{key:k}); }

const holdButtons = document.querySelectorAll('button[data-k]');
holdButtons.forEach(btn=>{
  const k = btn.dataset.k;
  btn.addEventListener('pointerdown', e=>{ e.preventDefault(); btn.classList.add('active'); keyDown(k); });
  btn.addEventListener('pointerup',   e=>{ e.preventDefault(); btn.classList.remove('active'); keyUp(k); });
  btn.addEventListener('pointerleave',e=>{ btn.classList.remove('active'); keyUp(k); });
});

document.getElementById('takeoff').onclick = ()=>fetch('/api/takeoff',{method:'POST'});
document.getElementById('land').onclick = ()=>fetch('/api/land',{method:'POST'});
document.getElementById('recover').onclick = ()=>fetch('/api/recover',{method:'POST'});
document.getElementById('flip_l').onclick = ()=>fetch('/api/flip',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dir:'l'})});
document.getElementById('flip_r').onclick = ()=>fetch('/api/flip',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dir:'r'})});
document.getElementById('flip_f').onclick = ()=>fetch('/api/flip',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dir:'f'})});
document.getElementById('flip_b').onclick = ()=>fetch('/api/flip',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dir:'b'})});

const map = new Set(['w','a','s','d','q','e','r','f','t','l','x',' ']);
window.addEventListener('keydown', (e)=>{
  const k = e.key.toLowerCase();
  if (map.has(k)) {
    e.preventDefault();
    keyDown(k === ' ' ? 'space' : k);
  }
});
window.addEventListener('keyup', (e)=>{
  const k = e.key.toLowerCase();
  if (map.has(k)) {
    e.preventDefault();
    keyUp(k === ' ' ? 'space' : k);
  }
});

async function refreshTelemetry(){
  try {
    const r = await fetch('/api/telemetry');
    const t = await r.json();
    document.getElementById('telemetry').textContent =
      `battery: ${t.battery ?? '-'} %\n` +
      `temperature: ${t.temperature ?? '-'} °C\n` +
      `height: ${t.height_cm ?? '-'} cm\n` +
      `tof: ${t.tof_cm ?? '-'} cm\n` +
      `barometer: ${t.barometer_cm ?? '-'} cm\n` +
      `flight time: ${t.flight_time_s ?? '-'} s\n` +
      `wifi snr: ${t.wifi_snr ?? '-'}\n` +
      `attitude p/r/y: ${t.pitch ?? '-'} / ${t.roll ?? '-'} / ${t.yaw ?? '-'}\n` +
      `velocity xyz: ${t.vgx ?? '-'} / ${t.vgy ?? '-'} / ${t.vgz ?? '-'}\n` +
      `accel xyz: ${t.agx ?? '-'} / ${t.agy ?? '-'} / ${t.agz ?? '-'}\n` +
      `flying: ${t.flying}\n` +
      `connected: ${t.connected}`;
  } catch {
    document.getElementById('telemetry').textContent = 'telemetry unavailable';
  }
}
setInterval(refreshTelemetry, 1000);
refreshTelemetry();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return Response(HTML, mimetype="text/html")


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
    global flying
    if TELLO is None:
        return jsonify(ok=False, error="controller not ready"), 503
    try:
        if not flying:
            TELLO.takeoff()
            flying = True
        return jsonify(ok=True, flying=flying)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/land")
def api_land():
    global flying
    if TELLO is None:
        return jsonify(ok=False, error="controller not ready"), 503
    try:
        if flying:
            TELLO.land()
            flying = False
        return jsonify(ok=True, flying=flying)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/flip")
def api_flip():
    if TELLO is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    direction = str(data.get("dir", "")).lower()
    if direction not in {"l", "r", "f", "b"}:
        return jsonify(ok=False, error="dir must be one of l|r|f|b"), 400
    try:
        TELLO.flip(direction)
        return jsonify(ok=True, dir=direction)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.get("/api/telemetry")
def api_telemetry():
    with telemetry_lock:
        return jsonify(telemetry)


@app.post("/api/recover")
def api_recover():
    ok, msg = recover_drone()
    return jsonify(ok=ok, message=msg)


@app.get("/video")
def video_feed():
    def gen():
        while running:
            with last_jpeg_lock:
                jpg = last_jpeg
            if jpg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.03)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ----- Drone threads -----
def video_loop():
    global last_jpeg
    while running:
        with frame_lock:
            fr = frame_reader
        if fr is None:
            time.sleep(0.05)
            continue

        frame = fr.frame
        if frame is None:
            time.sleep(0.02)
            continue
        out = frame[:, :, ::-1] if SWAP_CHANNELS else frame
        ok, jpg = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            with last_jpeg_lock:
                last_jpeg = jpg.tobytes()
        time.sleep(0.02)


def _as_int(v):
    try:
        return int(float(v))
    except Exception:
        return None


def telemetry_loop(tello: Tello):
    """
    Read telemetry from SDK state cache (UDP state stream), not from command queries.
    This avoids command-channel contention/timeouts that can freeze controls.
    """
    global last_state_seen
    while running:
        st = {}
        try:
            st = tello.get_current_state() or {}
        except Exception:
            st = {}

        if st:
            last_state_seen = time.time()
            with conn_lock:
                conn_state["connected"] = True

        temp = None
        tl = _as_int(st.get("templ"))
        th = _as_int(st.get("temph"))
        if tl is not None and th is not None:
            temp = int((tl + th) / 2)
        elif tl is not None:
            temp = tl
        elif th is not None:
            temp = th

        with conn_lock:
            connected_now = conn_state["connected"]

        with telemetry_lock:
            telemetry["battery"] = _as_int(st.get("bat"))
            telemetry["temperature"] = temp
            telemetry["height_cm"] = _as_int(st.get("h"))
            telemetry["tof_cm"] = _as_int(st.get("tof"))
            telemetry["barometer_cm"] = _as_int(st.get("baro"))
            telemetry["flight_time_s"] = _as_int(st.get("time"))
            # Not reliably supported on all firmware; keep nullable.
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

        time.sleep(0.5)


def recover_drone():
    global frame_reader, flying, last_state_seen
    if TELLO is None:
        return False, "controller not ready"
    t = TELLO
    try:
        try:
            t.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass
        try:
            t.land()
        except Exception:
            pass
        try:
            t.streamoff()
        except Exception:
            pass
        try:
            t.end()
        except Exception:
            pass

        time.sleep(1.0)
        t.connect()
        t.streamon()
        with frame_lock:
            frame_reader = t.get_frame_read()
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


def reconnect_loop(tello: Tello):
    global frame_reader, last_state_seen
    while running:
        now = time.time()
        stale = (now - last_state_seen) if last_state_seen else 9999

        with conn_lock:
            connected_now = conn_state["connected"]
            last_try = conn_state["last_reconnect"]

        if stale > RECONNECT_AFTER_S and (now - last_try) >= RECONNECT_RETRY_S:
            with conn_lock:
                conn_state["last_reconnect"] = now
                conn_state["connected"] = False

            try:
                tello.connect()
                tello.streamon()
                fr = tello.get_frame_read()
                with frame_lock:
                    frame_reader = fr
                last_state_seen = time.time()
                with conn_lock:
                    conn_state["connected"] = True
            except Exception:
                pass

        time.sleep(0.5)


def rc_loop(tello: Tello):
    global running, flying
    period = 1.0 / RC_HZ
    while running:
        t0 = time.time()

        # one-shot actions from terminal keys
        if consume_term_key("t") and not flying:
            try:
                tello.takeoff()
                flying = True
            except Exception:
                recover_drone()
        if consume_term_key("l") and flying:
            tello.land()
            flying = False
        if consume_term_key("esc"):
            running = False
            break

        # one-shot actions from web keys
        if has_key("t") and not flying:
            try:
                tello.takeoff()
                flying = True
            except Exception:
                recover_drone()
            remove_key("t")
        if has_key("l") and flying:
            tello.land()
            flying = False
            remove_key("l")

        lr = axis(has_key("d"), has_key("a")) * STICK
        fb = axis(has_key("w"), has_key("s")) * STICK
        ud = axis(has_key("r"), has_key("f")) * STICK
        yaw = axis(has_key("e"), has_key("q")) * STICK

        if has_key("space") or has_key("x"):
            lr = fb = ud = yaw = 0

        try:
            tello.send_rc_control(lr, fb, ud, yaw)
        except Exception:
            with conn_lock:
                conn_state["connected"] = False

        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def shutdown(tello: Tello):
    global running
    running = False
    try:
        tello.send_rc_control(0, 0, 0, 0)
    except Exception:
        pass
    if flying:
        try:
            tello.land()
        except Exception:
            pass
    try:
        tello.end()
    except Exception:
        pass


def main():
    global frame_reader, last_state_seen, TELLO
    logging.getLogger("djitellopy").setLevel(logging.CRITICAL)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    tello = Tello()
    TELLO = tello
    tello.connect()
    tello.streamon()
    with frame_lock:
        frame_reader = tello.get_frame_read()
    last_state_seen = time.time()
    with conn_lock:
        conn_state["connected"] = True

    atexit.register(lambda: shutdown(tello))

    threading.Thread(target=terminal_key_reader, daemon=True).start()
    threading.Thread(target=video_loop, daemon=True).start()
    threading.Thread(target=telemetry_loop, args=(tello,), daemon=True).start()
    threading.Thread(target=reconnect_loop, args=(tello,), daemon=True).start()
    threading.Thread(target=rc_loop, args=(tello,), daemon=True).start()

    print(f"http://{HTTP_HOST}:{HTTP_PORT}")

    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
