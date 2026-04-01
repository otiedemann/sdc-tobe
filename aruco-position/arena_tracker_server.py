"""
Arena Position Tracker Server
==============================
Reads live video from Parrot Anafi (RTSP stream decoded on Pi), runs ArUco
marker detection, triangulates drone position in arena coordinates, and
streams everything to a browser dashboard.

Run:
    python arena_tracker_server.py
    python arena_tracker_server.py --src rtsp://192.168.42.1/live --port 8099
    python arena_tracker_server.py --src 0 --calib camera_cal.npz
    python arena_tracker_server.py --telemetry http://192.168.1.20:8080/api/telemetry

Arguments:
    --src <url|int>     Camera source: RTSP URL or device index (default: rtsp://192.168.42.1/live)
    --port <n>          HTTP port (default: 8099)
    --calib <file>      Camera calibration .npz (camera_matrix, dist_coeffs)
    --fov <deg>         Camera horizontal FOV in degrees used for default intrinsics (default: 69)
    --detect <profile>  balanced | sensitive | strict  (default: balanced)
    --telemetry <url>   Olympe API telemetry URL for external velocity injection (optional)
    --latency <ms>      Initial display latency compensation in ms (default: 200)
"""

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco
from flask import Flask, Response, jsonify, request, send_file

# ── Import ArUco processor ────────────────────────────────────────────────────
_here = Path(__file__).parent
sys.path.insert(0, str(_here / "control-unit"))
try:
    from pi_position import HeadlessAruCoPositioning
except ImportError as exc:
    print(f"ERROR: Cannot import HeadlessAruCoPositioning from control-unit/pi_position.py: {exc}")
    sys.exit(1)

app = Flask(__name__, static_folder=None)
HTML_FILE = _here / "arena_tracker.html"

# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────

class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_cond = threading.Condition(self.lock)
        self.running = True

        # Latest annotated JPEG
        self.frame_jpg: bytes | None = None
        self.frame_seq: int = 0

        # Position data (arena coords: X±10, Y 0-10 depth, Z -1..1 height)
        self.pos: list | None = None
        self.dir_vec: list | None = None
        self.pos_ts: float = 0.0
        self.ref_markers: list = []
        self.marker_weights: dict = {}
        self.stale: bool = False

        # Velocity in arena coords [vx, vy, vz] m/s
        self.vel: list = [0.0, 0.0, 0.0]
        self.vel_ts: float = 0.0

        # Metrics
        self.fps: float = 0.0
        self.total_frames: int = 0

        # Settings (adjustable at runtime via /api/settings)
        self.latency_ms: float = 200.0
        self.camera_src: str = "rtsp://192.168.42.1/live"
        self.detect_profile: str = "balanced"
        self.fov_deg: float = 69.0

        # Velocity smoothing state (auto-derived from position diff)
        self._prev_pos: list | None = None
        self._prev_pos_ts: float = 0.0


st = _State()

# SSE listener queues
_sse_queues: list[queue.Queue] = []
_sse_lock = threading.Lock()


def _broadcast_sse(data: dict):
    msg = f"data: {json.dumps(data)}\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_queues:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            try:
                _sse_queues.remove(q)
            except ValueError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Camera / processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _default_camera_matrix(w: int, h: int, fov_deg: float) -> np.ndarray:
    """Approximate camera matrix from image dimensions and horizontal FOV."""
    f = (w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=float)


def _update_velocity_from_pos(new_pos: list, ts: float):
    """Derive arena-frame velocity from consecutive position estimates (EMA-smoothed)."""
    prev = st._prev_pos
    prev_ts = st._prev_pos_ts
    if prev is not None:
        dt = ts - prev_ts
        if 0.04 < dt < 1.5:
            raw = [(new_pos[i] - prev[i]) / dt for i in range(3)]
            raw = [max(-15.0, min(15.0, v)) for v in raw]  # clamp to realistic Anafi speed
            alpha = 0.25
            st.vel = [alpha * raw[i] + (1 - alpha) * st.vel[i] for i in range(3)]
            st.vel_ts = ts
    st._prev_pos = new_pos[:]
    st._prev_pos_ts = ts


def _annotate(frame: np.ndarray, corners, ids,
               pos, dir_vec, weights: dict, fps: float, stale: bool) -> np.ndarray:
    """Draw ArUco detections, position overlay and heading arrow onto frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Detected marker corners + IDs
    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(out, corners, ids)
        for i, mid in enumerate(ids.flatten()):
            mid_s = str(int(mid))
            wt = weights.get(mid_s)
            pts = corners[i].reshape(4, 2).astype(int)
            cx_m = int(pts[:, 0].mean())
            cy_m = int(pts[:, 1].mean())
            label = f"ID{int(mid)}" + (f" {wt:.2f}" if wt is not None else "")
            cv2.putText(out, label, (cx_m - 22, cy_m - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 255, 120), 1, cv2.LINE_AA)

    # Info overlay lines
    lines = []
    if pos:
        lines.append(f"X={pos[0]:+6.2f}m  Y={pos[1]:6.2f}m  Z={pos[2]:+5.2f}m")
    if dir_vec:
        hdg = math.degrees(math.atan2(dir_vec[0], dir_vec[1])) % 360
        lines.append(f"Heading: {hdg:.1f}\u00b0")
    n_ref = len(weights) if weights else 0
    tag = "  [STALE]" if stale else ""
    lines.append(f"Refs: {n_ref}   FPS: {fps:.1f}{tag}")

    for i, line in enumerate(lines):
        y = 22 + i * 24
        tw = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
        cv2.rectangle(out, (4, y - 17), (8 + tw, y + 6), (0, 0, 0), -1)
        color = (80, 200, 80) if not stale else (255, 180, 60)
        cv2.putText(out, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    # Direction arrow from frame center
    if dir_vec:
        cx_f, cy_f = w // 2, h // 2
        # Project arena XY movement direction onto image
        ax, ay = dir_vec[0], -dir_vec[1]   # flip arena Y → image Y
        n = math.sqrt(ax * ax + ay * ay) + 1e-9
        length = 55
        ex = int(cx_f + (ax / n) * length)
        ey = int(cy_f + (ay / n) * length)
        cv2.arrowedLine(out, (cx_f, cy_f), (ex, ey), (0, 210, 255), 2, tipLength=0.28)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Capture + detection loop
# ─────────────────────────────────────────────────────────────────────────────

def capture_loop(src, calib_file: str | None, fov_deg: float, detect_profile: str):
    """Background thread: read frames, run ArUco, publish JPEG + SSE."""
    cam_matrix = None
    dist_coeffs = np.zeros(5, dtype=float)

    if calib_file:
        try:
            d = np.load(calib_file)
            cam_matrix = d["camera_matrix"]
            dist_coeffs = d["dist_coeffs"]
            print(f"[tracker] Loaded calibration from {calib_file}")
        except Exception as e:
            print(f"[tracker] WARNING: calibration load failed: {e}")

    cap = None
    processor: HeadlessAruCoPositioning | None = None
    frame_ts_buf: list[float] = []
    detect_count = 0
    total_count = 0

    while st.running:
        if cap is None or not cap.isOpened():
            print(f"[tracker] Opening: {src}")
            cap = cv2.VideoCapture(src)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print("[tracker] Could not open camera, retrying in 3s...")
                time.sleep(3.0)
                continue

            # Sample one frame to get resolution
            ret, sample = cap.read()
            if not ret or sample is None:
                cap.release()
                cap = None
                time.sleep(1.0)
                continue

            hf, wf = sample.shape[:2]
            if cam_matrix is None:
                cam_matrix = _default_camera_matrix(wf, hf, fov_deg)
                print(f"[tracker] Default intrinsics for {wf}x{hf}, FOV={fov_deg}°: "
                      f"fx={cam_matrix[0,0]:.0f}")

            processor = HeadlessAruCoPositioning(cam_matrix, dist_coeffs,
                                                  detect_profile=detect_profile)
            print(f"[tracker] Ready (profile={detect_profile})")

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            cap = None
            time.sleep(0.1)
            continue

        ts = time.time()
        total_count += 1

        # ArUco detection + triangulation
        result = processor.process_frame(frame)

        # Re-detect markers separately for annotation (process_frame doesn't return corners)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = processor.detector.detectMarkers(gray)

        pos = dir_vec = None
        weights: dict = {}
        ref_markers: list = []
        stale = False

        if result:
            pos = result.get("cam")
            dir_vec = result.get("dir")
            weights = result.get("marker_weights", {})
            ref_markers = result.get("ref_markers", [])
            stale = result.get("stale", False)
            detect_count += 1

            with st.lock:
                if pos:
                    _update_velocity_from_pos(pos, ts)
                st.pos = pos
                st.dir_vec = dir_vec
                st.pos_ts = ts
                st.ref_markers = ref_markers[:]
                st.marker_weights = dict(weights)
                st.stale = stale

        # FPS rolling average over last 2 s
        frame_ts_buf.append(ts)
        frame_ts_buf = [t for t in frame_ts_buf if ts - t < 2.0]
        fps = len(frame_ts_buf) / 2.0

        with st.lock:
            st.fps = fps
            st.total_frames = total_count

        # Build annotated JPEG
        ann = _annotate(frame, corners, ids, pos, dir_vec, weights, fps, stale)
        _, jpg_enc = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 72])
        jpg_bytes = jpg_enc.tobytes()

        with st.frame_cond:
            st.frame_jpg = jpg_bytes
            st.frame_seq += 1
            st.frame_cond.notify_all()

        # SSE event
        with st.lock:
            vel_snap = st.vel[:]
            latency_snap = st.latency_ms

        _broadcast_sse({
            "ts": ts,
            "pos": pos,
            "dir": dir_vec,
            "vel": vel_snap,
            "latency_ms": latency_snap,
            "ref_markers": ref_markers,
            "marker_weights": weights,
            "stale": stale,
            "fps": round(fps, 1),
        })

    if cap:
        cap.release()
    print("[tracker] Capture loop exited.")


# ─────────────────────────────────────────────────────────────────────────────
# Optional: poll Olympe server for external velocity (dead-reckoning)
# ─────────────────────────────────────────────────────────────────────────────

def telemetry_poller(telemetry_url: str):
    """Poll Olympe /api/telemetry and inject velocity for dead-reckoning."""
    print(f"[tracker] Telemetry poller: {telemetry_url}")
    while st.running:
        try:
            with urllib.request.urlopen(telemetry_url, timeout=0.8) as resp:
                d = json.loads(resp.read())
            # Olympe reports speed_x (forward), speed_y (right), speed_z (down NED), yaw (deg)
            vfwd = float(d.get("speed_x", d.get("speedX", 0.0)))
            vright = float(d.get("speed_y", d.get("speedY", 0.0)))
            vdown = float(d.get("speed_z", d.get("speedZ", 0.0)))
            yaw = math.radians(float(d.get("yaw", 0.0)))
            # Rotate body-frame → arena XY
            vx = vfwd * math.cos(yaw) - vright * math.sin(yaw)
            vy = vfwd * math.sin(yaw) + vright * math.cos(yaw)
            vz = -vdown   # NED down → arena Z up
            alpha = 0.35
            with st.lock:
                st.vel[0] = alpha * vx + (1 - alpha) * st.vel[0]
                st.vel[1] = alpha * vy + (1 - alpha) * st.vel[1]
                st.vel[2] = alpha * vz + (1 - alpha) * st.vel[2]
                st.vel_ts = time.time()
        except Exception:
            pass
        time.sleep(0.15)   # ~7 Hz


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    if HTML_FILE.exists():
        return send_file(str(HTML_FILE))
    return "<h1>arena_tracker.html not found</h1>", 404


@app.get("/video")
def video_stream():
    """MJPEG stream of annotated camera frames."""
    def generate():
        last_seq = -1
        while st.running:
            with st.frame_cond:
                if not st.frame_cond.wait_for(
                        lambda: st.frame_seq != last_seq or not st.running, timeout=3.0):
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
                    continue
                if not st.running:
                    break
                jpg = st.frame_jpg
                last_seq = st.frame_seq
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


@app.get("/events")
def sse():
    """Server-Sent Events stream: position + velocity updates ~every frame."""
    q: queue.Queue = queue.Queue(maxsize=12)
    with _sse_lock:
        _sse_queues.append(q)

    def generate():
        try:
            # Send current state immediately on connect
            with st.lock:
                init = {
                    "ts": time.time(),
                    "pos": st.pos,
                    "dir": st.dir_vec,
                    "vel": st.vel[:],
                    "latency_ms": st.latency_ms,
                    "ref_markers": st.ref_markers[:],
                    "marker_weights": dict(st.marker_weights),
                    "stale": st.stale,
                    "fps": st.fps,
                }
            yield f"data: {json.dumps(init)}\n\n"

            while st.running:
                try:
                    msg = q.get(timeout=4.0)
                    yield msg
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _sse_lock:
                try:
                    _sse_queues.remove(q)
                except ValueError:
                    pass

    return Response(generate(),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/position")
def api_position():
    with st.lock:
        return jsonify({
            "pos": st.pos,
            "dir": st.dir_vec,
            "vel": st.vel[:],
            "ref_markers": st.ref_markers[:],
            "marker_weights": dict(st.marker_weights),
            "stale": st.stale,
            "fps": st.fps,
            "ts": st.pos_ts,
            "latency_ms": st.latency_ms,
        })


@app.post("/api/telemetry")
def api_telemetry_inject():
    """Inject arena-frame velocity directly (vx, vy, vz in m/s)."""
    d = request.get_json(silent=True) or {}
    with st.lock:
        st.vel = [float(d.get("vx", 0)), float(d.get("vy", 0)), float(d.get("vz", 0))]
        st.vel_ts = time.time()
    return jsonify(ok=True)


@app.get("/api/settings")
def api_settings_get():
    return jsonify({
        "latency_ms": st.latency_ms,
        "detect_profile": st.detect_profile,
        "camera_src": st.camera_src,
        "fov_deg": st.fov_deg,
    })


@app.post("/api/settings")
def api_settings_post():
    d = request.get_json(silent=True) or {}
    if "latency_ms" in d:
        with st.lock:
            st.latency_ms = max(0, float(d["latency_ms"]))
    return jsonify(ok=True, latency_ms=st.latency_ms)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Arena Position Tracker Server")
    p.add_argument("--src", default=os.getenv("CAMERA_SRC", "rtsp://192.168.42.1/live"),
                   help="Camera source: RTSP URL or device index integer")
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "8099")))
    p.add_argument("--calib", default=None, help="NPZ calibration file")
    p.add_argument("--fov", type=float, default=69.0, help="Camera horizontal FOV (degrees)")
    p.add_argument("--detect", default="balanced",
                   choices=["balanced", "sensitive", "strict"])
    p.add_argument("--telemetry", default=None,
                   help="Olympe telemetry URL for velocity (e.g. http://PI_IP:8080/api/telemetry)")
    p.add_argument("--latency", type=float, default=200.0,
                   help="Initial latency compensation in ms (default: 200)")
    args = p.parse_args()

    src = int(args.src) if args.src.isdigit() else args.src
    st.camera_src = str(src)
    st.detect_profile = args.detect
    st.fov_deg = args.fov
    st.latency_ms = args.latency

    print(f"[tracker] Arena Position Tracker")
    print(f"[tracker]   Camera:  {src}")
    print(f"[tracker]   Profile: {args.detect}")
    print(f"[tracker]   FOV:     {args.fov}°")
    print(f"[tracker]   Latency: {args.latency} ms (initial)")
    print(f"[tracker]   Port:    {args.port}")
    print(f"[tracker]   UI:      http://localhost:{args.port}/")

    cap_thread = threading.Thread(
        target=capture_loop,
        args=(src, args.calib, args.fov, args.detect),
        daemon=True, name="capture")
    cap_thread.start()

    if args.telemetry:
        tel_thread = threading.Thread(
            target=telemetry_poller, args=(args.telemetry,),
            daemon=True, name="telemetry")
        tel_thread.start()

    try:
        app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)
    finally:
        st.running = False


if __name__ == "__main__":
    main()
