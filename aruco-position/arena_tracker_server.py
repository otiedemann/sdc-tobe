"""
Arena Position Tracker Server
==============================
By default connects directly to a Parrot Anafi at 192.168.42.1 via the
Olympe SDK — no RTSP needed.  Olympe decodes the video via PDraw and
delivers raw YUV frames; telemetry (speed / attitude) is polled directly
from Olympe state without touching any external API server.

Alternatively you can point it at any OpenCV-compatible source (USB camera,
HTTP MJPEG from the olympe_pi_api_server, etc.) with --src.

Run:
    # Olympe-direct (default – Anafi on 192.168.42.1):
    python arena_tracker_server.py

    # Explicit Olympe drone IP:
    python arena_tracker_server.py --drone 192.168.42.1

    # Fallback – USB cam / HTTP MJPEG / RTSP (no Olympe needed):
    python arena_tracker_server.py --src 0
    python arena_tracker_server.py --src http://192.168.1.20:8080/api/video
    python arena_tracker_server.py --src rtsp://some-other-camera/live

    # Other options:
    python arena_tracker_server.py --port 8099 --calib camera_cal.npz
    python arena_tracker_server.py --fov 69 --detect sensitive --latency 300

Arguments:
    --drone <ip>        Anafi IP for Olympe-direct mode (default: 192.168.42.1)
    --src <url|int>     Override: use cv2.VideoCapture source instead of Olympe
    --port <n>          HTTP port (default: 8099)
    --calib <file>      Camera calibration .npz (camera_matrix, dist_coeffs)
    --fov <deg>         Camera horizontal FOV in degrees (default: 69)
    --detect <profile>  balanced | sensitive | strict  (default: balanced)
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
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco
from flask import Flask, Response, jsonify, request, send_file

# ── Optional Olympe import ────────────────────────────────────────────────────
try:
    import olympe
    from olympe.messages.ardrone3.PilotingState import (
        SpeedChanged,
        AttitudeChanged,
        AltitudeChanged,
        FlyingStateChanged,
    )
    from olympe.messages.common.CommonState import BatteryStateChanged
    HAS_OLYMPE = True
except ImportError:
    HAS_OLYMPE = False
    print("[tracker] WARNING: olympe not available – Olympe-direct mode disabled.")

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

        # Drone telemetry (Olympe-direct)
        self.battery_pct: int | None = None
        self.altitude_m: float | None = None
        self.yaw_deg: float | None = None
        self.flying_state: str | None = None

        # Metrics
        self.fps: float = 0.0
        self.total_frames: int = 0

        # Settings (adjustable at runtime via /api/settings)
        self.latency_ms: float = 200.0
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

    # Telemetry overlay (Olympe-direct)
    with st.lock:
        batt = st.battery_pct
        alt = st.altitude_m
        fly = st.flying_state
    telem_parts = []
    if batt is not None:
        telem_parts.append(f"Bat:{batt}%")
    if alt is not None:
        telem_parts.append(f"Alt:{alt:.1f}m")
    if fly:
        telem_parts.append(fly)
    if telem_parts:
        lines.append("  ".join(telem_parts))

    for i, line in enumerate(lines):
        y = 22 + i * 24
        tw = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
        cv2.rectangle(out, (4, y - 17), (8 + tw, y + 6), (0, 0, 0), -1)
        color = (80, 200, 80) if not stale else (255, 180, 60)
        cv2.putText(out, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    # Direction arrow from frame center
    if dir_vec:
        cx_f, cy_f = w // 2, h // 2
        ax, ay = dir_vec[0], -dir_vec[1]   # flip arena Y → image Y
        n = math.sqrt(ax * ax + ay * ay) + 1e-9
        length = 55
        ex = int(cx_f + (ax / n) * length)
        ey = int(cy_f + (ay / n) * length)
        cv2.arrowedLine(out, (cx_f, cy_f), (ex, ey), (0, 210, 255), 2, tipLength=0.28)

    return out


def _process_and_publish(frame: np.ndarray, ts: float,
                          processor: "HeadlessAruCoPositioning",
                          fps_buf: list) -> None:
    """Run ArUco detection on frame, update shared state, publish JPEG + SSE."""
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
    fps_buf.append(ts)
    while fps_buf and ts - fps_buf[0] > 2.0:
        fps_buf.pop(0)
    fps = len(fps_buf) / 2.0

    with st.lock:
        st.fps = fps
        st.total_frames += 1

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
        batt = st.battery_pct
        alt = st.altitude_m
        yaw = st.yaw_deg
        fly = st.flying_state

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
        "battery_pct": batt,
        "altitude_m": alt,
        "yaw_deg": yaw,
        "flying_state": fly,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Olympe-direct capture loop
# ─────────────────────────────────────────────────────────────────────────────

def capture_loop_olympe(drone_ip: str, calib_file: str | None,
                         fov_deg: float, detect_profile: str):
    """
    Connect to Parrot Anafi via Olympe, decode video with PDraw callbacks,
    poll telemetry from Olympe state — no RTSP, no external API needed.
    """
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

    # Queue carries (bgr_ndarray, timestamp) pairs from YUV callback → main loop
    _frame_q: queue.Queue = queue.Queue(maxsize=4)
    processor = None
    fps_buf: list[float] = []

    def _yuv_frame_cb(yuv_frame):
        """Called by Olympe PDraw in a worker thread for each decoded frame."""
        try:
            yuv_frame.ref()
        except Exception:
            pass
        try:
            # Detect pixel format (I420 default for Anafi 4K/720p)
            cv_cvt = cv2.COLOR_YUV2BGR_I420
            try:
                info = yuv_frame.info()
                yuv_fmt = None
                if isinstance(info, dict):
                    if "yuv" in info:
                        yuv_fmt = info["yuv"].get("format")
                    elif "raw" in info:
                        yuv_fmt = info["raw"].get("format")
                if yuv_fmt is not None:
                    for attr, flag in [("VDEF_I420", cv2.COLOR_YUV2BGR_I420),
                                       ("VDEF_NV12", cv2.COLOR_YUV2BGR_NV12)]:
                        c = getattr(olympe, attr, None)
                        if c is not None and c == yuv_fmt:
                            cv_cvt = flag
                            break
            except Exception:
                pass

            bgr = cv2.cvtColor(yuv_frame.as_ndarray(), cv_cvt)
            try:
                _frame_q.put_nowait((bgr, time.time()))
            except queue.Full:
                pass  # drop oldest implicitly
        except Exception as e:
            print(f"[tracker] YUV frame error: {e}")
        finally:
            try:
                yuv_frame.unref()
            except Exception:
                pass

    def _flush_cb(stream):
        """Called by PDraw when it needs to flush pending buffers (prevents stall)."""
        while True:
            try:
                _frame_q.get_nowait()
            except queue.Empty:
                break

    print(f"[tracker] Olympe-direct mode: connecting to {drone_ip}")

    while st.running:
        d = None
        streaming_started = False
        try:
            d = olympe.Drone(drone_ip)
            connected = d.connect()
            if not connected:
                print(f"[tracker] Olympe: could not connect to {drone_ip}, retrying in 5s...")
                time.sleep(5.0)
                continue

            print(f"[tracker] Olympe: connected to {drone_ip}")

            # Start video streaming via PDraw
            if hasattr(d, "streaming") and hasattr(d.streaming, "set_callbacks"):
                d.streaming.set_callbacks(
                    raw_cb=_yuv_frame_cb,
                    flush_raw_cb=_flush_cb,
                )
                d.streaming.start()
                streaming_started = True
            elif hasattr(d, "set_streaming_callbacks"):   # legacy Olympe API
                d.set_streaming_callbacks(raw_cb=_yuv_frame_cb)
                d.start_video_streaming()
                streaming_started = True
            else:
                print("[tracker] WARNING: Olympe streaming API not found on this version.")

            # ── Main processing loop ──────────────────────────────────────────
            while st.running:
                # ---- Telemetry polling (Olympe state) ----------------------
                try:
                    spd = d.get_state(SpeedChanged)
                    att = d.get_state(AttitudeChanged)

                    vfwd   = float(spd.get("speedX", 0.0))
                    vright = float(spd.get("speedY", 0.0))
                    vdown  = float(spd.get("speedZ", 0.0))
                    yaw_rad = float(att.get("yaw", 0.0))

                    # Rotate body-frame velocity → arena XY frame
                    vx =  vfwd * math.cos(yaw_rad) - vright * math.sin(yaw_rad)
                    vy =  vfwd * math.sin(yaw_rad) + vright * math.cos(yaw_rad)
                    vz = -vdown   # NED down → arena Z up

                    alpha = 0.35
                    with st.lock:
                        st.vel[0] = alpha * vx + (1 - alpha) * st.vel[0]
                        st.vel[1] = alpha * vy + (1 - alpha) * st.vel[1]
                        st.vel[2] = alpha * vz + (1 - alpha) * st.vel[2]
                        st.vel_ts = time.time()
                        st.yaw_deg = math.degrees(yaw_rad)
                except Exception:
                    pass

                try:
                    alt_state = d.get_state(AltitudeChanged)
                    with st.lock:
                        st.altitude_m = float(alt_state.get("altitude", 0.0))
                except Exception:
                    pass

                try:
                    bat_state = d.get_state(BatteryStateChanged)
                    with st.lock:
                        st.battery_pct = int(bat_state.get("percent", 0))
                except Exception:
                    pass

                try:
                    fly_state = d.get_state(FlyingStateChanged)
                    raw_fly = fly_state.get("state")
                    with st.lock:
                        st.flying_state = str(raw_fly).split(".")[-1] if raw_fly else None
                except Exception:
                    pass

                # ---- Frame processing ----------------------------------------
                try:
                    bgr, ts = _frame_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                if processor is None:
                    hf, wf = bgr.shape[:2]
                    if cam_matrix is None:
                        cam_matrix = _default_camera_matrix(wf, hf, fov_deg)
                        print(f"[tracker] Default intrinsics for {wf}x{hf}, "
                              f"FOV={fov_deg}°: fx={cam_matrix[0,0]:.0f}")
                    processor = HeadlessAruCoPositioning(
                        cam_matrix, dist_coeffs, detect_profile=detect_profile)
                    print(f"[tracker] Processor ready (profile={detect_profile})")

                _process_and_publish(bgr, ts, processor, fps_buf)

        except Exception as exc:
            print(f"[tracker] Olympe error: {exc}")
        finally:
            if d is not None:
                try:
                    if streaming_started:
                        if hasattr(d, "streaming"):
                            d.streaming.stop()
                        elif hasattr(d, "stop_video_streaming"):
                            d.stop_video_streaming()
                except Exception:
                    pass
                try:
                    d.disconnect()
                except Exception:
                    pass

        if st.running:
            print("[tracker] Olympe: reconnecting in 5s...")
            time.sleep(5.0)

    print("[tracker] Olympe capture loop exited.")


# ─────────────────────────────────────────────────────────────────────────────
# cv2.VideoCapture fallback loop
# ─────────────────────────────────────────────────────────────────────────────

def capture_loop_cv2(src, calib_file: str | None, fov_deg: float, detect_profile: str):
    """
    Fallback: read frames from any OpenCV-compatible source (USB cam, HTTP MJPEG,
    RTSP from a third-party cam, etc.).  Used when --src is given explicitly.
    """
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
    processor = None
    fps_buf: list[float] = []

    while st.running:
        if cap is None or not cap.isOpened():
            print(f"[tracker] Opening: {src}")
            cap = cv2.VideoCapture(src)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print("[tracker] Could not open source, retrying in 3s...")
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

            processor = HeadlessAruCoPositioning(
                cam_matrix, dist_coeffs, detect_profile=detect_profile)
            print(f"[tracker] Ready (profile={detect_profile})")

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            cap = None
            time.sleep(0.1)
            continue

        ts = time.time()
        _process_and_publish(frame, ts, processor, fps_buf)

    if cap:
        cap.release()
    print("[tracker] cv2 capture loop exited.")


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
    """Server-Sent Events stream: position + velocity + telemetry updates ~every frame."""
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
                    "battery_pct": st.battery_pct,
                    "altitude_m": st.altitude_m,
                    "yaw_deg": st.yaw_deg,
                    "flying_state": st.flying_state,
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
            "battery_pct": st.battery_pct,
            "altitude_m": st.altitude_m,
            "yaw_deg": st.yaw_deg,
            "flying_state": st.flying_state,
        })


@app.post("/api/telemetry")
def api_telemetry_inject():
    """Inject arena-frame velocity directly (vx, vy, vz in m/s) — for testing / external override."""
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
    p.add_argument("--drone", default=os.getenv("DRONE_IP", "192.168.42.1"),
                   help="Parrot Anafi IP for Olympe-direct mode (default: 192.168.42.1)")
    p.add_argument("--src", default=None,
                   help="Override: use cv2.VideoCapture source instead of Olympe "
                        "(e.g. 0, http://PI_IP:8080/api/video)")
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "8099")))
    p.add_argument("--calib", default=None, help="NPZ calibration file")
    p.add_argument("--fov", type=float, default=69.0, help="Camera horizontal FOV (degrees)")
    p.add_argument("--detect", default="balanced",
                   choices=["balanced", "sensitive", "strict"])
    p.add_argument("--latency", type=float, default=200.0,
                   help="Initial latency compensation in ms (default: 200)")
    args = p.parse_args()

    st.detect_profile = args.detect
    st.fov_deg = args.fov
    st.latency_ms = args.latency

    # Decide which capture backend to use
    use_olympe = (args.src is None) and HAS_OLYMPE

    print(f"[tracker] Arena Position Tracker")
    if use_olympe:
        print(f"[tracker]   Mode:    Olympe-direct (Anafi @ {args.drone})")
    else:
        src_raw = args.src if args.src else "cv2 (no --src given, Olympe unavailable)"
        src = int(args.src) if (args.src and args.src.isdigit()) else args.src
        print(f"[tracker]   Mode:    cv2.VideoCapture  src={src_raw}")
        if not HAS_OLYMPE and args.src is None:
            print("[tracker]   WARNING: olympe not installed; falling back to cv2 with no source.")
            print("[tracker]   Use --src to specify a camera URL or device index.")
    print(f"[tracker]   Profile: {args.detect}")
    print(f"[tracker]   FOV:     {args.fov}°")
    print(f"[tracker]   Latency: {args.latency} ms (initial)")
    print(f"[tracker]   Port:    {args.port}")
    print(f"[tracker]   UI:      http://localhost:{args.port}/")

    if use_olympe:
        cap_thread = threading.Thread(
            target=capture_loop_olympe,
            args=(args.drone, args.calib, args.fov, args.detect),
            daemon=True, name="capture-olympe")
    else:
        src = int(args.src) if (args.src and args.src.isdigit()) else (args.src or 0)
        cap_thread = threading.Thread(
            target=capture_loop_cv2,
            args=(src, args.calib, args.fov, args.detect),
            daemon=True, name="capture-cv2")

    cap_thread.start()

    try:
        app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)
    finally:
        st.running = False


if __name__ == "__main__":
    main()
