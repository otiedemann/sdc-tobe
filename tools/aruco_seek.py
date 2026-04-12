#!/usr/bin/env python3
"""
ArUco Marker Seek & Approach Tool
===================================

Autonomous drone tool that:
  1. Takes off
  2. Rotates 360 degrees scanning for any ArUco marker
  3. Once a marker is found, slowly approaches it
  4. Hovers 1 m in front of the marker

Uses the unified API server exclusively (HTTP + SSE).
Designed for Parrot Anafi drones via the flightctrl API servers.

Usage:
    python aruco_seek.py --api http://flightctrl1:8080
    python aruco_seek.py --api http://flightctrl1:8080 --hover-distance 1.5
    python aruco_seek.py --api http://flightctrl1:8080 --takeoff-height 1.2

Safety:
    - Press Ctrl+C at any time to abort and land
    - Automatic land on timeout (default 120s)
    - Automatic land if position lost for too long
    - Heartbeat sent every 400 ms to keep watchdog alive
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

# ─── Configuration defaults ──────────────────────────────────────────────────

DEFAULT_API_BASE = "http://flightctrl1:8080"
TAKEOFF_HEIGHT_M = 1.0        # altitude to hold during search
HOVER_DISTANCE_M = 1.5        # stop this far in front of the marker
SCAN_YAW_SPEED = 15           # RC yaw percent during search rotation (0-100)
APPROACH_SPEED = 8            # RC forward percent during approach (0-100)
RC_TICK_MS = 200              # RC command duration per tick
CONTROL_HZ = 5                # main loop frequency
HEARTBEAT_INTERVAL = 0.4      # seconds between heartbeats
POSITION_TIMEOUT_S = 5.0      # land if no position for this long
MISSION_TIMEOUT_S = 120.0     # total mission timeout
SETTLE_TIME_S = 3.0           # time to wait after takeoff before scanning
ALTITUDE_TOLERANCE_M = 0.3    # acceptable altitude error
WALL_SAFE_DISTANCE_M = 1.2   # hard minimum distance from any wall
WALL_BRAKE_DISTANCE_M = 2.5  # start braking when this close to a wall

# ─── Calibration curve for 18cm ArUco markers ───────────────────────────────
# Power-law fit from Distanzkalibrierung (calibration chart):
#   distance = CALIB_A * pixel_size ^ (-CALIB_B)
#   distance = 109.1653 * (1/pixel)^0.8973
#            = 109.1653 * pixel^(-0.8973)
CALIB_A = 109.1653           # coefficient
CALIB_B = 0.8973             # exponent


def pixel_distance_estimate(avg_pixel_size: float) -> float:
    """Estimate distance (m) from marker pixel size using calibration curve."""
    if avg_pixel_size <= 0:
        return float("inf")
    return CALIB_A * avg_pixel_size ** (-CALIB_B)


# ─── Globals ─────────────────────────────────────────────────────────────────

_abort = threading.Event()
_api_base = DEFAULT_API_BASE


# ─── API helpers ─────────────────────────────────────────────────────────────

def api_post(path: str, body: dict | None = None, timeout: float = 12.0) -> dict:
    """POST JSON to the API server and return parsed response."""
    data = json.dumps(body or {}).encode()
    req = Request(
        f"{_api_base}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"ok": False, "error": str(e)}


def api_get(path: str, timeout: float = 2.0) -> dict:
    """GET JSON from the API server."""
    try:
        with urlopen(f"{_api_base}{path}", timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"ok": False, "error": str(e)}


def send_rc(lr: int = 0, fb: int = 0, ud: int = 0, yaw: int = 0,
            duration_ms: int = RC_TICK_MS) -> dict:
    """Send a single RC (PCMD) command."""
    return api_post("/api/rc", {
        "lr": _clamp(lr), "fb": _clamp(fb),
        "ud": _clamp(ud), "yaw": _clamp(yaw),
        "duration_ms": duration_ms,
    })


def send_rc_stop():
    """Zero all RC axes."""
    send_rc(0, 0, 0, 0)


def _clamp(v: int, lo: int = -100, hi: int = 100) -> int:
    return max(lo, min(hi, int(v)))


# ─── Client-side ArUco detection from video feed ───────────────────────────

class VideoMarkerTracker:
    """
    Background thread that grabs JPEG frames from the MJPEG video stream
    and runs ArUco detection locally. Provides marker pixel centers and
    sizes without needing server-side support.
    """

    def __init__(self, api_base: str):
        self._url = f"{api_base}/api/position/video"
        self._lock = threading.Lock()
        self._markers: dict = {}  # {id: {"center": [cx,cy], "px_size": float}}
        self._frame_w: int = 1280
        self._frame_h: int = 720
        self._last_ts: float = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True, name="vid-track")
        self._running = True
        self._detector = None

    def start(self):
        self._thread.start()

    def stop(self):
        self._running = False

    def get_marker(self, marker_id: int) -> Optional[dict]:
        """Return {"center": [cx,cy], "px_size": float} or None."""
        with self._lock:
            return self._markers.get(marker_id)

    def get_all(self) -> dict:
        """Return {id: {"center": ..., "px_size": ...}} for all visible markers."""
        with self._lock:
            return dict(self._markers)

    @property
    def frame_size(self) -> tuple:
        with self._lock:
            return self._frame_w, self._frame_h

    @property
    def age(self) -> float:
        with self._lock:
            return time.time() - self._last_ts if self._last_ts > 0 else float("inf")

    def _init_detector(self):
        try:
            import cv2
            aruco = cv2.aruco
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
            params = aruco.DetectorParameters()
            # Relax thresholds for small/distant markers
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 23
            params.adaptiveThreshWinSizeStep = 5
            params.minMarkerPerimeterRate = 0.01
            self._detector = aruco.ArucoDetector(aruco_dict, params)
            return True
        except Exception as e:
            log(f"[vid-track] OpenCV ArUco init failed: {e}")
            return False

    def _run(self):
        import http.client
        from urllib.parse import urlparse
        try:
            import cv2
            import numpy as np
        except ImportError:
            log("[vid-track] OpenCV not available — tracker disabled")
            return

        if not self._init_detector():
            return

        log("[vid-track] Client-side ArUco tracker started")

        while self._running and not _abort.is_set():
            try:
                parsed = urlparse(self._url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=10)
                conn.request("GET", parsed.path)
                resp = conn.getresponse()

                buf = b""
                while self._running and not _abort.is_set():
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    buf += chunk

                    # Find complete JPEG frame in MJPEG stream
                    while True:
                        # MJPEG boundary: --frame\r\nContent-Type: image/jpeg\r\n\r\n
                        hdr_idx = buf.find(b"\r\n\r\n")
                        if hdr_idx < 0:
                            break
                        jpeg_start = hdr_idx + 4

                        # Find next boundary
                        next_boundary = buf.find(b"--frame", jpeg_start)
                        if next_boundary < 0:
                            break  # incomplete frame, need more data

                        jpeg_data = buf[jpeg_start:next_boundary]
                        buf = buf[next_boundary:]

                        # Decode and detect
                        try:
                            arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if frame is None:
                                continue
                            h, w = frame.shape[:2]
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            corners, ids, _ = self._detector.detectMarkers(gray)

                            new_markers = {}
                            if ids is not None and len(ids) > 0:
                                for i, mid_arr in enumerate(ids):
                                    mid = int(mid_arr[0])
                                    pts = corners[i].reshape(4, 2)
                                    side_lens = [float(np.linalg.norm(
                                        pts[j] - pts[(j + 1) % 4])) for j in range(4)]
                                    px_size = float(np.mean(side_lens))
                                    cx = float(np.mean(pts[:, 0]))
                                    cy = float(np.mean(pts[:, 1]))
                                    new_markers[mid] = {
                                        "center": [round(cx, 1), round(cy, 1)],
                                        "px_size": round(px_size, 2),
                                    }

                            with self._lock:
                                self._markers = new_markers
                                self._frame_w = w
                                self._frame_h = h
                                self._last_ts = time.time()
                        except Exception:
                            pass
                conn.close()
            except Exception as e:
                if self._running:
                    time.sleep(1.0)


# ─── Position SSE listener ──────────────────────────────────────────────────

class PositionListener:
    """
    Background thread that consumes the position SSE stream and keeps
    the latest snapshot available via .get().
    """

    def __init__(self, api_base: str):
        self._url = f"{api_base}/api/position/events"
        self._lock = threading.Lock()
        self._latest: Optional[dict] = None
        self._last_ts: float = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True, name="pos-sse")
        self._running = True

    def start(self):
        self._thread.start()

    def stop(self):
        self._running = False

    def get(self) -> Optional[dict]:
        """Return latest position snapshot, or None if stale/unavailable."""
        with self._lock:
            return self._latest

    @property
    def age(self) -> float:
        """Seconds since last position update."""
        with self._lock:
            if self._last_ts == 0:
                return float("inf")
            return time.time() - self._last_ts

    def _run(self):
        import http.client
        from urllib.parse import urlparse

        while self._running and not _abort.is_set():
            try:
                parsed = urlparse(self._url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=10)
                conn.request("GET", parsed.path)
                resp = conn.getresponse()

                buf = ""
                while self._running and not _abort.is_set():
                    chunk = resp.read(1).decode("utf-8", errors="replace")
                    if not chunk:
                        break
                    buf += chunk
                    # SSE messages are delimited by double newline
                    while "\n\n" in buf:
                        msg, buf = buf.split("\n\n", 1)
                        for line in msg.strip().split("\n"):
                            if line.startswith("data:"):
                                payload = line[5:].strip()
                                try:
                                    data = json.loads(payload)
                                    with self._lock:
                                        self._latest = data
                                        self._last_ts = time.time()
                                except json.JSONDecodeError:
                                    pass
                conn.close()
            except Exception as e:
                log(f"[pos-sse] connection error: {e}")
                time.sleep(1.0)


# ─── Heartbeat thread ───────────────────────────────────────────────────────

def _heartbeat_loop():
    """Send heartbeat to keep the drone watchdog alive."""
    while not _abort.is_set():
        try:
            api_get("/api/heartbeat", timeout=0.5)
        except Exception:
            pass
        _abort.wait(HEARTBEAT_INTERVAL)


# ─── Telemetry helpers ───────────────────────────────────────────────────────

def get_telemetry() -> dict:
    return api_get("/api/telemetry")


def get_arena_config() -> dict:
    return api_get("/api/arena/config")


def get_marker_positions(arena_cfg: dict) -> dict[int, dict]:
    """
    Parse arena config into {marker_id: {x, y, z, wall}} dict.
    """
    markers = {}
    for m in arena_cfg.get("markers", []):
        mid = int(m["id"])
        markers[mid] = {
            "x": float(m["x"]),
            "y": float(m["y"]),
            "z": float(m["z"]),
            "wall": m.get("wall", "front"),
        }
    return markers


# ─── Wall / arena boundary awareness ────────────────────────────────────────

class ArenaBounds:
    """
    Axis-aligned arena boundary computed from arena config.

    Walls:
      front: Y = y_min
      back:  Y = y_max
      left:  X = x_min
      right: X = x_max
    """

    def __init__(self, arena_cfg: dict):
        w = float(arena_cfg.get("arena_width_m", 20))
        d = float(arena_cfg.get("arena_height_m", 10))
        # Arena is centred on X, starts at Y=0
        self.x_min = -w / 2.0
        self.x_max = w / 2.0
        self.y_min = 0.0
        self.y_max = d
        self.width = w
        self.depth = d

    def wall_distances(self, x: float, y: float) -> dict[str, float]:
        """Return signed distance to each wall (positive = inside arena)."""
        return {
            "front": y - self.y_min,       # distance from front wall (Y=0)
            "back":  self.y_max - y,        # distance from back wall
            "left":  x - self.x_min,        # distance from left wall
            "right": self.x_max - x,        # distance from right wall
        }

    def nearest_wall(self, x: float, y: float) -> tuple[str, float]:
        """Return (wall_name, distance) for the closest wall."""
        dists = self.wall_distances(x, y)
        return min(dists.items(), key=lambda kv: kv[1])

    def clamp_target(self, tx: float, ty: float, safe_dist: float) -> tuple[float, float]:
        """Clamp a target position to be at least safe_dist from every wall."""
        tx = max(self.x_min + safe_dist, min(self.x_max - safe_dist, tx))
        ty = max(self.y_min + safe_dist, min(self.y_max - safe_dist, ty))
        return tx, ty

    def wall_brake_factor(self, x: float, y: float,
                          vx: float, vy: float) -> float:
        """
        Compute a 0..1 speed multiplier based on wall proximity.

        Returns 1.0 when far from walls, scales down to 0.0 when at
        WALL_SAFE_DISTANCE from any wall the drone is moving toward.
        Only brakes for walls the drone is approaching (not retreating from).
        """
        dists = self.wall_distances(x, y)
        factor = 1.0

        # Check each wall — only brake if moving toward it
        # front (Y=y_min): approaching if vy < 0
        if vy < 0 and dists["front"] < WALL_BRAKE_DISTANCE_M:
            f = max(0.0, (dists["front"] - WALL_SAFE_DISTANCE_M) /
                         (WALL_BRAKE_DISTANCE_M - WALL_SAFE_DISTANCE_M))
            factor = min(factor, f)

        # back (Y=y_max): approaching if vy > 0
        if vy > 0 and dists["back"] < WALL_BRAKE_DISTANCE_M:
            f = max(0.0, (dists["back"] - WALL_SAFE_DISTANCE_M) /
                         (WALL_BRAKE_DISTANCE_M - WALL_SAFE_DISTANCE_M))
            factor = min(factor, f)

        # left (X=x_min): approaching if vx < 0
        if vx < 0 and dists["left"] < WALL_BRAKE_DISTANCE_M:
            f = max(0.0, (dists["left"] - WALL_SAFE_DISTANCE_M) /
                         (WALL_BRAKE_DISTANCE_M - WALL_SAFE_DISTANCE_M))
            factor = min(factor, f)

        # right (X=x_max): approaching if vx > 0
        if vx > 0 and dists["right"] < WALL_BRAKE_DISTANCE_M:
            f = max(0.0, (dists["right"] - WALL_SAFE_DISTANCE_M) /
                         (WALL_BRAKE_DISTANCE_M - WALL_SAFE_DISTANCE_M))
            factor = min(factor, f)

        return factor

    def __repr__(self):
        return (f"ArenaBounds(x=[{self.x_min:.1f}, {self.x_max:.1f}], "
                f"y=[{self.y_min:.1f}, {self.y_max:.1f}])")


# ─── Coordinate math ────────────────────────────────────────────────────────

def compute_approach_target(
    marker: dict,
    hover_dist: float,
) -> tuple[float, float, float, float]:
    """
    Compute a target position that is `hover_dist` metres in front of a wall
    marker, and a yaw angle that faces the marker.

    Returns (x, y, z, yaw_rad) in arena frame.

    Wall normals (direction pointing INTO the arena):
      front (Y=0 wall):  normal = (0, +1, 0)   → target is at marker + (0, +hover_dist, 0)
      back  (Y=max wall): normal = (0, -1, 0)   → target is at marker + (0, -hover_dist, 0)
      left  (X=-max wall): normal = (+1, 0, 0)  → target is at marker + (+hover_dist, 0, 0)
      right (X=+max wall): normal = (-1, 0, 0)  → target is at marker + (-hover_dist, 0, 0)
    """
    wall = marker.get("wall", "front")
    mx, my, mz = marker["x"], marker["y"], marker["z"]

    # Normal vector pointing inward (away from wall, into arena)
    normals = {
        "front": (0.0, 1.0),
        "back":  (0.0, -1.0),
        "left":  (1.0, 0.0),
        "right": (-1.0, 0.0),
    }
    nx, ny = normals.get(wall, (0.0, 1.0))

    # Target position: hover_dist metres in front of the marker (into arena)
    tx = mx + nx * hover_dist
    ty = my + ny * hover_dist
    tz = mz  # match marker height

    # Yaw: face the marker (opposite of normal direction)
    # In our frame: yaw = atan2(Y, X), CCW positive
    # We want to face toward the marker from the target → direction = (-nx, -ny)
    face_yaw = math.atan2(-ny, -nx)

    return tx, ty, tz, face_yaw


def world_to_body(vx_w: float, vy_w: float, yaw: float) -> tuple[float, float]:
    """Rotate world-frame velocity to body frame. Same as controller.py."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    vx_body = c * vx_w + s * vy_w
    vy_body = -s * vx_w + c * vy_w
    return vx_body, vy_body


def wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


# ─── Logging ─────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ─── State machine phases ───────────────────────────────────────────────────

class Phase:
    PREFLIGHT = "preflight"
    TAKEOFF = "takeoff"
    SETTLE = "settle"
    SCAN = "scan"
    ALIGN = "align"       # yaw to face marker before approaching
    APPROACH = "approach"
    HOVER = "hover"
    LAND = "land"
    DONE = "done"
    ABORT = "abort"


# ─── Main mission logic ─────────────────────────────────────────────────────

def run_mission(
    api_base: str,
    hover_distance: float = HOVER_DISTANCE_M,
    takeoff_height: float = TAKEOFF_HEIGHT_M,
    mission_timeout: float = MISSION_TIMEOUT_S,
    scan_speed: int = SCAN_YAW_SPEED,
    approach_speed: int = APPROACH_SPEED,
    record: bool = False,
):
    global _api_base
    _api_base = api_base

    log(f"ArUco Seek & Approach")
    log(f"  API server:      {api_base}")
    log(f"  Hover distance:  {hover_distance} m")
    log(f"  Takeoff height:  {takeoff_height} m")
    log(f"  Approach speed:  {approach_speed} (RC %)")
    log(f"  Timeout:         {mission_timeout} s")
    log(f"  Ctrl+C to abort and land at any time")
    log("")

    # ── Pre-flight checks ────────────────────────────────────────────────

    log("Pre-flight: checking API connection...")
    hb = api_get("/api/heartbeat")
    if not hb.get("ok"):
        log(f"FATAL: Cannot reach API server at {api_base}")
        log(f"  Response: {hb}")
        return False

    log(f"  Drone type: {hb.get('drone_type', '?')}")
    log(f"  Connected:  {hb.get('connected')}")
    log(f"  Flying:     {hb.get('flying')}")

    if not hb.get("connected"):
        log("FATAL: Drone not connected to API server")
        return False

    if hb.get("flying"):
        log("WARNING: Drone is already flying! Will skip takeoff.")

    # Visual servoing mode — no arena config needed
    log("Pre-flight: visual servoing mode (no arena config dependency)")
    log(f"  Calibration: dist = {CALIB_A} * px^(-{CALIB_B})")
    log(f"  Hover distance: {hover_distance}m  approach speed: {approach_speed}")

    # Start video stream — required for ArUco detection
    # The positioning loop processes frames from the video callback;
    # without an active video stream, no frames are fed to OpenCV.
    log("Pre-flight: starting video stream (MJPEG)...")
    vid_result = api_post("/api/video/start", {"mode": "mjpeg"})
    if vid_result.get("ok"):
        log(f"  Video stream started: {vid_result.get('mode', '?')}")
    else:
        log(f"  WARNING: Video start failed: {vid_result.get('error', '?')}")
        log(f"  ArUco detection may not work without video!")

    # Enable position tracking (must be before recording so frames flow)
    log("Pre-flight: enabling position tracking...")
    api_post("/api/position/config", {"enabled": True})

    # Start position listener
    pos_listener = PositionListener(api_base)
    pos_listener.start()
    log("  Position SSE listener started")

    # Start client-side video marker tracker (local ArUco detection)
    vid_tracker = VideoMarkerTracker(api_base)
    vid_tracker.start()
    log("  Video marker tracker started (client-side ArUco)")

    # Start heartbeat
    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True, name="heartbeat")
    hb_thread.start()
    log("  Heartbeat thread started")

    # Wait for first position frame so the server knows the actual resolution
    time.sleep(1.5)

    # Start video recording if requested (after position is enabled and frames
    # are flowing so the server knows the actual frame resolution)
    _recording = False
    if record:
        log("Pre-flight: starting video recording...")
        rec_result = api_post("/api/video/record/start", {"raw": False})
        if rec_result.get("ok"):
            _recording = True
            log(f"  Recording to: {rec_result.get('path', '?')}")
        else:
            log(f"  WARNING: Recording failed: {rec_result.get('error', '?')}")

    # ── State machine ────────────────────────────────────────────────────

    phase = Phase.TAKEOFF if not hb.get("flying") else Phase.SETTLE
    mission_start = time.time()
    phase_start = time.time()
    scan_start_yaw = None
    scan_total_rotation = 0.0
    scan_last_yaw = None
    target_marker_id: Optional[int] = None
    approach_start_time = 0.0
    _marker_lost_ticks = 0
    _approach_stable_ticks = 0
    tick_interval = 1.0 / CONTROL_HZ

    log(f"\n{'='*60}")
    log(f"Mission start — phase: {phase}")
    log(f"{'='*60}\n")

    _last_tick = time.time()
    try:
        while phase not in (Phase.DONE, Phase.ABORT) and not _abort.is_set():
            # ── Tick timing (at top so 'continue' never skips it) ───
            now = time.time()
            sleep_time = max(0.0, tick_interval - (now - _last_tick))
            if sleep_time > 0:
                _abort.wait(sleep_time)
            _last_tick = time.time()

            # ── Global timeout ───────────────────────────────────────
            elapsed = time.time() - mission_start
            if elapsed > mission_timeout:
                log(f"TIMEOUT: Mission exceeded {mission_timeout}s — landing")
                phase = Phase.LAND

            # ── Get current state ────────────────────────────────────
            pos_data = pos_listener.get()
            pos = pos_data.get("pos") if pos_data else None
            pos_age = pos_listener.age

            # ── TAKEOFF ──────────────────────────────────────────────
            if phase == Phase.TAKEOFF:
                log("Taking off...")
                result = api_post("/api/takeoff")
                if result.get("ok"):
                    log("  Takeoff command accepted")
                    phase = Phase.SETTLE
                    phase_start = time.time()
                else:
                    log(f"  Takeoff failed: {result.get('error', '?')}")
                    phase = Phase.ABORT
                time.sleep(1.0)

            # ── SETTLE — wait for stable hover ───────────────────────
            elif phase == Phase.SETTLE:
                settle_elapsed = time.time() - phase_start
                if settle_elapsed < SETTLE_TIME_S:
                    # Gently hold altitude via RC (no horizontal movement)
                    # Let the flight controller stabilise
                    if settle_elapsed > 1.0:
                        # After 1s, try to reach target altitude
                        tel = get_telemetry()
                        height_cm = tel.get("height_cm") or 0
                        height_m = height_cm / 100.0
                        alt_err = takeoff_height - height_m
                        ud = _clamp(int(alt_err * 40), -30, 30)
                        send_rc(0, 0, ud, 0)
                    status = f"settling... {settle_elapsed:.1f}/{SETTLE_TIME_S}s"
                    if pos:
                        status += f"  pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                    log(status)
                else:
                    log("Settle complete — starting scan")
                    phase = Phase.SCAN
                    phase_start = time.time()
                    scan_start_yaw = None
                    scan_total_rotation = 0.0
                    scan_last_yaw = None

            # ── SCAN — rotate 360 looking for markers ────────────────
            elif phase == Phase.SCAN:
                # Check both SSE data and client-side tracker for markers
                seen_markers = []
                if pos_data:
                    seen_markers = pos_data.get("seen_markers", [])

                # Also check vid_tracker (client-side ArUco detection)
                vid_all = vid_tracker.get_all()

                # Merge: prefer vid_tracker data (always has pixel info)
                all_detected = {}  # {id: {"px_size": .., "center": ..}}
                for mid_str, info in (pos_data or {}).get("marker_pixel_sizes", {}).items():
                    mid = int(mid_str)
                    centers = (pos_data or {}).get("marker_centers", {})
                    all_detected[mid] = {
                        "px_size": info if isinstance(info, (int, float)) else 0,
                        "center": centers.get(mid_str),
                    }
                # vid_tracker overrides (more reliable pixel data)
                for mid, info in vid_all.items():
                    all_detected[mid] = info
                # Also include SSE seen_markers (may not have pixel info)
                for mid in seen_markers:
                    if mid not in all_detected:
                        all_detected[mid] = {"px_size": 0, "center": None}

                if all_detected:
                    # Found marker(s)! Pick the largest (closest) one.
                    best_id = None
                    best_px = 0
                    for mid, info in all_detected.items():
                        mpx = info.get("px_size", 0) or 0
                        if best_id is None or mpx > best_px:
                            best_id = mid
                            best_px = mpx

                    target_marker_id = best_id
                    send_rc_stop()
                    best_info = all_detected[best_id]
                    est_dist = pixel_distance_estimate(best_px) if best_px > 0 else "?"
                    log(f"MARKER FOUND! ID={target_marker_id}  "
                        f">>> {est_dist} <<<  "
                        f"px={best_px:.0f}  center={best_info.get('center', 'N/A')}  "
                        f"[{'vid_tracker' if best_id in vid_all else 'SSE'}]  "
                        f"all_ids={list(all_detected.keys())}")
                    phase = Phase.ALIGN
                    approach_start_time = time.time()
                    phase_start = time.time()
                    _marker_lost_ticks = 0
                    _align_centered_ticks = 0
                else:
                    # No markers seen — keep rotating
                    # Track total rotation to detect full 360
                    tel = get_telemetry()
                    yaw_deg = tel.get("yaw") or 0.0
                    yaw_rad = math.radians(yaw_deg)

                    if scan_last_yaw is not None:
                        d_yaw = wrap_pi(yaw_rad - scan_last_yaw)
                        scan_total_rotation += abs(d_yaw)
                    scan_last_yaw = yaw_rad

                    if scan_total_rotation > 2 * math.pi + 0.3:
                        # Completed full rotation — no markers found
                        send_rc_stop()
                        log("SCAN COMPLETE: No markers found after 360 degree rotation")
                        log("Landing...")
                        phase = Phase.LAND
                    else:
                        # Keep rotating slowly CW
                        # Also try to hold altitude
                        height_cm = tel.get("height_cm") or 0
                        height_m = height_cm / 100.0
                        alt_err = takeoff_height - height_m
                        ud = _clamp(int(alt_err * 40), -25, 25)

                        send_rc(0, 0, ud, scan_speed)
                        rot_deg = math.degrees(scan_total_rotation)
                        log(f"Scanning... rotated {rot_deg:.0f}/360 deg  "
                            f"alt={height_m:.2f}m  target_alt={takeoff_height:.1f}m")

            # ── ALIGN — yaw to face the marker dead-on before approaching ──
            #
            # Pure rotation + altitude: no forward/lateral movement.
            # Only transitions to APPROACH once marker is centred in image.
            #
            elif phase == Phase.ALIGN:
                # Timeout: if we can't align within 15s, try approaching anyway
                if time.time() - phase_start > 15.0:
                    log("ALIGN timeout — proceeding to approach")
                    phase = Phase.APPROACH
                    phase_start = time.time()
                    continue

                # Get marker data (same priority: vid_tracker > SSE)
                mk_center = None
                mk_px_size = 0.0
                frame_w, frame_h = 1280, 720
                data_src = "none"

                vid_info = vid_tracker.get_marker(target_marker_id)
                if vid_info and vid_tracker.age < 1.0:
                    mk_center = vid_info["center"]
                    mk_px_size = vid_info["px_size"]
                    frame_w, frame_h = vid_tracker.frame_size
                    data_src = "vid_tracker"
                if mk_px_size <= 0 and pos_data:
                    centers = pos_data.get("marker_centers", {})
                    px_sizes = pos_data.get("marker_pixel_sizes", {})
                    sse_center = centers.get(str(target_marker_id))
                    sse_px = px_sizes.get(str(target_marker_id), 0.0)
                    if sse_center and sse_px > 0:
                        mk_center = sse_center
                        mk_px_size = sse_px
                        frame_w = pos_data.get("frame_w") or 1280
                        frame_h = pos_data.get("frame_h") or 720
                        data_src = "SSE"

                if mk_center is None or mk_px_size <= 0:
                    # No pixel data — check if marker is at least in seen list
                    marker_in_seen = pos_data and target_marker_id in pos_data.get("seen_markers", [])
                    if not marker_in_seen:
                        _marker_lost_ticks += 1
                        send_rc_stop()
                        if _marker_lost_ticks > CONTROL_HZ * 5:
                            log("ALIGN: marker lost — returning to scan")
                            phase = Phase.SCAN
                            phase_start = time.time()
                            scan_total_rotation = 0.0
                            scan_last_yaw = None
                        else:
                            log(f"ALIGN: marker not visible (lost {_marker_lost_ticks} ticks)")
                    else:
                        # Marker visible but no pixel data — skip alignment
                        log("ALIGN: no pixel data — proceeding to approach")
                        phase = Phase.APPROACH
                        phase_start = time.time()
                    continue

                _marker_lost_ticks = 0
                mk_cx, mk_cy = mk_center[0], mk_center[1]
                img_cx, img_cy = frame_w / 2.0, frame_h / 2.0
                err_x = (mk_cx - img_cx) / img_cx
                err_y = (mk_cy - img_cy) / img_cy
                est_dist = pixel_distance_estimate(mk_px_size)

                # Pure yaw to face the marker
                yaw_P = 45.0
                yaw_rc = _clamp(int(err_x * yaw_P), -35, 35)

                # Also correct altitude to match marker height
                alt_P = 35.0
                ud = _clamp(int(-err_y * alt_P), -25, 25)

                # No forward/lateral movement during alignment
                send_rc(0, 0, ud, yaw_rc)

                # Check if centred enough to proceed
                if abs(err_x) < 0.08 and abs(err_y) < 0.15:
                    _align_centered_ticks += 1
                else:
                    _align_centered_ticks = 0

                if _align_centered_ticks >= CONTROL_HZ * 1:  # 1s centred
                    log(f"ALIGNED!  >>> {est_dist:.2f}m <<<  "
                        f"err_x={err_x:+.3f}  err_y={err_y:+.3f}  "
                        f"— proceeding to approach")
                    phase = Phase.APPROACH
                    phase_start = time.time()
                    _approach_stable_ticks = 0
                else:
                    log(f"Align:  >>> {est_dist:.2f}m <<<  "
                        f"err_x={err_x:+.2f} err_y={err_y:+.2f}  "
                        f"yaw={yaw_rc} ud={ud}  "
                        f"centered={_align_centered_ticks}/{CONTROL_HZ}  [{data_src}]")

            # ── APPROACH — visual servoing ──────────────────────────────
            #
            # Strategy: use the marker's position IN THE CAMERA IMAGE to
            # steer the drone. No world-coordinate position needed.
            #   - Marker left/right of center → yaw to centre it
            #   - Marker above/below center → adjust altitude
            #   - Marker pixel size → distance estimate → forward speed
            # This works in ANY room, even without arena config.
            #
            # Data sources (in priority order):
            #   1. vid_tracker (client-side ArUco on MJPEG frames)
            #   2. SSE marker_pixel_sizes / marker_centers (server-side)
            #   3. SSE seen_markers (no pixel data — fallback creep)
            #
            elif phase == Phase.APPROACH:
                # Approach timeout
                if time.time() - approach_start_time > 60.0:
                    send_rc_stop()
                    log("APPROACH TIMEOUT after 60s — landing")
                    phase = Phase.LAND
                    continue

                # ── Gather marker data from all sources ──
                mk_center = None
                mk_px_size = 0.0
                marker_in_seen = False
                frame_w, frame_h = 1280, 720  # defaults
                data_src = "none"

                # Source 1: vid_tracker (client-side, always has pixel data)
                vid_info = vid_tracker.get_marker(target_marker_id)
                if vid_info and vid_tracker.age < 1.0:
                    mk_center = vid_info["center"]
                    mk_px_size = vid_info["px_size"]
                    frame_w, frame_h = vid_tracker.frame_size
                    marker_in_seen = True
                    data_src = "vid_tracker"

                # Source 2: SSE pixel data (fallback if vid_tracker has nothing)
                if mk_px_size <= 0 and pos_data:
                    centers = pos_data.get("marker_centers", {})
                    px_sizes = pos_data.get("marker_pixel_sizes", {})
                    sse_center = centers.get(str(target_marker_id))
                    sse_px = px_sizes.get(str(target_marker_id), 0.0)
                    if sse_center and sse_px > 0:
                        mk_center = sse_center
                        mk_px_size = sse_px
                        frame_w = pos_data.get("frame_w") or 1280
                        frame_h = pos_data.get("frame_h") or 720
                        marker_in_seen = True
                        data_src = "SSE"

                # Source 3: SSE seen_markers (no pixel data)
                if not marker_in_seen and pos_data:
                    seen = pos_data.get("seen_markers", [])
                    if target_marker_id in seen:
                        marker_in_seen = True
                        data_src = "SSE-seen-only"

                has_pixel_data = mk_center is not None and mk_px_size > 0

                if not has_pixel_data and not marker_in_seen:
                    # Marker truly not visible — hold still and wait
                    _marker_lost_ticks += 1
                    send_rc_stop()
                    if _marker_lost_ticks > CONTROL_HZ * 5:  # 5 seconds
                        log(f"MARKER LOST for {_marker_lost_ticks / CONTROL_HZ:.0f}s "
                            f"— returning to scan")
                        phase = Phase.SCAN
                        phase_start = time.time()
                        scan_total_rotation = 0.0
                        scan_last_yaw = None
                    else:
                        log(f"Approach: marker not visible "
                            f"(lost {_marker_lost_ticks} ticks) — holding")
                    continue

                _marker_lost_ticks = 0  # reset — marker is visible

                if has_pixel_data:
                    # ── Full visual servoing ──
                    mk_cx, mk_cy = mk_center[0], mk_center[1]
                    img_cx, img_cy = frame_w / 2.0, frame_h / 2.0

                    # Normalised error: -1.0 (left/top) to +1.0 (right/bottom)
                    err_x = (mk_cx - img_cx) / img_cx
                    err_y = (mk_cy - img_cy) / img_cy

                    est_dist = pixel_distance_estimate(mk_px_size)

                    # YAW: centre marker horizontally → face the marker directly
                    # Strong P-gain so the drone actively rotates to face it
                    yaw_P = 40.0
                    yaw_rc = _clamp(int(err_x * yaw_P), -30, 30)

                    # LATERAL: strafe to keep marker centred while yaw catches up
                    # This prevents the drone from arcing — it flies straight at
                    # the marker instead of curving toward it
                    lr_P = 15.0
                    lr = _clamp(int(err_x * lr_P), -12, 12)

                    # ALTITUDE: match marker altitude — keep it exactly centred
                    # vertically. Higher P-gain + wider clamp for strong correction.
                    alt_P = 35.0
                    ud = _clamp(int(-err_y * alt_P), -25, 25)

                    # FORWARD/BACKWARD: proportional distance controller
                    # Positive dist_err = too far → fly forward
                    # Negative dist_err = too close → fly backward (away from wall!)
                    dist_err = est_dist - hover_distance
                    dist_P = 8.0
                    fb = _clamp(int(dist_err * dist_P), -approach_speed, approach_speed)

                    # Don't fly forward unless well-aligned with marker
                    if abs(err_x) > 0.15 and fb > 0:
                        fb = 0  # yaw to re-align first

                    # Transition to HOVER once stable near target distance
                    if abs(dist_err) < 0.3 and abs(err_x) < 0.15:
                        _approach_stable_ticks += 1
                        if _approach_stable_ticks >= CONTROL_HZ * 1:  # 1s stable
                            log(f"ARRIVED!  >>> {est_dist:.2f}m <<<  "
                                f"(target {hover_distance}m)  px={mk_px_size:.0f}  "
                                f"[{data_src}]")
                            phase = Phase.HOVER
                            phase_start = time.time()
                            _approach_stable_ticks = 0
                    else:
                        _approach_stable_ticks = 0

                    send_rc(lr, fb, ud, yaw_rc)
                    log(f"Approach  >>> {est_dist:.2f}m <<<  (target {hover_distance}m)  "
                        f"px={mk_px_size:.0f}  err_x={err_x:+.2f} err_y={err_y:+.2f}  "
                        f"RC lr={lr} fb={fb} ud={ud} yaw={yaw_rc}  [{data_src}]")

                else:
                    # ── Fallback: no pixel data, but marker in seen_markers ──
                    # Creep forward slowly — we know the marker is visible
                    # to the camera but the server doesn't send pixel coords.
                    # Hold altitude via telemetry, no yaw correction.
                    tel = get_telemetry()
                    height_cm = tel.get("height_cm") or 0
                    height_m = height_cm / 100.0
                    alt_err = takeoff_height - height_m
                    ud = _clamp(int(alt_err * 30), -15, 15)

                    fb = approach_speed  # steady creep forward
                    send_rc(0, fb, ud, 0)
                    log(f"Approach (no-pixel fallback): marker {target_marker_id} "
                        f"in seen_markers, creeping fb={fb}  alt={height_m:.2f}m")

            # ── HOVER — hold position facing marker ─────────────────
            elif phase == Phase.HOVER:
                # ── Gather marker data from all sources ──
                mk_center = None
                mk_px_size = 0.0
                marker_in_seen = False
                frame_w, frame_h = 1280, 720
                data_src = "none"

                # Source 1: vid_tracker
                vid_info = vid_tracker.get_marker(target_marker_id)
                if vid_info and vid_tracker.age < 1.0:
                    mk_center = vid_info["center"]
                    mk_px_size = vid_info["px_size"]
                    frame_w, frame_h = vid_tracker.frame_size
                    marker_in_seen = True
                    data_src = "vid_tracker"

                # Source 2: SSE pixel data
                if mk_px_size <= 0 and pos_data:
                    centers = pos_data.get("marker_centers", {})
                    px_sizes = pos_data.get("marker_pixel_sizes", {})
                    sse_center = centers.get(str(target_marker_id))
                    sse_px = px_sizes.get(str(target_marker_id), 0.0)
                    if sse_center and sse_px > 0:
                        mk_center = sse_center
                        mk_px_size = sse_px
                        frame_w = pos_data.get("frame_w") or 1280
                        frame_h = pos_data.get("frame_h") or 720
                        marker_in_seen = True
                        data_src = "SSE"

                # Source 3: SSE seen_markers
                if not marker_in_seen and pos_data:
                    seen = pos_data.get("seen_markers", [])
                    if target_marker_id in seen:
                        marker_in_seen = True
                        data_src = "SSE-seen-only"

                has_pixel_data = mk_center is not None and mk_px_size > 0

                if not has_pixel_data and not marker_in_seen:
                    _marker_lost_ticks += 1
                    send_rc_stop()
                    if _marker_lost_ticks > CONTROL_HZ * 8:
                        log(f"MARKER LOST during hover for "
                            f"{_marker_lost_ticks / CONTROL_HZ:.0f}s — landing")
                        phase = Phase.LAND
                    else:
                        log(f"HOVER: marker not visible "
                            f"(lost {_marker_lost_ticks} ticks) — holding")
                    continue

                _marker_lost_ticks = 0

                if has_pixel_data:
                    mk_cx, mk_cy = mk_center[0], mk_center[1]
                    img_cx, img_cy = frame_w / 2.0, frame_h / 2.0
                    err_x = (mk_cx - img_cx) / img_cx
                    err_y = (mk_cy - img_cy) / img_cy
                    est_dist = pixel_distance_estimate(mk_px_size)

                    # YAW: keep marker centred → drone faces it directly
                    # Strong P-gain for tight heading lock
                    yaw_P = 35.0
                    yaw_rc = _clamp(int(err_x * yaw_P), -25, 25)

                    # LATERAL: strafe to stay centred on the marker
                    lr_P = 12.0
                    lr = _clamp(int(err_x * lr_P), -10, 10)

                    # ALTITUDE: match marker altitude — keep marker exactly
                    # at vertical centre of image. Strong P-gain.
                    alt_P = 35.0
                    ud = _clamp(int(-err_y * alt_P), -25, 25)

                    # DISTANCE: actively maintain hover_distance
                    # Positive dist_err = too far → creep forward
                    # Negative dist_err = too close → back away from wall!
                    dist_err = est_dist - hover_distance
                    dist_P = 8.0
                    fb = _clamp(int(dist_err * dist_P), -15, 10)

                    send_rc(lr, fb, ud, yaw_rc)
                    hover_dur = time.time() - phase_start
                    log(f"HOVER  >>> {est_dist:.2f}m <<<  (target {hover_distance}m)  "
                        f"px={mk_px_size:.0f}  err_x={err_x:+.2f} err_y={err_y:+.2f}  "
                        f"RC lr={lr} fb={fb} ud={ud} yaw={yaw_rc}  "
                        f"ID={target_marker_id}  t={hover_dur:.0f}s  [{data_src}]")
                else:
                    # Fallback hover: just hold position, no pixel data
                    tel = get_telemetry()
                    height_cm = tel.get("height_cm") or 0
                    height_m = height_cm / 100.0
                    alt_err = takeoff_height - height_m
                    ud = _clamp(int(alt_err * 25), -10, 10)
                    send_rc(0, 0, ud, 0)
                    hover_dur = time.time() - phase_start
                    log(f"HOVER (no-pixel fallback): marker {target_marker_id} visible  "
                        f"alt={height_m:.2f}m  t={hover_dur:.0f}s  [Ctrl+C to land]")

            # ── LAND ─────────────────────────────────────────────────
            elif phase == Phase.LAND:
                send_rc_stop()
                time.sleep(0.2)
                log("Landing...")
                result = api_post("/api/land")
                log(f"  Land result: {result}")
                time.sleep(3.0)
                phase = Phase.DONE

    except KeyboardInterrupt:
        log("\n\nCTRL+C — ABORTING MISSION")
        phase = Phase.ABORT

    # ── Cleanup ──────────────────────────────────────────────────────────

    if phase == Phase.ABORT:
        send_rc_stop()
        time.sleep(0.2)
        log("Emergency landing...")
        api_post("/api/land")
        time.sleep(2.0)

    pos_listener.stop()
    vid_tracker.stop()
    _abort.set()

    # Stop recording
    if _recording:
        try:
            rec_stop = api_post("/api/video/record/stop")
            if rec_stop.get("ok"):
                log(f"Recording saved: {rec_stop.get('frames', '?')} frames → {rec_stop.get('path', '?')}")
            else:
                log(f"Recording stop error: {rec_stop.get('error', '?')}")
        except Exception:
            pass

    # Stop video stream
    try:
        api_post("/api/video/stop")
        log("Video stream stopped")
    except Exception:
        pass

    log(f"\n{'='*60}")
    log(f"Mission ended — phase: {phase}")
    elapsed = time.time() - mission_start
    log(f"Total time: {elapsed:.1f}s")
    log(f"{'='*60}")

    return phase == Phase.DONE


# ─── Signal handlers ─────────────────────────────────────────────────────────

def _signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    log("\nSignal received — setting abort flag")
    _abort.set()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ArUco Marker Seek & Approach — autonomous drone tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --api http://flightctrl1:8080
    %(prog)s --api http://flightctrl1:8080 --hover-distance 1.5
    %(prog)s --api http://flightctrl1:8080 --scan-speed 20 --timeout 180

Safety:
    Press Ctrl+C at any time to abort and land the drone.
    The drone will also auto-land on timeout or position loss.
        """,
    )
    parser.add_argument(
        "--api", default=DEFAULT_API_BASE,
        help=f"API server base URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--hover-distance", type=float, default=HOVER_DISTANCE_M,
        help=f"Distance to hover in front of marker (default: {HOVER_DISTANCE_M}m)",
    )
    parser.add_argument(
        "--takeoff-height", type=float, default=TAKEOFF_HEIGHT_M,
        help=f"Takeoff/scan altitude (default: {TAKEOFF_HEIGHT_M}m)",
    )
    parser.add_argument(
        "--scan-speed", type=int, default=SCAN_YAW_SPEED,
        help=f"Yaw rotation speed during scan, 0-100 (default: {SCAN_YAW_SPEED})",
    )
    parser.add_argument(
        "--approach-speed", type=int, default=APPROACH_SPEED,
        help=f"Forward RC percent during approach, 1-100 (default: {APPROACH_SPEED})",
    )
    parser.add_argument(
        "--timeout", type=float, default=MISSION_TIMEOUT_S,
        help=f"Mission timeout in seconds (default: {MISSION_TIMEOUT_S})",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Record video (with ArUco overlay) during the mission for debugging",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without executing (check API connectivity only)",
    )

    args = parser.parse_args()

    # Install signal handler for clean abort
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.dry_run:
        global _api_base
        _api_base = args.api
        log("DRY RUN — checking connectivity only")
        hb = api_get("/api/heartbeat")
        log(f"  Heartbeat: {hb}")
        arena = get_arena_config()
        markers = get_marker_positions(arena)
        log(f"  Arena: {len(markers)} markers")
        for mid, mk in sorted(markers.items()):
            tx, ty, tz, tyaw = compute_approach_target(mk, args.hover_distance)
            log(f"    Marker {mid:2d}: ({mk['x']:6.2f}, {mk['y']:6.2f}, {mk['z']:5.2f}) "
                f"wall={mk['wall']:6s} → hover at ({tx:.2f}, {ty:.2f}, {tz:.2f}) "
                f"yaw={math.degrees(tyaw):.0f}deg")
        pos = api_get("/api/position")
        log(f"  Position: {pos}")
        log("DRY RUN complete — no takeoff")
        return

    success = run_mission(
        api_base=args.api,
        hover_distance=args.hover_distance,
        takeoff_height=args.takeoff_height,
        mission_timeout=args.timeout,
        scan_speed=args.scan_speed,
        approach_speed=args.approach_speed,
        record=args.record,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
