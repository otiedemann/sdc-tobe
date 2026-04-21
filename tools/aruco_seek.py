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
    python aruco_seek.py --api http://flightctrl1:8080 --marker 5
    python aruco_seek.py --api http://flightctrl1:8080 --hover-distance 1.5 --marker 12
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
HOVER_DISTANCE_M = 2.0        # stop this far in front of the marker (SDC ops default)
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
MARKER_SIZE_M = 0.5           # physical marker size in meters (SDC arena: 0.5m = 50×50cm)
MARKER_SIZE_CALIB_M = 0.18    # marker size the CALIB_A/CALIB_B power-law was fit for
                              # — DO NOT change unless you refit the calibration curve.
                              # Distances for differently-sized markers are auto-scaled
                              # by (MARKER_SIZE_M / MARKER_SIZE_CALIB_M) ** CALIB_B.

# ─── HOVER controller tuning ────────────────────────────────────────────────
# These set how "twitchy" vs "soft" the in-front-of-marker hold is.
# Defaults tuned for "stay still, follow slowly" rather than "track sharply".
HOVER_EMA_ALPHA       = 0.30   # 0=no update, 1=no smoothing. Lower = smoother
HOVER_DEADBAND_X      = 0.05   # |err_x| below this → no yaw command
HOVER_DEADBAND_Y      = 0.06   # |err_y| below this → no altitude command
HOVER_DEADBAND_SKEW   = 0.04   # |skew| below this → no lateral command. Small value so the
                              # drone corrects for perspective distortion and ends up
                              # perpendicular to the marker face. Raise to 0.10+ if drone
                              # roll introduces too much noise.
HOVER_DEADBAND_DIST_M = 0.20   # |dist err| below this → no fwd/back command
HOVER_YAW_P           = 10.0   # was 35 — gentle
HOVER_SKEW_P          = 25.0   # lateral gain per unit of perspective skew. Strong enough to
                              # actually drive the drone perpendicular to the marker. Previous
                              # 8.0 was too weak — at skew=0.2 it only produced 1.6 RC which
                              # was below rc_min, so the drone stayed sideways forever.
HOVER_ALT_P           = 25.0   # was 60 — gentle altitude hold
HOVER_DIST_P          = 20.0   # aggressive: at dist_err=2m this already produces 40 RC,
                              # clamped to fb_max=45 → near-full-speed even at moderate
                              # distances. Drone eases off as dist_err shrinks below 1m.
HOVER_YAW_MAX         = 15     # was 40 — clamp magnitude
HOVER_LR_MAX          = 20     # RC% cap on lateral correction — enough authority to
                              # actively strafe into the marker's normal while approaching.
HOVER_UD_MAX          = 25     # was 50
HOVER_FB_MAX          = 45     # full-cruise RC when far. PD naturally slows the drone as
                              # dist_err shrinks below ~1m so this isn't a "final hover"
                              # speed — it's the cap during long approaches across the
                              # arena.
HOVER_FB_BACK_MAX     = 45     # matching backward authority for overshoot recovery / setting
                              # up a head-on approach by retreating and coming back in.
HOVER_RC_MIN          = 2      # |rc| < this → output 0 (kill micro-jitter)

# ── IMU damping (D-term) — opposes actual body motion to kill oscillation ──
# vgx/vgy/vgz from Anafi are body-frame in cm/s (Parrot NED: x fwd, y right, z down).
# yaw rate is computed by differencing consecutive yaw samples.
# Larger D = stronger damping (more sluggish, less overshoot).
HOVER_D_LR            = 0.40   # RC% per cm/s lateral velocity
HOVER_D_FB            = 0.40   # RC% per cm/s forward  velocity
HOVER_D_UD            = 0.50   # RC% per cm/s vertical velocity
HOVER_D_YAW           = 0.40   # RC% per deg/s yaw rate
HOVER_VEL_MAX_AGE_S   = 1.0    # ignore IMU velocity older than this

# ─── Calibration curve (originally fit for 18cm markers) ────────────────────
# Power-law fit from Distanzkalibrierung (calibration chart):
#   distance = CALIB_A * pixel_size ^ (-CALIB_B)
#   distance = 109.1653 * (1/pixel)^0.8973
#            = 109.1653 * pixel^(-0.8973)
#
# This was fit with an 18cm marker. For a marker of different physical size S,
# the same pixel reading corresponds to a distance scaled by (S / 0.18) ** CALIB_B,
# because pixel_size ∝ physical_size / distance at fixed focal length. We apply
# that scaling automatically below — so you only need to set MARKER_SIZE_M.
CALIB_A = 109.1653           # coefficient (calibrated at MARKER_SIZE_CALIB_M)
CALIB_B = 0.8973             # exponent


def pixel_distance_estimate(avg_pixel_size: float, marker_size_m: float = None) -> float:
    """Estimate distance (m) from marker pixel size using calibration curve.

    The calibration was fit with an 18cm marker. For any other marker size we
    scale the result by (marker_size_m / MARKER_SIZE_CALIB_M) ** CALIB_B so the
    same power-law holds. Pass `marker_size_m` explicitly to override the
    module-level MARKER_SIZE_M at call time (useful for per-drone overrides).
    """
    if avg_pixel_size <= 0:
        return float("inf")
    s = MARKER_SIZE_M if marker_size_m is None else float(marker_size_m)
    scale = (s / MARKER_SIZE_CALIB_M) ** CALIB_B
    return CALIB_A * scale * avg_pixel_size ** (-CALIB_B)


def edge_boost(err: float, threshold: float = 0.6) -> float:
    """
    Extra gain when marker is near the edge of the image.

    Linear ramp from 1.0 at |err|=threshold to 3.0 at |err|=1.0.
    Returns a multiplier >= 1.0.  Below threshold, returns 1.0.
    """
    a = abs(err)
    if a <= threshold:
        return 1.0
    return 1.0 + 2.0 * (a - threshold) / (1.0 - threshold)


def marker_skew(left_len: float, right_len: float) -> float:
    """
    Compute perspective skew from left/right edge lengths of a marker.

    When viewing a marker straight-on, left_len ≈ right_len → skew ≈ 0.
    When the drone is to the LEFT of the marker normal:
      - right edge appears longer (closer) → skew > 0 → need to strafe RIGHT
    When the drone is to the RIGHT:
      - left edge appears longer → skew < 0 → need to strafe LEFT

    Returns: normalised skew in [-1, +1].  Positive = strafe right needed.
    """
    total = left_len + right_len
    if total < 1.0:
        return 0.0
    return (right_len - left_len) / total


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

    def __init__(self, api_base: str, endpoint: str = "/api/video"):
        # IMPORTANT: default to the RAW video feed, not /api/position/video.
        # The position feed is annotated (drawDetectedMarkers overlays green
        # lines on the marker borders and puts text inside the marker body),
        # which corrupts the marker pattern and breaks local re-detection.
        # Consume the raw camera stream so our client-side detector sees
        # uncorrupted marker images.
        self._url = f"{api_base}{endpoint}"
        self._lock = threading.Lock()
        self._markers: dict = {}  # {id: {"center": [cx,cy], "px_size": float, "corners": [[x,y]*4], "left_len": float, "right_len": float}}
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

    @property
    def is_active(self) -> bool:
        """True if the tracker thread is running and has produced at least one frame."""
        with self._lock:
            return self._last_ts > 0

    def get_marker(self, marker_id: int) -> Optional[dict]:
        """Return {"center": [cx,cy], "px_size": float, "left_len": float, "right_len": float} or None."""
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
                log(f"[vid-track] Connected to {self._url} (status {resp.status})")

                buf = b""
                _frames_decoded = 0
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
                                    # ArUco corner order: TL(0), TR(1), BR(2), BL(3)
                                    side_lens = [float(np.linalg.norm(
                                        pts[j] - pts[(j + 1) % 4])) for j in range(4)]
                                    px_size = float(np.mean(side_lens))
                                    cx = float(np.mean(pts[:, 0]))
                                    cy = float(np.mean(pts[:, 1]))
                                    # Left edge: TL→BL (pts[0]→pts[3])
                                    # Right edge: TR→BR (pts[1]→pts[2])
                                    left_len = float(np.linalg.norm(pts[0] - pts[3]))
                                    right_len = float(np.linalg.norm(pts[1] - pts[2]))
                                    new_markers[mid] = {
                                        "center": [round(cx, 1), round(cy, 1)],
                                        "px_size": round(px_size, 2),
                                        "left_len": round(left_len, 2),
                                        "right_len": round(right_len, 2),
                                    }

                            with self._lock:
                                self._markers = new_markers
                                self._frame_w = w
                                self._frame_h = h
                                self._last_ts = time.time()
                            _frames_decoded += 1
                            if _frames_decoded == 1:
                                n_ids = len(new_markers)
                                log(f"[vid-track] First frame decoded: {w}x{h}, {n_ids} markers")
                        except Exception as e:
                            if _frames_decoded == 0:
                                log(f"[vid-track] Frame decode error: {e}")
                conn.close()
            except Exception as e:
                if self._running:
                    log(f"[vid-track] Connection error: {e} — retrying in 2s")
                    time.sleep(2.0)


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
    target_marker: Optional[int] = None,
):
    global _api_base
    _api_base = api_base

    log(f"ArUco Seek & Approach")
    log(f"  API server:      {api_base}")
    log(f"  Marker size:     {MARKER_SIZE_M*100:.0f}×{MARKER_SIZE_M*100:.0f} cm")
    log(f"  Target marker:   {target_marker if target_marker is not None else 'any (first found)'}")
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

    # Raise server altitude fence so the drone can reach marker height.
    # Default MAX_ALTITUDE_M on the server is only 2.0m which blocks climbing.
    desired_max_alt = 6.0  # enough for markers at z=4.0m + margin
    log(f"Pre-flight: raising server altitude limit to {desired_max_alt}m...")
    settings_result = api_post("/api/settings", {"max_altitude_m": desired_max_alt})
    if settings_result.get("ok"):
        new_alt = settings_result.get("max_altitude_m", "?")
        log(f"  Server max altitude set to {new_alt}m")
    else:
        log(f"  WARNING: Could not raise altitude limit: {settings_result}")

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

    # Wait for first position frame and verify vid_tracker is alive
    time.sleep(2.0)
    if vid_tracker.is_active:
        vw, vh = vid_tracker.frame_size
        log(f"  vid_tracker OK: receiving {vw}x{vh} frames (age={vid_tracker.age:.1f}s)")
        vm = vid_tracker.get_all()
        if vm:
            log(f"  vid_tracker sees markers: {list(vm.keys())}")
    else:
        log("  WARNING: vid_tracker NOT active — client-side ArUco detection not working!")
        log("  Will rely on server SSE data (may lack pixel info on old server)")

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
    target_alt: float = takeoff_height   # will be updated to marker z once found
    approach_start_time = 0.0
    _marker_lost_ticks = 0
    _approach_stable_ticks = 0
    tick_interval = 1.0 / CONTROL_HZ

    # HOVER EMA filter state — None means "uninitialised, take next sample as-is"
    hover_filter_x: Optional[float] = None
    hover_filter_y: Optional[float] = None
    hover_filter_skew: Optional[float] = None
    hover_filter_dist: Optional[float] = None
    # Yaw-rate tracker (computed from consecutive yaw samples; degrees/sec)
    hover_prev_yaw_deg: Optional[float] = None
    hover_prev_yaw_t: float = 0.0
    hover_yaw_rate_dps: float = 0.0  # EMA-filtered yaw rate

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

                # Filter for specific target marker if requested
                if target_marker is not None:
                    candidates = {mid: info for mid, info in all_detected.items()
                                  if mid == target_marker}
                    if all_detected and not candidates:
                        # We see markers but not the one we want — log and keep scanning
                        other_ids = list(all_detected.keys())
                        log(f"Scanning... see markers {other_ids} but looking for ID={target_marker}")
                else:
                    candidates = all_detected

                if candidates:
                    # Found marker(s)! Pick the largest (closest) one.
                    best_id = None
                    best_px = 0
                    for mid, info in candidates.items():
                        mpx = info.get("px_size", 0) or 0
                        if best_id is None or mpx > best_px:
                            best_id = mid
                            best_px = mpx

                    target_marker_id = best_id
                    send_rc_stop()
                    best_info = candidates[best_id]
                    est_dist = pixel_distance_estimate(best_px) if best_px > 0 else "?"
                    est_str = f"{est_dist:.2f}m" if isinstance(est_dist, float) else est_dist
                    log(f"MARKER FOUND! ID={target_marker_id}  "
                        f">>> {est_str} <<<  "
                        f"px={best_px:.0f}  marker_size={MARKER_SIZE_M*100:.0f}cm  "
                        f"center={best_info.get('center', 'N/A')}  "
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

                # Perspective skew: are we looking at the marker from an angle?
                skew = 0.0
                if vid_info and "left_len" in vid_info and "right_len" in vid_info:
                    skew = marker_skew(vid_info["left_len"], vid_info["right_len"])

                # YAW: centre marker horizontally
                # Boost gain when marker is near image edge to prevent losing it
                yaw_P = 45.0 * edge_boost(err_x)
                yaw_rc = _clamp(int(err_x * yaw_P), -50, 50)

                # LATERAL: strafe to get in front of the marker (correct skew)
                skew_P = 25.0
                lr = _clamp(int(skew * skew_P), -15, 15)

                # ALTITUDE: visual servoing — descend/climb to match marker height.
                # Use strong gain + edge boost.  Also update target_alt for fallback.
                tel = get_telemetry()
                height_m = (tel.get("height_cm") or 0) / 100.0
                alt_P = 60.0 * edge_boost(err_y)
                ud = _clamp(int(-err_y * alt_P), -50, 50)
                # Update target_alt estimate from visual: when marker is centred
                # vertically, the drone is at marker height.
                if abs(err_y) < 0.10:
                    target_alt = height_m  # lock in current height as target

                send_rc(lr, 0, ud, yaw_rc)

                # Check if centred AND square (no skew) to proceed
                aligned = abs(err_x) < 0.08 and abs(err_y) < 0.15 and abs(skew) < 0.06
                if aligned:
                    _align_centered_ticks += 1
                else:
                    _align_centered_ticks = 0

                if _align_centered_ticks >= CONTROL_HZ * 1:  # 1s centred+square
                    log(f"ALIGNED!  >>> {est_dist:.2f}m <<<  "
                        f"err_x={err_x:+.3f}  skew={skew:+.3f}  "
                        f"— proceeding to approach")
                    phase = Phase.APPROACH
                    phase_start = time.time()
                    _approach_stable_ticks = 0
                else:
                    log(f"Align:  >>> {est_dist:.2f}m <<<  "
                        f"px={mk_px_size:.0f} ({MARKER_SIZE_M*100:.0f}cm marker)  "
                        f"err_x={err_x:+.2f} err_y={err_y:+.2f} skew={skew:+.2f}  "
                        f"alt={height_m:.2f}m  "
                        f"RC lr={lr} yaw={yaw_rc} ud={ud}  "
                        f"ok={_align_centered_ticks}/{CONTROL_HZ}  [{data_src}]")

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

                    # Perspective skew: detect if approaching from an angle
                    skew = 0.0
                    if vid_info and "left_len" in vid_info and "right_len" in vid_info:
                        skew = marker_skew(vid_info["left_len"], vid_info["right_len"])

                    # YAW: centre marker horizontally → face the marker directly
                    # Boost gain when marker nears image edge to keep it in view
                    yaw_P = 40.0 * edge_boost(err_x)
                    yaw_rc = _clamp(int(err_x * yaw_P), -50, 50)

                    # LATERAL: strafe to correct perspective skew
                    skew_P = 20.0
                    err_x_P = 12.0
                    lr = _clamp(int(skew * skew_P + err_x * err_x_P), -15, 15)

                    # ALTITUDE: visual servoing to match marker height
                    tel = get_telemetry()
                    height_m = (tel.get("height_cm") or 0) / 100.0
                    alt_P = 60.0 * edge_boost(err_y)
                    ud = _clamp(int(-err_y * alt_P), -50, 50)
                    # Update target_alt for fallback ticks
                    if abs(err_y) < 0.10:
                        target_alt = height_m

                    # FORWARD/BACKWARD: proportional distance controller
                    dist_err = est_dist - hover_distance
                    dist_P = 8.0
                    fb = _clamp(int(dist_err * dist_P), -approach_speed, approach_speed)

                    # Don't fly forward unless well-aligned with marker
                    if (abs(err_x) > 0.15 or abs(skew) > 0.10) and fb > 0:
                        fb = 0  # align first, then approach

                    # Transition to HOVER once stable near target distance + aligned
                    if abs(dist_err) < 0.3 and abs(err_x) < 0.15 and abs(skew) < 0.08 and abs(err_y) < 0.20:
                        _approach_stable_ticks += 1
                        if _approach_stable_ticks >= CONTROL_HZ * 1:  # 1s stable
                            log(f"ARRIVED!  >>> {est_dist:.2f}m <<<  "
                                f"(target {hover_distance}m)  skew={skew:+.3f}  "
                                f"alt={height_m:.2f}m  err_y={err_y:+.3f}  "
                                f"[{data_src}]")
                            phase = Phase.HOVER
                            phase_start = time.time()
                            _approach_stable_ticks = 0
                            # Reset HOVER EMA so stale APPROACH errors don't kick the start
                            hover_filter_x = None
                            hover_filter_y = None
                            hover_filter_skew = None
                            hover_filter_dist = None
                            hover_prev_yaw_deg = None
                            hover_prev_yaw_t = 0.0
                            hover_yaw_rate_dps = 0.0
                    else:
                        _approach_stable_ticks = 0

                    send_rc(lr, fb, ud, yaw_rc)
                    # Expected px size at target distance: invert dist = A * px^(-B)
                    # → px = (A / dist)^(1/B)
                    expect_px = (CALIB_A / hover_distance) ** (1.0 / CALIB_B) if hover_distance > 0 else 0
                    log(f"Approach  >>> {est_dist:.2f}m <<<  (target {hover_distance}m)  "
                        f"px={mk_px_size:.0f} (expect {expect_px:.0f}px @{hover_distance}m for {MARKER_SIZE_M*100:.0f}cm marker)  "
                        f"err_x={err_x:+.2f} err_y={err_y:+.2f} skew={skew:+.2f}  "
                        f"alt={height_m:.2f}m  "
                        f"RC lr={lr} fb={fb} ud={ud} yaw={yaw_rc}  [{data_src}]")

                else:
                    # ── Fallback: no pixel data, but marker in seen_markers ──
                    # Try vid_tracker one more time (might have data but age > 1s)
                    # If vid_tracker has any recent data at all, use it for yaw
                    fb_yaw = 0
                    fb_lr = 0
                    fallback_vid = vid_tracker.get_marker(target_marker_id)
                    if fallback_vid and vid_tracker.age < 3.0:
                        fw, fh = vid_tracker.frame_size
                        fcx = fallback_vid["center"][0]
                        ferr_x = (fcx - fw / 2.0) / (fw / 2.0)
                        fb_yaw = _clamp(int(ferr_x * 40), -25, 25)
                        fb_lr = _clamp(int(ferr_x * 12), -10, 10)
                        log(f"Approach (fallback+vid): marker {target_marker_id}  "
                            f"err_x={ferr_x:+.2f}  yaw={fb_yaw}  lr={fb_lr}  "
                            f"vid_age={vid_tracker.age:.1f}s")
                    else:
                        log(f"Approach (blind fallback): marker {target_marker_id} "
                            f"in seen_markers, NO pixel data — creeping forward  "
                            f"vid_active={vid_tracker.is_active}  vid_age={vid_tracker.age:.1f}s")

                    tel = get_telemetry()
                    height_cm = tel.get("height_cm") or 0
                    height_m = height_cm / 100.0
                    # Hold last known good altitude (target_alt tracks visual estimate)
                    alt_err = target_alt - height_m
                    ud = _clamp(int(alt_err * 25), -30, 30)

                    fb = approach_speed  # steady creep forward
                    send_rc(fb_lr, fb, ud, fb_yaw)
                    log(f"  fallback alt={height_m:.2f}m (hold {target_alt:.2f}m)  ud={ud}")

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
                    raw_err_x = (mk_cx - img_cx) / img_cx
                    raw_err_y = (mk_cy - img_cy) / img_cy
                    raw_dist = pixel_distance_estimate(mk_px_size)
                    raw_skew = 0.0
                    if vid_info and "left_len" in vid_info and "right_len" in vid_info:
                        raw_skew = marker_skew(vid_info["left_len"], vid_info["right_len"])

                    # ── EMA smoothing on camera measurements ──
                    a = HOVER_EMA_ALPHA
                    hover_filter_x    = raw_err_x if hover_filter_x    is None else (1 - a) * hover_filter_x    + a * raw_err_x
                    hover_filter_y    = raw_err_y if hover_filter_y    is None else (1 - a) * hover_filter_y    + a * raw_err_y
                    hover_filter_skew = raw_skew  if hover_filter_skew is None else (1 - a) * hover_filter_skew + a * raw_skew
                    hover_filter_dist = raw_dist  if hover_filter_dist is None else (1 - a) * hover_filter_dist + a * raw_dist

                    err_x    = hover_filter_x
                    err_y    = hover_filter_y
                    skew     = hover_filter_skew
                    est_dist = hover_filter_dist

                    # ── Deadbands: within these zones the P-term is silent ──
                    def _db(e, dz):
                        if abs(e) < dz: return 0.0
                        return e - math.copysign(dz, e)
                    err_x_eff    = _db(err_x, HOVER_DEADBAND_X)
                    err_y_eff    = _db(err_y, HOVER_DEADBAND_Y)
                    skew_eff     = _db(skew,  HOVER_DEADBAND_SKEW)
                    dist_err     = est_dist - hover_distance
                    dist_err_eff = _db(dist_err, HOVER_DEADBAND_DIST_M)

                    # ── IMU: read body velocity and yaw rate for the D-term ──
                    # vgx/vgy/vgz come from Anafi SpeedChanged in cm/s (NED body frame:
                    # x fwd+, y right+, z down+). Use them to OPPOSE current motion.
                    tel = get_telemetry()
                    vx_cms = float(tel.get("vgx") or 0.0)        # forward+
                    vy_cms = float(tel.get("vgy") or 0.0)        # right+
                    vz_cms = float(tel.get("vgz") or 0.0)        # down+
                    # Compute yaw rate from successive yaw samples (deg/s)
                    yaw_now_deg = float(tel.get("yaw") or 0.0)
                    t_now = time.time()
                    if hover_prev_yaw_deg is not None:
                        dt = t_now - hover_prev_yaw_t
                        if 0.01 < dt < 1.0:
                            d_yaw = wrap_pi(math.radians(yaw_now_deg - hover_prev_yaw_deg))
                            inst_rate = math.degrees(d_yaw) / dt
                            # EMA-smooth the yaw rate (it's noisy)
                            hover_yaw_rate_dps = (1 - a) * hover_yaw_rate_dps + a * inst_rate
                    hover_prev_yaw_deg = yaw_now_deg
                    hover_prev_yaw_t   = t_now

                    height_m = (tel.get("height_cm") or 0) / 100.0

                    # ── PD on each axis: P from camera, D from IMU ──
                    # Sign convention for damping (rc - D*velocity_in_rc_direction):
                    #   lr+  = right →  damp by  -D*vy_cms     (vy+ = right)
                    #   fb+  = fwd   →  damp by  -D*vx_cms     (vx+ = fwd)
                    #   ud+  = up    →  damp by  +D*vz_cms     (vz+ = down → push UP to oppose)
                    #   yaw+ = CW    →  damp by  -D*yaw_rate
                    yaw_rc_f = err_x_eff * HOVER_YAW_P  - HOVER_D_YAW * hover_yaw_rate_dps
                    lr_f     = skew_eff  * HOVER_SKEW_P - HOVER_D_LR  * vy_cms
                    ud_f     = -err_y_eff * HOVER_ALT_P + HOVER_D_UD  * vz_cms
                    fb_f     = dist_err_eff * HOVER_DIST_P - HOVER_D_FB * vx_cms

                    yaw_rc = _clamp(int(round(yaw_rc_f)), -HOVER_YAW_MAX, HOVER_YAW_MAX)
                    lr     = _clamp(int(round(lr_f)),     -HOVER_LR_MAX,  HOVER_LR_MAX)
                    ud     = _clamp(int(round(ud_f)),     -HOVER_UD_MAX,  HOVER_UD_MAX)
                    fb     = _clamp(int(round(fb_f)),     -HOVER_FB_BACK_MAX, HOVER_FB_MAX)

                    # Kill micro-jitter: outputs below threshold → 0
                    if abs(yaw_rc) < HOVER_RC_MIN: yaw_rc = 0
                    if abs(lr)     < HOVER_RC_MIN: lr     = 0
                    if abs(ud)     < HOVER_RC_MIN: ud     = 0
                    if abs(fb)     < HOVER_RC_MIN: fb     = 0

                    # Update target_alt fallback when well-centred vertically
                    if abs(err_y) < 0.08:
                        target_alt = height_m

                    send_rc(lr, fb, ud, yaw_rc)
                    hover_dur = time.time() - phase_start
                    expect_px = (CALIB_A / hover_distance) ** (1.0 / CALIB_B) if hover_distance > 0 else 0
                    log(f"HOVER  >>> {est_dist:.2f}m <<<  (target {hover_distance}m)  "
                        f"px={mk_px_size:.0f} (expect {expect_px:.0f}px for {MARKER_SIZE_M*100:.0f}cm marker)  "
                        f"err_x={err_x:+.2f} err_y={err_y:+.2f} skew={skew:+.2f}  "
                        f"alt={height_m:.2f}m  "
                        f"v=({vx_cms:+.0f},{vy_cms:+.0f},{vz_cms:+.0f})cm/s  "
                        f"yaw_rate={hover_yaw_rate_dps:+.0f}d/s  "
                        f"RC lr={lr} fb={fb} ud={ud} yaw={yaw_rc}  "
                        f"ID={target_marker_id}  t={hover_dur:.0f}s  [{data_src}]")
                else:
                    # ── Fallback hover: NO pixel data → IMU-only "stay still" ──
                    # We can't see the marker, but the IMU still tells us how we
                    # are drifting. Pure D-control on velocity holds position.
                    tel = get_telemetry()
                    vx_cms = float(tel.get("vgx") or 0.0)
                    vy_cms = float(tel.get("vgy") or 0.0)
                    vz_cms = float(tel.get("vgz") or 0.0)
                    height_m = (tel.get("height_cm") or 0) / 100.0

                    # Tiny optional yaw correction from a stale vid_tracker hit
                    fb_yaw = 0
                    fallback_vid = vid_tracker.get_marker(target_marker_id)
                    if fallback_vid and vid_tracker.age < 3.0:
                        fw, fh = vid_tracker.frame_size
                        fcx = fallback_vid["center"][0]
                        ferr_x = (fcx - fw / 2.0) / (fw / 2.0)
                        if abs(ferr_x) > HOVER_DEADBAND_X:
                            fb_yaw = _clamp(int(ferr_x * HOVER_YAW_P * 0.5),
                                            -HOVER_YAW_MAX, HOVER_YAW_MAX)

                    # Pure-D damping on lateral and forward (oppose any drift)
                    lr_f = -HOVER_D_LR * vy_cms
                    fb_f = -HOVER_D_FB * vx_cms
                    # Altitude: fall back to the last good visual altitude (target_alt)
                    alt_err = target_alt - height_m
                    ud_f = alt_err * HOVER_ALT_P + HOVER_D_UD * vz_cms

                    lr = _clamp(int(round(lr_f)), -HOVER_LR_MAX, HOVER_LR_MAX)
                    fb = _clamp(int(round(fb_f)), -HOVER_FB_BACK_MAX, HOVER_FB_MAX)
                    ud = _clamp(int(round(ud_f)), -HOVER_UD_MAX, HOVER_UD_MAX)
                    if abs(lr)     < HOVER_RC_MIN: lr     = 0
                    if abs(fb)     < HOVER_RC_MIN: fb     = 0
                    if abs(ud)     < HOVER_RC_MIN: ud     = 0
                    if abs(fb_yaw) < HOVER_RC_MIN: fb_yaw = 0

                    send_rc(lr, fb, ud, fb_yaw)
                    hover_dur = time.time() - phase_start
                    log(f"HOVER (IMU-fallback): marker {target_marker_id}  "
                        f"v=({vx_cms:+.0f},{vy_cms:+.0f},{vz_cms:+.0f})cm/s  "
                        f"alt={height_m:.2f}m (hold {target_alt:.2f}m)  "
                        f"RC lr={lr} fb={fb} ud={ud} yaw={fb_yaw}  "
                        f"t={hover_dur:.0f}s  vid_active={vid_tracker.is_active}")

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


# ─── OBSERVE mode — read-only, no takeoff, no RC commands ──────────────────

def run_observe(
    api_base: str,
    target_marker: Optional[int] = None,
    hover_distance: float = HOVER_DISTANCE_M,
    rate_hz: float = 5.0,
):
    """
    OBSERVE MODE — completely passive.

    Does NOT take off, does NOT send any RC command, does NOT land.
    The drone stays exactly where it is (hand-held by you, or already
    flying under phone/RC control).  The script just:
      1. Subscribes to camera frames + position SSE + telemetry
      2. Runs the SAME HOVER PD math (camera P + IMU D)
      3. Prints what it WOULD send, plus all raw inputs

    Use this to tune HOVER_* constants safely, or to verify camera+IMU
    fusion before flying for real.

    Press Ctrl+C to exit.
    """
    global _api_base
    _api_base = api_base

    log("=" * 70)
    log("OBSERVE MODE — NO TAKEOFF · NO LAND · NO RC COMMANDS WILL BE SENT")
    log("=" * 70)
    log(f"  API server:      {api_base}")
    log(f"  Target marker:   {target_marker if target_marker is not None else 'auto (largest visible)'}")
    log(f"  Hover distance:  {hover_distance} m  (used to compute fb_p only)")
    log(f"  Update rate:     {rate_hz} Hz")
    log(f"  Move the drone BY HAND to verify camera + IMU response.")
    log(f"  Ctrl+C to exit.")
    log("")

    # Pre-flight: API connectivity
    hb = api_get("/api/heartbeat")
    if not hb.get("ok"):
        log(f"FATAL: cannot reach API server at {api_base}")
        return False
    log(f"  Drone type:  {hb.get('drone_type', '?')}")
    log(f"  Connected:   {hb.get('connected')}")
    log(f"  Flying:      {hb.get('flying')}  (state will NOT be changed)")

    # Start video stream so the marker tracker has frames.
    # This does NOT make the drone fly — it only enables the camera feed.
    log("Starting video stream (MJPEG) and position SSE...")
    api_post("/api/video/start", {"mode": "mjpeg"})
    api_post("/api/position/config", {"enabled": True})

    pos_listener = PositionListener(api_base)
    pos_listener.start()
    vid_tracker = VideoMarkerTracker(api_base)
    vid_tracker.start()
    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True, name="heartbeat")
    hb_thread.start()
    log("  Heartbeat / position SSE / video tracker started")

    # Wait for first frame
    time.sleep(2.0)
    if vid_tracker.is_active:
        vw, vh = vid_tracker.frame_size
        all_seen = vid_tracker.get_all()
        log(f"  vid_tracker OK: {vw}x{vh}, {len(all_seen)} marker(s) visible: {list(all_seen.keys())}")
    else:
        log("  WARNING: vid_tracker NOT active yet — try moving the drone so a marker is visible")

    # ── Observe loop ──
    tick_interval = 1.0 / max(0.5, rate_hz)
    last_tick = time.time()
    filt_x = filt_y = filt_skew = filt_dist = None
    prev_yaw_deg: Optional[float] = None
    prev_yaw_t: float = 0.0
    yaw_rate_dps: float = 0.0
    chosen_id: Optional[int] = target_marker  # may be None initially → pick largest
    last_log_summary = ""

    try:
        while not _abort.is_set():
            now = time.time()
            sleep_time = max(0.0, tick_interval - (now - last_tick))
            if sleep_time > 0:
                _abort.wait(sleep_time)
            last_tick = time.time()

            # ── Pick / refresh target marker ──
            vid_all = vid_tracker.get_all()
            if not vid_all:
                msg = (f"OBS: no markers visible "
                       f"(vid_active={vid_tracker.is_active}, age={vid_tracker.age:.1f}s)")
                if msg != last_log_summary:
                    log(msg)
                    last_log_summary = msg
                # Still print IMU even when no marker
                tel = get_telemetry()
                vx_cms = float(tel.get("vgx") or 0.0)
                vy_cms = float(tel.get("vgy") or 0.0)
                vz_cms = float(tel.get("vgz") or 0.0)
                pitch  = float(tel.get("pitch") or 0.0)
                roll   = float(tel.get("roll")  or 0.0)
                yaw_d  = float(tel.get("yaw")   or 0.0)
                hgt_m  = (tel.get("height_cm") or 0) / 100.0
                bat    = tel.get("battery", "?")
                log(f"     IMU: v=({vx_cms:+.0f},{vy_cms:+.0f},{vz_cms:+.0f})cm/s "
                    f"att=(p{pitch:+.1f},r{roll:+.1f},y{yaw_d:+.1f})  "
                    f"alt={hgt_m:.2f}m  bat={bat}%")
                continue

            if chosen_id is None or chosen_id not in vid_all:
                # No target locked yet, OR previous lock dropped — pick the largest
                new_id = max(vid_all.items(), key=lambda kv: kv[1].get("px_size", 0))[0]
                if chosen_id != new_id:
                    log(f"OBS: tracking marker ID={new_id}  (visible: {sorted(vid_all.keys())})")
                chosen_id = new_id
                filt_x = filt_y = filt_skew = filt_dist = None  # reset filters

            vid_info = vid_tracker.get_marker(chosen_id)
            if not vid_info or vid_tracker.age > 1.0:
                log(f"OBS: marker {chosen_id} stale "
                    f"(vid_age={vid_tracker.age:.1f}s; visible IDs={sorted(vid_all.keys())})")
                continue

            cx, cy = vid_info["center"]
            px_size = float(vid_info["px_size"])
            left_len = float(vid_info.get("left_len", 0))
            right_len = float(vid_info.get("right_len", 0))
            fw, fh = vid_tracker.frame_size

            # ── Camera measurements ──
            raw_err_x = (cx - fw / 2.0) / (fw / 2.0)
            raw_err_y = (cy - fh / 2.0) / (fh / 2.0)
            raw_dist  = pixel_distance_estimate(px_size)
            raw_skew  = marker_skew(left_len, right_len)

            # EMA smoothing (same as HOVER)
            a = HOVER_EMA_ALPHA
            filt_x    = raw_err_x if filt_x    is None else (1 - a) * filt_x    + a * raw_err_x
            filt_y    = raw_err_y if filt_y    is None else (1 - a) * filt_y    + a * raw_err_y
            filt_skew = raw_skew  if filt_skew is None else (1 - a) * filt_skew + a * raw_skew
            filt_dist = raw_dist  if filt_dist is None else (1 - a) * filt_dist + a * raw_dist

            # Deadbands
            def _db(e, dz):
                if abs(e) < dz: return 0.0
                return e - math.copysign(dz, e)
            err_x_eff    = _db(filt_x, HOVER_DEADBAND_X)
            err_y_eff    = _db(filt_y, HOVER_DEADBAND_Y)
            skew_eff     = _db(filt_skew, HOVER_DEADBAND_SKEW)
            dist_err     = filt_dist - hover_distance
            dist_err_eff = _db(dist_err, HOVER_DEADBAND_DIST_M)

            # ── IMU ──
            tel = get_telemetry()
            vx_cms  = float(tel.get("vgx") or 0.0)
            vy_cms  = float(tel.get("vgy") or 0.0)
            vz_cms  = float(tel.get("vgz") or 0.0)
            pitch   = float(tel.get("pitch") or 0.0)
            roll    = float(tel.get("roll")  or 0.0)
            yaw_now = float(tel.get("yaw")   or 0.0)
            hgt_m   = (tel.get("height_cm") or 0) / 100.0
            bat     = tel.get("battery", "?")

            t_now = time.time()
            if prev_yaw_deg is not None:
                dt = t_now - prev_yaw_t
                if 0.01 < dt < 1.0:
                    d_yaw = wrap_pi(math.radians(yaw_now - prev_yaw_deg))
                    inst_rate = math.degrees(d_yaw) / dt
                    yaw_rate_dps = (1 - a) * yaw_rate_dps + a * inst_rate
            prev_yaw_deg = yaw_now
            prev_yaw_t   = t_now

            # ── HOVER PD math (camera P + IMU D) ──
            yaw_p = err_x_eff    * HOVER_YAW_P
            yaw_d = -HOVER_D_YAW * yaw_rate_dps
            lr_p  = skew_eff     * HOVER_SKEW_P
            lr_d  = -HOVER_D_LR  * vy_cms
            ud_p  = -err_y_eff   * HOVER_ALT_P
            ud_d  = HOVER_D_UD   * vz_cms
            fb_p  = dist_err_eff * HOVER_DIST_P
            fb_d  = -HOVER_D_FB  * vx_cms

            yaw_rc = _clamp(int(round(yaw_p + yaw_d)), -HOVER_YAW_MAX,    HOVER_YAW_MAX)
            lr     = _clamp(int(round(lr_p  + lr_d)),  -HOVER_LR_MAX,     HOVER_LR_MAX)
            ud     = _clamp(int(round(ud_p  + ud_d)),  -HOVER_UD_MAX,     HOVER_UD_MAX)
            fb     = _clamp(int(round(fb_p  + fb_d)),  -HOVER_FB_BACK_MAX, HOVER_FB_MAX)
            if abs(yaw_rc) < HOVER_RC_MIN: yaw_rc = 0
            if abs(lr)     < HOVER_RC_MIN: lr     = 0
            if abs(ud)     < HOVER_RC_MIN: ud     = 0
            if abs(fb)     < HOVER_RC_MIN: fb     = 0

            # ── Output (3 lines per tick for readability) ──
            log(f"OBS id={chosen_id}  dist={filt_dist:.2f}m (err {dist_err:+.2f})  "
                f"px={px_size:.0f}  center=({cx:.0f},{cy:.0f})/{fw}x{fh}  "
                f"err=({filt_x:+.2f},{filt_y:+.2f})  skew={filt_skew:+.2f}")
            log(f"     IMU v=({vx_cms:+.0f},{vy_cms:+.0f},{vz_cms:+.0f})cm/s  "
                f"yaw_rate={yaw_rate_dps:+.0f}d/s  "
                f"att=(p{pitch:+.1f},r{roll:+.1f},y{yaw_now:+.1f})  "
                f"alt={hgt_m:.2f}m  bat={bat}%")
            log(f"     WOULD SEND lr={lr:+3d} (P{lr_p:+5.1f} D{lr_d:+5.1f})  "
                f"fb={fb:+3d} (P{fb_p:+5.1f} D{fb_d:+5.1f})  "
                f"ud={ud:+3d} (P{ud_p:+5.1f} D{ud_d:+5.1f})  "
                f"yaw={yaw_rc:+3d} (P{yaw_p:+5.1f} D{yaw_d:+5.1f})  [NOT SENT]")

    except KeyboardInterrupt:
        log("\nCtrl+C — exiting observe mode")

    # Cleanup — note we do NOT call /api/land here since we never took off
    pos_listener.stop()
    vid_tracker.stop()
    _abort.set()

    try:
        api_post("/api/video/stop")
    except Exception:
        pass

    log("Observe mode ended cleanly. Drone state was NOT modified.")
    return True


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
    %(prog)s --api http://flightctrl1:8080 --marker 5
    %(prog)s --api http://flightctrl1:8080 --hover-distance 1.5 --marker 12
    %(prog)s --api http://flightctrl1:8080 --scan-speed 20 --timeout 180

    # Observe-only — never takes off / sends RC. Move the drone by hand
    # and watch what the controller WOULD do. Great for tuning HOVER_*.
    %(prog)s --api http://flightctrl1:8080 --observe
    %(prog)s --api http://flightctrl1:8080 --observe --marker 5

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
        "--marker-size", type=float, default=MARKER_SIZE_M,
        help=(f"Physical marker size in meters (default: {MARKER_SIZE_M}m = "
              f"{MARKER_SIZE_M*100:.0f}cm). Distance calibration auto-scales."),
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
        "--marker", type=int, default=None,
        help="Specific marker ID to approach (default: approach any/closest marker)",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Record video (with ArUco overlay) during the mission for debugging",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without executing (check API connectivity only)",
    )
    parser.add_argument(
        "--observe", action="store_true",
        help=("Read-only mode: NO takeoff, NO land, NO RC commands sent. "
              "Subscribes to camera + IMU + position SSE, runs the HOVER PD "
              "math, and prints what it WOULD send. Move the drone by hand."),
    )
    parser.add_argument(
        "--observe-rate", type=float, default=5.0,
        help="Observe-mode update rate in Hz (default: 5)",
    )

    args = parser.parse_args()

    # Apply marker-size override so pixel_distance_estimate uses the right scale
    if args.marker_size is not None and abs(args.marker_size - MARKER_SIZE_M) > 1e-6:
        globals()["MARKER_SIZE_M"] = float(args.marker_size)
        log(f"  Marker size override: {MARKER_SIZE_M*100:.1f}cm "
            f"(distance scale = (S/{MARKER_SIZE_CALIB_M})^{CALIB_B} "
            f"= {(MARKER_SIZE_M/MARKER_SIZE_CALIB_M)**CALIB_B:.3f})")

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

    if args.observe:
        success = run_observe(
            api_base=args.api,
            target_marker=args.marker,
            hover_distance=args.hover_distance,
            rate_hz=args.observe_rate,
        )
        sys.exit(0 if success else 1)

    success = run_mission(
        api_base=args.api,
        hover_distance=args.hover_distance,
        takeoff_height=args.takeoff_height,
        mission_timeout=args.timeout,
        scan_speed=args.scan_speed,
        approach_speed=args.approach_speed,
        record=args.record,
        target_marker=args.marker,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
