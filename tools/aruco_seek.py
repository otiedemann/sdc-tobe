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
HOVER_DISTANCE_M = 1.0        # stop this far in front of the marker
SCAN_YAW_SPEED = 15           # RC yaw percent during search rotation (0-100)
APPROACH_MAX_SPEED = 20       # RC percent cap during approach
APPROACH_MIN_SPEED = 5        # RC percent floor during approach
RC_TICK_MS = 200              # RC command duration per tick
CONTROL_HZ = 5                # main loop frequency
HEARTBEAT_INTERVAL = 0.4      # seconds between heartbeats
POSITION_TIMEOUT_S = 5.0      # land if no position for this long
MISSION_TIMEOUT_S = 120.0     # total mission timeout
SETTLE_TIME_S = 3.0           # time to wait after takeoff before scanning
ARRIVAL_RADIUS_M = 0.25       # close enough to target
ALTITUDE_TOLERANCE_M = 0.3    # acceptable altitude error

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
):
    global _api_base
    _api_base = api_base

    log(f"ArUco Seek & Approach")
    log(f"  API server:     {api_base}")
    log(f"  Hover distance: {hover_distance} m")
    log(f"  Takeoff height: {takeoff_height} m")
    log(f"  Timeout:        {mission_timeout} s")
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

    # Load arena config to know marker positions
    log("Pre-flight: loading arena configuration...")
    arena_cfg = get_arena_config()
    marker_positions = get_marker_positions(arena_cfg)
    marker_size = arena_cfg.get("marker_size_m", "?")
    log(f"  {len(marker_positions)} markers configured, marker_size={marker_size}m")
    if not marker_positions:
        log("WARNING: No markers in arena config. Will use position SSE seen_markers only.")

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

    # Enable position tracking
    log("Pre-flight: enabling position tracking...")
    api_post("/api/position/config", {"enabled": True})

    # Start position listener
    pos_listener = PositionListener(api_base)
    pos_listener.start()
    log("  Position SSE listener started")

    # Start heartbeat
    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True, name="heartbeat")
    hb_thread.start()
    log("  Heartbeat thread started")

    # Wait briefly for first position data
    time.sleep(1.0)

    # ── State machine ────────────────────────────────────────────────────

    phase = Phase.TAKEOFF if not hb.get("flying") else Phase.SETTLE
    mission_start = time.time()
    phase_start = time.time()
    scan_start_yaw = None
    scan_total_rotation = 0.0
    scan_last_yaw = None
    target_marker_id: Optional[int] = None
    target_pos: Optional[tuple] = None  # (x, y, z, yaw)
    approach_start_time = 0.0
    tick_interval = 1.0 / CONTROL_HZ

    log(f"\n{'='*60}")
    log(f"Mission start — phase: {phase}")
    log(f"{'='*60}\n")

    try:
        while phase not in (Phase.DONE, Phase.ABORT) and not _abort.is_set():
            tick_start = time.time()

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
                # Check if position data sees any markers
                seen_markers = []
                if pos_data:
                    seen_markers = pos_data.get("seen_markers", [])

                if seen_markers:
                    # Found marker(s)! Pick the first one.
                    target_marker_id = seen_markers[0]
                    send_rc_stop()
                    log(f"MARKER FOUND! ID={target_marker_id}")

                    if target_marker_id in marker_positions:
                        mk = marker_positions[target_marker_id]
                        tx, ty, tz, tyaw = compute_approach_target(mk, hover_distance)
                        target_pos = (tx, ty, tz, tyaw)
                        log(f"  Marker at ({mk['x']:.1f}, {mk['y']:.1f}, {mk['z']:.1f}) wall={mk['wall']}")
                        log(f"  Approach target: ({tx:.2f}, {ty:.2f}, {tz:.2f}) yaw={math.degrees(tyaw):.0f} deg")
                        phase = Phase.APPROACH
                        approach_start_time = time.time()
                        phase_start = time.time()
                    else:
                        # Marker not in arena config — use position data directly
                        # Just hover where we are since we can't compute approach vector
                        log(f"  Marker ID {target_marker_id} not in arena config")
                        log(f"  Using current position as approach reference")
                        if pos:
                            # Hover at current XY, current Z
                            target_pos = (pos[0], pos[1], pos[2], 0.0)
                            phase = Phase.APPROACH
                            approach_start_time = time.time()
                            phase_start = time.time()
                        else:
                            log("  No position data — continuing scan")
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

            # ── APPROACH — fly toward target position ────────────────
            elif phase == Phase.APPROACH:
                if pos is None or pos_age > POSITION_TIMEOUT_S:
                    # Position lost
                    send_rc_stop()
                    if pos_age > POSITION_TIMEOUT_S:
                        log(f"POSITION LOST for {pos_age:.1f}s — landing for safety")
                        phase = Phase.LAND
                    else:
                        log("Waiting for position fix...")
                    continue

                tx, ty, tz, tyaw = target_pos
                cx, cy, cz = pos[0], pos[1], pos[2]

                # Position error
                ex = tx - cx
                ey = ty - cy
                ez = tz - cz
                dist_xy = math.sqrt(ex * ex + ey * ey)

                # Get current yaw from telemetry
                tel = get_telemetry()
                cur_yaw_deg = tel.get("yaw") or 0.0
                cur_yaw = math.radians(cur_yaw_deg)

                # Check if arrived
                if dist_xy < ARRIVAL_RADIUS_M and abs(ez) < ALTITUDE_TOLERANCE_M:
                    send_rc_stop()
                    log(f"ARRIVED at target! dist={dist_xy:.2f}m  alt_err={ez:.2f}m")
                    phase = Phase.HOVER
                    phase_start = time.time()
                    continue

                # Approach timeout (don't get stuck)
                if time.time() - approach_start_time > 60.0:
                    send_rc_stop()
                    log("APPROACH TIMEOUT after 60s — landing")
                    phase = Phase.LAND
                    continue

                # Compute RC commands (simplified P controller)
                # World frame velocity setpoint
                kp_xy = 0.6
                kp_z = 0.5
                kp_yaw = 1.2

                vx_sp = kp_xy * ex
                vy_sp = kp_xy * ey
                vz_sp = kp_z * ez

                # Clamp horizontal speed
                max_speed = 0.5  # m/s
                horiz = math.sqrt(vx_sp**2 + vy_sp**2)
                if horiz > max_speed:
                    vx_sp = vx_sp / horiz * max_speed
                    vy_sp = vy_sp / horiz * max_speed

                # Yaw toward target
                yaw_err = wrap_pi(tyaw - cur_yaw)
                yaw_rate = kp_yaw * yaw_err
                yaw_rate = max(-0.6, min(0.6, yaw_rate))

                # Slow down horizontal if yaw misaligned
                if abs(yaw_err) > 0.4:
                    vx_sp *= 0.2
                    vy_sp *= 0.2

                # World to body frame
                vx_body, vy_body = world_to_body(vx_sp, vy_sp, cur_yaw)

                # Map to RC percent
                fb = _clamp(int(vx_body / max_speed * APPROACH_MAX_SPEED))
                lr = _clamp(int(-vy_body / max_speed * APPROACH_MAX_SPEED))
                ud = _clamp(int(vz_sp / 0.5 * 30), -30, 30)
                yaw_rc = _clamp(int(-yaw_rate / 0.6 * 25), -30, 30)

                send_rc(lr, fb, ud, yaw_rc)

                log(f"Approach: dist={dist_xy:.2f}m  alt_err={ez:.2f}m  "
                    f"yaw_err={math.degrees(yaw_err):.0f}deg  "
                    f"RC fb={fb} lr={lr} ud={ud} yaw={yaw_rc}")

            # ── HOVER — hold position in front of marker ─────────────
            elif phase == Phase.HOVER:
                if pos is None or pos_age > POSITION_TIMEOUT_S:
                    send_rc_stop()
                    if pos_age > POSITION_TIMEOUT_S:
                        log(f"POSITION LOST during hover — landing")
                        phase = Phase.LAND
                    continue

                tx, ty, tz, tyaw = target_pos
                cx, cy, cz = pos[0], pos[1], pos[2]
                ex, ey, ez = tx - cx, ty - cy, tz - cz
                dist_xy = math.sqrt(ex**2 + ey**2)

                tel = get_telemetry()
                cur_yaw = math.radians(tel.get("yaw") or 0.0)

                # Light position hold
                kp = 0.4
                vx_sp = kp * ex
                vy_sp = kp * ey
                vz_sp = 0.3 * ez
                yaw_err = wrap_pi(tyaw - cur_yaw)
                yaw_rate = 0.8 * yaw_err

                vx_body, vy_body = world_to_body(vx_sp, vy_sp, cur_yaw)

                fb = _clamp(int(vx_body / 0.5 * 15), -20, 20)
                lr = _clamp(int(-vy_body / 0.5 * 15), -20, 20)
                ud = _clamp(int(vz_sp / 0.3 * 15), -20, 20)
                yaw_rc = _clamp(int(-yaw_rate / 0.6 * 15), -20, 20)

                send_rc(lr, fb, ud, yaw_rc)

                hover_dur = time.time() - phase_start
                log(f"HOVER: ({cx:.2f},{cy:.2f},{cz:.2f}) err_xy={dist_xy:.2f}m  "
                    f"facing marker {target_marker_id}  t={hover_dur:.0f}s  "
                    f"[Ctrl+C to land]")

            # ── LAND ─────────────────────────────────────────────────
            elif phase == Phase.LAND:
                send_rc_stop()
                time.sleep(0.2)
                log("Landing...")
                result = api_post("/api/land")
                log(f"  Land result: {result}")
                time.sleep(3.0)
                phase = Phase.DONE

            # ── Tick timing ──────────────────────────────────────────
            elapsed = time.time() - tick_start
            sleep_time = max(0.0, tick_interval - elapsed)
            if sleep_time > 0:
                _abort.wait(sleep_time)

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
    _abort.set()

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
        "--timeout", type=float, default=MISSION_TIMEOUT_S,
        help=f"Mission timeout in seconds (default: {MISSION_TIMEOUT_S})",
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
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
