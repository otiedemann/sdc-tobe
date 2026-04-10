"""
Anafi motion-input module for the controller-modular pipeline.

Consumed by main.py via::

    import input as input_module
    motion_sample = input_module.get_motion_input()

Returned dict format (matches fallback_get_motion_input in main.py)::

    {
        "timestamp": float,   # time.monotonic()
        "vx_body": float,     # body-frame forward velocity  (m/s, +X = forward)
        "vy_body": float,     # body-frame lateral velocity  (m/s, +Y = left)
        "vz_body": float,     # body-frame vertical velocity (m/s, +Z = down)
        "yaw_rate": float,    # yaw rate                     (rad/s, CCW positive)
    }

Conventions match the MotionEstimator constructed in main.py
(body_y_positive_is_left=True, body_z_positive_is_down=True,
yaw_positive_is_ccw=True).

Transport — tried in order:
  1. Olympe SDK  — connects directly to the Anafi, polls get_state() at ~25 Hz.
     Lowest latency.  Only available on x86/non-ARM and when parrot-olympe
     is installed.
  2. HTTP polling — reads from the unified Pi API server (/api/telemetry).
     Used when Olympe is unavailable or the direct connection fails.
     Requires the unified_api_server to be reachable.

Initialisation:
  Call ``input.init(anafi_ip)`` from main.py before the main loop.
  The background thread does not start until init() is called.
  Optional env-var fallback: ANAFI_IP / DRONE_IP / ANAFI_API_URL
  (only used when init() is never called — e.g. stand-alone testing).

Coordinate-system notes
-----------------------
Olympe SpeedChanged provides velocities in the NED world frame:
  speedX = North  (m/s)
  speedY = East   (m/s)
  speedZ = Down   (m/s)

Olympe AttitudeChanged provides:
  yaw = heading from North, CW positive (aviation convention), radians
  (positive yaw → rotating clockwise → facing East at yaw = π/2)

The MotionEstimator world frame:
  +X forward at yaw=0, +Y left, +Z up  (right-hand)

Transformation NED → body (with aviation yaw):
  vx_body =  cos(yaw) * speedX + sin(yaw) * speedY   # forward
  vy_body =  sin(yaw) * speedX - cos(yaw) * speedY   # left (= -east)
  vz_body =  speedZ                                    # down (passed as-is;
                                                        #  MotionEstimator
                                                        #  negates internally)
  yaw_rate = -d(yaw)/dt          # CW→CCW sign flip

If your Olympe build uses CCW-positive yaw, flip the sign of OLYMPE_YAW_CW
below and of YAW_RATE_SIGN.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Dict

# ============================================================
# Sign-convention knobs — adjust if readings are backwards
# ============================================================
OLYMPE_YAW_CW: bool = True   # True: Olympe yaw is CW-positive (aviation)
                               # False: Olympe yaw is CCW-positive (math)

# Multipliers applied to each axis; flip to -1 if a channel reads backwards
_SIGN_VX: float = 1.0
_SIGN_VY: float = 1.0
_SIGN_VZ: float = 1.0
_SIGN_YAW_RATE: float = 1.0

# ============================================================
# Configuration — set by init(), env vars used only as fallback
# ============================================================
_ANAFI_IP: str = ""
_API_BASE: str = ""
_initialized: bool = False

_OLYMPE_POLL_HZ: float = 25.0   # Olympe get_state poll rate
_HTTP_POLL_HZ: float = 10.0     # HTTP fallback poll rate

# ============================================================
# Shared state (written by background thread, read by main)
# ============================================================
_lock = threading.Lock()
_state: Dict[str, Any] = {
    "timestamp": 0.0,
    "vx_body": 0.0,
    "vy_body": 0.0,
    "vz_body": 0.0,
    "yaw_rate": 0.0,
    "source": "init",
}

# Previous yaw for finite-difference yaw_rate
_prev_yaw: float | None = None
_prev_yaw_ts: float | None = None


# ============================================================
# Coordinate conversion helpers
# ============================================================

def _ned_to_body(
    speed_x: float,
    speed_y: float,
    speed_z: float,
    yaw_rad: float,
) -> tuple[float, float, float]:
    """
    Convert NED world-frame velocities to body-frame velocities.

    Args:
        speed_x: North velocity (m/s).
        speed_y: East velocity (m/s).
        speed_z: Down velocity (m/s).
        yaw_rad: Heading from North, CW positive (aviation, radians).

    Returns:
        (vx_body, vy_body, vz_body) where:
          +vx = forward, +vy = left, +vz = down.
    """
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    vx = c * speed_x + s * speed_y   # forward
    vy = s * speed_x - c * speed_y   # left (= -east component)
    vz = speed_z                       # down (MotionEstimator negates internally)
    return vx, vy, vz


def _compute_yaw_rate(yaw_now: float, ts_now: float) -> float:
    """Finite-difference yaw_rate with wrap-around handling (rad/s, CCW positive)."""
    global _prev_yaw, _prev_yaw_ts

    yaw_rate = 0.0
    if _prev_yaw is not None and _prev_yaw_ts is not None:
        dt = ts_now - _prev_yaw_ts
        if dt > 1e-4:
            d_yaw = yaw_now - _prev_yaw
            # Wrap to [-π, π]
            while d_yaw > math.pi:
                d_yaw -= 2.0 * math.pi
            while d_yaw < -math.pi:
                d_yaw += 2.0 * math.pi
            # Olympe yaw is CW positive → negate to get CCW positive
            sign = -1.0 if OLYMPE_YAW_CW else 1.0
            yaw_rate = sign * d_yaw / dt

    _prev_yaw = yaw_now
    _prev_yaw_ts = ts_now
    return yaw_rate


def _update_state(
    vx_body: float,
    vy_body: float,
    vz_body: float,
    yaw_rate: float,
    source: str,
) -> None:
    now = time.monotonic()
    with _lock:
        _state["timestamp"] = now
        _state["vx_body"] = _SIGN_VX * vx_body
        _state["vy_body"] = _SIGN_VY * vy_body
        _state["vz_body"] = _SIGN_VZ * vz_body
        _state["yaw_rate"] = _SIGN_YAW_RATE * yaw_rate
        _state["source"] = source


# ============================================================
# Olympe transport
# ============================================================

try:
    import olympe
    from olympe.messages.ardrone3.PilotingState import (
        SpeedChanged,
        AttitudeChanged,
    )
    _HAS_OLYMPE = True
except Exception:
    _HAS_OLYMPE = False


def _olympe_loop() -> None:
    """Background thread: connect to Anafi via Olympe and poll state."""
    global _prev_yaw, _prev_yaw_ts

    interval = 1.0 / _OLYMPE_POLL_HZ
    drone: "olympe.Drone | None" = None

    while True:
        # --- Connect ---
        try:
            print(f"[input] Connecting to Anafi at {_ANAFI_IP} via Olympe…")
            drone = olympe.Drone(_ANAFI_IP)
            drone.connect()
            print("[input] Olympe connection established")
        except Exception as exc:
            print(f"[input] Olympe connect failed: {exc}; retrying in 5 s")
            time.sleep(5.0)
            continue

        # --- Poll loop ---
        consecutive_failures = 0
        while True:
            ts = time.monotonic()
            try:
                spd = drone.get_state(SpeedChanged)
                att = drone.get_state(AttitudeChanged)
            except Exception as exc:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print(f"[input] Olympe get_state failed {consecutive_failures}x: {exc}; reconnecting")
                    break
                time.sleep(interval)
                continue

            consecutive_failures = 0

            if spd and att:
                try:
                    speed_x = float(spd.get("speedX", 0.0))
                    speed_y = float(spd.get("speedY", 0.0))
                    speed_z = float(spd.get("speedZ", 0.0))

                    yaw_raw = float(att.get("yaw", 0.0))
                    # Olympe AttitudeChanged gives yaw in radians.
                    # If OLYMPE_YAW_CW: yaw is CW (aviation) → use directly.
                    # If not OLYMPE_YAW_CW: yaw is CCW (math) → negate for NED conversion.
                    yaw_for_rotation = yaw_raw if OLYMPE_YAW_CW else -yaw_raw

                    vx, vy, vz = _ned_to_body(speed_x, speed_y, speed_z, yaw_for_rotation)
                    yr = _compute_yaw_rate(yaw_raw, ts)

                    _update_state(vx, vy, vz, yr, source="olympe")
                except Exception:
                    pass

            elapsed = time.monotonic() - ts
            sleep_s = max(0.0, interval - elapsed)
            time.sleep(sleep_s)

        # Cleanup before reconnect
        try:
            drone.disconnect()
        except Exception:
            pass
        drone = None
        _prev_yaw = None
        _prev_yaw_ts = None
        time.sleep(2.0)


# ============================================================
# HTTP fallback transport
# ============================================================

def _http_loop() -> None:
    """Background thread: poll the unified API server for telemetry."""
    global _prev_yaw, _prev_yaw_ts

    interval = 1.0 / _HTTP_POLL_HZ
    url = f"{_API_BASE}/api/telemetry"

    try:
        import urllib.request
        import json as _json

        print(f"[input] HTTP fallback polling {url}")
        while True:
            ts = time.monotonic()
            try:
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    raw = _json.loads(resp.read().decode())

                # API returns velocities in cm/s → convert to m/s
                vgx = float(raw.get("vgx", 0.0)) / 100.0  # forward (NED X approx)
                vgy = float(raw.get("vgy", 0.0)) / 100.0  # lateral (NED Y approx)
                vgz = float(raw.get("vgz", 0.0)) / 100.0  # vertical (NED Z approx)
                yaw_deg = float(raw.get("yaw", 0.0))
                yaw_rad = math.radians(yaw_deg)

                # The unified API already names vgx/vgy as NED-world-frame speed.
                # yaw from the API is in degrees (AttitudeChanged converted to deg).
                vx, vy, vz = _ned_to_body(vgx, vgy, vgz, yaw_rad)
                yr = _compute_yaw_rate(yaw_rad, ts)

                _update_state(vx, vy, vz, yr, source="http")
            except Exception:
                pass  # keep running; zeros stay in place

            elapsed = time.monotonic() - ts
            time.sleep(max(0.0, interval - elapsed))

    except Exception as exc:
        print(f"[input] HTTP fallback failed to start: {exc}")


# ============================================================
# Module initialisation
# ============================================================

def _start_background() -> None:
    if _HAS_OLYMPE:
        t = threading.Thread(target=_olympe_loop, name="anafi_input_olympe", daemon=True)
    else:
        t = threading.Thread(target=_http_loop, name="anafi_input_http", daemon=True)
    t.start()


def init(anafi_ip: str | None = None, api_base_url: str | None = None) -> None:
    """
    Configure and start the telemetry background thread.

    Must be called once from main.py before the main loop.
    Subsequent calls are no-ops (thread is only started once).

    Args:
        anafi_ip:    Drone Wi-Fi IP, e.g. "192.168.42.1".
                     Falls back to ANAFI_IP / DRONE_IP env vars, then
                     the Anafi default.
        api_base_url: Base URL for the HTTP fallback transport,
                      e.g. "http://192.168.42.1:8080".
                      Derived from anafi_ip if omitted.
    """
    global _ANAFI_IP, _API_BASE, _initialized
    if _initialized:
        return
    _ANAFI_IP = (
        anafi_ip
        or os.getenv("ANAFI_IP")
        or os.getenv("DRONE_IP")
        or "192.168.42.1"
    )
    _API_BASE = api_base_url or f"http://{_ANAFI_IP}:8080"
    _initialized = True
    print(f"[input] init: ip={_ANAFI_IP}  api={_API_BASE}")
    _start_background()


# ============================================================
# Public API
# ============================================================

def get_motion_input() -> Dict[str, Any]:
    """
    Return the latest Anafi motion sample for the MotionEstimator.

    Fields:
      timestamp  – time.monotonic() of the sample
      vx_body    – forward velocity,  m/s (+X = forward)
      vy_body    – lateral velocity,  m/s (+Y = left)
      vz_body    – vertical velocity, m/s (+Z = down)
      yaw_rate   – yaw rate,          rad/s (CCW positive)
    """
    with _lock:
        return dict(_state)
