from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Dict

"""
Anafi motion-input module for the controller-modular pipeline.

Consumed by main.py via:

    import input as input_module
    motion_sample = input_module.get_motion_input()

Returned dict format:

    {
        "timestamp": float,   # time.monotonic()
        "vx_world": float,    # world-frame X velocity (m/s)
        "vy_world": float,    # world-frame Y velocity (m/s)
        "vz_world": float,    # world-frame Z velocity (m/s)
        "yaw": float,         # yaw angle (rad)
        "yaw_rate": float,    # yaw rate (rad/s)
        "source": str,        # "http"
    }

This module only uses HTTP polling against the unified Pi API server.

Expected API fields from /api/telemetry:
    vgx, vgy, vgz, yaw

Assumptions:
- vgx/vgy/vgz are already world-frame velocities
- yaw is provided in degrees by the API and converted to radians here

Adjust the unit conversion below if your API already returns m/s.
"""

# ============================================================
# Sign / convention knobs
# ============================================================
_SIGN_VX: float = 1.0
_SIGN_VY: float = 1.0
_SIGN_VZ: float = 1.0
_SIGN_YAW: float = 1.0
_SIGN_YAW_RATE: float = 1.0

# If the HTTP API returns velocities in cm/s, keep this at 0.01.
# If it already returns m/s, change to 1.0.
_VELOCITY_SCALE: float = 0.01

# If the HTTP API returns yaw in degrees, keep this True.
# If it already returns radians, change to False.
_YAW_IS_DEGREES: bool = True

# ============================================================
# Configuration
# ============================================================
_ANAFI_IP: str = ""
_API_BASE: str = ""
_initialized: bool = False

_HTTP_POLL_HZ: float = 10.0

# ============================================================
# Shared state
# ============================================================
_lock = threading.Lock()
_state: Dict[str, Any] = {
    "timestamp": 0.0,
    "vx_world": 0.0,
    "vy_world": 0.0,
    "vz_world": 0.0,
    "yaw": 0.0,
    "yaw_rate": 0.0,
    "source": "init",
}

_prev_yaw: float | None = None
_prev_yaw_ts: float | None = None


# ============================================================
# Helpers
# ============================================================

def _wrap_angle_pi(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def _compute_yaw_rate(yaw_now: float, ts_now: float) -> float:
    """Finite-difference yaw rate with wrap-around handling."""
    global _prev_yaw, _prev_yaw_ts

    yaw_rate = 0.0
    if _prev_yaw is not None and _prev_yaw_ts is not None:
        dt = ts_now - _prev_yaw_ts
        if dt > 1e-4:
            d_yaw = _wrap_angle_pi(yaw_now - _prev_yaw)
            yaw_rate = d_yaw / dt

    _prev_yaw = yaw_now
    _prev_yaw_ts = ts_now
    return yaw_rate


def _update_state(
    vx_world: float,
    vy_world: float,
    vz_world: float,
    yaw: float,
    yaw_rate: float,
    source: str,
) -> None:
    now = time.monotonic()
    with _lock:
        _state["timestamp"] = now
        _state["vx_world"] = _SIGN_VX * vx_world
        _state["vy_world"] = _SIGN_VY * vy_world
        _state["vz_world"] = _SIGN_VZ * vz_world
        _state["yaw"] = _SIGN_YAW * yaw
        _state["yaw_rate"] = _SIGN_YAW_RATE * yaw_rate
        _state["source"] = source


# ============================================================
# HTTP transport
# ============================================================

def _http_loop() -> None:
    """Background thread: poll the unified API server for telemetry."""
    global _prev_yaw, _prev_yaw_ts

    interval = 1.0 / _HTTP_POLL_HZ
    url = f"{_API_BASE}/api/telemetry"

    try:
        import json as _json
        import urllib.request

        print(f"[input] HTTP polling {url}")

        while True:
            loop_start = time.monotonic()

            try:
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    raw = _json.loads(resp.read().decode())

                vx_world = float(raw.get("vgx") or 0.0) * _VELOCITY_SCALE
                vy_world = float(raw.get("vgy") or 0.0) * _VELOCITY_SCALE
                vz_world = float(raw.get("vgz") or 0.0) * _VELOCITY_SCALE

                yaw_raw = float(raw.get("yaw") or 0.0)
                yaw = math.radians(yaw_raw) if _YAW_IS_DEGREES else yaw_raw
                yaw = _wrap_angle_pi(yaw)

                ts = time.monotonic()
                yaw_rate = _compute_yaw_rate(yaw, ts)

                _update_state(
                    vx_world=vx_world,
                    vy_world=vy_world,
                    vz_world=vz_world,
                    yaw=yaw,
                    yaw_rate=yaw_rate,
                    source="http",
                )

            except Exception:
                # Keep last valid state if request/parsing fails
                pass

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, interval - elapsed))

    except Exception as exc:
        print(f"[input] HTTP polling failed to start: {exc}")


# ============================================================
# Module initialisation
# ============================================================

def _start_background() -> None:
    t = threading.Thread(target=_http_loop, name="anafi_input_http", daemon=True)
    t.start()


def init(anafi_ip: str | None = None, api_base_url: str | None = None) -> None:
    """
    Configure and start the telemetry background thread.

    Must be called once from main.py before the main loop.
    Subsequent calls are no-ops.

    Args:
        anafi_ip: Drone / Pi IP, e.g. "192.168.42.1"
        api_base_url: Base URL for the HTTP transport,
                      e.g. "http://192.168.42.1:8080"
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
    _API_BASE = api_base_url or os.getenv("ANAFI_API_URL") or f"http://{_ANAFI_IP}:8080"

    _initialized = True
    print(f"[input] init: ip={_ANAFI_IP} api={_API_BASE}")
    _start_background()


# ============================================================
# Public API
# ============================================================

def get_motion_input() -> Dict[str, Any]:
    """
    Return the latest Anafi motion sample in world coordinates.

    Fields:
      timestamp  – time.monotonic() of the sample
      vx_world   – world-frame X velocity
      vy_world   – world-frame Y velocity
      vz_world   – world-frame Z velocity
      yaw        – yaw angle in rad
      yaw_rate   – yaw rate in rad/s
      source     – source identifier
    """
    with _lock:
        return dict(_state)
