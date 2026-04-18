from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict

import olympe
from olympe.messages.ardrone3.PilotingState import (
    SpeedChanged,
    AttitudeChanged,
)

"""
Anafi motion-input module for the controller-modular pipeline.

Consumed by main.py via:

    import input as input_module
    motion_sample = input_module.get_motion_input()

Returned dict format:

    {
        "timestamp": float,   # time.monotonic()
        "vx_body": float,     # body-frame forward velocity (m/s)
        "vy_body": float,     # body-frame right velocity (m/s)
        "vz_body": float,     # body-frame down velocity (m/s)
        "yaw":     float,     # yaw angle (rad)
        "yaw_rate": float,    # yaw rate (rad/s)
    }

This module subscribes to olympe event notifications directly on the drone
and requires no intermediate HTTP API server.

Olympe messages used:
    ardrone3.PilotingState.SpeedChanged   -> vx_body / vy_body / vz_body
    ardrone3.PilotingState.AttitudeChanged -> yaw (rad), yaw_rate (finite-diff)
"""

# ============================================================
# Sign / convention knobs
# ============================================================
_SIGN_VX: float = 1.0  # SpeedChanged.speedX  (forward, m/s)
_SIGN_VY: float = 1.0  # SpeedChanged.speedY  (right,   m/s)
_SIGN_VZ: float = 1.0  # SpeedChanged.speedZ  (down,    m/s)
_SIGN_YAW: float = 1.0
_SIGN_YAW_RATE: float = 1.0

# Attitude state for yaw-rate finite-diff (written only inside the attitude callback)
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


class MotionListener(olympe.EventListener):

    def __init__(self, drone):
        super().__init__(drone)

        print("[input] MotionListener initialized")

        # Internal state
        self.lock = threading.Lock()
        self.state: Dict[str, Any] = {
            "timestamp": 0.0,
            "vx_body": 0.0,
            "vy_body": 0.0,
            "vz_body": 0.0,
            "yaw": 0.0,
            "yaw_rate": 0.0
        }

        # Attitude state for yaw-rate finite-diff (written only inside the attitude callback)
        self.prev_yaw: float | None = None
        self.prev_yaw_ts: float | None = None
        self.att_lock = threading.Lock()

    @olympe.listen_event(AttitudeChanged(_policy="wait"))
    def on_attitude(self, event, scheduler):
        """
        Fired by olympe whenever AttitudeChanged is received from the drone.
        Provides roll/pitch/yaw in radians; we use yaw and compute yaw_rate.
        """
        global _prev_yaw, _prev_yaw_ts

        print(f"[input] AttitudeChanged: {event.args}")

        yaw = _wrap_angle_pi(_SIGN_YAW * float(event.args.get("yaw", 0.0)))
        now = time.monotonic()

        yaw_rate = 0.0
        with self.att_lock:
            if _prev_yaw is not None and _prev_yaw_ts is not None:
                dt = now - _prev_yaw_ts
                if dt > 1e-4:
                    d_yaw = _wrap_angle_pi(yaw - _prev_yaw)
                    yaw_rate = _SIGN_YAW_RATE * (d_yaw / dt)
            _prev_yaw = yaw
            _prev_yaw_ts = now

        with self.lock:
            self.state["yaw"] = yaw
            self.state["yaw_rate"] = yaw_rate
            self.state["timestamp"] = now

    @olympe.listen_event(SpeedChanged(_policy="wait"))
    def on_speed(self, event, scheduler):
        """
        Fired by olympe whenever SpeedChanged is received from the drone.
        speedX = forward (body), speedY = right (body), speedZ = down (body), all m/s.
        """
        print(f"[input] SpeedChanged: {event.args}")

        args = event.args
        vx = _SIGN_VX * float(args.get("speedX", 0.0))
        vy = _SIGN_VY * float(args.get("speedY", 0.0))
        vz = _SIGN_VZ * float(args.get("speedZ", 0.0))

        now = time.monotonic()
        with self.lock:
            self.state["vx_body"] = vx
            self.state["vy_body"] = vy
            self.state["vz_body"] = vz
            self.state["timestamp"] = now

    def get_motion_input(self) -> Dict[str, Any]:
        """
        Return the latest Anafi motion sample in body coordinates.

        Fields:
          timestamp  – time.monotonic() of the sample
          vx_body    – body-frame forward velocity (m/s)
          vy_body    – body-frame right velocity (m/s)
          vz_body    – body-frame down velocity (m/s)
          yaw        – yaw angle in rad
          yaw_rate   – yaw rate in rad/s
          source     – "olympe"
        """
        with self.lock:
            return dict(self.state)
