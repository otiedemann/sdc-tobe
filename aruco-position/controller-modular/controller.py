"""
Position controller for the Anafi drone.

Computes PCMD-compatible RC commands to fly the drone to a target (x, y, z)
in world frame and hold it there.  During transit, the drone yaws to face the
direction of travel before or while translating.

World-frame conventions (match MotionEstimator / main.py):
  +X  forward at start / yaw = 0
  +Y  left
  +Z  up
  yaw CCW positive (radians)

RC output dict (matches /api/rc endpoint and Olympe PCMD argument order):
  forward_back : int  -100..100   +100 = forward (pitch forward)
  left_right   : int  -100..100   +100 = right   (roll right)
  up_down      : int  -100..100   +100 = up       (gaz up)
  yaw          : int  -100..100   +100 = CW

Usage from main.py::

    import controller as ctrl_module

    # once, after drone is ready:
    ctrl_module.set_target(x=3.0, y=1.5, z=1.2)

    # in every control tick (e.g. the 25 Hz motion loop):
    rc = ctrl_module.update(last_prediction)
    if rc is not None and anafi_drone is not None:
        _send_pcmd(anafi_drone, rc)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# ============================================================
# Tunable gains and limits
# ============================================================

# P gains: position error (m) → velocity setpoint (m/s)
KP_XY: float = 0.75       # horizontal
KP_Z: float  = 0.75       # vertical

# P gain: yaw error (rad) → yaw rate setpoint (rad/s)
KP_YAW: float = 0.1

# Maximum velocity setpoints
MAX_HORIZ_SPEED: float = 0.15    # %  (horizontal)
MAX_VERT_SPEED: float  = 0.15    # %  (vertical)
MAX_YAW_RATE: float    = 0.15    # %

# Position acceptance radii (switch to HOVER phase)
ARRIVE_RADIUS_XY: float = 0.25  # m
ARRIVE_RADIUS_Z: float  = 0.25  # mq

# Yaw alignment during transit
TRANSIT_YAW_MIN_DIST: float    = 0.30  # m: only yaw toward target if farther than this
YAW_ALIGN_THRESHOLD: float     = 0.35  # rad: slow translation if |yaw_error| > this
YAW_ALIGN_SPEED_FACTOR: float  = 0.20  # fraction of max speed while rotating to align

# Uncertainty guard: freeze output if estimated position std > this
MAX_STD_POS: float = 0.40   # m  (set to math.inf to disable)

# Hard output clamp applied in send_pcmd_olympe before values reach the drone
PCMD_MAX: int = 30           # absolute maximum RC value sent (-PCMD_MAX .. +PCMD_MAX)

# Hysteresis: leave HOVER only if error grows beyond this
HOVER_EXIT_RADIUS_XY: float = ARRIVE_RADIUS_XY * 2.0


# ============================================================
# Internal helpers
# ============================================================

def _wrap_pi(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    while angle >  math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _clamp(value: float, lo: float, hi: float) -> int:
    """Clamp to [lo, hi] and return as int."""
    return int(max(lo, min(hi, value)))


def _world_to_body(vx_w: float, vy_w: float, yaw: float) -> tuple[float, float]:
    """
    Rotate world-frame horizontal velocity to body frame.

    World frame: +X forward at yaw=0, +Y left (CCW positive yaw).
    Body frame:  +X forward, +Y left.

    Returns (vx_body, vy_body).
    """
    c = math.cos(yaw)
    s = math.sin(yaw)
    vx_body =  c * vx_w + s * vy_w
    vy_body = -s * vx_w + c * vy_w
    return vx_body, vy_body


# ============================================================
# Phase enum
# ============================================================

class Phase:
    IDLE    = "idle"
    TRANSIT = "transit"
    HOVER   = "hover"


# ============================================================
# Controller class
# ============================================================

@dataclass
class _Target:
    x: float
    y: float
    z: float
    hold_yaw: Optional[float]  # rad, None = lock to arrival yaw


class PositionController:
    """
    P-based position controller.

    Position error  →  velocity setpoint  →  body frame  →  RC percent

    The velocity of the drone (from last_prediction["vx/vy/vz"]) acts as
    implicit derivative feedback, damping overshoot without an explicit D term.
    """

    def __init__(self) -> None:
        self._target: Optional[_Target] = None
        self._phase: str = Phase.IDLE
        self._arrival_yaw: Optional[float] = None   # locked when HOVER entered
        self._last_ts: Optional[float] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_target(
        self,
        x: float,
        y: float,
        z: float,
        hold_yaw: Optional[float] = None,
    ) -> None:
        """
        Set the target position and activate the controller.

        Args:
            x, y, z:   Target in world frame (metres).
            hold_yaw:  Desired heading at the target (radians, CCW positive).
                       None = keep the heading the drone has when it arrives.
        """
        self._target = _Target(x=x, y=y, z=z, hold_yaw=hold_yaw)
        self._phase = Phase.TRANSIT
        self._arrival_yaw = None
        print(f"[controller] Target set → x={x:.2f} y={y:.2f} z={z:.2f}"
              + (f" yaw={math.degrees(hold_yaw):.1f}°" if hold_yaw is not None else ""))

    def clear_target(self) -> None:
        """Deactivate the controller (outputs zero RC)."""
        self._target = None
        self._phase = Phase.IDLE
        self._arrival_yaw = None
        print("[controller] Target cleared — idle")

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def target(self) -> Optional[_Target]:
        return self._target

    # ------------------------------------------------------------------
    # Main update — call every control tick
    # ------------------------------------------------------------------

    def update(self, state: dict) -> Optional[dict]:
        """
        Compute RC commands for the current drone state.

        Args:
            state: ``last_prediction`` dict from main.py.
                   Required keys: x, y, z, yaw, vx, vy, vz, std_pos.

        Returns:
            RC dict ``{forward_back, left_right, up_down, yaw}`` in -100..100,
            or ``None`` when IDLE.
        """
        if self._target is None or self._phase == Phase.IDLE:
            return None

        # ---- extract current state ----
        cx  = float(state.get("x",       0.0))
        cy  = float(state.get("y",       0.0))
        cz  = float(state.get("z",       0.0))
        cyaw = float(state.get("yaw",    0.0))   # rad, CCW positive
        cvx = float(state.get("vx",      0.0))   # world frame m/s
        cvy = float(state.get("vy",      0.0))
        std = float(state.get("std_pos", 0.0))

        # ---- uncertainty guard ----
        if std > MAX_STD_POS:
            # Position estimate too uncertain — hold still
            return {"forward_back": 0, "left_right": 0, "up_down": 0, "yaw": 0}

        tgt = self._target

        # ---- position error ----
        ex = tgt.x - cx
        ey = tgt.y - cy
        ez = tgt.z - cz
        dist_xy = math.sqrt(ex * ex + ey * ey)

        # ---- phase transitions ----
        if self._phase == Phase.TRANSIT:
            if dist_xy <= ARRIVE_RADIUS_XY and abs(ez) <= ARRIVE_RADIUS_Z:
                self._phase = Phase.HOVER
                self._arrival_yaw = cyaw
                print(f"[controller] → HOVER  (err_xy={dist_xy:.2f} m)")

        elif self._phase == Phase.HOVER:
            if dist_xy > HOVER_EXIT_RADIUS_XY:
                self._phase = Phase.TRANSIT
                self._arrival_yaw = None
                print(f"[controller] → TRANSIT  (err_xy={dist_xy:.2f} m)")

        # ---- yaw setpoint ----

        # Use explicit requested yaw, or the heading we arrived with
        # desired_yaw = tgt.hold_yaw if tgt.hold_yaw is not None else cyaw

        if self._phase == Phase.HOVER:
            # Use explicitly requested yaw, or the heading we arrived with
            desired_yaw = (
                tgt.hold_yaw
                if tgt.hold_yaw is not None
                else (self._arrival_yaw if self._arrival_yaw is not None else cyaw)
            )
        else:
            # TRANSIT: face the direction of travel when far enough away
            if dist_xy >= TRANSIT_YAW_MIN_DIST:
                desired_yaw = math.atan2(ey, ex)   # bearing to target in world frame
            else:
                # Close to target — stop yawing, just translate the last bit
                desired_yaw = cyaw

        yaw_error    = _wrap_pi(desired_yaw - cyaw)
        yaw_rate_sp  = KP_YAW * yaw_error
        yaw_rate_sp  = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, yaw_rate_sp))

        # ---- horizontal velocity setpoint (world frame) ----
        vx_sp = KP_XY * ex
        vy_sp = KP_XY * ey
        # Subtract current velocity for implicit damping
        vx_sp -= cvx
        vy_sp -= cvy
        # Scale down while rotating to align
        if self._phase == Phase.TRANSIT and abs(yaw_error) > YAW_ALIGN_THRESHOLD:
            scale = YAW_ALIGN_SPEED_FACTOR
        else:
            scale = 1.0
        vx_sp *= scale
        vy_sp *= scale
        # Clamp magnitude
        horiz_sp = math.sqrt(vx_sp * vx_sp + vy_sp * vy_sp)
        if horiz_sp > MAX_HORIZ_SPEED:
            vx_sp = vx_sp / horiz_sp * MAX_HORIZ_SPEED
            vy_sp = vy_sp / horiz_sp * MAX_HORIZ_SPEED

        # ---- vertical velocity setpoint ----
        vz_sp = KP_Z * ez                               # up positive (world +Z = up)
        vz_sp = max(-MAX_VERT_SPEED, min(MAX_VERT_SPEED, vz_sp))

        # ---- world → body frame ----
        vx_body, vy_body = _world_to_body(vx_sp, vy_sp, cyaw)

        # ---- map to RC percent ----
        #   forward_back: +100 = forward = +vx_body
        #   left_right:   +100 = right   = -vy_body (vy_body is left+)
        #   up_down:      +100 = up      = +vz_sp   (world Z is up)
        #   yaw:          +100 = CW      = -yaw_rate_sp (our yaw_rate is CCW+)
        fb  = _clamp(vx_body * 100.0, -100, 100)
        lr  = _clamp(-vy_body * 100.0, -100, 100)
        ud  = _clamp(vz_sp * 100.0, -100, 100)
        yaw = _clamp(-yaw_rate_sp * 100.0 , -100, 100)

        return {
            "forward_back": fb,
            "left_right":   lr,
            "up_down":      ud,
            "yaw":          yaw,
            # debug extras (not sent to drone)
            "_phase":       self._phase,
            "_err_xy":      round(dist_xy, 3),
            "_err_z":       round(ez, 3),
            "_desired_yaw": round(math.degrees(desired_yaw), 1),
            "_yaw_err_deg": round(math.degrees(yaw_error), 1),
        }

    def status(self) -> dict:
        """Return a compact status dict for logging / UDP payload."""
        t = self._target
        return {
            "ctrl_phase":  self._phase,
            "ctrl_target": {"x": t.x, "y": t.y, "z": t.z} if t else None,
        }


# ============================================================
# Module-level singleton + convenience wrappers
# ============================================================

position_controller = PositionController()


def set_target(x: float, y: float, z: float, hold_yaw: Optional[float] = None) -> None:
    """Activate the controller with the given world-frame target."""
    position_controller.set_target(x, y, z, hold_yaw)


def clear_target() -> None:
    """Deactivate the controller."""
    position_controller.clear_target()


def update(state: dict) -> Optional[dict]:
    """
    Compute RC commands for the current drone state.
    Returns None when idle, otherwise a dict with RC values.
    """
    return position_controller.update(state)


def is_active() -> bool:
    return position_controller.phase != Phase.IDLE


# ============================================================
# Olympe PCMD helper (optional — call from main.py)
# ============================================================

# None = not yet tested, True = native API works, False = use raw PCMD fallback
_has_piloting_api: Optional[bool] = None
_pcmd_seq: int = 0


def _detect_piloting_api(drone) -> bool:
    """
    Test whether the native Olympe piloting API (start_piloting /
    piloting_pcmd / stop_piloting) actually works on this drone object.

    Result is cached after the first successful or failed probe.
    """
    global _has_piloting_api
    if _has_piloting_api is not None:
        return _has_piloting_api
    if not callable(getattr(drone, "start_piloting", None)):
        _has_piloting_api = False
        print("[controller] Native piloting API not found — using raw d(PCMD(...))")
        return False
    try:
        drone.start_piloting()
        _has_piloting_api = True
        print("[controller] Native piloting API active (start_piloting / piloting_pcmd)")
        return True
    except Exception as exc:
        _has_piloting_api = False
        print(f"[controller] Native piloting API failed ({exc}) — using raw d(PCMD(...))")
        return False


def send_pcmd_olympe(drone, rc: dict) -> None:
    """
    Send an RC dict to the drone via Olympe.

    Prefers the native ``drone.piloting_pcmd()`` API (Olympe ≥ 7).
    Falls back to a raw ``drone(PCMD(...))`` message for older builds.

    Call this from main.py after calling update() when using
    an olympe.Drone object directly (no unified API server needed).

    Args:
        drone: olympe.Drone instance (already connected).
        rc:    Dict returned by update(), must contain the four RC keys.
    """
    global _pcmd_seq

    # Hard clamp: never send values beyond ±PCMD_MAX to the drone
    def _hard_clamp(v: int) -> int:
        return max(-PCMD_MAX, min(PCMD_MAX, v))

    roll  = _hard_clamp(rc["left_right"])
    pitch = _hard_clamp(rc["forward_back"])
    yaw   = _hard_clamp(rc["yaw"])
    gaz   = _hard_clamp(rc["up_down"])

    print(f"[PCMD] roll={roll:+4d}  pitch={pitch:+4d}  yaw={yaw:+4d}  gaz={gaz:+4d}", flush=True)

    if _detect_piloting_api(drone):
        try:
            drone.piloting_pcmd(roll, pitch, yaw, gaz, 0)
            return
        except Exception as exc:
            print(f"[controller] piloting_pcmd failed: {exc}")

    # Fallback: raw PCMD message
    try:
        from olympe.messages.ardrone3.Piloting import PCMD
        _pcmd_seq = (_pcmd_seq + 1) & 0x7FFFFFFF
        drone(PCMD(1, roll, pitch, yaw, gaz, _pcmd_seq))
    except Exception as exc:
        print(f"[controller] PCMD send failed: {exc}")
