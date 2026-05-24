"""Closed-loop RC navigation: turn a high-level RoleAction into stick RC.

Roles emit a :class:`~c2.models.RoleAction` (GOTO / HOVER / YAW_SCAN /
ORBIT / LAND / TAKEOFF / NONE). :func:`action_to_rc` converts that intent
into the four RC stick values ``(lr, fb, ud, yaw)`` the FC understands,
using a simple proportional controller on the drone's arena-frame error.
:func:`at_waypoint` is the matching arrival test.

============================================================================
RC SIGN CONVENTION  (verified against the proven closed-loop code in this
repo -- do NOT flip these without re-checking those references)
============================================================================
Reference 1: ``marker_mission/controller.py`` :meth:`_step_goto`
  (lines ~2644-2700). It computes, with ``yaw`` = drone heading CW from
  arena +y (the front wall):

      forward unit (arena) = (sin h, cos h)
      right   unit (arena) = (cos h, -sin h)      # forward rotated 90 deg CW
      err_fwd   = ex*sin(h) + ey*cos(h)
      err_right = ex*cos(h) - ey*sin(h)
      _send_rc(int(u_lat), int(u_fwd), int(u_ud), int(u_yaw))
        with u_fwd  +ve  =>  fb  +ve (forward),
             u_lat  +ve  =>  lr  +ve (right),
             u_ud   +ve  =>  ud  +ve (up).

Reference 2: ``c2_strategy/strategy.py`` :meth:`_rc_navigate`
  (lines ~619-660). Independently arrives at the SAME frame:

      target_angle = atan2(dx, dy)         # CW from +y, matches models
      rel = target_angle - heading
      fb = speed*cos(rel)   # forward
      lr = speed*sin(rel)   # right
  and yaw-scan uses a POSITIVE yaw value for the scan (RC_YAW_SCAN = +35),
  i.e. +yaw = clockwise, consistent with the FC contract.

Both references agree, so this module uses:

      h  = drone.heading_deg  (degrees CW from +y / front wall)
      e  = target - position  (arena metres)
      fwd_err   = e.x*sin(h) + e.y*cos(h)
      right_err = e.x*cos(h) - e.y*sin(h)
      fb  =  kp_xy * fwd_err          (+forward)
      lr  =  kp_xy * right_err        (+right)
      ud  =  kp_z  * (target.z - z)   (+up)
      yaw =  +ve  => rotate clockwise (CW)

All four axes are clamped to [-rc_max, +rc_max]. Gains and the clamp come
from ``cfg.strategy`` (rc_kp_xy, rc_kp_z, rc_kp_yaw, rc_max). If the drone
has no position fix we cannot close the loop, so we return all zeros.
============================================================================
"""

from __future__ import annotations

import math

try:
    # Normal use: imported as ``c2.navigation`` (package context present).
    from .config import C2Config
    from .models import (
        ActionKind,
        DroneState,
        RoleAction,
        Vec2,
        Vec3,
    )
except ImportError:  # pragma: no cover - direct ``python c2/navigation.py`` run
    # Fall back to absolute imports so the __main__ smoke block below also
    # works when this file is executed directly (no parent package). Put
    # the repo root on sys.path so ``import c2.*`` resolves.
    import os
    import sys

    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from c2.config import C2Config
    from c2.models import (  # type: ignore[no-redef]
        ActionKind,
        DroneState,
        RoleAction,
        Vec2,
        Vec3,
    )

# Per-call advance of the orbit phase angle (radians). Small + fixed so the
# orbit is smooth and robust regardless of tick rate; the drone chases a
# point that walks gently around the circle ahead of it.
_ORBIT_STEP_RAD = 0.20


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: int, hi: int) -> int:
    """Round to int and clamp to [lo, hi]."""
    iv = int(round(v))
    if iv < lo:
        return lo
    if iv > hi:
        return hi
    return iv


def _body_errors(e_x: float, e_y: float, heading_deg: float) -> tuple[float, float]:
    """Project an arena-frame error (e_x, e_y) onto the drone body axes.

    Returns (fwd_err, right_err) for a heading measured CW from +y.
    See the module convention block for the derivation.
    """
    h = math.radians(heading_deg)
    s, c = math.sin(h), math.cos(h)
    fwd_err = e_x * s + e_y * c
    right_err = e_x * c - e_y * s
    return fwd_err, right_err


def _ud_to_hold(target_z: float | None, pos_z: float, cfg: C2Config) -> int:
    """Up/down stick to drive current altitude toward ``target_z``.

    Returns 0 when no target altitude is given (let the FC's onboard
    stabiliser hold whatever altitude the drone is at).
    """
    if target_z is None:
        return 0
    rc_max = cfg.strategy.rc_max
    return _clamp(cfg.strategy.rc_kp_z * (target_z - pos_z), -rc_max, rc_max)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def action_to_rc(
    action: RoleAction,
    drone: DroneState,
    cfg: C2Config,
    memory: dict,
) -> tuple[int, int, int, int]:
    """Convert a :class:`RoleAction` into ``(lr, fb, ud, yaw)`` RC sticks.

    Each axis is clamped to ``[-cfg.strategy.rc_max, +cfg.strategy.rc_max]``.
    ``memory`` is the role's per-drone scratch dict (used here to persist
    the ORBIT phase angle across ticks). If the drone has no position fix
    we can't close the loop, so we return ``(0, 0, 0, 0)``.
    """
    rc_max = cfg.strategy.rc_max
    kp_xy = cfg.strategy.rc_kp_xy
    kp_z = cfg.strategy.rc_kp_z
    kp_yaw = cfg.strategy.rc_kp_yaw

    kind = action.kind

    # Actions that need no closed loop -----------------------------------
    if kind in (ActionKind.NONE, ActionKind.LAND, ActionKind.TAKEOFF):
        return (0, 0, 0, 0)

    # Without a position fix we cannot close any loop. Safest is to hold
    # the sticks at neutral and let the FC stabiliser keep station.
    pos = drone.position
    if pos is None:
        return (0, 0, 0, 0)

    heading = drone.heading_deg if drone.heading_deg is not None else 0.0

    # HOVER: hold altitude if a target.z was given, else full neutral ----
    if kind == ActionKind.HOVER:
        tgt_z = action.target.z if action.target is not None else None
        ud = _ud_to_hold(tgt_z, pos.z, cfg)
        return (0, 0, ud, 0)

    # YAW_SCAN: spin in place, holding altitude --------------------------
    if kind == ActionKind.YAW_SCAN:
        tgt_z = action.target.z if action.target is not None else pos.z
        ud = _ud_to_hold(tgt_z, pos.z, cfg)
        # Map yaw_rate (deg/s, CW +ve) to an rc yaw value. +yaw == CW per
        # the FC contract, so the sign of yaw_rate carries straight
        # through. A 30 deg/s scan with the default rc_kp_yaw (1.5) gives
        # 45 -> clamped to rc_max (50) -> a clearly non-zero scan stick.
        yaw = _clamp(action.yaw_rate * kp_yaw, -rc_max, rc_max)
        # Guard against a tiny gain rounding a real scan rate to 0: if a
        # non-zero rate was requested, keep at least a minimal stick in
        # the right direction so the drone actually rotates.
        if yaw == 0 and action.yaw_rate != 0.0:
            yaw = 1 if action.yaw_rate > 0 else -1
        return (0, 0, ud, yaw)

    # GOTO: closed-loop fly to action.target -----------------------------
    if kind == ActionKind.GOTO:
        if action.target is None:
            return (0, 0, 0, 0)
        e_x = action.target.x - pos.x
        e_y = action.target.y - pos.y
        fwd_err, right_err = _body_errors(e_x, e_y, heading)
        fb = _clamp(kp_xy * fwd_err, -rc_max, rc_max)
        lr = _clamp(kp_xy * right_err, -rc_max, rc_max)
        ud = _clamp(kp_z * (action.target.z - pos.z), -rc_max, rc_max)
        yaw = 0
        return (lr, fb, ud, yaw)

    # ORBIT: circle orbit_center at orbit_radius_m -----------------------
    if kind == ActionKind.ORBIT:
        return _orbit_rc(action, drone, cfg, memory)

    # Unknown kind: be safe.
    return (0, 0, 0, 0)


def _orbit_rc(
    action: RoleAction,
    drone: DroneState,
    cfg: C2Config,
    memory: dict,
) -> tuple[int, int, int, int]:
    """Circle ``action.orbit_center`` at ``action.orbit_radius_m``.

    Keeps an orbit phase angle in ``memory["orbit_angle"]``, advances it a
    small step each call, computes the next point on the circle and steers
    toward it exactly like GOTO. Adds a gentle yaw so the drone keeps
    facing roughly outward (away from the centre), which is what a scout
    wants while sweeping the arena. Robust to a missing centre/radius:
    falls back to a plain altitude hold.
    """
    rc_max = cfg.strategy.rc_max
    kp_xy = cfg.strategy.rc_kp_xy
    kp_z = cfg.strategy.rc_kp_z
    kp_yaw = cfg.strategy.rc_kp_yaw

    pos = drone.position
    assert pos is not None  # guaranteed by caller
    heading = drone.heading_deg if drone.heading_deg is not None else 0.0

    center = action.orbit_center
    radius = float(action.orbit_radius_m)
    tgt_z = action.target.z if action.target is not None else pos.z

    if center is None or radius <= 0.0:
        # Nothing sensible to orbit -- just hold altitude.
        return (0, 0, _ud_to_hold(tgt_z, pos.z, cfg), 0)

    # Seed / advance the orbit phase. Seed from the drone's CURRENT
    # bearing relative to the centre so we start by closing onto the
    # circle near where we already are (no big initial lunge).
    angle = memory.get("orbit_angle")
    if angle is None:
        angle = math.atan2(pos.y - center.y, pos.x - center.x)
    angle += _ORBIT_STEP_RAD
    # Keep it bounded so it never grows without limit over a long run.
    if angle > math.pi:
        angle -= 2.0 * math.pi
    elif angle < -math.pi:
        angle += 2.0 * math.pi
    memory["orbit_angle"] = angle

    # Next waypoint on the circle.
    wx = center.x + radius * math.cos(angle)
    wy = center.y + radius * math.sin(angle)

    e_x = wx - pos.x
    e_y = wy - pos.y
    fwd_err, right_err = _body_errors(e_x, e_y, heading)
    fb = _clamp(kp_xy * fwd_err, -rc_max, rc_max)
    lr = _clamp(kp_xy * right_err, -rc_max, rc_max)
    ud = _clamp(kp_z * (tgt_z - pos.z), -rc_max, rc_max)

    # Gentle yaw to face outward (radially away from the centre). Desired
    # heading is the bearing from centre to the drone, expressed CW from
    # +y to match the heading convention: heading = atan2(dx, dy).
    out_dx = pos.x - center.x
    out_dy = pos.y - center.y
    desired_heading = math.degrees(math.atan2(out_dx, out_dy))
    yaw_err = ((desired_heading - heading + 540.0) % 360.0) - 180.0
    # Use a gentle fraction of the yaw gain so the orbit stays smooth and
    # the facing correction doesn't fight the lateral track.
    yaw = _clamp(0.5 * kp_yaw * yaw_err, -rc_max, rc_max)

    return (lr, fb, ud, yaw)


def at_waypoint(drone: DroneState, target: Vec3, cfg: C2Config) -> bool:
    """True when the drone has arrived at ``target``.

    Arrival means BOTH:
      * xy distance  < cfg.strategy.arrival_tol_m
      * |z error|    < cfg.strategy.altitude_tol_m

    With no position fix we cannot claim arrival -> False.
    """
    pos = drone.position
    if pos is None:
        return False
    xy_dist = math.hypot(target.x - pos.x, target.y - pos.y)
    z_err = abs(target.z - pos.z)
    return (
        xy_dist < cfg.strategy.arrival_tol_m
        and z_err < cfg.strategy.altitude_tol_m
    )


# ---------------------------------------------------------------------------
# tiny smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    # Build a fake drone + a couple of actions and print the resulting RC.
    cfg = C2Config()  # defaults pull arena geometry from marker_mission

    # Drone at origin, facing the front wall (+y), at 1.0 m altitude.
    drone = DroneState(
        drone_id="sim1",
        position=Vec3(0.0, 0.0, 1.0),
        heading_deg=0.0,
    )

    # GOTO a point 2 m forward (+y) and 1 m right (+x), climb to 2 m.
    goto = RoleAction(kind=ActionKind.GOTO, target=Vec3(1.0, 2.0, 2.0))
    print("GOTO  (expect +fb forward, +lr right, +ud up, yaw 0):")
    print("   ", action_to_rc(goto, drone, cfg, {}))

    # YAW_SCAN at 30 deg/s CW, holding 1.5 m.
    scan = RoleAction(
        kind=ActionKind.YAW_SCAN, yaw_rate=30.0, target=Vec3(0.0, 0.0, 1.5)
    )
    print("YAW_SCAN 30 deg/s CW (expect yaw clearly > 0, small +ud):")
    print("   ", action_to_rc(scan, drone, cfg, {}))

    # HOVER holding 1.5 m (drone currently at 1.0 m -> climb a bit).
    hover = RoleAction(kind=ActionKind.HOVER, target=Vec3(0.0, 0.0, 1.5))
    print("HOVER hold z=1.5 (expect lr=fb=yaw=0, +ud):")
    print("   ", action_to_rc(hover, drone, cfg, {}))

    # ORBIT the arena centre at 2.5 m radius, 4 m altitude.
    mem: dict = {}
    orbit = RoleAction(
        kind=ActionKind.ORBIT,
        orbit_center=Vec2(0.0, 0.0),
        orbit_radius_m=2.5,
        target=Vec3(0.0, 0.0, 4.0),
    )
    print("ORBIT center=(0,0) r=2.5 z=4 (expect non-trivial sticks, +ud):")
    print("   ", action_to_rc(orbit, drone, cfg, mem))
    print("    orbit_angle persisted:", round(mem.get("orbit_angle", 0.0), 3))

    # NONE / no-position safety.
    print("NONE (expect 0,0,0,0):", action_to_rc(RoleAction.none(), drone, cfg, {}))
    lost = DroneState(drone_id="sim2", position=None, heading_deg=0.0)
    print("GOTO no-fix (expect 0,0,0,0):", action_to_rc(goto, lost, cfg, {}))

    # at_waypoint check.
    print("at_waypoint at target (expect True):",
          at_waypoint(DroneState("s", position=Vec3(1.0, 2.0, 2.0)),
                      Vec3(1.0, 2.0, 2.0), cfg))
    print("at_waypoint far    (expect False):",
          at_waypoint(drone, Vec3(5.0, 5.0, 2.0), cfg))
