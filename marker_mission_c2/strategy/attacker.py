"""ATTACKER role.

Operator assigns a target *slot* (1..6). The role pushes ONE complete
mission script that does everything end-to-end:

    TAKEOFF
    HEIGHT <capture_ascend_m>      # ascend for camera line-of-sight
    FB_IMU <close_in_m>            # cruise within ArUco range
    APPROACH <enemy-face-id> <approach_distance_m>
    HEIGHT <capture_ascend_m>      # re-assert post-approach
    FB_IMU <capture_forward_m>     # drift over the box
    YAW_IMU 180
    HOOVER <capture_hover_s>
    HEIGHT <home_alt>              # cruise altitude above boxes
    FB_IMU <rth_close_in_m>        # high-speed cruise toward home
    APPROACH <wall_marker> <2.0>   # brake on home-wall marker
    YAW_IMU 180                    # face enemy again so next attack starts correct
    LAND

The drone closes in toward the enemy side, ascends, locks onto the
marker via APPROACH, drifts over it, hovers, then cruises home at full
speed and brakes on a 0.5 m wall marker via a second APPROACH. After
the script fully ends (FC drops back to Phase.INIT), the role marks
itself done and the target is cleared.

OPERATOR CONVENTION
-------------------
The FB_IMU close-in assumes the drone is placed at takeoff with its
front camera pointing toward the enemy side (i.e. red drones face arena
+y, blue drones face -y). FB_IMU is forward-in-drone-frame, not arena
frame — without the right takeoff orientation the drone would fly off
into a wall. We don't try to align via YAW_IMU because the absolute
IMU↔arena yaw transform is unreliable (no compass calibration in sim,
magnetic_north_arena_yaw_deg drifts). Operator-set takeoff orientation
is simple and reliable.

We deliberately keep it as a single push (rather than separate attack +
RTH scripts) because pushing a second script while the first is still
running is rejected by marker_mission_c2's start endpoint. Mid-script
phase==IDLE (during HOOVER) is *not* "script finished" — it's just
station-keeping within the current mission.

Phases:
    idle    → no target assigned, or target already captured by us
    running → attack script pushed, drone executing the whole sequence
    done    → script ended (FC back to INIT); reset to idle

If the operator assigns a slot we already hold (holder == our_team) the
role short-circuits straight to ``done``.

Verify-by-scout (i.e. abort RTH if the capture is not confirmed) is a
future enhancement; for now the script always commits to the full
approach-capture-RTH-land sequence.
"""
from __future__ import annotations

import logging
import math

from .roles import Decision, Role, RoleContext, noop, push, register
from .settings import face_id

logger = logging.getLogger(__name__)


# Physical target slot positions in arena frame (x, y) — z is fixed at
# 1.0 m by the box pedestals and isn't needed for the close-in distance.
# Pulled from tools/sphinx-arena/default_target_layout.json and the
# §1.1 regs: red slots 4-6 at y=-7.5, blue slots 1-3 at y=+7.5, all
# spread across x ∈ {-3, 0, +3}.
SLOT_POSITIONS_M: dict[int, tuple[float, float]] = {
    1: (-3.0, 7.5),  2: (0.0, 7.5),  3: (3.0, 7.5),
    4: (-3.0, -7.5), 5: (0.0, -7.5), 6: (3.0, -7.5),
}

# Stop the FB_IMU close-in this far short of the target so APPROACH has
# room to do its precision dance. 7 m gives 18 cm markers comfortable
# headroom for detection — empirically ArUco picks them up at <8 m on
# the sim's 70° FOV / 1280×720 cam.
CLOSE_IN_STANDOFF_M: float = 7.0

# mission_script enforces |FB_IMU| <= 5 m per step (anti-typo safety
# cap). But empirically chained 5m moveBy steps stall short (one round
# of 3×5+1.5 commanded → only ~7 m actual). Cap our chunk size at 4 m
# to stay well clear of that regime while respecting the parser cap.
FB_IMU_CHUNK_M: float = 4.0

# Home-wall markers for RTH braking. 0.5 m markers on each team's
# home wall, LOW position (2.0 m altitude) — close to cruise altitude
# for stable APPROACH lock. Maps team → (marker_id, (arena_x, arena_y)).
HOME_WALL_MARKER: dict[str, tuple[int, tuple[float, float]]] = {
    "red": (13, (0.0, -10.0)),
    "blue": (9, (0.0, 10.0)),
}

RTH_APPROACH_DISTANCE_M: float = 2.0


def _format_script(*lines: str) -> str:
    return "\n".join(line for line in lines if line) + "\n"


def _chunk_close_in(total_m: float) -> list[float]:
    """Split a total forward-cruise distance into <= FB_IMU_CHUNK_M chunks.

    Drops chunks below 0.1 m (parser min is 0.01 m but tiny steps add
    latency for no useful motion). Returns [] for non-positive totals.
    """
    out: list[float] = []
    remaining = float(total_m)
    while remaining > FB_IMU_CHUNK_M:
        out.append(FB_IMU_CHUNK_M)
        remaining -= FB_IMU_CHUNK_M
    if remaining >= 0.1:
        out.append(remaining)
    return out


def _close_in_distance_m(home_xy: tuple[float, float], slot: int) -> float:
    """Straight-line distance from home to target slot, minus standoff.

    Returns 0.0 if the slot is unknown or if home is already inside the
    standoff (don't emit a degenerate FB_IMU that drives backward).
    """
    target = SLOT_POSITIONS_M.get(int(slot))
    if target is None:
        return 0.0
    dx = target[0] - home_xy[0]
    dy = target[1] - home_xy[1]
    raw = math.hypot(dx, dy)
    return max(0.0, raw - CLOSE_IN_STANDOFF_M)


def _rth_close_in_distance_m(slot: int, our_team: str) -> float:
    """Distance from target slot to home-wall marker, minus standoff."""
    target = SLOT_POSITIONS_M.get(int(slot))
    if target is None:
        return 0.0
    entry = HOME_WALL_MARKER.get(our_team)
    if entry is None:
        return 0.0
    _, wall_xy = entry
    dx = wall_xy[0] - target[0]
    dy = wall_xy[1] - target[1]
    raw = math.hypot(dx, dy)
    return max(0.0, raw - CLOSE_IN_STANDOFF_M)


def _full_attack_script(
    ctx: RoleContext, attack_marker_id: int, slot: int
) -> str:
    """Close in toward the enemy side, capture, RTH via wall marker, land.

    Script shape:
        TAKEOFF
        HEIGHT <capture_ascend_m>      # ascend for camera line-of-sight
        FB_IMU <close_in_m>            # cruise within ArUco range
        APPROACH <id> <approach_d>     # precision lock + slow advance
        HEIGHT <capture_ascend_m>      # re-assert post-approach
        FB_IMU <capture_forward_m>     # drift over the box
        YAW_IMU 180                    # turn to face home
        HOOVER <capture_hover_s>       # capture window
        HEIGHT <home_alt>              # cruise altitude above boxes
        FB_IMU <rth_close_in_m>        # high-speed cruise toward home
        APPROACH <wall_marker> <2.0>   # brake on home-wall marker
        YAW_IMU 180                    # face enemy before landing
        LAND

    RTH uses APPROACH on the home-wall marker instead of TO_HOME.
    The position estimator drifts ~5 m at cruise speeds, making
    TO_HOME overshoot into the back wall. Per-marker IPPE distance
    is reliable — APPROACH on a 0.5 m wall marker brakes precisely.

    The final YAW_IMU 180 (after the wall-marker APPROACH) is the
    key to a repeatable multi-attack sequence: without it, the
    drone lands facing the home wall (the natural result of the
    APPROACH on a back-wall marker), and the NEXT attack's TAKEOFF
    inherits that heading — FB_IMU then drives backward into the
    home wall. Two YAW_IMUs per attack keeps the spawn-orientation
    convention (drone faces enemy) consistent across runs.
    """
    m = ctx.match
    approach_d = max(0.2, float(m.approach_distance_m))
    ascend = max(0.6, float(m.capture_ascend_m))
    forward = max(0.1, float(m.capture_forward_m))
    hover_s = max(1.0, float(m.capture_hover_s))
    home = (
        ctx.match.home_red if ctx.our_team == "red" else ctx.match.home_blue
    )
    home_alt = max(0.6, float(ctx.drone.home_alt_m or home.alt))
    close_in = _close_in_distance_m((home.x, home.y), slot)
    close_in_steps = "\n".join(
        f"FB_IMU {chunk:.2f}" for chunk in _chunk_close_in(close_in)
    )
    wall_entry = HOME_WALL_MARKER.get(ctx.our_team)
    if wall_entry:
        wall_marker_id, _ = wall_entry
        rth_close_in = _rth_close_in_distance_m(slot, ctx.our_team)
        rth_steps = "\n".join(
            f"FB_IMU {chunk:.2f}" for chunk in _chunk_close_in(rth_close_in)
        )
        rth_lines = (
            f"HEIGHT {home_alt:.2f}",
            rth_steps,
            f"APPROACH {wall_marker_id} {RTH_APPROACH_DISTANCE_M:.2f}",
            # YAW_IMU 180 so the drone lands facing the ENEMY (not home).
            # Without this, after one full attack the drone is facing the
            # home wall (because of the mid-script YAW_IMU 180 + RTH);
            # the next takeoff would then drive FB_IMU straight into the
            # home wall. Net is two YAW_IMU 180s per attack — one to
            # face home for RTH, one to face the enemy again before
            # landing — which keeps every subsequent attack consistent
            # with the operator's spawn convention (drone faces enemy).
            "YAW_IMU 180",
            "LAND",
        )
    else:
        rth_lines = (
            f"TO_HOME {home_alt:.2f}",
            "LAND",
        )
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {ascend:.2f}",
        close_in_steps,
        f"APPROACH {int(attack_marker_id)} {approach_d:.2f}",
        f"HEIGHT {ascend:.2f}",
        f"FB_IMU {forward:.2f}",
        "YAW_IMU 180",
        f"HOOVER {hover_s:.1f}",
        *rth_lines,
    )


def _enemy_face_for(slot: int, our_team: str) -> int:
    enemy = "blue" if our_team == "red" else "red"
    return face_id(slot, enemy)


def _slot_is_ours(ctx: RoleContext, slot: int) -> bool:
    return ctx.markers.slot_holder(slot) == ctx.our_team


class AttackerRole(Role):
    name = "attacker"

    def decide(self, ctx: RoleContext) -> Decision:
        rs = ctx.role_state
        drone = ctx.drone

        if not drone.enabled:
            return noop("drone disabled")
        if not drone.team:
            return noop("attacker: no team assigned")
        if not ctx.state.drone_connected:
            return noop("attacker: drone not connected")

        slot = rs.target_slot

        # ---- idle: waiting for a target slot ------------------------------
        if rs.phase in ("", "idle", "done"):
            if slot is None:
                return noop("attacker: idle (no target slot)")
            if _slot_is_ours(ctx, slot):
                rs.advance_phase(
                    "done", f"slot {slot} already captured by us — no action"
                )
                return noop(f"attacker: slot {slot} already ours")
            attack_id = _enemy_face_for(slot, ctx.our_team)
            rs.last_attack_marker_id = attack_id
            return push(
                _full_attack_script(ctx, attack_id, slot),
                new_phase="running",
                reason=f"attacker: full attack on slot {slot} (id={attack_id})",
            )

        # ---- running: waiting for the entire script to finish -------------
        if rs.phase == "running":
            if slot is None:
                # Target cleared while in flight — let the script finish on
                # its own (it ends with LAND); we just reset our state.
                rs.advance_phase("idle", "target cleared mid-flight")
                return noop("attacker: target cleared mid-flight")
            # Only "init" reliably means the script has fully ended (FC has
            # landed and gone back to its resting state). Phase=IDLE
            # during the HOOVER step is mid-script and must not be
            # treated as completion.
            if ctx.state.phase in ("init", "done", ""):
                rs.target_slot = None
                rs.target_assigned_unix_s = None
                rs.advance_phase(
                    "done",
                    f"attack on slot {slot} complete (FC back to init)",
                )
                return noop("attacker: attack complete, target cleared")
            return noop(
                f"attacker: running (fc phase={ctx.state.phase})"
            )

        # Unknown phase — reset to idle.
        rs.advance_phase("idle", f"unknown phase {rs.phase}")
        return noop("attacker: reset to idle")


register(AttackerRole())
