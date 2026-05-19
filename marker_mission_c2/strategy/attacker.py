"""ATTACKER role.

Operator assigns a target *slot* (1..6). The role pushes ONE complete
mission script that does everything end-to-end:

    TAKEOFF
    HEIGHT <capture_ascend_m>           # ascend for camera line-of-sight
    FB_RC 100 <close_in_dur_s>          # FAST cruise within ArUco range
    FB_BRAKE <enemy-face-id> 2.0 100 5  # vision-tripped brake on target
    FB_RC <drift_rc> <drift_dur_s>      # short drift over the box
    YAW_IMU 180
    HOOVER <capture_hover_s>
    HEIGHT <home_alt>                   # cruise altitude above boxes
    FB_RC 100 <rth_dur_s>               # FAST cruise toward home
    FB_BRAKE <wall_marker> 2.5 100 5    # vision-tripped brake on home wall
    YAW_IMU 180                         # face enemy again so next attack starts correct
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

# Long cruises use FB_RC (raw forward stick) instead of FB_IMU (Olympe
# moveBy). FB_IMU is firmware-internal closed-loop and caps around
# 0.5 m/s in the sim — fine for short fine-positioning, terrible for
# a 15+ m cruise. FB_RC at stick=100 pins the forward channel to
# ±100 (the same units as the operator joystick: each unit ≈ 4 cm/s
# of commanded ground speed at level flight, so 100 ≈ 4 m/s) and the
# DSL auto-brakes after the requested duration. We follow it with
# APPROACH which uses per-marker IPPE distance for the precision
# stop — so even if cruise overshoots a metre, APPROACH catches it.
CRUISE_RC_STICK: int = 100

# Calibrated cruise speed for converting "distance to cover" → FB_RC
# duration. Empirically ~4 m/s at stick=100 on the sim's Anafi; will
# need re-calibration if RC->m/s ratio differs on hardware. Set
# slightly conservative so the FB_RC ends short of the target and
# APPROACH does the final close-in without backing up.
CRUISE_SPEED_M_PER_S: float = 4.0

# Don't emit FB_RC for trivial distances. Sub-half-second sticks add
# latency without meaningful motion (FC takes longer to switch from
# stick=0 → stick=100 → stick=0 than the move itself).
MIN_CRUISE_DURATION_S: float = 0.4

# Short raw-stick drift after APPROACH to settle over the box without
# waiting for another full PD cycle. ~25 cm at stick=60 (≈2.4 m/s)
# for 0.3 s = 0.7 m forward — same intent as the old FB_IMU 0.70 but
# 5x faster.
DRIFT_OVER_BOX_RC: int = 60
DRIFT_OVER_BOX_DURATION_S: float = 0.3

# Home-wall markers for RTH braking. 0.5 m markers on each team's
# home wall, LOW position (2.0 m altitude) — close to cruise altitude
# for stable APPROACH lock. Maps team → (marker_id, (arena_x, arena_y)).
HOME_WALL_MARKER: dict[str, tuple[int, tuple[float, float]]] = {
    "red": (13, (0.0, -10.0)),
    "blue": (9, (0.0, 10.0)),
}

RTH_APPROACH_DISTANCE_M: float = 2.0

# FB_BRAKE tunings. The DSL primitive cruises at full stick and brakes
# as soon as the named marker's IPPE distance < stop_m. We pick the
# stop distances to leave room for the auto-brake to settle without
# overshooting into the target box / wall:
#   - target slot face: 0.19 m marker, drone closes at ~4 m/s, brake
#     settles in ~1 m → 2.0 m stop leaves the drone hovering ~1 m
#     short of the box (FB_RC drift then puts us over it).
#   - home-wall marker: 0.5 m marker, same brake distance → 2.5 m
#     stop leaves the drone safely in the home zone.
# Timeout is a hard upper bound that flips into brake even if the
# marker is never seen (vision dropout → don't open-loop into a wall).
FB_BRAKE_TARGET_STOP_M: float = 2.0
FB_BRAKE_WALL_STOP_M: float = 2.5
FB_BRAKE_STICK: int = 100
FB_BRAKE_TIMEOUT_S: float = 5.0


def _format_script(*lines: str) -> str:
    return "\n".join(line for line in lines if line) + "\n"


def _cruise_duration_s(distance_m: float) -> float:
    """Convert a cruise distance to FB_RC stick-pin duration.

    Uses CRUISE_SPEED_M_PER_S as the calibrated stick=100 ground
    speed. Returns 0.0 for distances below MIN_CRUISE_DURATION_S
    worth of motion (so we skip the FB_RC step entirely and let
    APPROACH cover the whole remainder). The DSL auto-brake adds
    ~1 s to whatever we emit here.
    """
    if distance_m <= 0.0:
        return 0.0
    raw_dur = float(distance_m) / CRUISE_SPEED_M_PER_S
    if raw_dur < MIN_CRUISE_DURATION_S:
        return 0.0
    return raw_dur


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
        HEIGHT <capture_ascend_m>           # ascend for camera line-of-sight
        FB_RC 100 <close_in_dur_s>          # FAST cruise within ArUco range
        FB_BRAKE <id> 2.0 100 5             # vision-tripped brake on target
        FB_RC <drift_rc> <drift_dur_s>      # short drift over the box
        YAW_IMU 180                         # turn to face home
        HOOVER <capture_hover_s>            # capture window (~0.3-1 s)
        HEIGHT <home_alt>                   # cruise altitude above boxes
        FB_RC 100 <rth_dur_s>               # FAST cruise toward home
        FB_BRAKE <wall_marker> 2.5 100 5    # vision-tripped brake on home wall
        YAW_IMU 180                         # face enemy before landing
        LAND

    Speed-optimised design notes:

    Two-stage closing on every target / wall:

      1. FB_RC at full stick for the bulk of the distance (~4 m/s
         in sim — 8x the closed-loop moveBy speed). Open-loop, so
         it can overshoot/undershoot by a metre or two.
      2. FB_BRAKE on the relevant marker. The DSL primitive cruises
         at full stick AND watches the marker's IPPE distance; the
         moment d < stop_m the FC zeros sticks and enters its
         standard IMU-velocity brake. Replaces APPROACH (which was
         60-75 s of slow-zone PD per leg → 5 s of vision-tripped
         brake instead).

    RTH uses FB_BRAKE on the home-wall marker instead of TO_HOME.
    The position estimator drifts ~5 m at cruise speeds, making
    TO_HOME overshoot into the back wall. Per-marker IPPE distance
    is reliable — braking on a 0.5 m wall marker stops the drone
    precisely in the home zone regardless of estimator state.

    The final YAW_IMU 180 (after the wall-marker APPROACH) is the
    key to a repeatable multi-attack sequence: without it, the
    drone lands facing the home wall (the natural result of the
    APPROACH on a back-wall marker), and the NEXT attack's TAKEOFF
    inherits that heading — the next FB_RC would drive backward
    into the home wall. Two YAW_IMUs per attack keeps the
    spawn-orientation convention (drone faces enemy) consistent
    across runs.

    The post-APPROACH HEIGHT re-assert was removed because APPROACH
    already holds altitude via its own PD; the short FB_RC drift
    over the box is what actually puts us on top of the target.
    """
    m = ctx.match
    approach_d = max(0.2, float(m.approach_distance_m))
    ascend = max(0.6, float(m.capture_ascend_m))
    # Capture hover floor lowered to 0.3 s. The intent of the hover is
    # "give the scout a chance to register the new face" — at 5 Hz
    # vision that's 1-2 frames, plenty. Long hovers just burn budget.
    hover_s = max(0.3, float(m.capture_hover_s))
    home = (
        ctx.match.home_red if ctx.our_team == "red" else ctx.match.home_blue
    )
    home_alt = max(0.6, float(ctx.drone.home_alt_m or home.alt))

    # Attack-leg cruise: distance from home to target slot, minus a
    # standoff so APPROACH starts from inside its reliable detection
    # range. Emitted as ONE FB_RC step at full forward stick — much
    # faster than chained moveBy chunks.
    close_in = _close_in_distance_m((home.x, home.y), slot)
    close_in_dur = _cruise_duration_s(close_in)
    close_in_step = (
        f"FB_RC {CRUISE_RC_STICK} {close_in_dur:.2f}" if close_in_dur > 0 else ""
    )

    wall_entry = HOME_WALL_MARKER.get(ctx.our_team)
    if wall_entry:
        wall_marker_id, _ = wall_entry
        rth_close_in = _rth_close_in_distance_m(slot, ctx.our_team)
        rth_dur = _cruise_duration_s(rth_close_in)
        rth_cruise_step = (
            f"FB_RC {CRUISE_RC_STICK} {rth_dur:.2f}" if rth_dur > 0 else ""
        )
        rth_lines = (
            f"HEIGHT {home_alt:.2f}",
            rth_cruise_step,
            f"FB_BRAKE {wall_marker_id} {FB_BRAKE_WALL_STOP_M:.2f} "
            f"{FB_BRAKE_STICK} {FB_BRAKE_TIMEOUT_S:.1f}",
            # YAW_IMU 180 so the drone lands facing the ENEMY (not home).
            # Without this, after one full attack the drone is facing the
            # home wall (because of the mid-script YAW_IMU 180 + RTH);
            # the next takeoff would then drive forward straight into
            # the home wall. Net is two YAW_IMU 180s per attack — one
            # to face home for RTH, one to face the enemy again before
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
        close_in_step,
        # FB_BRAKE on the target's face marker. Replaces the old
        # APPROACH — closed-loop PD with slow-zone settle was the
        # single biggest time-sink in the script (~60-75 s vs 5 s
        # for FB_BRAKE). FB_BRAKE just pins the stick and snaps to
        # brake the moment vision sees the marker within
        # FB_BRAKE_TARGET_STOP_M; the drone ends roughly 1 m short
        # of the box and the FB_RC drift below puts us on top of it.
        f"FB_BRAKE {int(attack_marker_id)} {FB_BRAKE_TARGET_STOP_M:.2f} "
        f"{FB_BRAKE_STICK} {FB_BRAKE_TIMEOUT_S:.1f}",
        f"FB_RC {DRIFT_OVER_BOX_RC} {DRIFT_OVER_BOX_DURATION_S:.2f}",
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
