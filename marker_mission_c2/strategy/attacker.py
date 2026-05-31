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
    HEIGHT <cruise_alt>                 # climb back to distinct cruise alt
    FB_RC 100 <rth_dur_s>               # FAST cruise toward home
    FB_BRAKE <wall_marker> 2.5 100 5 x y# vision/world brake on home wall
    YAW_IMU 180                         # face enemy again so next attack starts correct
    HOOVER <rearm>                      # brief home hover — NO land

The drone closes in toward the enemy side, ascends, locks onto the
marker via APPROACH, drifts over it, hovers, then cruises home at full
speed and brakes on a 0.5 m wall marker via a second APPROACH. The script
ends with a brief home hover and NO land: the drone must stay airborne the
whole match, so it hovers at phase=done in the home zone until re-dispatched.
When the script ends the role marks itself done and the target is cleared.

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

from .roles import (
    Decision, Role, RoleContext, noop, push, register, home_park_xy,
)
from .settings import face_id

logger = logging.getLogger(__name__)


# Physical target slot positions in arena frame (x, y) — z is fixed at
# 1.0 m by the box pedestals and isn't needed for the close-in distance.
# Per SDC26 v7 regs §1.4.2 Table 1: boxes 1-3 live in the RED home zone
# (-Y end, default markers 41/42/43), boxes 4-6 in the BLUE home zone
# (+Y end, default markers 34/35/36), all spread across x in {-3, 0, +3}.
SLOT_POSITIONS_M: dict[int, tuple[float, float]] = {
    1: (-3.0, -7.5), 2: (0.0, -7.5), 3: (3.0, -7.5),
    4: (-3.0,  7.5), 5: (0.0,  7.5), 6: (3.0,  7.5),
}

# Stop the FB_IMU close-in this far short of the target so APPROACH has
# room to do its precision dance. 7 m gives 18 cm markers comfortable
# headroom for detection — empirically ArUco picks them up at <8 m on
# the sim's 70° FOV / 1280×720 cam.
CLOSE_IN_STANDOFF_M: float = 7.0

# Long cruises use FB_RC (raw forward stick) instead of FB_IMU (Olympe
# moveBy). FB_IMU is firmware-internal closed-loop and caps around
# 0.5 m/s in the sim — fine for short fine-positioning, terrible for
# a 15+ m cruise. FB_RC pins the forward channel to a stick value;
# the DSL auto-brakes after the requested duration. We follow it
# with APPROACH or FB_BRAKE for the precision stop.
#
# Stick=100 (~4 m/s) crashed the drone into walls repeatedly because
# (a) full forward pitch tilts the camera DOWN, hiding markers, and
# (b) Anafi can't decelerate from 4 m/s in less than ~3 m → the
# brake step overshoots into whatever's in front. Stick=50 (~2 m/s)
# is the conservative default; brake distance is ~1 m, marker
# detection works even with pitch.
CRUISE_RC_STICK: int = 50

# Calibrated cruise speed for converting "distance to cover" → FB_RC
# duration. ~2 m/s at stick=50 on the sim's Anafi.
CRUISE_SPEED_M_PER_S: float = 2.0

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
# Slight upward stick during the drift to keep the drone above the
# box top during the brief sideways momentum-bleed.
DRIFT_OVER_BOX_UD_RC: int = 20

# v7 §1.4.3 capture rule: the attacker must hover over the target box
# for ≥ 2 s within the 1-2 m RFID detection band for the flip to
# register. 2.5 s gives a 25 % safety margin over the regulation floor
# (sim tick rounding + brief drift outside capture radius can both eat
# 100-300 ms of the dwell window). Emitted as a HOOVER step right after
# the drift, before YAW_IMU 180 / RTH — uses only the existing FC verb
# (per the "no new FC functions, build strictly the C2" constraint).
CAPTURE_HOLD_HOVER_S: float = 2.5

# Combined-motion sticks for the "climb + cruise" RC step that
# replaces dedicated HEIGHT + FB_RC sequences. Anafi accepts all four
# sticks on a single PCMD frame — sending fb + ud simultaneously
# saves a full HEIGHT step (~10-20 s) per leg by parallelising the
# vertical and horizontal motion. The ud value is chosen so the
# drone climbs the configured ascend distance over roughly the same
# duration the fb stick covers the cruise distance:
#
#   capture ascend (TAKEOFF → camera-level): ~0.6 m at ~0.3 m/s = 2 s
#   RTH ascend (camera-level → above boxes): ~1.5 m at ~0.4 m/s = 4 s
#
# The cruise itself is faster (~4 m/s) so the ud stick is sized to
# *finish climbing by the time we're in marker-detection range* —
# any residual climb after FB_BRAKE trips would just lift us off
# the box.
CRUISE_UP_UD_RC: int = 80      # combined fb + climb for attack leg
RTH_CRUISE_UP_UD_RC: int = 60  # combined fb + climb for RTH leg

# Initial fast-climb step before any forward motion. Anafi takeoff
# settles at ~0.9 m; slot box tops are at 1.4 m. If we start cruising
# forward immediately, the drone tilts pitch-forward (which also
# pitches the camera DOWN), and even if it survives the dip, the
# camera can lose the target marker because it's looking at the
# ground instead of ahead. The brief dedicated climb fixes both:
# safe altitude clearance AND a level camera for vision.
INITIAL_CLIMB_UD_RC: int = 100   # full up-stick
INITIAL_CLIMB_DURATION_S: float = 1.2  # ~0.5m climb → ~1.4m total alt

# 180° turns use the firmware-internal YAW_IMU (api.rotate), NOT
# open-loop YAW_RC. We tried YAW_RC for speed (~2 s vs 5 s) but it's
# too imprecise: YAW_RC 100 for 1.0 s only rotated ~95° in testing
# (the firmware ramps the yaw rate up over the first ~0.5 s and the
# auto-brake eats the tail), AND the fast spin destabilised altitude
# hold (drone lost ~1.4 m and sank to the floor on the RTH leg).
# A wrong heading sends the whole RTH cruise sideways into open
# arena. YAW_IMU's closed-loop "rotate exactly N degrees and confirm"
# is worth the extra ~3 s per turn — heading accuracy is critical
# because the cruise that follows is open-loop.

# Home-wall markers for RTH braking. 0.5 m markers on each team's
# home wall, LOW position (2.0 m altitude) — close to cruise altitude
# for stable APPROACH lock. Maps team → (marker_id, (arena_x, arena_y)).
HOME_WALL_MARKER: dict[str, tuple[int, tuple[float, float]]] = {
    "red": (13, (0.0, -10.0)),
    "blue": (9, (0.0, 10.0)),
}

RTH_APPROACH_DISTANCE_M: float = 2.0

# Default approach speed (forward RC stick, 1-100) the strategy passes
# to AUTO_ATTACK. ~4 cm/s per unit, so 55 ≈ 2.2 m/s — a conservative
# cruise that keeps the camera level enough for vision and lets the
# brake settle within the arena guard's margin. Raise for a faster
# run in an open arena.
AUTO_ATTACK_APPROACH_SPEED: int = 55

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
# FB_BRAKE stops the drone roughly stop_m from the marker BY VISION,
# or (stop_m + WORLD_FALLBACK_MARGIN_M) from the world-known position
# as a safety net. We pick stop_m larger than strictly needed so the
# brake-settle distance (~1 m at fb=50) plus position-estimator drift
# (~2 m) doesn't push the drone into the wall behind the marker.
# Target stop is tighter (1.5 m) — vision drives the final approach
# now (world fallback is suppressed while vision is live), so we can
# safely close in. Wall stop stays conservative (3.0 m) since the
# wall brake leans on the world fallback (vision of the wall marker
# is less reliable at cruise pitch) and we must not clip the wall.
FB_BRAKE_TARGET_STOP_M: float = 1.5
FB_BRAKE_WALL_STOP_M: float = 3.0
# FB_BRAKE drives at the SAME stick as cruise (so we don't accelerate
# back to full speed during the brake-search phase, which then has
# more momentum to bleed off). Sticking to ~2 m/s throughout keeps
# the system in a regime where the auto-brake actually works.
FB_BRAKE_STICK: int = 50
# Timeout is now generous (was 4.0). The drone cruises slower than
# the nominal calibration (~1 m/s effective), so a tight timeout
# fired BEFORE the vision/world trip and the drone braked short of
# the target. With the arena guard as the hard wall backstop, a long
# FB_BRAKE timeout is safe: vision/world will trip well within it,
# and if both somehow fail the guard stops the drone at the wall.
FB_BRAKE_TIMEOUT_S: float = 10.0

# After returning home, hover briefly (registers v7 §1.4.4 home-zone presence)
# and then simply END the script — WITHOUT landing. The drones must stay in
# the air for the WHOLE match (operator requirement), so we never emit LAND.
# When the script ends the FC drops to phase=done while still hovering in the
# home zone; the strategy then re-dispatches the drone for its NEXT attack and
# the new mission's TAKEOFF is a no-op (already airborne). Ending the script
# is what frees the drone for re-dispatch — landing was never needed for that;
# the OLD ~10-minute home hover just never ended, so the drone got stuck in
# one mission and could fly only ONE attack per match. 2 s ≈ 2 C2 ticks.
HOME_REARM_HOVER_S: float = 2.0


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
        RC 0 100 80 0 <close_in_dur>        # COMBINED climb + cruise (no HEIGHT)
        FB_BRAKE <id> 2.0 100 5             # vision-tripped brake on target
        RC 0 60 20 0 0.3                    # drift over box (fwd + slight up)
        YAW_RC 100 1.0                      # fast 180° turn (~2 s vs YAW_IMU 5 s)
        RC 0 100 60 0 <rth_dur>             # COMBINED climb + RTH cruise (no HEIGHT,
                                            # no HOOVER — starts immediately)
        FB_BRAKE <wall_marker> 2.5 100 5    # vision-tripped brake on home wall
        YAW_RC 100 1.0                      # fast 180° turn back to enemy-facing
        HOOVER <rearm>                      # brief home hover — NO land (stay airborne)

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
    # capture_ascend_m and capture_hover_s are still read for compat
    # but no longer drive separate HEIGHT / HOOVER steps — combined
    # RC handles both. Kept here so config writes don't crash if the
    # operator still flips these in the dashboard.
    _ = max(0.6, float(m.capture_ascend_m))
    _ = max(0.0, float(m.capture_hover_s))
    home = (
        ctx.match.home_red if ctx.our_team == "red" else ctx.match.home_blue
    )
    _ = max(0.6, float(ctx.drone.home_alt_m or home.alt))

    # Attack-leg cruise duration (distance ÷ calibrated cruise speed).
    # Pure FB_RC (no simultaneous up-stick) — the dedicated UD_RC
    # climb step before this puts us at safe altitude; adding ud
    # here would force the drone to keep pitching forward AND up,
    # which destabilises camera pointing and the velocity controller.
    close_in = _close_in_distance_m((home.x, home.y), slot)
    close_in_dur = _cruise_duration_s(close_in)
    close_in_step = (
        f"FB_RC {CRUISE_RC_STICK} {close_in_dur:.2f}"
        if close_in_dur > 0 else ""
    )

    # Slot box position in arena frame — used by FB_BRAKE's
    # world-position fallback so the drone can brake even when vision
    # can't see the target face marker (slot faces aren't in
    # arena_config; the FC wouldn't otherwise know where they are).
    target_xy = SLOT_POSITIONS_M.get(int(slot))
    target_world_hint = (
        f" {target_xy[0]:g} {target_xy[1]:g}" if target_xy else ""
    )

    # This drone's DISTINCT cruise altitude (from the runner's team-keyed
    # altitude grid — >= 0.30 m from every other drone of BOTH teams). Used
    # for the outbound HEIGHT climb AND the RTH climb so the drone holds its
    # own height across the neutral zone in EACH direction and never shares a
    # transit altitude with another drone. Floored above the 1.4 m box tops;
    # falls back to the bare UD_RC climb when no deconflicted altitude is set.
    cruise_alt = max(1.4, float(ctx.cruise_alt_m)) if ctx.cruise_alt_m else 0.0
    height_step = f"HEIGHT {cruise_alt:.2f}" if cruise_alt else ""

    # End-of-run = return to OUR home zone and confirm home-zone presence
    # (v7 §1.4.4) with a brief hover — then the script just ENDS. No LAND:
    # the drone must stay airborne the whole match, so it hovers at phase=done
    # in the home zone until the strategy re-dispatches it for the next attack.
    # The short hover (vs the old ~10-min one) is what lets the script end so
    # the drone becomes a free attacker again — without ever touching down.
    home_rearm = f"HOOVER {HOME_REARM_HOVER_S:.1f}"
    wall_entry = HOME_WALL_MARKER.get(ctx.our_team)
    if wall_entry:
        wall_marker_id, w_xy = wall_entry
        rth_close_in = _rth_close_in_distance_m(slot, ctx.our_team)
        rth_dur = _cruise_duration_s(rth_close_in)
        # Pure FB_RC for RTH at the same conservative cruise stick.
        rth_cruise_step = (
            f"FB_RC {CRUISE_RC_STICK} {rth_dur:.2f}"
            if rth_dur > 0 else ""
        )
        rth_lines = (
            # Climb back to our DISTINCT cruise altitude BEFORE crossing the
            # neutral zone home. Without this the drone returns at the flat
            # 1.5 m capture height, so every returning drone (ours AND the
            # enemy's, heading the opposite way) shares 1.5 m — a head-on
            # collision band. The wall brake below carries an explicit world
            # hint, so it still stops us correctly at any altitude.
            height_step,
            # NO HOOVER between YAW and RTH cruise — start the run
            # back immediately per user's "much faster flight" req.
            rth_cruise_step,
            f"FB_BRAKE {wall_marker_id} {FB_BRAKE_WALL_STOP_M:.2f} "
            f"{FB_BRAKE_STICK} {FB_BRAKE_TIMEOUT_S:.1f} "
            f"{w_xy[0]:g} {w_xy[1]:g}",
            "YAW_IMU 180",
            # Confirm home presence, then end the script (NO land) — stay
            # airborne, re-arm, and re-attack on the next dispatch.
            home_rearm,
        )
    else:
        rth_lines = (
            f"TO_HOME {ctx.drone.home_alt_m or 3.0:.2f}",
            # Confirm home presence, then end the script (NO land) — stay
            # airborne, re-arm, and re-attack on the next dispatch.
            home_rearm,
        )
    # v7 §1.4.3 capture needs the drone in the 1-2 m RFID detection band.
    # If the deconflicted cruise altitude is above that band (large rosters
    # push scouts/attackers up to ~3 m+), we MUST descend just before the
    # target brake or the capture never registers. Aim for the middle of
    # the band (1.5 m) — well inside z_min/z_max and clear of box tops.
    CAPTURE_DESCENT_ALT_M = 1.5
    return _format_script(
        "TAKEOFF",
        # CLIMB FIRST — Anafi settles at ~0.9m post-takeoff; slot
        # boxes top out at 1.4m. Forward motion BEFORE we clear that
        # height pitches the drone (and its camera) down, with two
        # consequences: (a) drone risks clipping its own home-side
        # slot boxes on the way out, (b) the camera looks at the
        # ground instead of forward → vision misses the target.
        # 1.2 s of up-stick + auto-brake gets us safely to ~1.4 m.
        f"UD_RC {INITIAL_CLIMB_UD_RC} {INITIAL_CLIMB_DURATION_S:g}",
        # Rise to this drone's distinct cruise altitude (deconfliction).
        height_step,
        # Combined RC for the long cruise.
        close_in_step,
        # Descend into the 1-2 m capture band BEFORE braking on the
        # target. Without this, an attacker deconflicted to >2 m cruise
        # arrives above the RFID detection band and the capture rule
        # (z in [1, 2] m) never triggers.
        f"HEIGHT {CAPTURE_DESCENT_ALT_M:.2f}",
        # FB_BRAKE on the target's face marker, with explicit world
        # coords passed in so the controller's world-fallback brake
        # works (slot face markers aren't in arena_config).
        f"FB_BRAKE {int(attack_marker_id)} {FB_BRAKE_TARGET_STOP_M:.2f} "
        f"{FB_BRAKE_STICK} {FB_BRAKE_TIMEOUT_S:.1f}{target_world_hint}",
        # Drift over the box (brief forward push + slight up-stick
        # to stay above the box top during momentum bleed).
        f"RC 0 {DRIFT_OVER_BOX_RC} {DRIFT_OVER_BOX_UD_RC} 0 "
        f"{DRIFT_OVER_BOX_DURATION_S:.2f}",
        # Settle precisely over the box centre. FB_BRAKE stops at
        # FB_BRAKE_TARGET_STOP_M (~1.5 m) from the box; the drift above
        # pushes us forward but momentum + position noise can leave the
        # drone outside the capture radius (sim: 0.8 m; the RFID detector
        # zone in real life is similarly tight). TO is precise xy goto
        # (existing FC verb) with ARRIVE_XY=0.25 m — well inside the
        # capture radius. Without this the next HOOVER can dwell *next
        # to* the box instead of *over* it, and the flip never registers.
        f"TO {target_xy[0]:g} {target_xy[1]:g}" if target_xy else "",
        # v7 §1.4.3 capture rule: the drone must hover ≥ 2 s within the
        # 1-2 m RFID band over the box for the flip to register. Without
        # this step the brief drift above isn't enough dwell — the box
        # never changes colour. Uses the existing HOOVER FC verb (no new
        # FC functions); 2.5 s = 2 s requirement + safety margin against
        # tick rounding / brief drift outside the capture radius.
        f"HOOVER {CAPTURE_HOLD_HOVER_S:.1f}",
        "YAW_IMU 180",
        *rth_lines,
    )


def _auto_attack_script(
    ctx: RoleContext, attack_marker_id: int, slot: int
) -> str:
    """DEPRECATED / UNUSED — retained for reference only.

    The C2 no longer emits the FC's monolithic ``AUTO_ATTACK`` maneuver:
    per the "build strictly the C2, compose from existing FC primitives"
    rule, the attacker now uses :func:`_full_attack_script` (TAKEOFF /
    UD_RC / HEIGHT / FB_RC / FB_BRAKE / RC / YAW_IMU / LAND). This
    function is kept only so the AUTO_ATTACK choreography notes aren't
    lost; nothing calls it.

    Reactive end-to-end attack via the FC's AUTO_ATTACK phase.

    Instead of a fixed choreography (climb N s, cruise M s, brake, …),
    we hand the FC a single AUTO_ATTACK step carrying the target slot
    marker + its arena position and the home-wall marker + its arena
    position. The FC then runs a closed-loop state machine (climb →
    go-target → capture → go-home → land) that steers continuously,
    homing on vision (drift-free) when the marker is in view and on
    the ArUco arena position when it isn't, with the always-on arena
    guard preventing wall contact. This adapts every tick instead of
    committing to pre-computed timings — the architecture the operator
    asked for after the fixed-choreography runs proved unreliable.

    Script shape:
        TAKEOFF
        AUTO_ATTACK <tgt_id> <tx> <ty> <home_id> <hx> <hy> <alt_m> <speed>

    (AUTO_ATTACK ends with LAND internally, so no trailing LAND.)

    Altitude comes from the per-drone ``attack_alt_m`` (floored above
    the box tops); approach speed is AUTO_ATTACK_APPROACH_SPEED. Both
    are tunable so the operator can fly higher/slower in a tight arena
    or lower/faster in an open one.
    """
    target_xy = SLOT_POSITIONS_M.get(int(slot)) or (0.0, 0.0)
    wall_entry = HOME_WALL_MARKER.get(ctx.our_team)
    if wall_entry is None:
        # Shouldn't happen (both teams mapped); fall back to old script.
        return _full_attack_script(ctx, attack_marker_id, slot)
    home_marker_id, home_xy = wall_entry
    # Cruise altitude: per-drone attack_alt_m, floored at 1.4 m so we
    # always clear the slot boxes. Approach speed: a conservative
    # forward-stick default (operator can raise for a faster run).
    # Prefer the deconflicted cruise altitude (distinct per drone) so two
    # attackers transit at different heights; floor at 1.4 m to always
    # clear the slot boxes. Fall back to the per-drone attack_alt_m.
    altitude_m = max(1.4, float(ctx.cruise_alt_m or ctx.drone.attack_alt_m or 1.6))
    approach_speed = AUTO_ATTACK_APPROACH_SPEED
    return _format_script(
        "TAKEOFF",
        f"AUTO_ATTACK {int(attack_marker_id)} "
        f"{target_xy[0]:g} {target_xy[1]:g} "
        f"{int(home_marker_id)} {home_xy[0]:g} {home_xy[1]:g} "
        f"{altitude_m:g} {approach_speed}",
    )


def _return_home_script(ctx: RoleContext) -> str:
    """Fly back into our home zone and hover (no land) — used during a BANK.

    Existing FC verbs only (TAKEOFF / HEIGHT / TO / HOOVER); TAKEOFF is a no-op
    when already airborne. Brings an idle/out attacker home so the team can
    secure the 5-pt attempt (all drones home before the box is recaptured).
    """
    cruise_alt = max(1.4, float(ctx.cruise_alt_m)) if ctx.cruise_alt_m else 1.6
    hx, hy = home_park_xy(ctx.our_team)
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {cruise_alt:.2f}",
        f"TO {hx:g} {hy:g}",
        f"HEIGHT {cruise_alt:.2f}",
        f"HOOVER {HOME_REARM_HOVER_S:.1f}",
    )


def _enemy_face_for(slot: int, our_team: str) -> int:
    enemy = "blue" if our_team == "red" else "red"
    return face_id(slot, enemy)


def _slot_is_ours(ctx: RoleContext, slot: int) -> bool:
    # DECAYED belief (same as the planner): only treat the slot as ours if a
    # drone has actually SEEN it ours recently. A stale "we hold it" must not
    # cancel a capture run — the enemy can re-capture the instant we leave and
    # we can't see the slot from home. Once the belief decays to "unknown" the
    # attacker flies back out to re-check / re-capture it (continuous play).
    return ctx.markers.effective_holder(slot) == ctx.our_team


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

        # ---- BANK: we just captured an enemy box — get home to secure the
        # 5-pt attempt (all drones must return home before recapture). Don't
        # start any new attack; bring this drone home if it's out, hold if home.
        if ctx.team_phase == "bank":
            rs.target_slot = None        # abandon any pending attack intent
            rs.target_assigned_unix_s = None
            if ctx.state.phase == "running":
                # Mid-run: its own script ends with an RTH into home, so just
                # let it finish — interrupting would waste the in-flight run.
                return noop("attacker: bank — finishing run (ends home)")
            if ctx.in_home_now:
                rs.advance_phase("done", "bank — home, holding")
                return noop("attacker: bank — home, holding (secured)")
            # Idle/done but not home (e.g. was waiting out) — fly home now.
            return push(
                _return_home_script(ctx),
                new_phase="returning",
                reason="attacker: bank — return home to secure the attempt",
            )

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
            # The FC can only START a mission from INIT (on the ground). Since
            # runs now end with a home HOVER (we never land mid-match), a drone
            # that already flew a run is still airborne and CANNOT begin a new
            # mission — pushing would just be rejected. Hold instead. (Doing
            # several no-land runs needs them chained into one mission.)
            if ctx.state.phase not in ("init", "done", ""):
                return noop(
                    f"attacker: airborne (fc phase={ctx.state.phase}); "
                    f"holding — FC must be INIT to start a new run"
                )
            # v7 §1.4.4: a drone may only start a new scoring attempt after it
            # has been detected back in its own home zone. Refuse to push until
            # in_home_now flips True (computed from world_position_m + team).
            if not ctx.in_home_now:
                return noop(
                    "attacker: not yet detected back in home zone "
                    "(v7 §1.4.4 requires home-zone presence before re-attempt)"
                )
            attack_id = _enemy_face_for(slot, ctx.our_team)
            rs.last_attack_marker_id = attack_id
            # Compose the attack from EXISTING basic FC verbs (TAKEOFF /
            # UD_RC / HEIGHT / FB_RC / FB_BRAKE / RC / YAW_IMU / LAND).
            # We intentionally do NOT use the FC's monolithic AUTO_ATTACK
            # state machine: the C2 owns the choreography and only orders
            # the FC around with primitives it already provides.
            return push(
                _full_attack_script(ctx, attack_id, slot),
                new_phase="running",
                reason=f"attacker: capture slot {slot} (id={attack_id})",
            )

        # ---- running: waiting for the entire script to finish -------------
        if rs.phase == "running":
            if slot is None:
                # Target cleared while in flight — let the script finish on
                # its own (it ends with a brief home hover, no land); we just
                # reset our state.
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
