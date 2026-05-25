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

    # End-of-run = return to OUR home zone and HOVER — never LAND during a
    # match (SDC26 regs: the drone must return to the home zone and remain
    # inside to validate the capture; landing is not required and wastes the
    # rest of the match). We hold with a long HOOVER (the FC only
    # safety-lands once a script ends, i.e. after this hover, which we size
    # to ~a full match). The FC's arena guard keeps the hover inside bounds.
    home_hold_s = max(30.0, float(getattr(ctx.match, "home_hover_s", 600.0)))
    home_hold = f"HOOVER {home_hold_s:.0f}"
    wall_entry = HOME_WALL_MARKER.get(ctx.our_team)
    if wall_entry:
        wall_marker_id, _w_xy = wall_entry
        rth_close_in = _rth_close_in_distance_m(slot, ctx.our_team)
        rth_dur = _cruise_duration_s(rth_close_in)
        # Pure FB_RC for RTH at the same conservative cruise stick.
        rth_cruise_step = (
            f"FB_RC {CRUISE_RC_STICK} {rth_dur:.2f}"
            if rth_dur > 0 else ""
        )
        rth_lines = (
            # NO HOOVER between YAW and RTH cruise — start the run
            # back immediately per user's "much faster flight" req.
            rth_cruise_step,
            f"FB_BRAKE {wall_marker_id} {FB_BRAKE_WALL_STOP_M:.2f} "
            f"{FB_BRAKE_STICK} {FB_BRAKE_TIMEOUT_S:.1f}",
            "YAW_IMU 180",
            # Hold in the home zone (validates the capture) — do NOT land.
            home_hold,
        )
    else:
        rth_lines = (
            f"TO_HOME {ctx.drone.home_alt_m or 3.0:.2f}",
            # Hold in the home zone — do NOT land.
            home_hold,
        )
    # Height deconfliction: settle at this drone's DISTINCT cruise
    # altitude (assigned by the runner, floored above the 1.4 m box
    # tops) using the existing HEIGHT verb, so two attackers never
    # transit the neutral zone at the same height. Falls back to the
    # bare UD_RC climb when no deconflicted altitude is supplied.
    cruise_alt = max(1.4, float(ctx.cruise_alt_m)) if ctx.cruise_alt_m else 0.0
    height_step = f"HEIGHT {cruise_alt:.2f}" if cruise_alt else ""
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
        # FB_BRAKE on the target's face marker, with explicit world
        # coords passed in so the controller's world-fallback brake
        # works (slot face markers aren't in arena_config).
        f"FB_BRAKE {int(attack_marker_id)} {FB_BRAKE_TARGET_STOP_M:.2f} "
        f"{FB_BRAKE_STICK} {FB_BRAKE_TIMEOUT_S:.1f}{target_world_hint}",
        # Drift over the box (brief forward push + slight up-stick
        # to stay above the box top during momentum bleed).
        f"RC 0 {DRIFT_OVER_BOX_RC} {DRIFT_OVER_BOX_UD_RC} 0 "
        f"{DRIFT_OVER_BOX_DURATION_S:.2f}",
        "YAW_IMU 180",
        # No HOOVER — RTH starts immediately. Capture detection is
        # the scout role's job; the attacker just commits and runs.
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
