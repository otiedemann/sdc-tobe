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

from . import arena_state
from .roles import (
    Decision, Role, RoleContext, noop, push, stop_cmd, register, home_park_xy,
    enemy_heading_deg,
)
from .settings import face_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arena-derived geometry (LIVE — follows the active arena, see arena_state).
#
# ``SLOT_POSITIONS_M`` (target box xy, for the C2's bookkeeping + dashboard map)
# and ``HOME_WALL_MARKER`` (per-team home back-wall marker) used to be hardcoded
# literals here. They now read through ``arena_state`` so a live arena switch
# (real<->gvz) is reflected immediately. They stay exposed under these names —
# a thin ``_LiveMap`` keeps the ``[team]`` / ``.get(slot)`` call sites (here and
# the ``from .attacker import ...`` in defender.py / planner.py) working
# unchanged, while always returning the CURRENT arena's values. The ATTACK
# itself doesn't use slot xy — it homes on the ArUco marker via APPROACH — so
# these only drive bookkeeping + the map.
class _LiveMap:
    """Read-only mapping view that re-derives from a thunk on every access, so
    it always reflects the active arena (never a stale snapshot)."""

    __slots__ = ("_fn",)

    def __init__(self, fn):
        self._fn = fn                       # fn() -> dict

    def get(self, key, default=None):
        return self._fn().get(key, default)

    def __getitem__(self, key):
        return self._fn()[key]

    def __contains__(self, key):
        return key in self._fn()

    def __iter__(self):
        return iter(self._fn())

    def __len__(self):
        return len(self._fn())

    def keys(self):
        return self._fn().keys()

    def values(self):
        return self._fn().values()

    def items(self):
        return self._fn().items()


SLOT_POSITIONS_M = _LiveMap(arena_state.slot_positions)
HOME_WALL_MARKER = _LiveMap(lambda: arena_state.current().home_wall_marker)

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

# v7 §1.4.3 capture rule: the attacker must hover over the target box for ≥ 2 s
# within the 1-2 m RFID band for the flip to register. 3.5 s gives a solid
# margin over the 2 s floor — the descent into the band (from cruise altitude)
# eats the first ~1 s of the HOOVER, so a 2.5 s hover left only ~1.5 s of
# qualifying dwell and captures landed intermittently. 3.5 s ensures ≥ 2 s in
# the band even after the descent. Existing FC verb only.
CAPTURE_HOLD_HOVER_S: float = 3.5

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

# (Home-wall markers for RTH braking now live in ``arena_state`` and are
# exposed above as the live ``HOME_WALL_MARKER`` proxy — derived per arena as
# the back-wall marker for red / front-wall marker for blue, at flight-camera
# height. real + gvz both resolve to red=13 / blue=9.)

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
# safely close in. Wall stop is 5.0 m: the home wall is at |y|=10 and the
# home zone is |y| in [5, 10], so braking 5 m short of the wall puts the
# drone at |y|≈5 — just BARELY inside the home zone, NOT deep against the
# wall. Operator: "they only need to be slightly in the home zone, 5 m from
# the wall is fine" (was 3.0 m, which parked them deep at |y|≈7 and risked
# clipping the back wall on overshoot).
FB_BRAKE_TARGET_STOP_M: float = 1.5
FB_BRAKE_WALL_STOP_M: float = 5.0
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

# ── Vision-based APPROACH standoffs (no absolute positions) ────────────────
# Capture move (operator-specified technique): APPROACH the box's face marker
# to ATTACK_STANDOFF_M (~1 m), RISE to the capture altitude (clearing the 0.73 m
# box top), then FB_IMU forward ~(standoff - 0.10 m) to sit exactly OVER the box
# centre, then hover. APPROACH homes purely on vision (no absolute position);
# FB_IMU is a closed-loop relative move (also no absolute position). The drone
# is never lower than the box top until it's positioned over the box.
ATTACK_STANDOFF_M: float = 1.0
# How far to fly forward after rising, to end up over the box: the marker was
# ATTACK_STANDOFF_M ahead, minus 10 cm so we stop just shy of the far edge and
# sit over the box centre (operator: "the distance the marker was away - 10 cm").
OVER_BOX_FORWARD_M: float = ATTACK_STANDOFF_M - 0.10
# Capture altitude over the box: the RFID detects in the 1-2 m band; the
# operator wants 1-2 m, "better 1.5 m max". 1.5 m sits mid-band — a full 0.77 m
# above the 0.73 m open box top and 0.5 m below the 2 m ceiling — so the box
# reliably registers the drone with margin on both sides.
CAPTURE_ALT_M: float = 1.5
# Minimum safe altitude anywhere inside a home zone: the target boxes are 0.73 m
# tall (open), so we must always be above that to avoid clipping a box. We
# transit/capture well above this; this is the floor for any in-zone HEIGHT.
BOX_CLEARANCE_ALT_M: float = 1.0
# RTH: APPROACH the home back-wall marker and stop this far short of it (the
# operator asked for ~3-4 m). 3.5 m -> the wall is at |y|=10, home zone is
# |y| in [5,10], so braking 3.5 m short parks the drone at |y|≈6.5 — comfortably
# inside home (clears the §1.4.4 home-presence gate) and well off the wall.
RTH_WALL_STANDOFF_M: float = 3.5


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
    # This drone's DISTINCT cruise altitude (team-keyed grid, >= 0.30 m from
    # every other drone of both teams) for the out-and-back transit. Floored
    # above the box tops; falls back to a safe default if not deconflicted.
    cruise_alt = max(1.4, float(ctx.cruise_alt_m)) if ctx.cruise_alt_m else 1.6
    height_step = f"HEIGHT {cruise_alt:.2f}"

    # End-of-run: hover briefly at home (confirms v7 §1.4.4 presence) then the
    # script ENDS — no LAND, stay airborne for the next dispatch.
    home_rearm = f"HOOVER {HOME_REARM_HOVER_S:.1f}"

    # RTH leg: VISION-HOME on the home-wall marker (APPROACH), not a world goto.
    # The drone can't know its absolute position in the live arena, so it visually
    # acquires the wall marker and closes to a standoff — stopping shallow inside
    # home, never overshooting into the wall.
    wall_entry = HOME_WALL_MARKER.get(ctx.our_team)
    if wall_entry:
        wall_marker_id, _w_xy = wall_entry
        rth_lines = (
            height_step,                       # climb to our transit altitude
            f"YAW {enemy_heading_deg(ctx.our_team) + 180:g}",  # face home wall
            # APPROACH homes on the wall marker by vision and stops
            # RTH_WALL_STANDOFF_M short of it (~5 m -> just inside home).
            f"APPROACH {wall_marker_id} {RTH_WALL_STANDOFF_M:.2f}",
            f"YAW {enemy_heading_deg(ctx.our_team):g}",        # re-face enemy
            home_rearm,
        )
    else:
        rth_lines = (home_rearm,)

    # ── VISION-BASED ATTACK ────────────────────────────────────────────────
    # The whole capture is driven by APPROACH <marker_id> <distance>: the FC
    # visually acquires the marker (rotating to search if needed) and closes to
    # the standoff — NO absolute coordinates, so it works in the live arena
    # where the drone only knows what ArUco markers it sees. It physically
    # cannot overshoot into a wall: it stops at a standoff from a marker it can
    # see. (The old script used TO <x,y> / FB_RC <duration> / world-coord
    # brakes, all of which assume a known absolute position the drone doesn't
    # have — that's what drove drones into walls on bad position estimates.)
    return _format_script(
        "TAKEOFF",
        # Face the enemy side first (absolute YAW) so the camera looks toward
        # the target boxes — APPROACH then acquires the marker straight ahead
        # instead of having to spin a full search.
        f"YAW {enemy_heading_deg(ctx.our_team):g}",
        # Climb to the transit altitude (clears the 0.73 m boxes + keeps the
        # camera level so it can see the target marker across the arena).
        height_step,
        # CAPTURE TECHNIQUE (operator-specified):
        # 1) VISION-HOME on the box's face marker and stop ~1 m short of it.
        #    The FC searches (rotates) for the marker if not yet in view.
        f"APPROACH {int(attack_marker_id)} {ATTACK_STANDOFF_M:.2f}",
        # 2) RISE to the capture altitude (1-1.5 m) — above the 0.73 m box top —
        #    BEFORE moving over the box, so we never clip it.
        f"HEIGHT {CAPTURE_ALT_M:.2f}",
        # 3) Fly FORWARD the distance the marker was away minus 10 cm, to sit
        #    exactly over the box centre. FB_IMU is a closed-loop relative move
        #    (sensor-fused, no absolute position).
        f"FB_IMU {OVER_BOX_FORWARD_M:.2f}",
        # 4) Hover over the box for >= 2 s so the RFID flips it to our colour
        #    (v7 §1.4.3).
        f"HOOVER {CAPTURE_HOLD_HOVER_S:.1f}",
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
    """Return into our home zone and hover (no land) — used during a BANK.

    VISION-BASED: APPROACH the home-wall marker (no absolute coords), so the
    drone homes on a marker it can see and stops a standoff short of the wall —
    shallow inside the home zone, never overshooting. TAKEOFF is a no-op when
    already airborne. Brings an out attacker home so the team can secure the
    5-pt attempt (all drones home before the box is recaptured).
    """
    cruise_alt = max(1.4, float(ctx.cruise_alt_m)) if ctx.cruise_alt_m else 1.6
    wall_entry = HOME_WALL_MARKER.get(ctx.our_team)
    if wall_entry is None:
        return _format_script(
            "TAKEOFF", f"HEIGHT {cruise_alt:.2f}",
            f"YAW {enemy_heading_deg(ctx.our_team):g}",
            f"HOOVER {HOME_REARM_HOVER_S:.1f}")
    wall_marker_id, _w_xy = wall_entry
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {cruise_alt:.2f}",
        f"YAW {enemy_heading_deg(ctx.our_team) + 180:g}",   # face the home wall
        f"APPROACH {wall_marker_id} {RTH_WALL_STANDOFF_M:.2f}",  # vision-home home
        f"YAW {enemy_heading_deg(ctx.our_team):g}",         # re-face the enemy
        f"HOOVER {HOME_REARM_HOVER_S:.1f}",
    )


def _slot_home_team(slot: int) -> str:
    """v7 §1.4.2 Table 1: boxes 1-3 are RED home, 4-6 are BLUE home."""
    return "red" if 1 <= int(slot) <= 3 else "blue"


def _enemy_face_for(slot: int, our_team: str) -> int:
    enemy = "blue" if our_team == "red" else "red"
    return face_id(slot, enemy)


def _recapture_script(ctx: RoleContext, slot: int) -> str:
    """Re-capture one of OUR OWN home boxes that the enemy flipped.

    VISION-BASED, like the attack: the enemy currently holds this box, so its
    visible face shows the ENEMY's marker for that slot. We APPROACH that marker
    (the FC searches for it by rotating, then homes — no absolute coords needed),
    descend into the RFID band, and hover to flip it back to our colour. Then
    re-acquire the home-wall marker to settle shallow inside home, facing the
    enemy, ready to attack/defend again. Never lands.

    0 pts (home recapture, regs §1.4.3) but it stops the enemy bleeding us and
    is required to ever reach the all-6 instant win.
    """
    cruise_alt = max(1.4, float(ctx.cruise_alt_m)) if ctx.cruise_alt_m else 1.6
    # The box shows the ENEMY face now (they hold it) -> APPROACH that marker.
    enemy_face = _enemy_face_for(int(slot), ctx.our_team)
    wall_entry = HOME_WALL_MARKER.get(ctx.our_team)
    settle = ([f"YAW {enemy_heading_deg(ctx.our_team) + 180:g}",
               f"APPROACH {wall_entry[0]} {RTH_WALL_STANDOFF_M:.2f}",
               f"YAW {enemy_heading_deg(ctx.our_team):g}"]
              if wall_entry else [])
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {cruise_alt:.2f}",
        # Same vision capture technique as the attack:
        # 1) VISION-HOME on the (enemy-coloured) face of our flipped box, ~1 m.
        f"APPROACH {enemy_face} {ATTACK_STANDOFF_M:.2f}",
        # 2) RISE to capture altitude (above the 0.73 m box) before moving over.
        f"HEIGHT {CAPTURE_ALT_M:.2f}",
        # 3) Fly forward over the box centre (marker distance - 10 cm).
        f"FB_IMU {OVER_BOX_FORWARD_M:.2f}",
        # 4) Hover to flip the box back to our colour.
        f"HOOVER {CAPTURE_HOLD_HOVER_S:.1f}",
        # Climb, re-acquire the home wall, settle shallow inside home.
        f"HEIGHT {cruise_alt:.2f}",
        *settle,
        f"HOOVER {HOME_REARM_HOVER_S:.1f}",
    )


def _slot_is_ours(ctx: RoleContext, slot: int) -> bool:
    # DECAYED belief (same as the planner): only treat the slot as ours if a
    # drone has actually SEEN it ours recently. A stale "we hold it" must not
    # cancel a capture run — the enemy can re-capture the instant we leave and
    # we can't see the slot from home. Once the belief decays to "unknown" the
    # attacker flies back out to re-check / re-capture it (continuous play).
    return ctx.markers.effective_holder(slot) == ctx.our_team


def _own_box_under_threat(ctx: RoleContext) -> "int | None":
    """Nearest-by-x enemy-held OWN slot that needs recapturing, or None.

    Same detection the defender uses (RAW last-seen holder; skip the 5-s
    post-capture lock) so a returning/home attacker can SELF-convert to a
    defender the instant it sees one of our boxes flipped — without waiting for
    the planner's idle-gated assignment. Slots a teammate is already
    recapturing this tick (``ctx.peer_recapture_slots``) are excluded so two
    drones never pile onto one box.
    """
    import time as _t
    now = _t.time()
    our = ctx.our_team
    enemy = "blue" if our == "red" else "red"
    px = ctx.state.world_position_m[0] if ctx.state.world_position_m else 0.0
    threats = []
    for sl in ctx.active_slots:
        if _slot_home_team(sl) != our:
            continue                       # only OUR home boxes
        if sl in ctx.peer_recapture_slots:
            continue                       # a teammate is already on it
        if ctx.markers.slot_holder(sl) != enemy:
            continue                       # not enemy-held
        if ctx.markers.slot_locked(sl, now):
            continue                       # in the 5-s post-capture lock
        sx = SLOT_POSITIONS_M.get(sl, (0.0, 0.0))[0]
        threats.append((abs(sx - px), sl))
    if not threats:
        return None
    threats.sort()
    return threats[0][1]                    # nearest by x


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

        # ---- RE-CAPTURE: the planner pulled this attacker to defend one of
        # OUR OWN flipped boxes (an "attacker becomes defender" run). It's a
        # 0-pt home flip but stops the enemy scoring, so it BEATS both the bank
        # recall and a fresh enemy attack. Detect it by the target being one of
        # OUR home slots; fly the heading-agnostic recapture, then the attacker
        # reverts to normal attacking once the target clears.
        slot0 = rs.target_slot
        if slot0 is not None and _slot_home_team(slot0) == ctx.our_team:
            if rs.phase == "running":
                if _slot_is_ours(ctx, slot0) or ctx.state.phase in ("init", "done", ""):
                    rs.target_slot = None
                    rs.target_assigned_unix_s = None
                    rs.advance_phase("done", f"slot {slot0} recaptured / run done")
                    return noop("attacker: recapture complete")
                return noop(f"attacker: recapturing own slot {slot0}")
            if _slot_is_ours(ctx, slot0):
                rs.target_slot = None
                rs.advance_phase("done", f"slot {slot0} already ours")
                return noop(f"attacker: own slot {slot0} already ours")
            if ctx.state.phase not in ("init", "done", ""):
                return noop(f"attacker: busy; will recapture own slot {slot0} once idle")
            return push(
                _recapture_script(ctx, slot0),
                new_phase="running",
                reason=f"attacker->defender: re-capture own slot {slot0}",
            )

        # ---- SELF-DEFEND: a RETURNING / HOME attacker that sees one of OUR
        # boxes flipped to the enemy converts to a defender NOW — it breaks off,
        # recaptures the box, and (via the branch above, next tick) reverts to
        # attacking once the box is ours again. We detect this OURSELVES every
        # tick: the planner's defend assignment is idle-gated and would wait for
        # our return script to finish. Guards:
        #  * only when we're home or heading home (the operator's "when the
        #    attackers return to our home zone" condition) AND physically on our
        #    own side — never abort a COMMITTED attacker in the enemy half, which
        #    must finish its capture;
        #  * peer_recapture_slots dedups vs the defender / other attackers so two
        #    drones never chase the same box.
        if slot0 is None or _slot_home_team(slot0) != ctx.our_team:
            returning = ctx.in_home_now or rs.phase == "returning_home"
            on_our_side = True
            pos = ctx.state.world_position_m
            if pos is not None:
                y = float(pos[1])
                on_our_side = (y <= 0.5) if ctx.our_team == "red" else (y >= -0.5)
            if returning and on_our_side:
                threat = _own_box_under_threat(ctx)
                if threat is not None:
                    # CLAIM the slot first (even on the break-off path) so the
                    # runner records it in recap_claims and the remaining drones
                    # this tick dedup against it — otherwise two returning
                    # attackers both break off for the same box. With the claim
                    # set, the recapture itself is then driven by the
                    # planner-assigned-recapture branch above on the next idle
                    # tick (target is now one of our own slots).
                    rs.target_slot = threat
                    rs.target_assigned_unix_s = None
                    # Break off the current run first (the FC only starts a new
                    # mission from idle); next tick the branch above pushes it.
                    if ctx.state.phase not in ("init", "done", ""):
                        return stop_cmd(
                            f"attacker->defender: own slot {threat} flipped — "
                            f"break off to recapture NOW")
                    return push(
                        _recapture_script(ctx, threat),
                        new_phase="running",
                        reason=f"attacker->defender: re-capture own slot {threat}",
                    )

        # ---- BANK: we just captured an enemy box — get home to secure the
        # 5-pt attempt (ALL drones must return home before the box is
        # recaptured). BUT do NOT abort an attacker that's already COMMITTED —
        # i.e. already in the enemy half (y past the neutral midline toward the
        # enemy). Yanking a committed drone home wastes a near-complete capture
        # AND it can't make it home in time anyway; letting it finish lands the
        # capture, then its own RTH brings it home. Only drones still on our
        # side / in neutral are recalled (they're close, so the all-home bank
        # completes fast). This is the key thrash fix: previously every bank
        # aborted all 3 attackers mid-flight so none ever reached the boxes.
        if ctx.team_phase == "bank":
            pos = ctx.state.world_position_m
            enemy_side = False
            if pos is not None:
                y = float(pos[1])
                # red attacks +Y, blue attacks -Y; "committed" = past midline.
                enemy_side = (y > 0.5) if ctx.our_team == "red" else (y < -0.5)
            if enemy_side and rs.phase == "running":
                # Committed: let the in-flight capture + its RTH finish.
                return noop("attacker: bank — committed in enemy half, "
                            "finishing capture then RTH")
            rs.target_slot = None        # not committed -> abandon attack intent
            rs.target_assigned_unix_s = None
            if ctx.in_home_now:
                if rs.phase not in ("returning_home", "done"):
                    rs.advance_phase("done", "bank — home, holding (secured)")
                return noop("attacker: bank — home, holding (secured)")
            return push(
                _return_home_script(ctx),
                new_phase="returning_home",
                reason="attacker: bank — return home to secure the attempt",
            )

        slot = rs.target_slot

        # ---- idle: waiting for a target slot ------------------------------
        if rs.phase in ("", "idle", "done"):
            if slot is None:
                # No target assigned. Per the spec the attacker must NEVER just
                # sit there. The §1.4.4 home gate means we can only LAUNCH a new
                # attack from home, so the productive thing to do while waiting
                # for the planner to hand us an enemy slot is to be AT HOME,
                # re-armed and ready to launch the instant one frees. If we're
                # not home (just finished a run out in neutral/enemy), fly home
                # now; if we're home, hold there ready. This continuous
                # home<->attack cycle keeps every attacker always either
                # attacking or repositioning to attack — never idle-in-place.
                if not ctx.in_home_now and ctx.state.phase in ("init", "done", ""):
                    return push(
                        _return_home_script(ctx),
                        new_phase="returning_home",
                        reason="attacker: no target — return home, ready to strike",
                    )
                return noop("attacker: home, ready — waiting for an enemy slot")
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

        # ---- returning_home: flying back to re-arm (bank or idle-out) ------
        if rs.phase == "returning_home":
            # Done the moment we're detected home (ready for the next attack)
            # OR the homeward script ends. Reset to idle so the next tick can
            # immediately launch a fresh attack on an assigned target.
            if ctx.in_home_now or ctx.state.phase in ("init", "done", ""):
                rs.advance_phase("idle", "home — re-armed, ready to attack")
                return noop("attacker: home, re-armed")
            return noop(f"attacker: returning home (fc phase={ctx.state.phase})")

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
