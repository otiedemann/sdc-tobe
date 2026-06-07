"""ATTACKER role.

Operator assigns a target *slot* (1..6). The role pushes ONE complete
mission script — the operator's canonical 5-step attack:

    TAKEOFF
    APPROACH <enemy-face-id> 1     # vision-home onto the box face, stop ~1 m short
    FB_UD_IMU 0.9 0.5              # ONE combined move: slide 0.9 m forward over the
                                   #   box WHILE rising 0.5 m — clears the 0.73 m top
                                   #   and parks in the 1-2 m RFID band → box flips
    YAW_IMU 180                    # turn to face home
    GO_HOME <wall-marker>          # vision-home onto the home back-wall marker,
                                   #   settle shallow inside the home zone

The drone vision-homes onto the enemy box face (APPROACH rotates to find
it — no absolute coordinates), slides up and over it in a single combined
FB_UD_IMU move (rising as it advances, so it clears the open box top and
sits in the RFID band to flip the box), turns around, and GO_HOMEs onto
the home back-wall marker. The script ENDS at GO_HOME, so the FC safety-
lands the drone inside its own home zone (satisfying the v7 §1.4.4 home-
presence gate); the next dispatch re-launches it with TAKEOFF.

OPERATOR CONVENTION
-------------------
The capture move assumes the drone is placed at takeoff with its front
camera pointing toward the enemy side (red drones face arena +y, blue
drones face -y) so the box face is in front. APPROACH will rotate to
search if the marker isn't already in view, and the FC has no absolute-
heading verb anyway (only the relative YAW_IMU), so no compass-based
alignment is attempted — operator-set takeoff orientation is simple and
reliable.

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
# to ATTACK_STANDOFF_M, then slide forward ~(standoff - 0.10 m) over the box
# WHILE rising (FB_UD_IMU). APPROACH homes purely on vision (no absolute
# position) and vision-aligns to the marker's height.
ATTACK_STANDOFF_M: float = 1.5
# Arrival distance tolerance (m) for the capture APPROACH: settle within ±this
# of ATTACK_STANDOFF_M instead of the tight ~5 cm deadband, so the stop is fast
# and not fussy (operator: "1.5 m with 20 cm deadband"). Still a capture (the
# approach climbs to the marker's height).
ATTACK_DIST_TOL_M: float = 0.20
# How far to fly forward after rising, to end up over the box: the marker was
# ATTACK_STANDOFF_M ahead, minus 10 cm so we stop just shy of the far edge and
# sit over the box centre. At a 1.5 m standoff this is 1.4 m (operator).
OVER_BOX_FORWARD_M: float = ATTACK_STANDOFF_M - 0.10
# How far to RISE while sliding forward over the box. FB_UD_IMU does the climb
# and the forward slide in ONE combined moveBy (rises WHILE advancing), so the
# drone is already above the 0.73 m box top by the time it's over the centre —
# no separate HEIGHT step, no clip. 0.5 m over the ~0.9 m takeoff/approach
# height lands it ~1.4 m, inside the 1-2 m RFID capture band. Operator default
# (FB_UD_IMU 0.9 0.5).
CAPTURE_RISE_M: float = 0.5
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
    """Operator's canonical attack: APPROACH the box, slide-up over it to flip,
    turn around, and GO_HOME onto the home back-wall marker.

    Default script (operator-specified, e.g. attacking face 31, home wall 13):

        TAKEOFF
        APPROACH 31 1                 # vision-home onto the box face, stop ~1 m short
        FB_UD_IMU 0.9 0.5             # ONE combined move: slide 0.9 m forward over the
                                      #   box WHILE rising 0.5 m — clears the 0.73 m top
                                      #   and parks in the 1-2 m RFID band → box flips
        YAW_IMU 180                   # turn to face home
        GO_HOME 13                    # vision-home onto the home back-wall marker,
                                      #   settle shallow inside the home zone

    This replaces the old 10-line form (HEIGHT / APPROACH / HEIGHT / FB_IMU /
    HOOVER / HEIGHT / APPROACH / YAW / HOOVER). The combined ``FB_UD_IMU`` does
    the rise + forward slide in a single closed-loop ``moveBy`` (rises WHILE
    advancing), so there is no separate HEIGHT/HOOVER and no risk of clipping the
    box on the way over it.

    Everything is VISION-BASED (APPROACH / GO_HOME home on ArUco markers, no
    absolute coordinates), so it works in the live arena where the drone only
    knows what markers it sees and can never open-loop into a wall.

    The script ENDS at GO_HOME: the FC's safety-land then sets the drone down
    inside its own home zone (satisfying the v7 §1.4.4 home-presence gate), and
    the next dispatch begins with TAKEOFF. ``marker_id`` and the home-wall marker
    are arena-derived, so this is identical in the real and GVZ arenas (face
    31-36/41-46, wall 13 red / 9 blue).
    """
    # Each attacker approaches the home-wall marker from a DIFFERENT bearing
    # (3 attackers -> -45 / 0 / +45 deg relative to the marker normal) so they
    # fan out to distinct points around it instead of converging on one standoff
    # and colliding. GO_HOME's 4th arg is that arrival heading (per-drone,
    # editable in the C2). 0 = head-on.
    rth_hdg = (ctx.rth_approach_angle_deg
               if ctx.rth_approach_angle_deg is not None else 0.0)

    # RTH leg: VISION-HOME on the home-wall marker (GO_HOME), not a world goto.
    # The drone can't know its absolute position in the live arena, so it visually
    # acquires the wall marker and closes to a standoff — stopping shallow inside
    # home, never overshooting into the wall. GO_HOME rotates to FIND the marker.
    wall_entry = HOME_WALL_MARKER.get(ctx.our_team)
    if wall_entry:
        wall_marker_id, _w_xy = wall_entry
        home_line = (f"GO_HOME {wall_marker_id} {RTH_WALL_STANDOFF_M:.2f} "
                     f"0.5 {rth_hdg:g}")
    else:
        # No wall marker mapped (shouldn't happen) — just hover briefly so the
        # script still ends cleanly in place rather than mid-capture.
        home_line = f"HOOVER {HOME_REARM_HOVER_S:.1f}"

    return _format_script(
        "TAKEOFF",
        # The operator orients the drone facing the enemy before take-off, and
        # APPROACH re-acquires the box marker by rotating if it isn't already in
        # view — so no explicit heading step is needed (the FC has no
        # absolute-heading verb anyway, only the relative YAW_IMU).
        # 1) VISION-HOME on the box's face marker and stop ~1.5 m short of it,
        #    settling within ±20 cm (fast, not fussy; still climbs to the marker).
        f"APPROACH {int(attack_marker_id)} {ATTACK_STANDOFF_M:.2f} "
        f"{ATTACK_DIST_TOL_M:.2f}",
        # 2) ONE combined move: slide forward over the box centre (marker was
        #    ~1.5 m ahead, minus 10 cm = 1.4 m) WHILE rising into the RFID band —
        #    rises as it advances, so it clears the 0.73 m box top without a
        #    separate HEIGHT step and flips the box to our colour.
        f"FB_UD_IMU {OVER_BOX_FORWARD_M:.2f} {CAPTURE_RISE_M:.2f}",
        # 3) Turn to face home for the return leg.
        "YAW_IMU 180",
        # 4) Vision-home onto the home back-wall marker; settle shallow inside
        #    home. The script ends here -> FC safety-lands in the home zone.
        home_line,
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
    # COLLISION AVOIDANCE on the way home (operator): fly the cruise at this
    # drone's DISTINCT deconflicted altitude so concurrently-returning attackers
    # are vertically separated; floored at box clearance, falls back to the flat
    # approach altitude for a single drone.
    cruise_alt = max(BOX_CLEARANCE_ALT_M, float(ctx.cruise_alt_m or
                                                ctx.drone.attack_alt_m or 1.0))
    # Per-drone arrival bearing so several attackers fan out around the wall
    # marker instead of converging (3 attackers -> -45 / 0 / +45 deg).
    rth_hdg = (ctx.rth_approach_angle_deg
               if ctx.rth_approach_angle_deg is not None else 0.0)
    wall_entry = HOME_WALL_MARKER.get(ctx.our_team)
    if wall_entry is None:
        return _format_script(
            "TAKEOFF", f"HEIGHT {cruise_alt:.2f}",
            f"HOOVER {HOME_REARM_HOVER_S:.1f}")
    wall_marker_id, _w_xy = wall_entry
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {cruise_alt:.2f}",
        # Vision-home onto the home-wall marker (GO_HOME rotates to find it) at
        # the per-drone arrival bearing, then a relative 180 deg turn to re-face
        # the enemy. No absolute YAW / no absolute coords.
        f"GO_HOME {wall_marker_id} {RTH_WALL_STANDOFF_M:.2f} 0.5 {rth_hdg:g}",
        "YAW_IMU 180",                                      # re-face the enemy
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
    # The box shows the ENEMY face now (they hold it) -> APPROACH that marker.
    enemy_face = _enemy_face_for(int(slot), ctx.our_team)
    wall_entry = HOME_WALL_MARKER.get(ctx.our_team)
    rth_hdg = (ctx.rth_approach_angle_deg
               if ctx.rth_approach_angle_deg is not None else 0.0)
    # After flipping our box back, turn to face home and GO_HOME onto the home
    # wall marker (it rotates to find it), settling shallow inside home. No
    # absolute-heading verb on the FC. Script ends -> safety-land at home.
    settle = (["YAW_IMU 180",
               f"GO_HOME {wall_entry[0]} {RTH_WALL_STANDOFF_M:.2f} 0.5 {rth_hdg:g}"]
              if wall_entry else [])
    return _format_script(
        "TAKEOFF",
        # Same vision capture technique as the attack:
        # 1) VISION-HOME on the (enemy-coloured) face of our flipped box, ~1.5 m
        #    (±20 cm band).
        f"APPROACH {enemy_face} {ATTACK_STANDOFF_M:.2f} {ATTACK_DIST_TOL_M:.2f}",
        # 2) ONE combined move: slide forward over the box centre WHILE rising
        #    into the RFID band — clears the 0.73 m box top and flips it back.
        f"FB_UD_IMU {OVER_BOX_FORWARD_M:.2f} {CAPTURE_RISE_M:.2f}",
        *settle,
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

        # ---- MANUAL TARGET MARKER (testing): the operator pinned an explicit
        # marker id to attack. Fly the standard attack script STRAIGHT at that
        # marker -- bypassing the slot->enemy-face mapping and all the slot-based
        # holder/recapture/bank logic below. One-shot: cleared when the run
        # finishes. Takes priority over everything else so a test always wins.
        manual_id = rs.target_marker_id
        if manual_id is not None:
            if rs.phase == "running":
                # Only "init"/"done" means the whole script (incl. the GO_HOME +
                # safety-land) has finished.
                if ctx.state.phase in ("init", "done", ""):
                    rs.target_marker_id = None
                    rs.advance_phase(
                        "done", f"manual attack on marker {manual_id} complete")
                    return noop("attacker: manual attack complete (marker cleared)")
                return noop(
                    f"attacker: manual attack on marker {manual_id} running "
                    f"(fc phase={ctx.state.phase})")
            # Not running yet: the FC can only START a mission from INIT (on the
            # ground / idle). We deliberately do NOT gate on in_home here -- a
            # test should launch from wherever the drone is placed.
            if ctx.state.phase not in ("init", "done", ""):
                return noop(
                    f"attacker: airborne (fc phase={ctx.state.phase}); "
                    f"manual attack waits for the FC to be INIT")
            rs.last_attack_marker_id = manual_id
            return push(
                _full_attack_script(ctx, manual_id, 0),
                new_phase="running",
                reason=f"attacker: MANUAL attack on marker {manual_id} (testing)",
            )

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
