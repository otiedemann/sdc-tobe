"""DEFENDER role.

A defender protects OUR target slots. When it has no active duty it WAITS,
airborne, in the neutral zone near our home boundary — poised to dash to any
of our home slots — and the instant one of our slots is flipped to the enemy's
colour it launches a re-capture run on that slot.

Design choices:

* **No new FC maneuvers.** Re-capturing our own slot is mechanically
  identical to attacking an enemy slot — APPROACH the box's *currently-showing*
  face (which is the enemy's face, since the enemy holds it), rise above the
  box top, move over it, hover to flip it back to our colour, then return on
  the back-wall marker. So the defender reuses the attacker's vision-based
  capture script builder (``TAKEOFF / HEIGHT / YAW / APPROACH / FB_IMU /
  HOOVER``) verbatim — just pointed at one of *our* slots.

* **Hovers on-station in the neutral zone (never camps, never patrols).** With
  no threat it holds at a single fixed point just inside the neutral zone on
  our side (x=0, |y|=4.5), FACING THE BACK WALL and positioned by a vision
  ``APPROACH`` on our back-wall marker — no absolute ``TO`` (so no GPS-style
  position) and no left-right sweep. The neutral zone is empty airspace — NOT
  above any box — so this never trips the regs §1.3 "dead duck over a target"
  rule, and |y|=4.5 keeps the defender just OUTSIDE our home zone (the 5-pt
  "all drones out" condition holds) yet as close as legally possible to our
  home slots for minimum recapture latency.

* **Recaptures the instant a box flips.** Every tick the defender checks the
  marker tracker itself (it does NOT wait for the planner); the moment one of
  our boxes shows the enemy colour it breaks the hold and dashes in to
  recapture that exact slot — see :meth:`DefenderRole.decide` step (1). Target
  selection: the operator (MANUAL) or the AUTO planner may also set
  ``role_state.target_slot``, but the per-tick self-detection is what makes the
  response immediate. Once the slot flips back to us (or the script finishes)
  it resets to idle and returns to its neutral-zone station.

Phases:
    idle/holding → no threat; hovering on-station (neutral, facing back wall)
    recapturing  → re-capture script pushed; drone executing the sequence
    done         → script ended; reset to idle (-> returns to neutral station)
"""
from __future__ import annotations

import logging

from .roles import (
    Decision, Role, RoleContext, noop, push, stop_cmd, register, home_park_xy,
    enemy_heading_deg,
)
# Reuse the attacker's primitive-composed capture script + helpers so the
# defender adds NO new flight maneuvers of its own.
from .attacker import (
    _recapture_script, _slot_is_ours, HOME_REARM_HOVER_S, _format_script,
    SLOT_POSITIONS_M, HOME_WALL_MARKER,
)

logger = logging.getLogger(__name__)

# Where an idle defender waits when on SORTIE: x=0 (centred so the dash to any
# of our 3 home slots is symmetric), |y| = 4.5 m. The neutral zone is
# y∈[-5, +5]; 4.5 m is 0.5 m into the neutral zone from our home boundary
# (|y|=5) — the closest legal neutral-zone point to home, AND crucially OUTSIDE
# our home zone so the defender satisfies the 5-pt "all drones out" condition
# while the attackers are scoring.
NEUTRAL_WAIT_Y_M: float = 4.5
# How close (m, horizontal) the defender must be to its waiting point to count
# as "already waiting" (so we don't re-push the hold script every tick).
_WAIT_ARRIVE_M: float = 1.5
# Outward nudge (m, body-relative) the defender uses to leave the home zone
# before it settles on the back-wall marker. APPROACH can only pull the drone
# IN toward the wall, never push it out, so after a recapture (which ends deep
# in home, ~|y|=6.5) we first fly this far toward neutral — facing the enemy —
# then APPROACH the back-wall marker from the neutral side. 4 m clears the
# home boundary (|y|=5) from the post-recapture standoff with margin.
DEFENDER_EXIT_LEG_M: float = 4.0
# Dwell (s) the defender holds on-station per pushed hold script (re-pushed when
# it lapses or the drone drifts off-station). A threat breaks the hold instantly
# via the recapture path (stop_cmd at the top of decide), so a long dwell never
# delays a recapture — it only keeps the mission/command log clean by avoiding
# redundant re-pushes while the defender sits idle in neutral.
DEFENDER_HOLD_HOVER_S: float = 30.0


def _slot_home_team(slot: int) -> str:
    """v7 §1.4.2 Table 1: boxes 1-3 RED home, 4-6 BLUE home."""
    return "red" if 1 <= slot <= 3 else "blue"


def _threatened_own_slot(ctx: RoleContext) -> int | None:
    """The nearest-by-x enemy-held OWN slot that needs recapturing, or None.

    The defender detects threats ITSELF every tick from the marker tracker —
    it does NOT wait for the planner to assign a target (the planner can only
    assign when the FC is idle, but our patrol keeps the FC 'running', so a
    planner-only path would never fire — the exact bug where the defender
    patrolled while our boxes stayed enemy-held). Uses the RAW last-seen holder
    so a box stays "threatened" until a drone actually sees it flipped back.
    Skips slots inside the 5-s post-capture lock (can't recapture yet).
    """
    import time as _t
    now = _t.time()
    our = ctx.our_team
    px = ctx.state.world_position_m[0] if ctx.state.world_position_m else 0.0
    threats = []
    for sl in ctx.active_slots:
        if _slot_home_team(sl) != our:
            continue                       # only OUR home boxes
        if sl in ctx.peer_recapture_slots:
            continue                       # a returning attacker is already on it
        if ctx.markers.slot_holder(sl) != ("blue" if our == "red" else "red"):
            continue                       # not enemy-held
        if ctx.markers.slot_locked(sl, now):
            continue                       # in post-capture lock
        sx = SLOT_POSITIONS_M.get(sl, (0.0, 0.0))[0]
        threats.append((abs(sx - px), sl))
    if not threats:
        return None
    threats.sort()
    return threats[0][1]                    # nearest by x


def _wait_xy(ctx: RoleContext) -> tuple[float, float]:
    """The defender's waiting point for the current team phase.

    sortie -> neutral zone (outside home), poised to recapture — and outside
              home so the team's 5-pt "all drones out" condition holds.
    bank   -> our home zone, returning with the team to secure the 5-pt attempt.
    """
    if ctx.team_phase == "bank":
        return ctx.home_park_xy or home_park_xy(ctx.our_team)
    y = -NEUTRAL_WAIT_Y_M if ctx.our_team == "red" else NEUTRAL_WAIT_Y_M
    return (0.0, y)


def _on_station_sortie(ctx: RoleContext) -> bool:
    """True if the defender is already holding its sortie station.

    The vision ``APPROACH`` on the back-wall marker pins the drone to a fixed
    *standoff arc* around the marker, NOT to an exact (0, |y|) point — its x is
    left wherever the last recapture ended. So "on station" is judged by depth/
    arc, not an exact xy: airborne, in the neutral zone (just OUTSIDE our home,
    so the 5-pt "all out" condition holds), and ~``standoff`` from the marker.
    Judging by the arc (not a point) is what lets the hover settle and stop
    re-pushing every tick when it parks slightly off-centre.
    """
    pos = ctx.state.world_position_m
    if not pos or len(pos) < 3 or float(pos[2]) <= 0.5:
        return False
    cur_y = float(pos[1])
    in_neutral = cur_y > -5.0 if ctx.our_team == "red" else cur_y < 5.0
    if not in_neutral:
        return False
    _mid, (mx, my) = HOME_WALL_MARKER[ctx.our_team]
    standoff = abs(my) - NEUTRAL_WAIT_Y_M
    d = ((float(pos[0]) - mx) ** 2 + (cur_y - my) ** 2) ** 0.5
    return abs(d - standoff) <= _WAIT_ARRIVE_M


def _is_inward_of_wait(ctx: RoleContext) -> bool:
    """True if the drone is DEEPER into our home than the neutral station.

    Happens right after a recapture (the recapture run ends settled near our
    home wall). APPROACH on the back-wall marker can only pull us further IN,
    so in that case the hold script must first nudge OUT toward neutral with a
    relative forward move before APPROACHing the marker from the neutral side.
    Uses the C2's own position estimate purely to PICK the script — the actual
    motion is still vision/relative, never an absolute TO.
    """
    pos = ctx.state.world_position_m
    if not pos or len(pos) < 2:
        return False
    cur_y = float(pos[1])
    wy = -NEUTRAL_WAIT_Y_M if ctx.our_team == "red" else NEUTRAL_WAIT_Y_M
    margin = 0.5
    return cur_y < wy - margin if ctx.our_team == "red" else cur_y > wy + margin


def _wait_script(ctx: RoleContext, *, need_exit: bool = False) -> str:
    """Defender holding pattern (no land, stays airborne all match).

    BANK: fly home and hover (return with the team to secure the 5-pt attempt).

    SORTIE: HOVER at a single fixed station just inside the neutral zone
    (x=0, |y|=``NEUTRAL_WAIT_Y_M``), FACING THE BACK WALL, positioned by a
    vision ``APPROACH`` on our back-wall marker — NO left-right patrol and NO
    absolute ``TO``, so the hold is drift-free and needs no GPS-style position.
    The instant one of our boxes is flipped the decide() recapture path breaks
    this hold and dashes in. ``need_exit`` prepends an outward nudge for when
    we're currently deeper in home than the station (APPROACH alone can't push
    us back out toward neutral). FC verbs only (TAKEOFF / HEIGHT / YAW_IMU /
    FB_IMU / APPROACH / HOOVER); TAKEOFF is a no-op when already airborne.
    """
    cruise_alt = max(1.5, float(ctx.cruise_alt_m or ctx.drone.attack_alt_m or 1.6))

    # BANK: single point at home, hover (handled by _wait_xy -> home).
    if ctx.team_phase == "bank":
        wx, wy = _wait_xy(ctx)
        return _format_script(
            "TAKEOFF",
            f"HEIGHT {cruise_alt:.2f}",
            f"TO {wx:g} {wy:g}",
            f"HEIGHT {cruise_alt:.2f}",
            f"HOOVER {HOME_REARM_HOVER_S:.1f}",
        )

    # SORTIE: hover on-station, facing the back wall, set by a vision APPROACH
    # on the back-wall marker. standoff = (wall depth) - (station depth) so the
    # APPROACH stops us exactly at |y|=NEUTRAL_WAIT_Y_M (0.5 m into neutral).
    home_marker_id, (_mx, my) = HOME_WALL_MARKER[ctx.our_team]
    standoff = abs(my) - NEUTRAL_WAIT_Y_M               # 10 - 4.5 = 5.5 m
    lines = ["TAKEOFF", f"HEIGHT {cruise_alt:.2f}"]
    if need_exit:
        # We're inward of the station (e.g. just recaptured). Come OUT toward
        # neutral by moving FORWARD while facing the enemy (+Y). The FC has no
        # absolute-heading verb, only the relative YAW_IMU — and a fixed 180 deg
        # flip would inherit any off-axis heading left by the prior APPROACH and
        # drift the FB_IMU sideways into a wall. So compute the EXACT relative
        # turn from the live arena heading (this turn runs right after TAKEOFF,
        # so no rotation intervenes). Fall back to the tracked-facing 180 flip
        # if no heading telemetry is available.
        want = enemy_heading_deg(ctx.our_team)
        cur = ctx.state.heading_deg
        if cur is not None:
            delta = ((want - float(cur) + 180.0) % 360.0) - 180.0   # [-180,180]
            if abs(delta) > 3.0:
                lines.append(f"YAW_IMU {delta:.0f}")
        elif ctx.role_state.facing != "enemy":
            lines.append("YAW_IMU 180")
        lines.append(f"FB_IMU {DEFENDER_EXIT_LEG_M:.2f}")
    lines += [
        # APPROACH rotates to find the back-wall marker itself and ends facing
        # it, so no explicit "face the wall" step is needed (and the FC has no
        # absolute-heading verb). standoff lands us at |y|=NEUTRAL_WAIT_Y_M.
        f"APPROACH {home_marker_id} {standoff:.2f}",         # settle on-station
        f"HOOVER {DEFENDER_HOLD_HOVER_S:.1f}",               # hold, facing wall
    ]
    return _format_script(*lines)


class DefenderRole(Role):
    name = "defender"

    def decide(self, ctx: RoleContext) -> Decision:
        rs = ctx.role_state
        drone = ctx.drone

        if not drone.enabled:
            return noop("drone disabled")
        if not drone.team:
            return noop("defender: no team assigned")
        if not ctx.state.drone_connected:
            return noop("defender: drone not connected")

        # ---- (1) RE-CAPTURE — top priority, IMMEDIATE, beats everything -----
        # Detect the threat OURSELVES every tick (don't wait for the planner —
        # our patrol keeps the FC 'running' so the planner's idle-gate would
        # never assign us). If one of our boxes is enemy-held, recapture it NOW.
        threat = _threatened_own_slot(ctx)

        if rs.phase == "recapturing":
            # Mid-recapture run. Finished when the slot reads ours again, or the
            # FC script ended -> drop back to patrol (re-evaluated next tick).
            tgt = rs.target_slot
            if tgt is None or _slot_is_ours(ctx, tgt) or \
                    ctx.state.phase in ("init", "done", ""):
                rs.target_slot = None
                rs.advance_phase("idle", "recapture done -> back to patrol")
                return noop("defender: recapture complete")
            return noop(f"defender: recapturing slot {tgt} "
                        f"(fc phase={ctx.state.phase})")

        if threat is not None:
            # We want to recapture `threat`. The FC can only START a mission
            # from idle (init/done) on real hardware (a start while running is
            # 409-rejected). If we're mid-patrol/bank, STOP first (1 tick),
            # which returns the FC to idle; next tick we push the recapture.
            if ctx.state.phase not in ("init", "done", ""):
                return stop_cmd(f"defender: own slot {threat} flipped — "
                                f"break off to recapture NOW")
            rs.target_slot = threat
            dec = push(
                _recapture_script(ctx, threat),
                new_phase="recapturing",
                reason=f"defender: re-capture own slot {threat} (enemy-held)",
            )
            rs.facing = "enemy"   # recapture ends with a 180 deg turn re-facing enemy
            return dec

        # No threat -> clear any stale recapture intent.
        rs.target_slot = None

        # ---- (2) BANK, no threat — return home to secure the 5-pt attempt --
        if ctx.team_phase == "bank":
            if ctx.in_home_now:
                if rs.phase != "waiting":
                    rs.advance_phase("waiting", "bank — home, holding (secured)")
                return noop("defender: bank — home, holding (secured)")
            if rs.phase == "banking_home" and ctx.state.phase not in (
                    "init", "done", ""):
                return noop("defender: bank — en route home")
            dec = push(
                _wait_script(ctx),       # _wait_xy -> home during bank
                new_phase="banking_home",
                reason="defender: bank — return home to secure the attempt",
            )
            rs.facing = "home"       # banked home, roughly facing our back wall
            return dec

        # ---- (3) SORTIE, no threat — HOLD a fixed neutral-zone station ------
        # Hover just inside the neutral zone, FACING THE BACK WALL, positioned
        # by a vision APPROACH on the back-wall marker (no patrol, no TO). Hold
        # while the script runs and we're on-station; re-push only once it
        # lapses or we've drifted off (a threat is handled above, immediately).
        if (rs.phase == "holding"
                and ctx.state.phase not in ("init", "done", "")
                and _on_station_sortie(ctx)):
            return noop("defender: holding station in neutral, facing back wall")
        dec = push(
            _wait_script(ctx, need_exit=_is_inward_of_wait(ctx)),
            new_phase="holding",
            reason="defender: hold neutral station (APPROACH back-wall marker)",
        )
        rs.facing = "home"           # hold ends facing the back wall
        return dec


register(DefenderRole())
