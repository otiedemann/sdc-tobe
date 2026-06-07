"""DEFENDER role.

A defender protects OUR target slots. When it has no active duty it WAITS,
airborne, in front of a SIDE-wall marker nearest our home (12/14 for red,
10/16 for blue in the competition hall), at the inner edge of our home zone —
watching our home boxes — and the instant one of them is flipped to the enemy's
colour (red: faces 31/32/33 visible; blue: 44/45/46) it launches a re-capture
run on that box using the SAME maneuver the attackers use on enemy boxes.

Design choices:

* **No new FC maneuvers.** Re-capturing our own box is mechanically identical
  to attacking an enemy box — APPROACH the box's *currently-showing* face
  (which is the enemy's face, since the enemy holds it), slide up and over it
  to flip it back to our colour, then GO_HOME. So the defender reuses the
  attacker's vision-based capture script builder (``_recapture_script``)
  verbatim — just pointed at one of *our* slots.

* **Waits in front of a side-wall marker (never camps over a box, never
  patrols).** With no threat it holds ~2.5 m in front of a side marker nearest
  our home, positioned by a vision ``GO_HOME`` (no absolute ``TO``) which HOLDS
  altitude (a positioning hop, so it does not climb to the 2 m marker). The
  side markers are DERIVED per arena (``arena_state.side_markers``), so this is
  correct in both the competition hall (red 12/14, blue 10/16) and GVZ (swapped)
  with no code change.

* **Recaptures the instant a box flips.** Every tick the defender checks the
  marker tracker itself (it does NOT wait for the planner); the moment one of
  our boxes shows the enemy colour it breaks the hold and dashes in to
  recapture that exact slot — see :meth:`DefenderRole.decide` step (1). The
  marker tracker is fed by ALL our drones (incl. the scout), so the defender
  detects the flip even without seeing the box itself. Once the slot flips back
  to us (or the script finishes) it resets to idle and returns to its station.

Phases:
    idle/holding → no threat; hovering in front of a side marker
    recapturing  → re-capture script pushed; drone executing the sequence
    done         → script ended; reset to idle (-> returns to side-marker station)
"""
from __future__ import annotations

import logging

from . import arena_state
from .roles import (
    Decision, Role, RoleContext, noop, push, stop_cmd, register, home_park_xy,
)
# Reuse the attacker's primitive-composed capture script + helpers so the
# defender adds NO new flight maneuvers of its own.
from .attacker import (
    _recapture_script, _slot_is_ours, HOME_REARM_HOVER_S, _format_script,
    SLOT_POSITIONS_M, HOME_WALL_MARKER,
)

logger = logging.getLogger(__name__)

# SORTIE wait station: hover in front of a SIDE-wall marker nearest our home
# (derived per arena -> red 12/14, blue 10/16 in the competition hall), just at
# the inner edge of our home zone, watching our home boxes. GO_HOME homes on the
# marker by vision (no absolute TO) and HOLDS altitude (positioning), so the
# defender does not climb to the 2 m marker. The instant one of our boxes flips
# to the enemy colour the decide() recapture path breaks this hold and runs the
# attacker's capture maneuver on it.
DEFENDER_SIDE_STANDOFF_M: float = 2.5   # "in front of" the side marker
DEFENDER_SIDE_TOL_M: float = 0.5        # loose arrival band

# Bank-park depth: |y| of the defender's home hold during a BANK recall, 0.5 m
# INTO neutral from the inner home boundary (used only by the bank fallback).
NEUTRAL_WAIT_Y_M: float = 4.5
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


def _pick_side_marker(ctx: RoleContext) -> int | None:
    """Which side-wall marker (12/14 red, 10/16 blue in the competition hall) to
    GO_HOME onto this cycle. Prefer the one already in use if still in view
    (stable), else any visible configured side marker, else the previous/first
    one — GO_HOME rotates to find it regardless. Returns None only if the arena
    has no side markers (then the caller falls back to the back-wall hold)."""
    cands = arena_state.side_markers(ctx.our_team)
    if not cands:
        return None
    vis: set[int] = set()
    for mid in ctx.state.visible_marker_ids:
        try:
            vis.add(int(mid))
        except (TypeError, ValueError):
            pass
    last = ctx.role_state.scratch.get("side_marker")
    if last in vis and last in cands:
        return last
    for m in cands:
        if m in vis:
            return m
    return last if last in cands else cands[0]


def _wait_script(ctx: RoleContext, *, side_marker: "int | None" = None) -> str:
    """Defender holding pattern (no land, stays airborne all match).

    BANK: fly home and hover (return with the team to secure the 5-pt attempt).

    SORTIE: hover IN FRONT OF a SIDE-wall marker nearest our home (12/14 red,
    10/16 blue in the competition hall), at the inner edge of our home zone,
    watching our home boxes. Positioned by a vision ``GO_HOME`` on the side
    marker — NO absolute ``TO`` — which HOLDS altitude (it's a positioning hop,
    so the defender does NOT climb to the 2 m marker). The instant one of our
    boxes flips to the enemy colour the decide() recapture path breaks this hold
    and runs the attacker's capture maneuver on it. GO_HOME rotates to find the
    marker itself, so no pre-orientation is needed; FC verbs only (TAKEOFF /
    HEIGHT / GO_HOME / HOOVER); TAKEOFF is a no-op when already airborne.
    """
    cruise_alt = max(1.5, float(ctx.cruise_alt_m or ctx.drone.attack_alt_m or 1.6))

    # BANK: come home and hover. Vision-home INTO the home zone with GO_HOME on
    # our back-wall marker (NO absolute TO) — the FC rotates to find the marker
    # and stops a shallow standoff short of it, landing the defender just inside
    # home. standoff = wall depth - park depth.
    if ctx.team_phase == "bank":
        _wx, wy = _wait_xy(ctx)
        home_marker_id, (_mx, my) = HOME_WALL_MARKER[ctx.our_team]
        standoff = max(0.5, abs(my) - abs(wy))          # 10 - 5.5 = 4.5 m
        return _format_script(
            "TAKEOFF",
            f"HEIGHT {cruise_alt:.2f}",
            f"GO_HOME {home_marker_id} {standoff:.2f} 0.5",
            f"HOOVER {HOME_REARM_HOVER_S:.1f}",
        )

    # SORTIE: hover in front of a side-wall marker, watching our home boxes.
    if side_marker is None:
        side_marker = _pick_side_marker(ctx)
    if side_marker is None:
        # No side markers in this arena — fall back to a shallow back-wall hold.
        home_marker_id, (_mx, my) = HOME_WALL_MARKER[ctx.our_team]
        standoff = max(0.5, abs(my) - 4.5)
        return _format_script(
            "TAKEOFF", f"HEIGHT {cruise_alt:.2f}",
            f"GO_HOME {home_marker_id} {standoff:.2f} 0.5",
            f"HOOVER {DEFENDER_HOLD_HOVER_S:.1f}",
        )
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {cruise_alt:.2f}",
        # GO_HOME (loose, holds altitude) onto the side marker -> hover ~2.5 m in
        # front of it at the inner edge of our home zone. Rotates to find it.
        f"GO_HOME {int(side_marker)} {DEFENDER_SIDE_STANDOFF_M:.2f} "
        f"{DEFENDER_SIDE_TOL_M:g}",
        f"HOOVER {DEFENDER_HOLD_HOVER_S:.1f}",
    )


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

        # ---- (3) SORTIE, no threat — HOLD in front of a side-wall marker ----
        # Hover in front of a side-wall marker nearest our home (12/14 red,
        # 10/16 blue), at the inner edge of our home zone, watching our boxes.
        # GO_HOME (no TO) homes on the marker by vision and holds altitude. Hold
        # while the script runs; re-push only once it lapses (the HOOVER dwell
        # ends and the FC goes idle) so we re-centre periodically. A threat is
        # handled above, immediately (stop_cmd -> recapture).
        if (rs.phase == "holding"
                and ctx.state.phase not in ("init", "done", "")):
            return noop("defender: holding in front of side marker "
                        f"{rs.scratch.get('side_marker')}, watching our boxes")
        marker = _pick_side_marker(ctx)
        rs.scratch["side_marker"] = marker
        dec = push(
            _wait_script(ctx, side_marker=marker),
            new_phase="holding",
            reason=(f"defender: hold in front of side marker {marker}"
                    if marker is not None
                    else "defender: hold (back-wall fallback)"),
        )
        rs.facing = "enemy"          # GO_HOME ends facing the side marker
        return dec


register(DefenderRole())
