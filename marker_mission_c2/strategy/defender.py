"""DEFENDER role.

A defender protects OUR target slots. When it has no active duty it WAITS,
airborne, in the neutral zone near our home boundary — poised to dash to any
of our home slots — and the instant one of our slots is flipped to the enemy's
colour it launches a re-capture run on that slot.

Design choices:

* **No new FC maneuvers.** Re-capturing our own slot is mechanically
  identical to attacking an enemy slot — fly to the box, brake on the
  *currently-showing* face (which is the enemy's face, since the enemy
  holds it), drift over it, hover to flip it back to our colour, then
  return. So the defender reuses the attacker's primitive-composed script
  builder (``TAKEOFF / UD_RC / FB_RC / FB_BRAKE / RC / YAW_IMU``) verbatim —
  just pointed at one of *our* slots.

* **Waits in the neutral zone (never camps).** With no threat it holds at a
  fixed point just inside the neutral zone on our side (x=0, |y|=4.5), at its
  deconflicted cruise altitude. The neutral zone is empty airspace — NOT above
  any box — so this never trips the regs §1.3 "dead duck over a target" rule,
  and it puts the defender as close as legally possible to our home slots for
  minimum recapture latency. (The earlier design flew an anti-camp patrol over
  our own boxes; waiting in neutral is both cleaner — no dead-duck risk — and
  faster to respond.)

* **Target selection.** The operator (MANUAL) or the AUTO planner sets
  ``role_state.target_slot`` to the threatened own-slot. The defender
  only acts while that slot is held by the enemy; once it flips back to
  us (or the script finishes), it resets to idle and returns to its
  neutral-zone waiting point.

Phases:
    idle/waiting → no threat; holding at the neutral-zone waiting point
    running      → re-capture script pushed; drone executing the sequence
    done         → script ended; reset to idle (-> returns to neutral wait)
"""
from __future__ import annotations

import logging

from .roles import (
    Decision, Role, RoleContext, noop, push, register, home_park_xy,
)
# Reuse the attacker's primitive-composed capture script + helpers so the
# defender adds NO new flight maneuvers of its own.
from .attacker import (
    _recapture_script, _slot_is_ours, HOME_REARM_HOVER_S, _format_script,
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


def _slot_home_team(slot: int) -> str:
    """v7 §1.4.2 Table 1: boxes 1-3 RED home, 4-6 BLUE home."""
    return "red" if 1 <= slot <= 3 else "blue"


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


def _at_wait(ctx: RoleContext) -> bool:
    """True if the defender is already airborne at the current waiting point."""
    pos = ctx.state.world_position_m
    if not pos or len(pos) < 3:
        return False
    wx, wy = _wait_xy(ctx)
    dx, dy = float(pos[0]) - wx, float(pos[1]) - wy
    return (dx * dx + dy * dy) ** 0.5 <= _WAIT_ARRIVE_M and float(pos[2]) > 0.5


# Left-right patrol bounds (x) for the sortie neutral-zone sweep. The arena is
# 10 m wide (x in [-5, +5]); ±3.5 keeps the defender clear of the side walls
# while covering the full width in front of our three home boxes (x in {-3,0,3}).
_PATROL_X_MAX: float = 3.5
# How many full left-right cycles per pushed patrol script. Each leg is ~7 m at
# ~2 m/s ~= 3.5 s, so a cycle ~7 s; 40 cycles ~= a full 10-min match per push,
# but the script re-pushes whenever the FC reaches phase=done anyway.
_PATROL_CYCLES: int = 40


def _wait_script(ctx: RoleContext) -> str:
    """Defender holding pattern (no land, stays airborne all match).

    On BANK: fly home and hover (return with the team to secure the 5-pt
    attempt). On SORTIE: PATROL the neutral zone LEFT-RIGHT in front of our
    home boxes — sweeping x from -3.5 to +3.5 and back, continuously — so the
    defender is always moving and equally close to any of our three boxes the
    instant one is flipped (operator spec: "fly left to right"). Existing FC
    verbs only (TAKEOFF / HEIGHT / TO). TAKEOFF is a no-op when airborne.
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

    # SORTIE: continuous left-right patrol along the neutral-zone front line.
    y = -NEUTRAL_WAIT_Y_M if ctx.our_team == "red" else NEUTRAL_WAIT_Y_M
    lines = ["TAKEOFF", f"HEIGHT {cruise_alt:.2f}"]
    for _ in range(max(1, _PATROL_CYCLES)):
        lines.append(f"TO {_PATROL_X_MAX:g} {y:g}")     # sweep right
        lines.append(f"TO {-_PATROL_X_MAX:g} {y:g}")    # sweep left
    return "\n".join(lines) + "\n"


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

        slot = rs.target_slot

        # ---- (1) RE-CAPTURE — top priority, BEATS the bank recall ----------
        # The planner assigns a threatened own slot. Defence is a 0-pt home
        # flip but stops the enemy scoring and is needed for the all-6 win, so
        # it overrides everything else. Fly the heading-agnostic recapture
        # (world-frame TO to the box) the instant the FC is idle.
        if slot is not None and slot in ctx.active_slots:
            if rs.phase == "running":
                # Mid-recapture: done when the slot reads ours, or FC finished.
                if _slot_is_ours(ctx, slot) or ctx.state.phase in ("init", "done", ""):
                    rs.target_slot = None
                    rs.target_assigned_unix_s = None
                    rs.advance_phase("done", f"slot {slot} recaptured / run done")
                    return noop("defender: recapture complete, return to wait")
                return noop(f"defender: recapturing slot {slot} "
                            f"(fc phase={ctx.state.phase})")
            if _slot_is_ours(ctx, slot):
                rs.target_slot = None
                rs.advance_phase("done", f"slot {slot} already ours")
                return noop(f"defender: slot {slot} already ours")
            if ctx.state.phase not in ("init", "done", ""):
                return noop(f"defender: busy (fc phase={ctx.state.phase}); "
                            f"will recapture slot {slot} once idle")
            return push(
                _recapture_script(ctx, slot),
                new_phase="running",
                reason=f"defender: re-capture slot {slot} (held by enemy)",
            )

        # ---- (2) BANK, no threat — return home to secure the 5-pt attempt --
        if ctx.team_phase == "bank":
            if ctx.in_home_now and _at_wait(ctx):
                if rs.phase != "waiting":
                    rs.advance_phase("waiting", "bank — home, holding (secured)")
                return noop("defender: bank — home, holding (secured)")
            if ctx.state.phase not in ("init", "done", ""):
                return noop(f"defender: bank — en route home "
                            f"(fc phase={ctx.state.phase})")
            return push(
                _wait_script(ctx),       # _wait_xy -> home during bank
                new_phase="waiting",
                reason="defender: bank — return home to secure the attempt",
            )

        # ---- (3) SORTIE, no threat — PATROL the neutral zone left-right -----
        # The patrol script is a long left-right sweep; while the FC is running
        # it, just let it keep sweeping. Only (re)push when the FC is idle
        # (init/done) — i.e. at startup or after a patrol script fully ends.
        if ctx.state.phase not in ("init", "done", ""):
            return noop(f"defender: patrolling neutral zone "
                        f"(fc phase={ctx.state.phase})")
        return push(
            _wait_script(ctx),
            new_phase="patrolling",
            reason="defender: left-right patrol of the neutral zone",
        )


register(DefenderRole())
