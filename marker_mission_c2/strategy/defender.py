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
    _recapture_script, _slot_is_ours, HOME_REARM_HOVER_S,
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


def _wait_script(ctx: RoleContext) -> str:
    """Fly to the current waiting point and hold there (no land).

    Existing FC verbs only (TAKEOFF / HEIGHT / TO / HOOVER). TAKEOFF is a
    no-op when already airborne. The script ends with a brief hover so it
    reaches phase=done — at which the FC holds position (keeps hovering) AND
    the drone is free for the planner to re-dispatch on a re-capture. We never
    emit LAND: the defender stays airborne the whole match.
    """
    cruise_alt = max(1.5, float(ctx.cruise_alt_m or ctx.drone.attack_alt_m or 1.6))
    wx, wy = _wait_xy(ctx)
    lines = [
        "TAKEOFF",
        f"HEIGHT {cruise_alt:.2f}",
        f"TO {wx:g} {wy:g}",
        f"HEIGHT {cruise_alt:.2f}",
        f"HOOVER {HOME_REARM_HOVER_S:.1f}",
    ]
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

        # ---- BANK: the team just captured an enemy box and is securing the
        # 5-pt attempt — EVERY drone must get home, the defender included. Its
        # waiting point during bank is already home (_wait_xy), so just send it
        # there now; abandon any recapture intent until we're banking again.
        if ctx.team_phase == "bank":
            rs.target_slot = None
            rs.target_assigned_unix_s = None
            if ctx.in_home_now:
                if rs.phase != "waiting":
                    rs.advance_phase("waiting", "bank — home, holding (secured)")
                return noop("defender: bank — home, holding (secured)")
            return push(
                _wait_script(ctx),       # _wait_xy -> home during bank
                new_phase="waiting",
                reason="defender: bank — return home to secure the attempt",
            )

        slot = rs.target_slot

        # ---- idle: waiting for a threatened own slot ----------------------
        if rs.phase in ("", "idle", "done"):
            if slot is None:
                # No assigned threat — WAIT, airborne, at the phase's waiting
                # point: neutral zone on SORTIE (poised to recapture, and OUT
                # of home so the team's 5-pt condition holds), or our home zone
                # on BANK (returning with the team to secure the attempt). If
                # already parked there, do nothing (FC keeps hovering at
                # phase=done). The FC starts a script only from INIT/done.
                where = "home zone" if ctx.team_phase == "bank" else "neutral zone"
                if _at_wait(ctx):
                    return noop(f"defender: waiting in {where} ({ctx.team_phase})")
                if ctx.state.phase not in ("init", "done", ""):
                    return noop(
                        f"defender: airborne (fc phase={ctx.state.phase}); "
                        f"holding — FC must be INIT/done to reposition to wait"
                    )
                return push(
                    _wait_script(ctx),
                    new_phase="waiting",
                    reason=f"defender: hold in {where} ({ctx.team_phase})",
                )
            # Only defend OUR slots; ignore a mis-assigned enemy slot.
            if slot not in ctx.active_slots:
                return noop(f"defender: slot {slot} not active")
            if _slot_is_ours(ctx, slot):
                rs.advance_phase("done", f"slot {slot} already ours — no action")
                return noop(f"defender: slot {slot} already ours")
            # The FC starts a script only when idle (phase init/done) — i.e.
            # parked at the neutral waiting point (or mid-reposition having just
            # finished). If it's mid-run (e.g. still flying to the wait point),
            # hold until that ends, THEN launch the recapture.
            if ctx.state.phase not in ("init", "done", ""):
                return noop(
                    f"defender: busy (fc phase={ctx.state.phase}); "
                    f"will launch re-capture once idle"
                )
            # NOTE: unlike the attacker, we deliberately do NOT gate on
            # in_home_now here. The defender waits in the NEUTRAL zone, and a
            # recapture of one of OUR slots is a 0-pt home defence (not an
            # enemy-zone scoring attempt that §1.4.4 guards against camping) —
            # the drone flies INTO its home zone to do it. Gating on home
            # presence would strand the neutral-waiting defender and it could
            # never recapture, defeating the whole role.
            # Enemy holds our slot -> re-capture its currently-showing
            # (enemy) face with the attacker's primitive script.
            recap_id = _enemy_face_for(slot, ctx.our_team)
            rs.last_attack_marker_id = recap_id
            return push(
                _full_attack_script(ctx, recap_id, slot),
                new_phase="running",
                reason=f"defender: re-capture slot {slot} (id={recap_id})",
            )

        # ---- running: wait for the script to fully finish -----------------
        if rs.phase == "running":
            if slot is not None and _slot_is_ours(ctx, slot):
                # Re-captured (a scout confirmed our face) — let the script
                # finish its RTH on its own (ends hovering in home, no land);
                # reset intent so the idle branch returns us to the neutral
                # waiting point.
                rs.target_slot = None
                rs.target_assigned_unix_s = None
                rs.advance_phase("done", f"slot {slot} re-captured")
                return noop("defender: slot re-captured, returning to wait")
            if ctx.state.phase in ("init", "done", ""):
                rs.target_slot = None
                rs.target_assigned_unix_s = None
                rs.advance_phase("done", "re-capture run complete (FC back to init)")
                return noop("defender: run complete")
            return noop(f"defender: running (fc phase={ctx.state.phase})")

        rs.advance_phase("idle", f"unknown phase {rs.phase}")
        return noop("defender: reset to idle")


register(DefenderRole())
