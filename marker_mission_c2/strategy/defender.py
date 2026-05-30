"""DEFENDER role.

A defender protects OUR target slots. It sits idle (on the ground, no
script) until one of our slots is flipped to the enemy's colour, then it
launches a re-capture run on that slot and returns home.

Design choices:

* **No new FC maneuvers.** Re-capturing our own slot is mechanically
  identical to attacking an enemy slot — fly to the box, brake on the
  *currently-showing* face (which is the enemy's face, since the enemy
  holds it), drift over it, hover to flip it back to our colour, then
  RTH + land. So the defender reuses the attacker's primitive-composed
  script builder (``TAKEOFF / UD_RC / FB_RC / FB_BRAKE / RC / YAW_IMU /
  LAND``) verbatim — just pointed at one of *our* slots.

* **Never camps.** When no slot is threatened the defender returns a
  noop (stays grounded / holds), so it never "sits" over a box — which
  the regs penalise (DQ for a stationary "dead duck" over a target).

* **Target selection.** The operator (MANUAL) or the AUTO planner sets
  ``role_state.target_slot`` to the threatened own-slot. The defender
  only acts while that slot is held by the enemy; once it flips back to
  us (or the script finishes), it resets to idle and the target clears.

Phases mirror the attacker:
    idle    → no threat assigned, or assigned slot already ours
    running → re-capture script pushed; drone executing the sequence
    done    → script ended (FC back to init); reset to idle
"""
from __future__ import annotations

import logging

from .roles import Decision, Role, RoleContext, noop, push, register
# Reuse the attacker's primitive-composed capture script + helpers so the
# defender adds NO new flight maneuvers of its own.
from .attacker import (
    _full_attack_script, _enemy_face_for, _slot_is_ours, SLOT_POSITIONS_M,
    HOME_REARM_HOVER_S,
)

logger = logging.getLogger(__name__)


def _slot_home_team(slot: int) -> str:
    """v7 §1.4.2 Table 1: boxes 1-3 RED home, 4-6 BLUE home."""
    return "red" if 1 <= slot <= 3 else "blue"


# Anti dead-duck patrol: regs §1.3 forbid "passive sitting or hovering
# directly above the boxes". The defender pass-through dwell stays well
# below the regs' anti-camp threshold (which the org doesn't quantify but
# which we interpret as ~2 s). 0.5 s is enough for the box's RFID system
# to register a defender presence (1-2 m altitude band) and interrupt any
# in-progress enemy hover (per v7 §1.4.3 "no defending drone is detected").
_PATROL_PASS_HOLD_S: float = 0.5

# How many full triangle cycles per pushed patrol script. The match is
# 10 minutes (600 s); each cycle takes roughly (3 boxes * (transit ~2 s
# + 0.5 s pass-through)) ~= 7.5 s. 8 cycles ~ 1 min per push.
_PATROL_CYCLES: int = 8


def _patrol_script(ctx: RoleContext) -> str:
    """Generate a continuous-triangle patrol script for the home boxes.

    Uses only existing FC verbs (TAKEOFF / HEIGHT / TO / HOOVER) — the
    drone climbs, then loops the 3 own-side boxes, dipping to the capture
    altitude band briefly at each one (deliberately short so it never
    counts as parking), then ends at home with a sustained hover so the
    script never lands during the match (regs §1.3 + our own no-land rule).
    """
    our = ctx.our_team
    home_slots = [sl for sl in ctx.active_slots if _slot_home_team(sl) == our]
    # Sort so the drone always visits in the same x order — easier to
    # eyeball in the 3D view and gives a predictable cycle time.
    home_slots.sort(key=lambda sl: SLOT_POSITIONS_M.get(sl, (0.0, 0.0))[0])

    # Use this drone's DISTINCT deconflicted altitude (team-keyed grid, >=
    # 0.30 m from every other drone). Floor at 1.5 m so we clear the 1.4 m
    # box tops — NOT 2.0 m, which would pull low-rung defenders up onto a
    # shared 2.0 m band and defeat the deconfliction.
    cruise_alt = max(1.5, float(ctx.cruise_alt_m or ctx.drone.attack_alt_m or 2.2))
    detect_alt = 1.5   # middle of the 1-2 m RFID detection band

    lines: list[str] = ["TAKEOFF", f"HEIGHT {cruise_alt:.2f}"]
    for _ in range(max(1, _PATROL_CYCLES)):
        for sl in home_slots:
            x, y = SLOT_POSITIONS_M[sl]
            lines.append(f"TO {x:g} {y:g}")
            lines.append(f"HEIGHT {detect_alt:.2f}")
            lines.append(f"HOOVER {_PATROL_PASS_HOLD_S:.1f}")
            lines.append(f"HEIGHT {cruise_alt:.2f}")
    # Forward-stage at the home/neutral boundary so the defender can react
    # fast (req 6). We stay 1 m INSIDE our home zone (y = ±5 is the
    # boundary; -6 / +6 keeps the regs §1.4.4 home-presence gate True so
    # the re-capture mission can launch the instant a slot is flipped),
    # while sitting as close as legally possible to the neutral zone for
    # minimum dispatch latency.
    home_y = -6.0 if our == "red" else 6.0
    lines.append(f"TO 0 {home_y:g}")
    lines.append(f"HEIGHT {cruise_alt:.2f}")
    # Confirm home-zone presence (v7 §1.4.4) then LAND so the FC returns to
    # INIT and the planner can dispatch this defender on a re-capture the
    # instant the scout reports one of our slots flipped to the enemy. The
    # old long home hover kept the FC in a never-ending mission, so the
    # defender could never re-arm to recapture (and a free-defender was
    # never available to the planner's defend_uncap play).
    lines.append(f"HOOVER {HOME_REARM_HOVER_S:.1f}")
    lines.append("LAND")
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

        # ---- idle: waiting for a threatened own slot ----------------------
        if rs.phase in ("", "idle", "done"):
            if slot is None:
                # No assigned threat — run an anti dead-duck patrol that
                # passes through every own-slot RFID zone briefly, then
                # ends at home with a sustained hover. Same FC INIT +
                # home-presence gates as the re-capture path.
                if ctx.state.phase not in ("init", "done", ""):
                    return noop(
                        f"defender: airborne (fc phase={ctx.state.phase}); "
                        f"holding — FC must be INIT to start a patrol"
                    )
                if not ctx.in_home_now:
                    return noop(
                        "defender: not in home zone yet; cannot start patrol"
                    )
                return push(
                    _patrol_script(ctx),
                    new_phase="patrolling",
                    reason="defender: anti-camp patrol of own slots",
                )
            # Only defend OUR slots; ignore a mis-assigned enemy slot.
            if slot not in ctx.active_slots:
                return noop(f"defender: slot {slot} not active")
            if _slot_is_ours(ctx, slot):
                rs.advance_phase("done", f"slot {slot} already ours — no action")
                return noop(f"defender: slot {slot} already ours")
            # The FC starts a mission only from INIT (on the ground). Runs end
            # with a home HOVER (never land mid-match), so an airborne drone
            # can't begin a new re-capture run — hold until it's back at INIT.
            if ctx.state.phase not in ("init", "done", ""):
                return noop(
                    f"defender: airborne (fc phase={ctx.state.phase}); "
                    f"holding — FC must be INIT to start a re-capture"
                )
            # v7 §1.4.4 home-presence: drone must be detected in its own home
            # zone before starting a new scoring attempt.
            if not ctx.in_home_now:
                return noop(
                    "defender: not yet detected back in home zone "
                    "(v7 §1.4.4 requires home-zone presence before re-attempt)"
                )
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
                # finish its RTH/LAND on its own; reset our intent.
                rs.target_slot = None
                rs.target_assigned_unix_s = None
                rs.advance_phase("done", f"slot {slot} re-captured")
                return noop("defender: slot re-captured, returning")
            if ctx.state.phase in ("init", "done", ""):
                rs.target_slot = None
                rs.target_assigned_unix_s = None
                rs.advance_phase("done", "re-capture run complete (FC back to init)")
                return noop("defender: run complete")
            return noop(f"defender: running (fc phase={ctx.state.phase})")

        rs.advance_phase("idle", f"unknown phase {rs.phase}")
        return noop("defender: reset to idle")


register(DefenderRole())
