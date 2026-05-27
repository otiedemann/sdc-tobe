"""ANCHOR role — defensive recapture of captured home-zone slots.

The anchor drone responds when the enemy captures one of OUR home-zone
boxes. It immediately flies to that box, performs a brief fly-over to
trigger the mechanical recapture, then returns to its home zone to
complete the RTH cycle.

Regulation notes (§ = SDC Regulations v5.0.0 / "v7"):
  §1.4.3  Home-zone recapture:
            "A home-zone box changes back to your team's color IMMEDIATELY
            when recaptured."  The capture is triggered when our drone
            hovers over the box for ≥ MIN_CAPTURE_HOVER_S (2 s).
  §1.4.3  Home-zone recapture scores ZERO game points:
            "you receive 0 game points. This rule prevents defensive game
            point farming." → the anchor role is purely defensive/strategic;
            it prevents the enemy from banking points on our home boxes.
  §1.3 / §1.4  "Sitting dead duck" (continuous passive hovering OVER boxes)
            is EXPLICITLY PROHIBITED and leads to disqualification.
            The anchor uses AUTO_ATTACK (a brief fly-over pass), NOT a
            continuous hover. The drone must land after each recapture.

Script shape (delegated to _auto_attack_script from attacker.py)::

    TAKEOFF
    HEIGHT   <attack_alt_m>
    ... (fly to box, brief hover for RFID detection, RTH, land)

The script is identical to an attack run because the flight geometry —
fly to box, hover ≥ 2 s, return home — is the same.

Concretely:
  - If the enemy has captured slot S in our zone, the box shows the
    ENEMY face (e.g. 3x for blue when we are red).
  - We fly over it, hover long enough for RFID detection → box flips
    back to OUR colour immediately (§1.4.3 home-zone rule).
  - We then RTH so the next scoring cycle can begin.

The coordinator (MatchStateMachine) assigns the target slot via
runner.assign_target() — identical API to the attacker.

Phases: idle → running → done  (same as AttackerRole).
"""
from __future__ import annotations

import logging

from .attacker import (
    HOME_WALL_MARKER,
    SLOT_POSITIONS_M,
    _auto_attack_script,
    _enemy_face_for,
)
from .roles import Decision, Role, RoleContext, noop, push, register

logger = logging.getLogger(__name__)


class AnchorRole(Role):
    """Defensive role: recaptures our own home-zone boxes when the enemy
    flips them.  The coordinator assigns target slots via
    runner.assign_target() exactly as it does for attackers.
    """

    name = "anchor"

    def decide(self, ctx: RoleContext) -> Decision:
        rs = ctx.role_state
        drone = ctx.drone

        if not drone.enabled:
            return noop("anchor: drone disabled")
        if not drone.team:
            return noop("anchor: no team assigned")
        if not ctx.state.drone_connected:
            return noop("anchor: drone not connected")

        slot = rs.target_slot

        # ── idle: waiting for coordinator to assign a recapture slot ─
        if rs.phase in ("", "idle", "done"):
            if slot is None:
                return noop("anchor: idle (no recapture target)")
            # Check if the slot still needs recapturing
            # (coordinator may have already handled it, or it was never
            # actually stolen — guard against spurious assignments).
            if ctx.markers.slot_holder(slot) == ctx.our_team:
                # Already ours — clear and stay idle.
                rs.advance_phase(
                    "done",
                    f"anchor: slot {slot} already ours — skipping",
                )
                return noop(f"anchor: slot {slot} already ours")

            attack_id = _enemy_face_for(slot, ctx.our_team)
            rs.last_attack_marker_id = attack_id
            script = _auto_attack_script(ctx, attack_id, slot)
            return push(
                script,
                new_phase="running",
                reason=f"anchor: recapture slot {slot} (id={attack_id})",
            )

        # ── running: wait for the script to finish ────────────────────
        if rs.phase == "running":
            if slot is None:
                rs.advance_phase("idle", "anchor: target cleared mid-flight")
                return noop("anchor: target cleared mid-flight")
            if ctx.state.phase in ("init", "done", ""):
                rs.target_slot = None
                rs.target_assigned_unix_s = None
                rs.advance_phase(
                    "done",
                    f"anchor: recapture of slot {slot} complete",
                )
                return noop("anchor: recapture complete, back to idle")
            return noop(f"anchor: running (fc phase={ctx.state.phase})")

        rs.advance_phase("idle", f"anchor: unknown phase {rs.phase}")
        return noop("anchor: reset to idle")


register(AnchorRole())
