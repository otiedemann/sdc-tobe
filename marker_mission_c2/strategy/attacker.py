"""ATTACKER role.

Operator assigns a target *slot* (1..6). The role decides which ArUco
face to approach based on the slot's current holder, then runs the
canonical capture script:

    TAKEOFF
    APPROACH <enemy-face-id> <approach_distance_m>
    HEIGHT <capture_ascend_m>
    FB_IMU <capture_forward_m>
    HOOVER <capture_hover_s>

After the script ends, we wait for a scout to confirm the slot now shows
*our* face (i.e. ``markers.slot_holder(slot) == our_team``). Falls
through to RTH after ``VERIFY_TIMEOUT_S`` so we never sit forever.

Phase machine:

    idle    → no target assigned, or target already captured by us
    attack  → script pushed, waiting for FC to finish
    verify  → script done, waiting for scout confirmation
    rth     → returning to home
    done    → mission complete; reset to idle

If the operator assigns a slot we already hold (holder == our_team) the
role skips straight to ``done`` — no need to attack.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from .roles import Decision, Role, RoleContext, noop, push, register, stop_cmd
from .settings import face_id

logger = logging.getLogger(__name__)


VERIFY_TIMEOUT_S = 30.0


def _format_script(*lines: str) -> str:
    return "\n".join(line for line in lines if line) + "\n"


def _attack_script(ctx: RoleContext, attack_marker_id: int) -> str:
    """Approach the *visible enemy face*, ascend, drift over, hover."""
    m = ctx.match
    approach_d = max(0.2, float(m.approach_distance_m))
    ascend = max(0.6, float(m.capture_ascend_m))
    forward = max(0.1, float(m.capture_forward_m))
    hover_s = max(2.0, float(m.capture_hover_s))
    return _format_script(
        "TAKEOFF",
        f"APPROACH {int(attack_marker_id)} {approach_d:.2f}",
        f"HEIGHT {ascend:.2f}",
        f"FB_IMU {forward:.2f}",
        f"HOOVER {hover_s:.0f}",
    )


def _rth_script(ctx: RoleContext) -> str:
    drone = ctx.drone
    home = ctx.match.home_red if drone.team == "red" else ctx.match.home_blue
    alt = max(0.6, float(home.alt))
    return _format_script(
        f"TO {home.x:.2f} {home.y:.2f} {alt:.2f}",
        "HOOVER 1",
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
            attack_id = _enemy_face_for(slot, ctx.our_team)
            rs.last_attack_marker_id = attack_id
            return push(
                _attack_script(ctx, attack_id),
                new_phase="attack",
                reason=f"attacker: attack slot {slot} (id={attack_id})",
            )

        # ---- attack: waiting for script to finish -------------------------
        if rs.phase == "attack":
            if slot is None:
                return stop_cmd("attacker: target cleared mid-attack")
            if ctx.state.phase in ("init", "idle", "landed"):
                rs.advance_phase("verify", "attack script finished")
                return noop("attacker: awaiting scout verification")
            return noop(f"attacker: attacking (fc phase={ctx.state.phase})")

        # ---- verify: scout confirms our face is up ------------------------
        if rs.phase == "verify":
            if slot is None:
                return stop_cmd("attacker: target cleared during verify")
            if _slot_is_ours(ctx, slot):
                return push(
                    _rth_script(ctx),
                    new_phase="rth",
                    reason=f"attacker: slot {slot} captured (our face up), RTH",
                )
            elapsed = time.time() - rs.phase_started_unix_s
            if elapsed > VERIFY_TIMEOUT_S:
                return push(
                    _rth_script(ctx),
                    new_phase="rth",
                    reason=f"attacker: verify timeout on slot {slot}, RTH anyway",
                )
            return noop(
                f"attacker: verify slot {slot} (waiting for scout, t+{elapsed:.0f}s)"
            )

        # ---- rth: returning to home ---------------------------------------
        if rs.phase == "rth":
            if ctx.state.phase in ("init", "idle", "landed"):
                rs.target_slot = None
                rs.target_assigned_unix_s = None
                rs.advance_phase("done", "rth complete, target cleared")
                return noop("attacker: home, target cleared")
            return noop(f"attacker: RTH (fc phase={ctx.state.phase})")

        # Unknown phase — reset to idle.
        rs.advance_phase("idle", f"unknown phase {rs.phase}")
        return noop("attacker: reset to idle")


register(AttackerRole())
