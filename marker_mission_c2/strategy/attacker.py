"""ATTACKER role.

Phases:
    idle      → waiting for the operator to assign a target marker.
    attack    → script pushed: TAKEOFF, APPROACH <id> <approach_distance>,
                HEIGHT <capture_ascend>, FB_IMU <capture_forward>, HOOVER
                <capture_hover_s>. The drone ends up hovering over the
                target.
    verify    → script finished, waiting for the marker tracker to flag
                ``captured=True`` (no scout has seen the marker for
                CAPTURE_COOLDOWN_S, see :mod:`.markers`). Times out after
                ``VERIFY_TIMEOUT_S`` so the drone doesn't sit indefinitely
                if no scout is up.
    rth       → script pushed: TO <home.x> <home.y> <home_alt> (fast
                cruise). Waits for the FC to finish.
    done      → mission complete; falls back to idle so the operator can
                assign the next target.

Targets are assigned by the operator via the web UI. The runner stamps
``role_state.target_marker_id`` and the role drives everything from there.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from .roles import Decision, Role, RoleContext, noop, push, register, stop_cmd

logger = logging.getLogger(__name__)


# How long we'll wait in `verify` for a scout to confirm the capture before
# giving up and returning to home anyway. Operator can always re-assign.
VERIFY_TIMEOUT_S = 30.0


def _format_script(*lines: str) -> str:
    return "\n".join(line for line in lines if line) + "\n"


def _attack_script(ctx: RoleContext, target_id: int) -> str:
    """Approach the target, ascend, drift forward over it, then hover."""
    m = ctx.match
    approach_d = max(0.2, float(m.approach_distance_m))
    ascend = max(0.6, float(m.capture_ascend_m))
    forward = max(0.1, float(m.capture_forward_m))
    hover_s = max(2.0, float(m.capture_hover_s))
    return _format_script(
        "TAKEOFF",
        f"APPROACH {int(target_id)} {approach_d:.2f}",
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


def _is_target_captured(ctx: RoleContext, target_id: int) -> bool:
    """The marker tracker says the target has been knocked over."""
    status = ctx.markers.status_for(target_id)
    return bool(status and status.captured)


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

        target_id = rs.target_marker_id

        # ---- idle: waiting for a target -----------------------------------
        if rs.phase in ("", "idle", "done"):
            if target_id is None:
                return noop("attacker: idle (no target)")
            script = _attack_script(ctx, target_id)
            return push(
                script,
                new_phase="attack",
                reason=f"attacker: attack target {target_id}",
            )

        # ---- attack: waiting for script to finish -------------------------
        if rs.phase == "attack":
            if target_id is None:
                # Operator pulled the target. Cancel mission and reset.
                return stop_cmd("attacker: target cleared mid-attack")
            # FC has finished the attack script (back to init/idle/landed)?
            if ctx.state.phase in ("init", "idle", "landed"):
                rs.advance_phase("verify", "attack script finished")
                return noop("attacker: awaiting capture verification")
            return noop(f"attacker: attacking (fc phase={ctx.state.phase})")

        # ---- verify: scout confirms capture -------------------------------
        if rs.phase == "verify":
            if target_id is None:
                return stop_cmd("attacker: target cleared during verify")
            if _is_target_captured(ctx, target_id):
                rth = _rth_script(ctx)
                return push(
                    rth,
                    new_phase="rth",
                    reason=f"attacker: target {target_id} captured, RTH",
                )
            elapsed = time.time() - rs.phase_started_unix_s
            if elapsed > VERIFY_TIMEOUT_S:
                rth = _rth_script(ctx)
                return push(
                    rth,
                    new_phase="rth",
                    reason=f"attacker: verify timeout, RTH anyway",
                )
            return noop(
                f"attacker: verify (waiting for scout, t+{elapsed:.0f}s)"
            )

        # ---- rth: returning to home ---------------------------------------
        if rs.phase == "rth":
            if ctx.state.phase in ("init", "idle", "landed"):
                rs.target_marker_id = None
                rs.target_assigned_unix_s = None
                rs.advance_phase("done", "rth complete, target cleared")
                return noop("attacker: home, target cleared")
            return noop(f"attacker: RTH (fc phase={ctx.state.phase})")

        # Unknown phase — reset to idle.
        rs.advance_phase("idle", f"unknown phase {rs.phase}")
        return noop("attacker: reset to idle")


register(AttackerRole())
