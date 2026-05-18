"""SCOUT role.

Flight pattern:
    TAKEOFF
    TO  <neutral.x> <neutral.y> <scout_alt_m>
    SCOUT
    HOOVER <scout_hover_s>

The ``SCOUT`` DSL verb performs a slow 360° yaw rotation in place — fast
enough to finish in seconds, slow enough that the ArUco detector can lock
onto each marker as it sweeps past. We then hover briefly so the marker
tracker has time to capture a fresh observation before the script ends and
we re-push.

The runner is responsible for harvesting ``visible_marker_ids`` from the C2
state stream and feeding it to :class:`MarkerTracker`; the role itself just
keeps the drone airborne and rotating.
"""
from __future__ import annotations

import logging
from typing import Optional

from .roles import Decision, Role, RoleContext, noop, push, register

logger = logging.getLogger(__name__)


def _format_script(*lines: str) -> str:
    return "\n".join(line for line in lines if line) + "\n"


def _compose_scout_script(ctx: RoleContext) -> Optional[str]:
    drone = ctx.drone
    if not drone.team:
        return None
    neutral = (
        ctx.match.neutral_red if drone.team == "red" else ctx.match.neutral_blue
    )
    alt = max(0.6, float(drone.scout_alt_m))
    hover_s = max(2.0, float(ctx.match.scout_hover_s))
    return _format_script(
        "TAKEOFF",
        f"TO {neutral.x:.2f} {neutral.y:.2f} {alt:.2f}",
        "SCOUT",
        f"HOOVER {hover_s:.0f}",
    )


class ScoutRole(Role):
    name = "scout"

    def decide(self, ctx: RoleContext) -> Decision:
        if not ctx.drone.enabled:
            return noop("drone disabled")
        if not ctx.drone.team:
            return noop("scout: no team assigned")
        if not ctx.state.drone_connected:
            return noop("scout: drone not connected")

        script = _compose_scout_script(ctx)
        if script is None:
            return noop("scout: cannot compose script")

        # Re-push when:
        #   (a) we've never pushed for this role, or
        #   (b) the FC reports phase=init (script finished) and enough time
        #       has passed since the last push to avoid hammering.
        import time

        now = time.time()
        rs = ctx.role_state
        never_pushed = not rs.last_pushed_script
        fc_finished = ctx.state.phase in ("init", "idle", "landed")
        long_enough = (now - rs.last_pushed_unix_s) >= 2.0

        if never_pushed:
            return push(script, new_phase="scouting", reason="scout: initial push")
        if fc_finished and long_enough:
            return push(script, new_phase="scouting", reason="scout: loop re-push")
        return noop(f"scout: airborne in phase={ctx.state.phase}")


register(ScoutRole())
