"""SCOUT role.

Flight pattern:
    TAKEOFF
    HEIGHT <scout_alt_m>
    SCOUT
    HOOVER <scout_hover_s>

We deliberately skip a precision ``TO`` step here. The marker_mission
``TO`` step requires sub-5cm horizontal positioning and 1s of settling
before it'll advance, which is unrealistic when the only visible markers
are far-wall ones. The drone would then hang at step 2/4 forever and
never reach SCOUT.

Instead we use ``HEIGHT`` (uses the on-board altimeter, not ArUco) to
get the drone up to scout altitude, then ``SCOUT`` to do the slow 360°
yaw rotation in place. After takeoff the Anafi already auto-stabilises
in place, so this is plenty good for "rotate and observe."

The runner harvests ``visible_marker_ids`` from the C2 state stream and
feeds it to :class:`MarkerTracker`; the role just keeps the drone
airborne and rotating.
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
    alt = max(0.6, float(drone.scout_alt_m))
    hover_s = max(2.0, float(ctx.match.scout_hover_s))
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {alt:.2f}",
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
