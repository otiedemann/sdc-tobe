"""SCOUT role.

Flight pattern (one long script per push, avoids the land/takeoff yo-yo):

    TAKEOFF
    HEIGHT <scout_alt_m>
    SCOUT
    HOOVER <inter_scout_hover_s>
    SCOUT
    HOOVER <inter_scout_hover_s>
    ...  (repeated SCOUT_CYCLES times)

Why a single long script instead of TAKEOFF + SCOUT + HOOVER repeated?
marker_mission auto-lands at end-of-script (see controller._advance_script
"safety LAND"), so re-pushing a short script after every rotation makes
the drone land + takeoff between scans. Instead we push one ~10 minute
mission of continuous SCOUT/HOOVER cycles. The Anafi battery dies before
the script does, and the strategy re-pushes a fresh script after the
drone is back on the ground.

We also intentionally drop the precision ``TO`` step. marker_mission's
``TO`` requires sub-5cm horizontal precision + 1s settle, which is
unrealistic from far-wall ArUco only. The drone would hang at step 2/N
forever and never reach SCOUT. ``HEIGHT`` uses the on-board altimeter
(no ArUco needed) so it always settles.

Per-FC config dependency: the FC's ``default_height_m`` tune (climb-to
altitude after TAKEOFF) should be set LOWER than ``scout_alt_m`` —
otherwise the drone climbs to that altitude first, then descends to
``scout_alt_m`` via the HEIGHT step (visible as an up-then-down). Set
default_height_m ≈ 1.0 m on each FC via /api/tune/apply +
/api/tune/save (or directly edit ``~/.marker_mission/config.json``).

The runner harvests ``visible_marker_ids`` from the C2 state stream and
feeds it to :class:`MarkerTracker`; the role just keeps the drone
airborne and rotating.
"""
from __future__ import annotations

import logging
from typing import Optional

from .roles import Decision, Role, RoleContext, noop, push, register

logger = logging.getLogger(__name__)


# Number of SCOUT/HOOVER pairs to chain in a single push. Each SCOUT
# step is roughly SCOUT_MAX_DRIVE_S (~30s) of slow yaw, then we
# HOOVER scout_hover_s seconds between rotations so the marker tracker
# can settle and the position estimator can re-acquire if it briefly
# lost the active marker. 30 cycles × (30 + 2) s = ~16 minutes;
# battery dies first.
SCOUT_CYCLES = 30


def _format_script(*lines: str) -> str:
    return "\n".join(line for line in lines if line) + "\n"


def _compose_scout_script(ctx: RoleContext) -> Optional[str]:
    drone = ctx.drone
    if not drone.team:
        return None
    alt = max(0.6, float(drone.scout_alt_m))
    inter_s = max(1.0, float(ctx.match.scout_hover_s))
    lines = ["TAKEOFF", f"HEIGHT {alt:.2f}"]
    for _ in range(SCOUT_CYCLES):
        lines.append("SCOUT")
        lines.append(f"HOOVER {inter_s:.0f}")
    return _format_script(*lines)


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
