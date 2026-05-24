"""SCOUT role.

Flight pattern (one continuous yaw drive, no breaks between rotations):

    TAKEOFF
    HEIGHT <scout_alt_m>
    RC 0 0 0 <scout_yaw_stick> <scout_drive_duration_s>

We deliberately pick the ``RC`` verb over chained ``SCOUT`` verbs. Each
``SCOUT`` step in marker_mission auto-brakes when 360° accumulates,
then we'd HOOVER, then the next SCOUT starts from a hover — that's the
visible "stop and restart" between rotations the operator was seeing.
``RC`` pins the yaw stick for the full duration with no intermediate
braking, so the drone yaws continuously until the timer expires.

The drone keeps rotating for ``scout_drive_duration_s`` seconds per
push (default 600 s ≈ 10 minutes), then the script ends and the FC
safety-lands. The strategy then re-pushes a fresh script after the
drone has truly returned to Phase.INIT (handled by the
``fc_finished`` gate below).

Two precision steps are deliberately skipped:

* ``TO`` (precision arena-frame goto). Requires sub-5cm horizontal +
  1s settle — unrealistic from far-wall ArUco only; the drone would
  hang at step 2/N. We just stay at the takeoff position.
* The ``SCOUT`` verb itself — see above.

``HEIGHT`` uses the on-board altimeter (no ArUco needed) so it always
settles. The FC's ``default_height_m`` tune should be set LOWER than
``scout_alt_m`` (e.g. 1.0 m) so the drone climbs cleanly through it
on takeoff rather than overshooting and descending.
"""
from __future__ import annotations

import logging
import time

from .roles import Decision, Role, RoleContext, noop, push, register

logger = logging.getLogger(__name__)


def _format_script(*lines: str) -> str:
    return "\n".join(line for line in lines if line) + "\n"


def _compose_scout_script(ctx: RoleContext) -> str | None:
    drone = ctx.drone
    if not drone.team:
        return None
    # Prefer the deconflicted cruise altitude (distinct per drone) so two
    # scouts/loiterers never share a height; fall back to the drone's own
    # scout_alt_m when deconfliction isn't available.
    alt = max(0.6, float(ctx.cruise_alt_m if ctx.cruise_alt_m else drone.scout_alt_m))
    yaw_stick = int(ctx.match.scout_yaw_stick)
    # Clamp to RC channel range; the FC bypasses cfg caps for the RC
    # verb so we are the only limit on operator-typed values.
    yaw_stick = max(-100, min(100, yaw_stick))
    duration_s = max(10.0, float(ctx.match.scout_drive_duration_s))
    # Optionally fly to the arena centre (0,0) first, via the FC's
    # existing TO verb, so the scout watches all six slots from the
    # middle. The operator can disable this (match.scout_center=False)
    # if TO is flaky on their FC — in the sim the drone spawns at 0,0
    # so it's already centred without TO.
    center_step = "TO 0 0" if bool(getattr(ctx.match, "scout_center", True)) else ""
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {alt:.2f}",
        center_step,
        f"RC 0 0 0 {yaw_stick} {duration_s:.0f}",
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

        # Push only when the FC is truly idle — i.e. phase == init/done.
        # Phase.IDLE is mid-script HOOVER and must NOT trigger a push (the
        # C2 would reject it). The "never_pushed" guard alone isn't enough
        # after a strategy restart: the FC may still be running an old
        # mission from before the restart, and pushing while it runs
        # produces the script_push_failed storm we saw.
        now = time.time()
        rs = ctx.role_state
        never_pushed = not rs.last_pushed_script
        fc_finished = ctx.state.phase in ("init", "done", "")
        long_enough = (now - rs.last_pushed_unix_s) >= 2.0

        if not fc_finished:
            return noop(f"scout: FC busy (phase={ctx.state.phase})")
        if never_pushed:
            return push(script, new_phase="scouting", reason="scout: initial push")
        if long_enough:
            return push(script, new_phase="scouting", reason="scout: loop re-push")
        return noop("scout: waiting for cooldown")


register(ScoutRole())
