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

from .roles import (
    Decision, Role, RoleContext, noop, push, register, home_park_xy,
)

logger = logging.getLogger(__name__)


def _format_script(*lines: str) -> str:
    return "\n".join(line for line in lines if line) + "\n"


def _scout_target_xy(ctx: RoleContext) -> tuple[float, float]:
    """Where the scout rotates this team-phase.

    sortie -> arena centre (0,0): watches all six boxes from the middle while
              the attackers are out (and, being in neutral, the scout itself is
              outside our home zone — needed for the 5-pt all-out condition).
    bank   -> our home zone: the scout must come home with everyone else to
              secure the attempt; it keeps rotating there to watch our own
              boxes (enemy boxes are out of vision range from home).
    """
    if ctx.team_phase == "bank":
        return ctx.home_park_xy or home_park_xy(ctx.our_team)
    return (0.0, 0.0)


def _compose_scout_script(ctx: RoleContext, tx: float, ty: float) -> str | None:
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
    # Fly to the target point first (via the FC's existing TO verb), then
    # rotate in place. TAKEOFF is a no-op when already airborne (we never land).
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {alt:.2f}",
        f"TO {tx:g} {ty:g}",
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

        rs = ctx.role_state
        # The scout tracks WHICH point it's rotating at via its role phase, so
        # a team-phase flip (centre <-> home) redirects it even mid-rotate.
        tx, ty = _scout_target_xy(ctx)
        desired_phase = "scout_home" if ctx.team_phase == "bank" else "scout_center"
        fc_finished = ctx.state.phase in ("init", "done", "")

        # Already rotating at the right point -> nothing to do.
        if rs.phase == desired_phase and not fc_finished:
            return noop(f"scout: rotating at {desired_phase[6:]} ({ctx.team_phase})")

        # Otherwise (re)launch the rotate toward the desired point NOW. When the
        # team phase just flipped, the scout is mid-rotate at the WRONG point;
        # we push the new go-to-point script straight away, which OVERRIDES the
        # running rotate — instant redirect, no stop-then-wait dance (that cost
        # the team an extra tick or two getting the scout home each bank).
        # _apply_decision throttles an identical re-push, so once it's heading
        # to / rotating at the right point this just falls through to the noop
        # above and never hammers the FC.
        script = _compose_scout_script(ctx, tx, ty)
        if script is None:
            return noop("scout: cannot compose script")
        return push(
            script,
            new_phase=desired_phase,
            reason=f"scout: rotate at {desired_phase[6:]} ({ctx.team_phase})",
        )


register(ScoutRole())
