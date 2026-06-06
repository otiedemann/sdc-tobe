"""SCOUT role — self-centering scout that watches all six target boxes.

The scout's job is to sit *roughly in the centre of the arena* and rotate so
it keeps every target box's ArUco face in view (feeding the slot tracker).

Why not just ``TO 0 0``?
------------------------
The FC's ``TO`` (absolute arena-frame goto) only translates while the
world-position fix is *fresh*, which needs continuous sight of arena markers.
From the centre, looking at far-wall ArUco, the fix goes stale every few
frames, so ``TO`` falls back to a yaw-scan-in-place and the drone never
actually reaches the centre — it "seems not to find the centre". So we do NOT
use absolute goto here.

Self-centering by vision-homing
-------------------------------
Instead we recenter with ``APPROACH_HOME`` onto our own back-wall marker — the
one marker that is always present and that the FC's APPROACH auto-rotates to
find (frame-independent vision homing, robust to a stale/mirrored heading).
APPROACH_HOME closes to a *loose* standoff distance equal to the wall→centre
distance, so the scout ends up on the arena centreline (≈ 0,0) facing home,
then spins up.

After takeoff the scout SCANs (one slow 360°) and counts how many of the six
target markers it actually sees. If it is missing some, it recenters again and
re-scans (up to a few tries) — "fly closer so all six come into view". Once it
sees them all (or has tried enough), it commits to a long continuous rotation
in place. The whole loop is driven tick-by-tick from :meth:`ScoutRole.decide`
using ``role_state.scratch`` to remember what it has seen and how many recenter
attempts it has made.

APPROACH homes along the drone's current bearing to the wall marker, so it
controls the DEPTH (the long arena axis, along which the six boxes are spread)
precisely and pulls the scout onto the mid-field band (≈ y 0). Any residual
lateral (x) offset from a drifted start is left roughly as-is — fine, because
the arena is narrow in x and "around the centre" is all the scouting rotation
needs. A scout that starts near the centre (the expected case) ends up well
centred; this is not a replacement for absolute localisation.

``team_phase``
--------------
* ``sortie`` — centre of the arena (outside our home zone, watching all six
  boxes while the attackers are out — needed for the all-out 5-pt condition).
* ``bank``   — recall: the scout must come home with everyone else to secure
  the attempt, so it APPROACH_HOMEs *into* the home zone (shallow standoff) and
  rotates there, watching our own boxes.
"""
from __future__ import annotations

import logging
import math
import time

from . import arena_state
from .roles import (
    Decision, Role, RoleContext, noop, push, register, home_park_xy,
)
from .settings import slot_for_face

logger = logging.getLogger(__name__)

# Loose APPROACH_HOME arrival band (m): "be roughly there", no precise hold.
RECENTER_TOL_M = 0.6
# A pushed script is only treated as "finished" once it has been running for
# at least this long — filters the command-latency window where the FC still
# reports phase init/done right after a push (avoids a false "scan done").
SCRIPT_GRACE_S = 6.0
# How many recenter+rescan attempts before we give up and rotate best-effort.
MAX_RECENTER_TRIES = 3


def _format_script(*lines: str) -> str:
    return "\n".join(line for line in lines if line) + "\n"


def _scout_alt(ctx: RoleContext) -> float:
    drone = ctx.drone
    return max(0.6, float(ctx.cruise_alt_m if ctx.cruise_alt_m else drone.scout_alt_m))


def _yaw_stick(ctx: RoleContext) -> int:
    return max(-100, min(100, int(ctx.match.scout_yaw_stick)))


def _rotate_duration(ctx: RoleContext) -> float:
    return max(10.0, float(ctx.match.scout_drive_duration_s))


def _standoff_to(ctx: RoleContext, target_xy: tuple[float, float]):
    """(home_wall_marker_id, distance) so an APPROACH_HOME onto our back-wall
    marker lands the scout at ``target_xy``. None if the wall marker is unknown.

    The scout homes head-on onto the wall marker (APPROACH centres it in view),
    so its lateral position converges on the marker's x (≈ centreline). The
    standoff distance sets how far IN FRONT of the wall it stops — we pick the
    wall→target distance so it stops at the desired depth.
    """
    hw = arena_state.home_wall_marker(ctx.our_team)
    if not hw:
        return None
    mid, (wx, wy) = hw
    dist = math.hypot(wx - target_xy[0], wy - target_xy[1])
    return int(mid), float(dist)


def _seek_script(ctx: RoleContext) -> str | None:
    """Recenter onto the arena centreline (vision-homing), then scan one 360°.

    Counts the markers it sees during the SCOUT turn (the runner streams
    visible_marker_ids into ``decide`` every tick). TAKEOFF is a no-op when
    already airborne — we never land between scans.
    """
    so = _standoff_to(ctx, (0.0, 0.0))
    if so is None:
        return None
    mid, dist = so
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {_scout_alt(ctx):.2f}",
        f"APPROACH_HOME {mid} {dist:.2f} {RECENTER_TOL_M:g}",
        "SCOUT",
    )


def _rotate_script(ctx: RoleContext) -> str | None:
    """Long continuous yaw drive in place (no inter-rotation braking)."""
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {_scout_alt(ctx):.2f}",
        f"RC 0 0 0 {_yaw_stick(ctx)} {_rotate_duration(ctx):.0f}",
    )


def _home_script(ctx: RoleContext) -> str | None:
    """Bank: APPROACH_HOME *into* our home zone (shallow standoff), then rotate
    there to keep watching our own boxes while the team secures the attempt."""
    park = ctx.home_park_xy or home_park_xy(ctx.our_team)
    so = _standoff_to(ctx, park)
    if so is None:
        return None
    mid, dist = so
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {_scout_alt(ctx):.2f}",
        f"APPROACH_HOME {mid} {dist:.2f} {RECENTER_TOL_M:g}",
        f"RC 0 0 0 {_yaw_stick(ctx)} {_rotate_duration(ctx):.0f}",
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
        now = time.time()
        # A pushed script is genuinely finished only AFTER the grace window,
        # so we never mistake the post-push command-latency (FC still reporting
        # init/done) for the scan having completed.
        fc_idle = ctx.state.phase in ("init", "done", "")
        since_push = now - (rs.last_pushed_unix_s or 0.0)
        script_done = fc_idle and since_push > SCRIPT_GRACE_S

        # Accumulate the target slots visible THIS tick (per-scout view).
        seen = rs.scratch.setdefault("seen_slots", set())
        for mid in ctx.state.visible_marker_ids:
            slot = slot_for_face(int(mid))
            if slot is not None:
                seen.add(slot)

        if ctx.team_phase == "bank":
            return self._decide_bank(ctx, rs, fc_idle, since_push)
        return self._decide_sortie(ctx, rs, script_done, seen)

    # -- sortie: self-center at the arena centre, then rotate ----------------
    def _decide_sortie(self, ctx: RoleContext, rs, script_done: bool,
                       seen: set) -> Decision:
        required = len(ctx.active_slots) or 6

        # Settled: rotating at centre. Re-verify centering only once the long
        # rotation script actually ends (duration elapsed) — then re-seek.
        if rs.phase == "scout_center":
            if not script_done:
                return noop(f"scout: rotating at centre ({len(seen)}/{required} seen)")
            rs.scratch["seen_slots"] = set()
            rs.scratch["tries"] = 0
            script = _seek_script(ctx)
            if script is None:
                return noop("scout: home wall marker unknown")
            return push(script, new_phase="scout_seek",
                        reason="scout: re-verify centre")

        # Seeking: recenter+scan running. Wait for the scan to finish.
        if rs.phase == "scout_seek":
            if not script_done:
                return noop(f"scout: seeking centre ({len(seen)}/{required} seen)")
            tries = int(rs.scratch.get("tries", 0))
            enough = len(seen) >= required
            if enough or tries >= MAX_RECENTER_TRIES:
                script = _rotate_script(ctx)
                if script is None:
                    return noop("scout: cannot compose rotate")
                reason = (
                    f"scout: centre confirmed ({len(seen)}/{required} boxes)"
                    if enough else
                    f"scout: centre best-effort ({len(seen)}/{required} "
                    f"after {tries} tries)"
                )
                return push(script, new_phase="scout_center", reason=reason)
            # Missing some boxes -> recenter and rescan.
            rs.scratch["tries"] = tries + 1
            rs.scratch["seen_slots"] = set()
            script = _seek_script(ctx)
            if script is None:
                return noop("scout: home wall marker unknown")
            return push(script, new_phase="scout_seek",
                        reason=f"scout: recenter ({len(seen)}/{required} boxes, "
                               f"try {tries + 1})")

        # Fresh / coming from bank: start a seek.
        rs.scratch["seen_slots"] = set()
        rs.scratch["tries"] = 0
        script = _seek_script(ctx)
        if script is None:
            return noop("scout: home wall marker unknown")
        return push(script, new_phase="scout_seek", reason="scout: seek centre")

    # -- bank: recall into our home zone and rotate --------------------------
    def _decide_bank(self, ctx: RoleContext, rs, fc_idle: bool,
                     since_push: float) -> Decision:
        # Already homing/rotating at home and the script is still running.
        if rs.phase == "scout_home" and not (fc_idle and since_push > SCRIPT_GRACE_S):
            return noop("scout: at home (bank)")
        script = _home_script(ctx)
        if script is None:
            return noop("scout: home wall marker unknown")
        return push(script, new_phase="scout_home", reason="scout: bank -> home")


register(ScoutRole())
