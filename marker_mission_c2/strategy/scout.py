"""SCOUT role — centres on the arena via the side-wall markers, then scans.

The scout's job is to sit *roughly in the centre of the arena* and rotate so
it keeps every target box's ArUco face in view (feeding the slot tracker).

Why not ``TO``?
---------------
The FC's ``TO`` (absolute arena-frame goto) only translates while the
world-position fix is *fresh*, which needs continuous sight of arena markers.
From the centre, looking at far-wall ArUco, the fix goes stale every few frames,
so ``TO`` falls back to a yaw-scan-in-place and the drone never actually reaches
the centre. So the C2 does NOT use ``TO`` — it positions purely with ``APPROACH``
(vision homing: the FC auto-rotates to find the marker, frame-independent, no
absolute position needed).

Centre positioning (operator-specified)
---------------------------------------
Markers **11** (right wall) and **15** (left wall) sit at x=±5, y=0, z=2 m — the
mid-field side markers, present in both the real and GVZ arenas. The scout:

  1. ``APPROACH``es whichever of 11 / 15 it can find, to a 6 m standoff — that
     lands it ~1 m past the arena centre on the y=0 line, at its SET
     ``scout_alt_m`` altitude.
  2. then rotates **continuously** in place (one long ``RC`` yaw drive) to scan
     every box — it does NOT land. The FC safety-lands the moment a script
     COMPLETES, so a "rotate N times then end" script would land the scout; the
     long drive keeps it airborne until EMERGENCY LAND or a team ``bank`` recall.
     If the FC ever does go idle, the C2 re-pushes (re-centres) automatically.

The side marker is chosen by :func:`_pick_center_marker` — the one already in
view if any, else either (APPROACH searches for it regardless).

``team_phase``
--------------
* ``sortie`` — centre of the arena (outside our home zone, watching all six
  boxes while the attackers are out — needed for the all-out 5-pt condition).
* ``bank``   — recall: the scout comes home with everyone else to secure the
  attempt, so it ``GO_HOME``s *into* the home zone (shallow standoff) and rotates
  there, watching our own boxes. (Still no ``TO``.)
"""
from __future__ import annotations

import logging
import math
import time

from . import arena_state
from .roles import (
    Decision, Role, RoleContext, noop, push, register, home_park_xy,
)

logger = logging.getLogger(__name__)

# Loose GO_HOME arrival band (m): "be roughly there", no precise hold.
RECENTER_TOL_M = 0.6
# A pushed script is only treated as "finished" once it has been running for
# at least this long — filters the command-latency window where the FC still
# reports phase init/done right after a push (avoids a false "cycle done").
SCRIPT_GRACE_S = 6.0

# ── Centre positioning via the mid-field side-wall markers ──────────────────
# Markers 11 (right wall) and 15 (left wall) sit at x=±5, y=0, z=2 m (camera
# height) in BOTH the real and GVZ arenas. APPROACHing either to
# CENTER_STANDOFF_M lands the scout ~1 m past the arena centre on the y=0 line —
# close enough to watch all six boxes. APPROACH auto-rotates to FIND its marker
# (vision homing), so this needs NO TO / absolute goto and no pre-orientation.
# Operator-specified positioning method.
# Default centre markers; overridden per hall by match.scout_center_markers.
CENTER_MARKERS: tuple[int, int] = (11, 15)
# 5 m head-on from a side marker (x=±5, y=0) lands the scout at the arena centre
# (x=0, y=0). Operator-specified.
CENTER_STANDOFF_M: float = 5.0
CENTER_APPROACH_TOL_M: float = 0.5     # loose GO_HOME arrival band
CENTER_APPROACH_HDG_DEG: float = 0.0   # head-on (0°)
# The scout rotates CONTINUOUSLY, not in discrete SCOUT turns. The FC SAFETY-
# LANDS the drone the instant a mission script completes, so a "rotate 3× then
# end" script lands the scout after 3 rotations. One long RC yaw drive keeps it
# spinning in place AND airborne for the whole match — only EMERGENCY LAND (or a
# team bank recall) brings it down. 3600 s ≫ any match ("effectively forever");
# if the FC ever does go idle the C2 re-pushes (re-centres) automatically.
SCOUT_ROTATE_DURATION_S: float = 3600.0


def _format_script(*lines: str) -> str:
    return "\n".join(line for line in lines if line) + "\n"


def _scout_alt(ctx: RoleContext) -> float:
    # Honor the operator's per-drone scout_alt_m (e.g. 1.0 m) so the scout flies
    # at the SET altitude. We deliberately do NOT use the deconflicted
    # cruise_alt_m here — that was lifting the scout to ~1.6 m and ignoring the
    # setting. A single centre-loitering scout has no peer to deconflict against.
    return max(0.6, float(ctx.drone.scout_alt_m or 1.0))


def _yaw_stick(ctx: RoleContext) -> int:
    # Rotation speed: per-drone scout_yaw_stick (editable in the dashboard), with
    # the old match-level value as a fallback for configs that predate it.
    stick = getattr(ctx.drone, "scout_yaw_stick", None) or ctx.match.scout_yaw_stick
    return max(1, min(100, int(stick)))


def _rotate_duration(ctx: RoleContext) -> float:
    return max(10.0, float(ctx.match.scout_drive_duration_s))


def _standoff_to(ctx: RoleContext, target_xy: tuple[float, float]):
    """(home_wall_marker_id, distance) so a GO_HOME onto our back-wall marker
    lands the scout at ``target_xy``. None if the wall marker is unknown. Used by
    the bank/return-home path."""
    hw = arena_state.home_wall_marker(ctx.our_team)
    if not hw:
        return None
    mid, (wx, wy) = hw
    dist = math.hypot(wx - target_xy[0], wy - target_xy[1])
    return int(mid), float(dist)


def _center_markers(ctx: RoleContext) -> tuple[int, ...]:
    """The configured side-wall centre markers for this hall (operator-editable
    in the C2: ``match.scout_center_markers``). Falls back to the module
    default 11/15 for configs that predate the setting."""
    cm = getattr(ctx.match, "scout_center_markers", None)
    out = tuple(int(m) for m in cm) if cm else ()
    return out or CENTER_MARKERS


def _pick_center_marker(ctx: RoleContext, rs) -> int:
    """Which side centre marker to APPROACH this cycle. Prefer the one we are
    already using if it is still in view (stable choice), else any currently
    visible configured centre marker, else the previous/default one — APPROACH
    will rotate to search for it regardless ("find one of the markers"). The
    candidate set comes from ``match.scout_center_markers`` (default 11/15),
    so it adapts per hall without a code change."""
    centers = _center_markers(ctx)
    vis = set()
    for mid in ctx.state.visible_marker_ids:
        try:
            vis.add(int(mid))
        except (TypeError, ValueError):
            pass
    last = rs.scratch.get("center_marker")
    if last in vis and last in centers:
        return last
    for m in centers:
        if m in vis:
            return m
    return last if last in centers else centers[0]


def _center_script(ctx: RoleContext, marker_id: int) -> str:
    """Fly to the arena centre via a side wall marker, then rotate continuously.

    Operator-specified centring: a LOOSE head-on (0°) approach to 5 m from a
    mid-field side marker (11/15 at x=±5, y=0) lands the scout at x=0,y=0. We use
    ``GO_HOME`` (loose ±tol arrival) rather than a tight ``APPROACH`` because the
    tight approach never settled at this standoff and stuck/drifted. The approach
    height-aligns the drone to the 2 m marker, so right after it we drop back to
    the SET ``scout_alt_m`` before spinning. The long RC yaw drive then keeps the
    scout spinning AND airborne — the FC safety-lands the instant a script
    COMPLETES, so we never let the script finish. TAKEOFF is a no-op once
    airborne."""
    alt = _scout_alt(ctx)
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {alt:.2f}",
        # Loose head-on approach to 5 m from the side marker -> arena centre.
        f"GO_HOME {int(marker_id)} {CENTER_STANDOFF_M:.2f} "
        f"{CENTER_APPROACH_TOL_M:g} {CENTER_APPROACH_HDG_DEG:g}",
        # The approach climbed us toward the 2 m marker — drop back to the set
        # scout altitude before spinning.
        f"HEIGHT {alt:.2f}",
        f"RC 0 0 0 {_yaw_stick(ctx)} {SCOUT_ROTATE_DURATION_S:.0f}",
    )


def _home_script(ctx: RoleContext) -> str | None:
    """Bank: GO_HOME *into* our home zone (shallow standoff), then rotate there
    to keep watching our own boxes while the team secures the attempt. No TO."""
    park = ctx.home_park_xy or home_park_xy(ctx.our_team)
    so = _standoff_to(ctx, park)
    if so is None:
        return None
    mid, dist = so
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {_scout_alt(ctx):.2f}",
        f"GO_HOME {mid} {dist:.2f} {RECENTER_TOL_M:g}",
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
        # A pushed script is genuinely finished only AFTER the grace window, so
        # the post-push command-latency (FC still reporting init/done) isn't
        # mistaken for the cycle having completed.
        fc_idle = ctx.state.phase in ("init", "done", "")
        since_push = now - (rs.last_pushed_unix_s or 0.0)
        script_done = fc_idle and since_push > SCRIPT_GRACE_S

        if ctx.team_phase == "bank":
            return self._decide_bank(ctx, rs, fc_idle, since_push)
        return self._decide_sortie(ctx, rs, script_done)

    # -- sortie: hold the set altitude and rotate continuously in place -------
    def _decide_sortie(self, ctx: RoleContext, rs, script_done: bool) -> Decision:
        # Fly to the centre via a side marker (GO_HOME 11/15 to 5 m, head-on),
        # drop to the set altitude, then spin continuously. Only re-push if the FC
        # goes idle (the long rotation elapsed or an emergency land), so the scout
        # keeps rotating until EMERGENCY LAND.
        if rs.phase == "scout_center" and not script_done:
            return noop(f"scout: centring via marker {rs.scratch.get('center_marker')}"
                        f" + rotating")
        marker = _pick_center_marker(ctx, rs)
        rs.scratch["center_marker"] = marker
        return push(_center_script(ctx, marker), new_phase="scout_center",
                    reason=f"scout: GO_HOME {marker} {CENTER_STANDOFF_M:g}m head-on -> "
                           f"centre, then rotate")

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
