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
mid-field side markers, present in both the real and GVZ arenas. Each cycle the
scout:

  1. ``APPROACH``es whichever of 11 / 15 it can find, to a 6 m standoff — that
     lands it ~1 m past the arena centre on the y=0 line.
  2. rotates 360° three times (3× ``SCOUT``) to scan every box.
  3. when the FC finishes the cycle, the C2 pushes a fresh one — the new
     ``APPROACH`` re-acquires a side marker and re-adjusts the scout back to the
     centre ("re-check and re-adjust position").

The side marker for each cycle is chosen by :func:`_pick_center_marker` — the
one already in view if any, else either (APPROACH searches for it regardless).

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
CENTER_MARKERS: tuple[int, int] = (11, 15)
CENTER_STANDOFF_M: float = 6.0
N_CENTER_ROTATIONS: int = 3


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
    """(home_wall_marker_id, distance) so a GO_HOME onto our back-wall marker
    lands the scout at ``target_xy``. None if the wall marker is unknown. Used by
    the bank/return-home path."""
    hw = arena_state.home_wall_marker(ctx.our_team)
    if not hw:
        return None
    mid, (wx, wy) = hw
    dist = math.hypot(wx - target_xy[0], wy - target_xy[1])
    return int(mid), float(dist)


def _pick_center_marker(ctx: RoleContext, rs) -> int:
    """Which side marker (11/15) to APPROACH this cycle. Prefer the one we are
    already using if it is still in view (stable choice), else any currently
    visible side marker, else the previous/default one — APPROACH will rotate
    to search for it regardless ("find one of the two markers")."""
    vis = set()
    for mid in ctx.state.visible_marker_ids:
        try:
            vis.add(int(mid))
        except (TypeError, ValueError):
            pass
    last = rs.scratch.get("center_marker")
    if last in vis:
        return last
    for m in CENTER_MARKERS:
        if m in vis:
            return m
    return last if last in CENTER_MARKERS else CENTER_MARKERS[0]


def _center_script(ctx: RoleContext, marker_id: int) -> str:
    """Vision-home onto a mid-field side marker to sit near the arena centre,
    then rotate 360° N times to scan. APPROACH searches (rotates) to acquire the
    marker itself, then closes to CENTER_STANDOFF_M head-on. TAKEOFF is a no-op
    once airborne — we never land between cycles."""
    return _format_script(
        "TAKEOFF",
        f"HEIGHT {_scout_alt(ctx):.2f}",
        f"APPROACH {int(marker_id)} {CENTER_STANDOFF_M:.2f}",
        *(["SCOUT"] * N_CENTER_ROTATIONS),
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

    # -- sortie: centre via a side marker, rotate 3×, re-check & re-adjust ----
    def _decide_sortie(self, ctx: RoleContext, rs, script_done: bool) -> Decision:
        # One cycle = APPROACH 11/15 to 6 m (vision-home to the centre) followed
        # by N×360° scans. When the FC finishes the cycle we push a fresh one:
        # its APPROACH re-acquires a side marker and re-adjusts the scout back to
        # the centre — the operator's "re-check and re-adjust position" step.
        if rs.phase == "scout_center" and not script_done:
            return noop(f"scout: centring via marker {rs.scratch.get('center_marker')}"
                        f" + {N_CENTER_ROTATIONS}×360°")
        marker = _pick_center_marker(ctx, rs)
        rs.scratch["center_marker"] = marker
        return push(_center_script(ctx, marker), new_phase="scout_center",
                    reason=f"scout: APPROACH {marker} {CENTER_STANDOFF_M:g}m -> "
                           f"centre, then {N_CENTER_ROTATIONS}×360°")

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
