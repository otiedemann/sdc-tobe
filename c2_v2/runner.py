"""C2 V2 runner — Phase 3.

A 1 Hz async tick loop that, for each enabled drone with a role, LAUNCHES its
never-land role script once the FC is idle. Because the role scripts never
complete (scout/defender rotate ~1 h; attacker chains 30 capture passes), each
drone is launched once and then flies its role for the whole match — it only
lands on the '?' emergency-land.

Phase 3 holds the role fixed (operator-assigned). The smart coordinator
(auto-assignment + 5/10/Special timing) lands in Phase 4 on top of this.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from marker_mission_c2.strategy import arena_state

from . import scripts as S

logger = logging.getLogger("c2_v2.runner")

TICK_S = 1.0
# After a push, treat the drone as "launching" for this long so we don't spam a
# second start while the FC transitions idle -> running.
PUSH_GRACE_S = 6.0
# FC phases that mean "idle, ready to start a new mission".
_IDLE_PHASES = ("", "init", "done")


class Runner:
    def __init__(self, pool, match) -> None:
        self.pool = pool
        self.match = match
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_push: Dict[str, float] = {}   # fc -> monotonic ts of last start
        self._last_action: Dict[str, str] = {}   # fc -> human reason (for UI)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="c2v2-runner")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

    async def emergency_land(self) -> Dict[str, bool]:
        """Land everyone NOW and disarm. The single most important command."""
        self.match.set_armed(False)
        res = await self.pool.emergency_land_all()
        for fc in res:
            self._last_action[fc] = "EMERGENCY LAND"
        logger.warning("c2_v2: EMERGENCY LAND ALL -> %s", res)
        return res

    def last_actions(self) -> Dict[str, str]:
        return dict(self._last_action)

    # ------------------------------------------------------------------ loop
    async def _loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                await self._tick()
            except Exception as exc:
                logger.exception("c2_v2 runner tick failed: %s", exc)
            await asyncio.sleep(max(0.1, TICK_S - (time.monotonic() - t0)))

    async def _tick(self) -> None:
        if not self.match.armed:
            return
        worlds = self.pool.worlds
        # Compute mover lanes (altitude) + attacker fan-out from current roles,
        # so role changes take effect immediately.
        movers = sorted(
            fc for fc, c in self.match.drones.items()
            if c.get("enabled") and c.get("role") in ("attacker", "defender"))
        mover_rank = {fc: i for i, fc in enumerate(movers)}
        attackers = sorted(
            fc for fc, c in self.match.drones.items()
            if c.get("enabled") and c.get("role") == "attacker")
        fan = _fan_out_angles(len(attackers))
        atk_hdg = {fc: fan[i] for i, fc in enumerate(attackers)}

        now = time.monotonic()
        for fc, cfg in self.match.drones.items():
            if not cfg.get("enabled"):
                continue
            role = cfg.get("role", "idle")
            if role == "idle":
                continue
            w = worlds.get(fc)
            if w is None or not w.connected:
                continue
            # Only START from idle, and not within the post-push grace window.
            if w.phase not in _IDLE_PHASES:
                continue
            if now - self._last_push.get(fc, -1e9) < PUSH_GRACE_S:
                continue
            script = self._build_role_script(fc, role, w,
                                             mover_rank.get(fc, 0),
                                             atk_hdg.get(fc, 0.0))
            if not script:
                continue
            ok, payload = await self.pool.push_and_start(fc, script)
            self._last_push[fc] = now
            self._last_action[fc] = (
                f"launch {role}" if ok else f"launch {role} FAILED: {payload}")
            logger.info("c2_v2: %s -> launch %s (%s)", fc, role,
                        "ok" if ok else f"FAIL {payload}")

    # ------------------------------------------------------------------ scripts
    def _build_role_script(self, fc: str, role: str, world,
                           mover_lane: int, rth_hdg: float) -> Optional[str]:
        team = self.match.our_team
        if role == "scout":
            center = _pick_visible(world, S.CENTER_MARKERS) or S.CENTER_MARKERS[0]
            return S.scout_script(center)
        if role == "defender":
            sides = arena_state.side_markers(team) or S.CENTER_MARKERS
            side = _pick_visible(world, sides) or sides[0]
            return S.defender_script(side, S.mover_alt(mover_lane))
        if role == "attacker":
            wall = arena_state.home_wall_marker(team)
            if not wall:
                return None
            wall_id = int(wall[0])
            faces = S.enemy_home_faces(team)
            return S.attacker_loop_script(faces, wall_id,
                                          S.mover_alt(mover_lane), rth_hdg)
        return None


def _fan_out_angles(n: int) -> List[float]:
    """RTH arrival bearings around the home wall marker: 1->[0]; 2->[-30,+30];
    3->[-30,0,+30]; more -> evenly across [-30,+30]."""
    if n <= 0:
        return []
    if n == 1:
        return [0.0]
    half = 30.0
    return [round(-half + (2 * half) * i / (n - 1), 1) for i in range(n)]


def _pick_visible(world, candidates) -> Optional[int]:
    """Return the first of ``candidates`` currently visible to this drone, else
    None (the APPROACH/GO_HOME will rotate to find it regardless)."""
    vis = set(world.visible_marker_ids or [])
    for m in candidates:
        if int(m) in vis:
            return int(m)
    return None
