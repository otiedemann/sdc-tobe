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
from .coordinator import Coordinator, DroneView

logger = logging.getLogger("c2_v2.runner")

TICK_S = 1.0
# After a push, treat the drone as "launching" for this long so we don't spam a
# second start while the FC transitions idle -> running.
PUSH_GRACE_S = 6.0
# A role change (manual or auto) re-tasks the drone (stop -> land -> relaunch).
# Stop it at most this often so we never thrash a drone between roles.
RETASK_COOLDOWN_S = 25.0
# In AUTO, only adopt a new desired role once the coordinator has wanted it
# consistently for this long — damps box-state flicker into stable roles.
ROLE_HOLD_S = 12.0
# FC phases that mean "idle, ready to start a new mission".
_IDLE_PHASES = ("", "init", "done")


class Runner:
    def __init__(self, pool, match) -> None:
        self.pool = pool
        self.match = match
        self.coordinator = Coordinator()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_push: Dict[str, float] = {}    # fc -> monotonic ts of last start
        self._launched_role: Dict[str, str] = {}  # fc -> role of the running script
        self._last_retask: Dict[str, float] = {}  # fc -> monotonic ts of last stop
        self._desired_role: Dict[str, tuple] = {}  # fc -> (role, since) AUTO hysteresis
        self._last_action: Dict[str, str] = {}    # fc -> human reason (for UI)

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
        now = time.monotonic()

        # AUTO: let the coordinator (re)assign roles from the game state, with
        # hold-time hysteresis so box-state flicker doesn't thrash roles.
        if self.match.auto:
            self._apply_coordinator(worlds, now)

        # Mover altitude lanes + attacker fan-out, recomputed from current roles.
        movers = sorted(
            fc for fc, c in self.match.drones.items()
            if c.get("enabled") and c.get("role") in ("attacker", "defender"))
        mover_rank = {fc: i for i, fc in enumerate(movers)}
        attackers = sorted(
            fc for fc, c in self.match.drones.items()
            if c.get("enabled") and c.get("role") == "attacker")
        fan = _fan_out_angles(len(attackers))
        atk_hdg = {fc: fan[i] for i, fc in enumerate(attackers)}

        for fc, cfg in self.match.drones.items():
            w = worlds.get(fc)
            if w is None or not w.connected:
                continue
            role = cfg.get("role", "idle")
            enabled = bool(cfg.get("enabled"))
            flying = w.phase not in _IDLE_PHASES
            launched = self._launched_role.get(fc)

            # A) Should NOT be flying (disabled or idle role) but is -> land it.
            if flying and (not enabled or role == "idle"):
                if now - self._last_retask.get(fc, -1e9) >= RETASK_COOLDOWN_S:
                    await self.pool.stop(fc)
                    self._last_retask[fc] = now
                    self._launched_role.pop(fc, None)
                    self._last_action[fc] = "stand down (land)"
                continue

            if not enabled or role == "idle":
                continue

            # B) Flying the WRONG role (a role change, manual or auto) -> re-task:
            #    stop (lands), then a later idle tick relaunches the new role.
            if flying:
                if launched is not None and launched != role:
                    if now - self._last_retask.get(fc, -1e9) >= RETASK_COOLDOWN_S:
                        await self.pool.stop(fc)
                        self._last_retask[fc] = now
                        self._launched_role.pop(fc, None)
                        self._last_action[fc] = f"re-task -> {role}"
                continue

            # C) Idle -> launch the current role's never-land script.
            if now - self._last_push.get(fc, -1e9) < PUSH_GRACE_S:
                continue
            script = self._build_role_script(fc, role, w,
                                             mover_rank.get(fc, 0),
                                             atk_hdg.get(fc, 0.0))
            if not script:
                continue
            ok, payload = await self.pool.push_and_start(fc, script)
            self._last_push[fc] = now
            if ok:
                self._launched_role[fc] = role
            self._last_action[fc] = (
                f"launch {role}" if ok else f"launch {role} FAILED: {payload}")
            logger.info("c2_v2: %s -> launch %s (%s)", fc, role,
                        "ok" if ok else f"FAIL {payload}")

    def _apply_coordinator(self, worlds, now: float) -> None:
        """Run the coordinator and adopt its role plan with per-drone hold-time
        hysteresis (so a momentary box-state flip doesn't re-task a drone)."""
        views = {
            fc: DroneView(
                fc=fc,
                connected=bool(worlds.get(fc) and worlds[fc].connected),
                enabled=bool(cfg.get("enabled")),
                current_role=cfg.get("role", "idle"))
            for fc, cfg in self.match.drones.items()
        }
        plan = self.coordinator.plan(self.match.our_team,
                                     self.match.holders(), views)
        self.match.coord_summary = plan.summary
        for fc, want in plan.roles.items():
            cur = self.match.drones.get(fc, {}).get("role")
            if want == cur:
                self._desired_role.pop(fc, None)
                continue
            prev = self._desired_role.get(fc)
            if prev is None or prev[0] != want:
                self._desired_role[fc] = (want, now)      # start the hold timer
                continue
            if now - prev[1] >= ROLE_HOLD_S:              # held long enough
                self.match.set_role(fc, want)
                self._desired_role.pop(fc, None)

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
