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
# An attacker can be LIVE re-targeted (new enemy boxes) via a mission splice —
# no landing — so this can be far shorter than RETASK_COOLDOWN_S. It only damps
# spamming splices while the scout's box-state settles.
SPLICE_COOLDOWN_S = 10.0
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
        self._last_splice: Dict[str, float] = {}  # fc -> monotonic ts of last splice
        self._desired_role: Dict[str, tuple] = {}  # fc -> (role, since) AUTO hysteresis
        self._atk_lane: Dict[str, int] = {}        # attacker fc -> its fixed swimlane (0,1,2)
        self._defender_recapture: Dict[str, int] = {}  # defender fc -> own slot being recaptured
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
        # hold-time hysteresis so box-state flicker doesn't thrash roles. When
        # AUTO is off, drop any pending hysteresis so a later AUTO-on starts
        # fresh (no stale desired-role from old game state).
        if self.match.auto:
            self._apply_coordinator(worlds, now)
        elif self._desired_role:
            self._desired_role.clear()

        # Role-based ALTITUDE + attacker fan-out, recomputed from current roles.
        # Attackers fly fixed lanes that never cross -> they all share ONE flat
        # altitude. Defenders fly a higher tier (so attackers cross beneath them),
        # vertically separated from each other.
        t = self.match.tunables
        attackers = sorted(
            fc for fc, c in self.match.drones.items()
            if c.get("enabled") and c.get("role") == "attacker")
        defenders = sorted(
            fc for fc, c in self.match.drones.items()
            if c.get("enabled") and c.get("role") == "defender")
        fan = _fan_out_angles(len(attackers))
        atk_hdg = {fc: fan[i] for i, fc in enumerate(attackers)}
        cruise_alt = {fc: S.attacker_alt(t) for fc in attackers}
        cruise_alt.update({fc: S.defender_alt(i, t) for i, fc in enumerate(defenders)})
        self._assign_lanes(attackers)

        for fc, cfg in self.match.drones.items():
            w = worlds.get(fc)
            if w is None or not w.connected:
                continue
            role = cfg.get("role", "idle")
            enabled = bool(cfg.get("enabled"))
            flying = w.phase not in _IDLE_PHASES
            launched = self._launched_role.get(fc)

            # OFF (disabled): COMPLETELY IGNORE it — never command it, and in
            # particular NEVER issue a LAND/stop. The operator turned it off to
            # exclude it from the strategy, not to land it; only EMERGENCY LAND
            # (which lands every drone) ever puts a disabled drone down. Don't
            # touch its launched-role bookkeeping either, so re-enabling resumes
            # cleanly without a re-task.
            if not enabled:
                self._last_action[fc] = "off — ignored (no land)"
                continue

            # A) Enabled but role is idle and it's flying -> stand it down (land).
            #    (An ENABLED drone with no role lands; a disabled one never does.)
            if flying and role == "idle":
                if now - self._last_retask.get(fc, -1e9) >= RETASK_COOLDOWN_S:
                    await self.pool.stop_drone(fc)
                    self._last_retask[fc] = now
                    self._launched_role.pop(fc, None)
                    self._defender_recapture.pop(fc, None)
                    self._last_action[fc] = "stand down (land)"
                continue

            if role == "idle":
                continue

            # B) Flying the WRONG role (a role change, manual or auto) -> re-task:
            #    stop (lands), then a later idle tick relaunches the new role.
            if flying:
                if launched is not None and launched != role:
                    if now - self._last_retask.get(fc, -1e9) >= RETASK_COOLDOWN_S:
                        await self.pool.stop_drone(fc)
                        self._last_retask[fc] = now
                        self._launched_role.pop(fc, None)
                        self._defender_recapture.pop(fc, None)
                        self._last_action[fc] = f"re-task -> {role}"
                    continue
                # Right role, still flying. ATTACKER flies its SELF-CONTAINED
                # fixed-target loop (fly out, capture, return home, hover, repeat)
                # -> nothing for the runner to do. DEFENDER reactively recaptures
                # one of our boxes the instant it flips, via a no-land splice.
                if role == "defender":
                    await self._maybe_recapture(fc, now, cruise_alt.get(fc, S.defender_alt(0, t)))
                continue

            # C) Idle -> launch the current role's never-land script.
            if now - self._last_push.get(fc, -1e9) < PUSH_GRACE_S:
                continue
            script = self._build_role_script(fc, role, w,
                                             cruise_alt.get(fc),
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
                                     self.match.holders(), views,
                                     baseline_def=self.match.tunables.baseline_defenders)
        self.match.coord_summary = plan.summary
        # Drop hysteresis timers for drones the coordinator no longer plans for
        # (disconnected / disabled) so a stale desired-role can't fire later.
        for fc in list(self._desired_role):
            if fc not in plan.roles:
                self._desired_role.pop(fc, None)
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
    def _neutral_standoff(self, team: str):
        """(home_wall_marker_id, standoff_m): a GO_HOME onto our home-wall marker
        stops the defender ``defender_neutral_m`` (operator: 5 m) off the marker,
        facing our boxes to scan them. NOTE the geometry: the marker is at y=±10
        and our home band ends ~5 m off it, so 5 m sits AT the home/neutral
        boundary (and with the GO_HOME arrival tolerance it can settle ~0.5 m into
        home). Raise defender_neutral_m to ~5.5-6 m to wait clearly out in the
        neutral zone. None if the wall marker is unknown."""
        wall = arena_state.home_wall_marker(team)
        if not wall:
            return None, None
        return int(wall[0]), float(self.match.tunables.defender_neutral_m)

    def _build_role_script(self, fc: str, role: str, world,
                           cruise_alt: Optional[float], rth_hdg: float) -> Optional[str]:
        team = self.match.our_team
        t = self.match.tunables
        alt = cruise_alt if cruise_alt is not None else S.attacker_alt(t)
        if role == "scout":
            center = _pick_visible(world, S.CENTER_MARKERS) or S.CENTER_MARKERS[0]
            return S.scout_script(center, t)
        if role == "defender":
            wall_id, standoff = self._neutral_standoff(team)
            if wall_id is None:
                return None
            # Wait just outside home in the neutral zone, scanning our 3 boxes;
            # the runner splices a recapture in the instant one flips.
            return S.defender_neutral_script(wall_id, standoff, alt, t)
        if role == "attacker":
            # OPERATOR-BOUND swimlane (per-drone field) overrides auto-assignment:
            # the two box slots the operator entered -> WAIT_AND_ATTACK both ends.
            manual = (self.match.drones.get(fc) or {}).get("swimlane")
            if manual and len(manual) == 2:
                enemy_face, our_face = S.swimlane_faces_for_boxes(
                    int(manual[0]), int(manual[1]), team)
                return S.attacker_swimlane_script(int(enemy_face), int(our_face), alt, t)
            lane = self._atk_lane.get(fc)
            if lane is None:
                # No manual swimlane and no free auto lane -> hover at home (no land).
                wall = arena_state.home_wall_marker(team)
                if not wall:
                    return None
                return S.attacker_park_script(int(wall[0]), alt, rth_hdg, t)
            # AUTO swimlane: shuttle this attacker's straight vertical lane between
            # its ENEMY box and OUR box, capturing/recapturing each whenever it
            # shows the enemy colour (WAIT_AND_ATTACK at both ends). REPEAT loop.
            enemy_face, our_face = S.swimlane_faces(int(lane), team)
            return S.attacker_swimlane_script(int(enemy_face), int(our_face), alt, t)
        return None

    def _assign_lanes(self, attackers: List[str]) -> None:
        """SWIMLANE assignment pinned to DRONE IDENTITY (not sort rank): each
        attacker owns ONE distinct vertical swimlane (0,1,2 = box 1<->4, 2<->5,
        3<->6). KEEP every existing binding, free the lane of a drone that stopped
        attacking, and give a freed lane only to a newly-appeared attacker (e.g. a
        defender PROMOTED to attacker takes the dropped attacker's lane). So a
        dropout never reshuffles the other attackers' lanes, and no two attackers
        ever share a lane. Extras beyond the 3 lanes get NO lane (they just hover).

        Attackers the operator MANUALLY bound to a swimlane (drones[fc]['swimlane'])
        are excluded from auto-assignment and don't hold an auto lane; if their
        binding matches a standard vertical pair, that lane is reserved so an
        auto-assigned attacker can't double up on it."""
        drones = self.match.drones
        manual = {fc for fc in attackers if (drones.get(fc) or {}).get("swimlane")}
        attacker_set = set(attackers)
        for old_fc in [f for f in self._atk_lane if f not in attacker_set or f in manual]:
            del self._atk_lane[old_fc]
        taken = set(self._atk_lane.values())
        for fc in manual:                    # reserve standard lanes used by bindings
            lane = _swimlane_std_lane((drones.get(fc) or {}).get("swimlane"))
            if lane is not None:
                taken.add(lane)
        free = [l for l in range(S.NUM_SWIMLANES) if l not in taken]
        for fc in sorted(attackers):         # deterministic fill order
            if fc not in manual and fc not in self._atk_lane and free:
                self._atk_lane[fc] = free.pop(0)

    async def _maybe_recapture(self, fc: str, now: float, cruise_alt: float) -> None:
        """DEFENDER reactive recapture: if one of OUR boxes is enemy-held (and
        not in its post-capture lock), splice a recapture leg into this airborne
        defender (no land) — dash in, flip it back, return to the neutral scan.
        Each defender handles one box at a time; a flip is assigned to at most
        one defender so two don't pile on the same box."""
        team = self.match.our_team
        enemy = self.match.enemy_team
        holders = self.match.holders()
        our_slots = (1, 2, 3) if team == "red" else (4, 5, 6)
        # Boxes that need recapture, not already claimed by another defender.
        claimed = {s for f, s in self._defender_recapture.items() if f != fc}
        flipped = [s for s in our_slots
                   if holders.get(s) == enemy
                   and not self.match.tracker.slot_locked(s)
                   and s not in claimed]
        # Done with our current box? (recaptured or no longer flipped) -> release.
        cur = self._defender_recapture.get(fc)
        if cur is not None and holders.get(cur) != enemy:
            self._defender_recapture.pop(fc, None)
            cur = None
        if cur is not None or not flipped:
            return                                  # busy, or nothing to do
        if now - self._last_splice.get(fc, -1e9) < SPLICE_COOLDOWN_S:
            return
        slot = flipped[0]
        face = S.our_box_enemy_faces(team).get(slot)
        wall_id, standoff = self._neutral_standoff(team)
        if face is None or wall_id is None:
            return
        t = self.match.tunables
        script = S.defender_recapture_splice(int(face), wall_id, standoff,
                                             cruise_alt, t)
        ok, payload = await self.pool.splice(fc, script)
        self._last_splice[fc] = now
        if ok:
            self._defender_recapture[fc] = slot
            self._last_action[fc] = f"recapture box {slot} (no land)"
            logger.info("c2_v2: %s -> defender recapture box %s", fc, slot)
        else:
            self._last_action[fc] = f"recapture splice rejected ({payload})"
            logger.info("c2_v2: %s -> recapture splice rejected (%s)", fc, payload)


def _swimlane_std_lane(swimlane) -> Optional[int]:
    """The standard vertical lane (0,1,2) a manual swimlane occupies, or None if
    it isn't a standard pair (box i <-> box i+3). Used to reserve the lane so
    auto-assignment can't double up on it."""
    if not swimlane or len(swimlane) != 2:
        return None
    lo, hi = sorted(int(s) for s in swimlane)
    return (lo - 1) if (lo in (1, 2, 3) and hi == lo + 3) else None


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
