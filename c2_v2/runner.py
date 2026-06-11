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

# Watchdog (stuck-drone detection + recovery, all roles) — see _watchdog.
# current_step_kind values where the drone is TRANSLATING (so a ~zero ground
# speed means it's stuck). Hover/wait/rotate steps (WAIT_AND_ATTACK, YAW, HEIGHT,
# RC, HOOVER, TAKEOFF, REPEAT, ...) are deliberately EXCLUDED so a legitimately
# holding attacker (e.g. WAIT_AND_ATTACK waiting for its box to flip) is never
# mistaken for frozen.
_TRANSLATING_STEP_KINDS = (
    "ATTACK", "APPROACH", "GO_HOME", "MOVE_FBUD", "MOVE_FB", "MOVE_LR", "TO")
# After handing off a LANDED attacker, give its replacement-role relaunch this
# long to get airborne before escalating to a firmware reboot.
WD_RECOVER_TAKEOFF_S = 15.0
# After a reboot, wait this long for the drone to come back + take off before
# giving up (drone returns in ~30-60 s).
WD_REBOOT_WAIT_S = 80.0


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
        # Watchdog state:
        self._wd_bad_since: Dict[str, tuple] = {}   # fc -> (reason, since_monotonic)
        self._wd_recover: Dict[str, dict] = {}      # fc -> recovery state machine
        self._wd_no_spare_at: Dict[str, float] = {}  # fc -> last "no spare" log ts (throttle)
        # Wake signal: arming (or any operator action that wants an immediate
        # launch) sets this so the loop ticks NOW instead of waiting out the 1 Hz
        # period — cuts up to ~1 s off press-to-takeoff. Set thread-safely via
        # request_tick() from the Flask thread.
        self._wake = asyncio.Event()
        self._loop_ref: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop_ref = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._loop(), name="c2v2-runner")

    def request_tick(self) -> None:
        """Ask the runner loop to tick immediately (thread-safe; callable from the
        Flask/web thread). Used on ARM so the launch push doesn't wait ~1 s for the
        next 1 Hz tick."""
        loop = self._loop_ref
        if loop is not None:
            loop.call_soon_threadsafe(self._wake.set)

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
            # Sleep out the 1 Hz period, but wake EARLY if request_tick() fired
            # (e.g. the operator just armed) so the launch happens immediately.
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(),
                                       timeout=max(0.1, TICK_S - (time.monotonic() - t0)))
            except asyncio.TimeoutError:
                pass

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

        # WATCHDOG: detect stuck attackers and hand their lane to a healthy
        # spare (runs before the role/lane computation below so the swap takes
        # effect THIS tick — the re-task loop then relaunches both drones).
        await self._watchdog(worlds, now)

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
            # The watchdog OWNS a drone it's rebooting — don't relaunch it (it
            # must stay landed for the firmware reboot to take).
            rec = self._wd_recover.get(fc)
            if rec is not None and rec.get("hold"):
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
            # Operator-set spotter target [marker_id, distance_m] overrides the
            # default (a visible centre marker at the default standoff).
            spot = (self.match.drones.get(fc) or {}).get("spotter")
            if spot and len(spot) == 2:
                mid, dist = int(spot[0]), float(spot[1])
            else:
                mid = _pick_visible(world, S.CENTER_MARKERS) or S.CENTER_MARKERS[0]
                dist = S.SPOTTER_DEFAULT_DISTANCE_M
            return S.scout_script(mid, dist, t)
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
        attacker owns ONE distinct vertical swimlane (0,1,2 = box 1<->6, 2<->5,
        3<->4, the clockwise corridors). KEEP every existing binding, free the
        lane of a drone that stopped
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

    # ------------------------------------------------------------------ watchdog
    async def _watchdog(self, worlds, now: float) -> None:
        """Detect a STUCK drone in ANY active role (unreachable / landed /
        stale-camera / frozen) and recover it:
          * a stuck ATTACKER with a healthy spare -> hand its swimlane to the
            spare (role swap), then recover the stuck drone in its new role;
          * any other stuck drone (scout/defender, or attacker with no spare) ->
            recover IN PLACE: stop -> relaunch, escalating to a firmware reboot
            (then relaunch) if it won't come back healthy."""
        t = self.match.tunables
        if int(getattr(t, "watchdog_enabled", 1)) <= 0:
            self._wd_bad_since.clear()
            self._wd_recover.clear()
            return

        thresh = {
            "unreachable": float(t.watchdog_unreach_s),
            "radio-lost": float(t.watchdog_unreach_s),
            "landed": float(t.watchdog_landed_s),
            "stale": float(t.watchdog_frozen_s),          # telemetry callback stalled
            "frozen": float(t.watchdog_frozen_s),
            "camera-stall": float(t.watchdog_camera_stall_s),  # FC self-heals first
        }
        for fc, cfg in list(self.match.drones.items()):
            role = cfg.get("role")
            # Watch every ENABLED, ACTIVE-role drone. Skip idle/off, and skip a
            # drone already in active recovery (don't re-trigger mid-recovery).
            if (not cfg.get("enabled") or role in (None, "idle")
                    or fc in self._wd_recover):
                self._wd_bad_since.pop(fc, None)
                continue
            reason = self._drone_unhealthy_reason(fc, worlds.get(fc), role, t)
            if reason is None:
                self._wd_bad_since.pop(fc, None)
                continue
            prev = self._wd_bad_since.get(fc)
            if prev is None or prev[0] != reason:
                self._wd_bad_since[fc] = (reason, now)     # (re)start confirm timer
                continue
            if now - prev[1] < thresh.get(reason, 10.0):
                continue                                   # not sustained yet
            self._wd_bad_since.pop(fc, None)
            await self._handle_stuck(fc, role, reason, worlds, now)

        # Drive the recovery escalation (relaunch -> reboot -> relaunch).
        await self._process_recovery(worlds, now)

    def _drone_unhealthy_reason(self, fc: str, w, role: str, t) -> Optional[str]:
        """Why this drone looks stuck, or None if healthy. Role-agnostic. Order:
        unreachable > radio-lost > landed > stale > camera-stall > frozen."""
        if w is None or not w.connected:
            return "unreachable"          # FC/host unreachable (or crashed)
        if not w.drone_connected:
            return "radio-lost"           # FC up, but the drone link dropped
        # Don't judge flying/stale/frozen until the drone is actually flying its
        # CURRENT role (so a drone still mid-(re)launch isn't falsely flagged).
        if self._launched_role.get(fc) != role:
            return None
        if not w.flying:
            return "landed"               # landed / crashed / beep-mode motor cut
        if w.telemetry_stalled:
            return "stale"                # drone telemetry callback not updating
        # CAMERA-STALL = vision frame frozen in flight. The FC restarts the
        # camera locally (~every 3s) and holds neutral RC meanwhile; this only
        # fires (after watchdog_camera_stall_s) if those restarts DON'T clear
        # it, escalating to the land+reboot recovery ladder.
        if w.camera_stalled:
            return "camera-stall"
        # FROZEN = ~zero IMU ground speed during a TRANSLATING step. A
        # legitimately hovering/rotating step (WAIT_AND_ATTACK, YAW, RC hold) is
        # excluded by the step filter so a waiting drone isn't flagged.
        #
        # The ATTACK step also wraps a SEARCH sub-phase: before it can translate
        # toward a target the drone hovers and yaws in place to acquire the
        # marker, so its ground speed is ~zero BY DESIGN there. That hover can
        # easily outlast watchdog_frozen_s, so the confirm timer alone won't save
        # us -> we'd false-flag a searching attacker as frozen and hand its lane
        # to a spare. Skip the frozen verdict while searching; a truly dead drone
        # is still caught by unreachable/radio-lost/landed/stale above (none of
        # which depend on ground speed).
        if (w.current_step_kind in _TRANSLATING_STEP_KINDS
                and w.phase != "search"):
            spd = w.ground_speed_cms
            if spd is not None and spd < float(t.watchdog_frozen_speed_cms):
                return "frozen"
        return None

    def _pick_takeover(self, worlds) -> Optional[str]:
        """Best healthy NON-attacker to take over a lane: prefer a currently
        FLYING drone (fastest to re-task), then by role (defender/scout before
        idle, to avoid waiting out a takeoff). Requires a live FC + drone link."""
        cands = []
        for fc, cfg in self.match.drones.items():
            if not cfg.get("enabled"):
                continue
            role = cfg.get("role", "idle")
            if role == "attacker":
                continue
            w = worlds.get(fc)
            if w is None or not w.connected or not w.drone_connected:
                continue
            flying_rank = 0 if w.flying else 1
            role_rank = {"defender": 0, "scout": 1, "idle": 2}.get(role, 3)
            cands.append((flying_rank, role_rank, fc))
        if not cands:
            return None
        cands.sort()
        return cands[0][2]

    def _attacker_lane_params(self, fc: str) -> Optional[list]:
        """The two box slots this attacker shuttles — its manual swimlane, or its
        auto-lane's clockwise corridor pair. None if it holds no lane."""
        manual = (self.match.drones.get(fc) or {}).get("swimlane")
        if manual and len(manual) == 2:
            return [int(manual[0]), int(manual[1])]
        lane = self._atk_lane.get(fc)
        return [lane + 1, 7 - (lane + 1)] if lane is not None else None

    async def _handle_stuck(self, fc: str, role: str, reason: str,
                            worlds, now: float) -> None:
        """Recover a confirmed-stuck drone. A stuck ATTACKER with a healthy spare
        hands its lane to the spare (role swap) so the lane stays covered, then
        recovers in its new role. Everyone else recovers IN PLACE."""
        if role == "attacker":
            params = self._attacker_lane_params(fc)
            takeover = self._pick_takeover(worlds) if params else None
            if params and takeover is not None:
                old_role = self.match.drones.get(takeover, {}).get("role", "idle")
                self.match.set_swimlane(takeover, params)
                self.match.set_role(takeover, "attacker")
                self.match.set_swimlane(fc, None)
                self.match.set_role(fc, old_role)
                self._atk_lane.pop(fc, None)
                tos = takeover.replace("flightctrl", "fc")
                self._last_action[takeover] = (
                    f"WATCHDOG takeover: attacker lane {params} "
                    f"(from {fc.replace('flightctrl', 'fc')})")
                self._last_action[fc] = (
                    f"WATCHDOG: stuck ({reason}) -> {tos} took lane {params}; "
                    f"now {old_role}, recovering")
                logger.warning("c2_v2 watchdog: %s stuck (%s) -> %s takes lane %s; "
                               "%s -> %s + recover", fc, reason, takeover, params,
                               fc, old_role)
                # Role changed -> the main loop re-tasks it (stop -> relaunch).
                # Register it for the reboot escalation in its new role.
                self._wd_recover[fc] = {"since": now, "stage": "relaunch",
                                        "hold": False, "reason": reason,
                                        "role": old_role}
                return
            if takeover is None and now - self._wd_no_spare_at.get(fc, -1e9) > 10.0:
                logger.warning("c2_v2 watchdog: %s attacker stuck (%s); no spare "
                               "-> in-place recovery", fc, reason)
                self._wd_no_spare_at[fc] = now
        # In-place recovery (scout/defender, or attacker with no spare).
        await self._kick_and_recover(fc, role, reason, worlds, now)

    async def _kick_and_recover(self, fc: str, role: str, reason: str,
                                worlds, now: float) -> None:
        """Force a fresh relaunch of a stuck drone's CURRENT role: if it's still
        flying (stale/frozen), stop it so the main loop relaunches from idle; then
        register it for the reboot escalation."""
        w = worlds.get(fc)
        if w is not None and w.connected and w.flying:
            await self.pool.stop_drone(fc)        # lands -> main loop relaunches role
            self._launched_role.pop(fc, None)
            self._last_retask[fc] = now
        self._wd_recover[fc] = {"since": now, "stage": "relaunch",
                                "hold": False, "reason": reason, "role": role}
        self._last_action[fc] = f"WATCHDOG: {role} stuck ({reason}) -> recovering in place"
        logger.warning("c2_v2 watchdog: %s %s stuck (%s) -> in-place recovery "
                       "(relaunch, reboot if needed)", fc, role, reason)

    async def _process_recovery(self, worlds, now: float) -> None:
        """Recovery state machine for any drone the watchdog benched/kicked:
          * relaunch  — wait for the main loop's relaunch to restore it; if not
                        healthy after WD_RECOVER_TAKEOFF_S, escalate to reboot.
          * reboot    — take ownership (hold the main loop off it): land it if
                        airborne (a firmware reboot mid-air = it falls), then
                        /api/reboot once on the ground.
          * post_reboot — release the hold; the main loop relaunches once it's
                        back. Give up after WD_REBOOT_WAIT_S.
        'Healthy' = reachable + drone link + flying + telemetry no longer stalled."""
        for fc in list(self._wd_recover):
            rec = self._wd_recover[fc]
            w = worlds.get(fc)
            healthy = (w is not None and w.connected and w.drone_connected
                       and w.flying and not w.telemetry_stalled
                       and not w.camera_stalled)
            if healthy and not rec.get("hold"):
                self._wd_recover.pop(fc, None)
                self._last_action[fc] = "WATCHDOG: recovered (healthy again)"
                continue
            elapsed = now - rec["since"]
            stage = rec["stage"]
            if stage == "relaunch":
                if elapsed >= WD_RECOVER_TAKEOFF_S:
                    rec.update(stage="reboot", hold=True, since=now)   # take ownership
            elif stage == "reboot":
                if w is None or not w.connected:
                    continue                          # offline (maybe rebooting) — wait
                if w.flying:
                    await self.pool.stop_drone(fc)    # land first; can't reboot mid-air
                    self._launched_role.pop(fc, None)
                    self._last_action[fc] = "WATCHDOG: still stuck -> landing to reboot"
                else:
                    ok, payload = await self.pool.reboot_drone(fc)
                    rec.update(stage="post_reboot", hold=False, since=now)
                    self._last_action[fc] = ("WATCHDOG: relaunch failed -> rebooting drone"
                                             if ok else f"WATCHDOG: reboot rejected ({payload})")
                    logger.warning("c2_v2 watchdog: %s reboot -> %s", fc,
                                   "sent" if ok else payload)
            elif stage == "post_reboot":
                if elapsed >= WD_REBOOT_WAIT_S:
                    self._wd_recover.pop(fc, None)
                    self._last_action[fc] = "WATCHDOG: reboot+relaunch failed — needs manual"
                    logger.error("c2_v2 watchdog: %s did not recover after reboot", fc)

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
    it isn't a standard clockwise pair. The clockwise corridors pair red slot r
    (1,2,3) with blue slot (7-r): 1<->6, 2<->5, 3<->4 -> lanes 0,1,2. Used to
    reserve the lane so auto-assignment can't double up on it."""
    if not swimlane or len(swimlane) != 2:
        return None
    lo, hi = sorted(int(s) for s in swimlane)
    return (lo - 1) if (lo in (1, 2, 3) and hi == 7 - lo) else None


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
