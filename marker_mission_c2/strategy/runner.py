"""Strategy main loop.

Runs an asyncio task that ticks every ``TICK_INTERVAL_S`` and, for each
known drone:
  1. Reads the latest C2 overview.
  2. Updates the marker tracker with the drone's ``visible_marker_ids``.
  3. Dispatches the drone's active role -> :class:`Decision`.
  4. Acts on the decision (push script / stop mission / no-op) via the
     :class:`C2Client`.

The runner is "armed" by the operator before it actually sends any
commands; before that it observes only. This is the safety gate that
prevents drones taking off at boot.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional

from .c2_client import C2Client
from .collision import CollisionAvoider, Track
from .markers import MarkerTracker
from .missions import MissionLog
from .roles import (
    Decision,
    DroneState,
    Role,
    RoleContext,
    RoleState,
    get as get_role,
    in_home_zone,
    push,
)
from .settings import SettingsStore

# Pull role side-effect imports so they register themselves.
from . import scout as _scout  # noqa: F401
from . import attacker as _attacker  # noqa: F401
from . import defender as _defender  # noqa: F401

logger = logging.getLogger(__name__)

TICK_INTERVAL_S = 1.0
EVENT_LOG_MAX = 200


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


@dataclass
class Event:
    unix_s: float
    kind: str
    drone: Optional[str]
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unix_s": self.unix_s,
            "kind": self.kind,
            "drone": self.drone,
            "text": self.text,
        }


class EventLog:
    """Thread-safe ring buffer of events. Keep small; mostly UI fodder."""

    def __init__(self, maxlen: int = EVENT_LOG_MAX) -> None:
        self._lock = threading.RLock()
        self._buf: Deque[Event] = collections.deque(maxlen=maxlen)

    def add(self, kind: str, text: str, drone: Optional[str] = None) -> None:
        ev = Event(unix_s=time.time(), kind=kind, drone=drone, text=text)
        with self._lock:
            self._buf.append(ev)
        logger.info("strategy event [%s] %s%s", kind,
                    f"{drone}: " if drone else "", text)

    def snapshot(self) -> List[Event]:
        with self._lock:
            return list(self._buf)

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.snapshot()]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class SwarmRunner:
    """Owns per-drone role state and pushes scripts on tick."""

    def __init__(
        self,
        *,
        settings: SettingsStore,
        c2: C2Client,
        markers: MarkerTracker,
        events: Optional[EventLog] = None,
        tick_interval_s: float = TICK_INTERVAL_S,
        mission_log: Optional[MissionLog] = None,
    ) -> None:
        self._settings = settings
        self._c2 = c2
        self._markers = markers
        self._events = events or EventLog()
        self._tick_interval_s = tick_interval_s
        # Records the exact mission scripts we push to the FCs (for the
        # operator to read back / copy-paste onto live drones). Always present
        # so callers don't have to null-check; file sink is optional.
        self._mission_log = mission_log or MissionLog()

        self._role_states: Dict[str, RoleState] = {}
        self._states_lock = threading.RLock()

        # Armed = strategy may push/stop scripts. When disarmed the runner
        # still observes (overview + marker tracker) but never commands.
        self._armed = False
        self._armed_lock = threading.RLock()

        # Mode: MANUAL (operator assigns every role+target) vs AUTO (the
        # runner auto-assigns TARGETS within the operator-set roles, and
        # reacts to events — e.g. an enemy capturing one of our slots
        # dispatches a free defender to re-capture it). AUTO only assigns
        # intent; the ``_armed`` gate still controls whether anything is
        # actually pushed to the FCs, so AUTO+disarmed is a safe preview.
        # Default ON so the moment the operator hits Arm the planner
        # immediately dispatches the v7 game-launch sortie (3 attackers ->
        # the 3 enemy slots) without an extra MANUAL->AUTO toggle.
        self._auto_mode = True
        self._auto_lock = threading.RLock()

        # Loop bookkeeping.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._tick_count = 0
        self._last_tick_unix_s = 0.0
        self._last_overview: Dict[str, Any] = {}
        # Last computed per-drone deconflicted cruise altitudes (for UI).
        self._last_cruise_alts: Dict[str, float] = {}
        # Runtime collision avoidance (our-team drones): predicts path/speed
        # conflicts each tick and dodges by altitude. ``_avoiding`` dedups the
        # collision_avoid event so it logs on a NEW conflict, not every tick.
        self._collision = CollisionAvoider()
        self._avoiding: Dict[str, float] = {}

        # Team scoring coordinator (v7 5-pt play). The whole team pulses
        # out-and-back together:
        #   "sortie" — all drones OUT of home, attackers capturing enemy boxes
        #              (5 pts need every drone outside home at the capture).
        #   "bank"   — we just captured an enemy box -> recall EVERY drone home
        #              to secure the attempt (5 pts need all home before the box
        #              is recaptured), then sortie again.
        # _last_cap_seen tracks the newest capture-event ts we've reacted to so
        # a single capture triggers exactly one bank.
        self._team_phase = "sortie"
        self._bank_since = 0.0
        self._bank_pending_since = 0.0
        self._last_cap_seen = 0.0
        # Wall-clock the last bank ENDED (sortie resumed). Banking is suppressed
        # for BANK_COOLDOWN_S after this so attackers get an uninterrupted
        # attack cycle between banks (no thrash). 0 = never banked yet.
        self._last_sortie_resumed = 0.0

    # ------------------------------------------------------------------
    # Public read API (called from Flask handlers, so must be thread-safe)
    # ------------------------------------------------------------------

    @property
    def events(self) -> EventLog:
        return self._events

    @property
    def mission_log(self) -> MissionLog:
        return self._mission_log

    def is_armed(self) -> bool:
        with self._armed_lock:
            return self._armed

    def arm(self, source: str = "operator") -> None:
        with self._armed_lock:
            if self._armed:
                return
            self._armed = True
        self._events.add("arm", f"runner armed by {source}")

    def disarm(self, source: str = "operator") -> None:
        with self._armed_lock:
            if not self._armed:
                return
            self._armed = False
        self._events.add("disarm", f"runner disarmed by {source}")

    def is_auto(self) -> bool:
        with self._auto_lock:
            return self._auto_mode

    def set_mode(self, auto: bool, source: str = "operator") -> None:
        auto = bool(auto)
        with self._auto_lock:
            if self._auto_mode == auto:
                return
            self._auto_mode = auto
        self._events.add(
            "mode", f"{'AUTO' if auto else 'MANUAL'} (by {source})"
        )

    def tick_count(self) -> int:
        return self._tick_count

    def role_state(self, fc_name: str) -> Optional[RoleState]:
        with self._states_lock:
            return self._role_states.get(fc_name)

    def role_states(self) -> Dict[str, RoleState]:
        with self._states_lock:
            return {fc: rs for fc, rs in self._role_states.items()}

    def reset_roster(self, source: str = "operator") -> int:
        """Drop every drone from the strategy roster and clear all per-drone
        role memory. Auto-adopt will re-fill from the C2 overview on the
        next tick with FRESH DroneSettings (current code defaults), which
        is the easiest way to flush stale persisted values (e.g. an old
        scout_alt_m of 1.8 m) and any ghost drones lingering from earlier
        runs. Returns the number of drones removed."""
        removed = self._settings.reset_drones()
        with self._states_lock:
            self._role_states.clear()
        self._last_cruise_alts = {}
        self._events.add(
            "roster_reset",
            f"roster cleared ({removed} drone(s)) by {source}; "
            f"auto-adopt will refill from C2 overview",
        )
        return removed

    def sync_role_state(self, fc_name: str) -> None:
        """Pull the drone's role from settings into RoleState immediately.

        Called by the web layer after a settings change so a follow-up
        :meth:`assign_target` doesn't get wiped by the runner's lazy reset
        on the next tick.
        """
        d = self._settings.drone(fc_name)
        if d is None:
            return
        with self._states_lock:
            rs = self._role_states.setdefault(
                fc_name, RoleState(fc_name=fc_name)
            )
            if rs.role != d.role:
                self._events.add(
                    "role",
                    f"{rs.role!r} -> {d.role!r} (sync)",
                    drone=fc_name,
                )
                rs.reset_for_role(d.role)

    def assign_target(self, fc_name: str, slot: Optional[int]) -> None:
        """Assign (or clear) an attacker's target slot (1..6)."""
        with self._states_lock:
            rs = self._role_states.setdefault(
                fc_name, RoleState(fc_name=fc_name)
            )
            rs.target_slot = int(slot) if slot is not None else None
            rs.target_assigned_unix_s = time.time() if slot is not None else None
            rs.last_attack_marker_id = None
            rs.advance_phase(
                "idle",
                reason=(
                    "target cleared" if slot is None
                    else f"slot {slot} assigned"
                ),
            )
        self._events.add(
            "target",
            f"{'cleared' if slot is None else f'assigned slot {slot}'}",
            drone=fc_name,
        )

    def snapshot(self) -> Dict[str, Any]:
        # Expose the last C2 overview (per-FC live telemetry: drone_connected,
        # battery, height, world_position, visible_marker_ids, FC phase, …)
        # so the strategy dashboard can render a *detailed* per-drone status
        # without a separate fetch. The overview is refreshed every tick by
        # ``_tick_once``; here we just pass through the latest snapshot.
        return {
            "armed": self.is_armed(),
            "auto": self.is_auto(),
            "mode": "auto" if self.is_auto() else "manual",
            "tick_count": self._tick_count,
            "last_tick_unix_s": self._last_tick_unix_s,
            "tick_interval_s": self._tick_interval_s,
            "cruise_alts": dict(self._last_cruise_alts),
            "team_phase": self._team_phase,
            "drones": {
                fc: rs.to_dict() for fc, rs in self.role_states().items()
            },
            "overview": dict(self._last_overview),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._loop_body(), name="strategy-runner")
        self._events.add("runner", "loop started (disarmed)")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        self._events.add("runner", "loop stopped")

    # ------------------------------------------------------------------
    # Loop body
    # ------------------------------------------------------------------

    async def _loop_body(self) -> None:
        try:
            while True:
                t0 = time.time()
                try:
                    await self._tick_once()
                except Exception:
                    logger.exception("strategy: tick crashed (continuing)")
                self._tick_count += 1
                self._last_tick_unix_s = time.time()
                elapsed = time.time() - t0
                await asyncio.sleep(max(0.05, self._tick_interval_s - elapsed))
        except asyncio.CancelledError:
            raise

    async def _tick_once(self) -> None:
        # 1) Pull settings snapshot once per tick (fresh each loop).
        s = self._settings.snapshot()
        # Keep the marker tracker's active slots in sync with config.
        self._markers.set_active_slots(s.markers.active_slots)

        # 2) Read C2 overview.
        ok, overview = await self._c2.overview()
        if not ok:
            return
        self._last_overview = overview

        # 3) Ingest visible markers from every connected drone.
        for fc_name, item in overview.items():
            state = (item or {}).get("state") or {}
            vmids = state.get("visible_marker_ids") or []
            try:
                ids = [int(x) for x in vmids]
            except (TypeError, ValueError):
                continue
            if ids:
                self._markers.ingest(fc_name, ids)

        # 3a) Auto-adopt: every FC the C2 reports that isn't in our roster
        # yet is added automatically — and we PRE-FILL team + role so the
        # operator only has to Arm instead of clicking through every drone.
        #
        # TEAM ISOLATION (regs §1.3 — "no access to enemy drones"): a red
        # strategy must never enrol blue drones (and vice versa). We sniff
        # team from the FC name prefix ("red*" → red, "blue*" → blue) and
        # SKIP drones whose prefix belongs to the other team. Anything
        # without a team prefix gets tagged with OUR team by default
        # (manual fleet — operator still controls the slot/role).
        #
        # ROLE SPREAD (req 6): the v7 launch plan is
        #   1 scout (rotates at the arena centre to watch slot states)
        # + 1 defender (anti-camp patrol of own slots, then home hover)
        # + N-2 attackers (one per enemy slot via the planner).
        # We hand out roles in that order across freshly-adopted OUR-team
        # drones, counting existing ones first so adoption is idempotent
        # across runner restarts.
        known = {d.fc_name for d in s.drones}
        our_team = (s.markers.our_team or "").lower()
        new_fcs = [fc for fc in overview.keys() if fc and fc not in known]
        if new_fcs:
            our_count = sum(1 for d in s.drones
                            if d.team and d.team.lower() == our_team and d.enabled)
            for fc in sorted(new_fcs):
                name = fc.lower()
                sniffed = ("red" if name.startswith("red")
                           else "blue" if name.startswith("blue")
                           else None)
                # Drop drones that explicitly belong to the OTHER team —
                # the strategy must not know they exist.
                if sniffed and our_team and sniffed != our_team:
                    self._events.add(
                        "adopt_skip",
                        f"ignored (team {sniffed} != ours {our_team or '?'})",
                        drone=fc,
                    )
                    continue
                team = sniffed or our_team or None
                # Role spread: 1st on our team -> scout, 2nd -> defender,
                # then attackers. With 5 drones we land on 1+1+3.
                if our_count == 0:
                    role = "scout"
                elif our_count == 1:
                    role = "defender"
                else:
                    role = "attacker"
                our_count += 1
                self._settings.update_drone(fc, team=team, role=role)
                self._events.add(
                    "adopt",
                    f"discovered from C2; pre-set team={team or '?'} role={role}",
                    drone=fc,
                )
            s = self._settings.snapshot()          # refresh so they dispatch this tick

        # 3a.5) Team scoring coordinator: flip between "sortie" (attack, all
        # drones out) and "bank" (we captured an enemy box -> recall all home
        # to secure the 5-pt attempt). Must run BEFORE the planner so a bank
        # suppresses new attacks this very tick.
        try:
            self._update_team_phase(s, overview)
        except Exception:
            logger.exception("strategy: team-phase update crashed (continuing)")

        # 3b) AUTO mode: auto-assign target slots within operator-set roles
        # based on the live slot states. Runs EVERY tick (including during a
        # bank) so a defensive RE-CAPTURE of one of our own flipped boxes is
        # never starved — the planner itself suppresses new enemy ATTACK plays
        # while banking, but always schedules recaptures (highest priority).
        if self.is_auto():
            try:
                self._auto_plan(s)
            except Exception:
                logger.exception("strategy: auto-plan crashed (continuing)")

        # 3b.5) v7 §1.4.3 Special-Maneuver detection. If our team holds all
        # six slots continuously for ≥ 5 s the match ends with an instant
        # win. We poll the marker tracker each tick and emit a one-shot
        # ``match_won`` event the moment the dwell crosses the threshold.
        try:
            our_team = (s.markers.our_team or "").lower()
            ms = self._markers.match_status(our_team, time.time())
            if ms.get("just_won"):
                self._events.add(
                    "match_won",
                    f"Special Maneuver: {our_team} held all 6 slots for "
                    f"{ms['dwell_s']:.1f}s — INSTANT WIN",
                )
        except Exception:
            logger.exception("strategy: match_status check crashed (continuing)")

        # 3c) Height deconfliction: assign each enabled drone a DISTINCT
        # cruise altitude so two drones never loiter/transit at the same
        # height. Roles read this via RoleContext.cruise_alt_m.
        cruise_alts = self._deconflicted_cruise_alts(s)
        self._last_cruise_alts = cruise_alts

        # 3c.5) Per-drone home-park lanes: spread OUR team across the home
        # zone's x so a bank recall doesn't converge everyone on one point
        # (which made them cross altitude bands and storm the collision
        # avoider). Stable order by fc_name -> stable lane assignment.
        home_parks = self._home_park_lanes(s)

        # 3d) Collision avoidance: predict our drones' paths from live position
        # + speed and, where two are on course to come within SAFETY_RADIUS,
        # tell the lower-priority one to change altitude. Computed once per tick
        # over OUR enabled drones with a known position (regs §1.3: we can't see
        # enemy drones, so they're out of scope — cross-team separation is
        # handled by the static red/blue altitude convention).
        avoid = self._collision_overrides(s, overview)

        # 4) Dispatch each known drone (drones from settings, not C2 — the
        # operator decides who plays; the C2 overview tells us if they're
        # online).
        for drone in s.drones:
            ds = DroneState.from_overview(drone.fc_name, overview.get(drone.fc_name) or {})
            with self._states_lock:
                rs = self._role_states.setdefault(
                    drone.fc_name, RoleState(fc_name=drone.fc_name)
                )
                if rs.role != drone.role:
                    self._events.add(
                        "role",
                        f"{rs.role!r} -> {drone.role!r}",
                        drone=drone.fc_name,
                    )
                    rs.reset_for_role(drone.role)

            # v7 §1.4.4 home-presence: True iff the drone is currently inside
            # its own home zone. False if we don't have a position yet (be
            # conservative — better to hold than push from an unknown spot).
            in_home_now = False
            if drone.team and ds.world_position_m is not None:
                x, y, _z = ds.world_position_m
                in_home_now = in_home_zone(drone.team, x, y)

            # Collision avoidance OVERRIDES the role this tick: if this drone is
            # predicted to conflict, change its altitude instead of running its
            # mission. BUT skip the override for a drone that is HOME and not
            # mid-attack: re-arming attackers all cluster near the home-park
            # point, which trips the avoider and FREEZES them in a HOOVER hold —
            # the "8-second stall facing the home wall" the operator saw. At
            # home the next attack launch (a TO toward the enemy box) naturally
            # spreads them out within a second, so a hold there is both
            # unnecessary and harmful. Only hold drones conflicting OUT in the
            # neutral/enemy transit corridor (the real collision risk).
            rs_now = self._role_states.get(drone.fc_name)
            mid_attack = bool(rs_now and rs_now.phase in ("running",))
            if drone.fc_name in avoid and not (in_home_now and not mid_attack):
                target, reason = avoid[drone.fc_name]
                await self._apply_decision(drone.fc_name, push(
                    f"HEIGHT {target:.2f}\nHOOVER 1.0",
                    reason=f"collision avoidance: {reason}",
                ))
                continue

            role = get_role(drone.role) or get_role("idle")
            if role is None:
                continue
            ctx = RoleContext(
                drone=drone,
                match=s.match,
                state=ds,
                markers=self._markers,
                role_state=rs,
                our_team=s.markers.our_team,
                active_slots=tuple(s.markers.active_slots),
                cruise_alt_m=cruise_alts.get(drone.fc_name),
                in_home_now=in_home_now,
                team_phase=self._team_phase,
                home_park_xy=home_parks.get(drone.fc_name),
            )

            try:
                decision = role.decide(ctx)
            except Exception:
                logger.exception(
                    "strategy: role %s for %s raised", role.name, drone.fc_name
                )
                continue

            await self._apply_decision(drone.fc_name, decision)

    # ------------------------------------------------------------------
    # AUTO mode planner
    # ------------------------------------------------------------------

    @staticmethod
    def _slot_home_team(slot: int) -> str:
        """Which team's zone a physical slot sits in.

        Per SDC26 v7 regs §1.4.2 Table 1: boxes 1-3 sit in the RED home
        zone (default markers 41/42/43), boxes 4-6 in the BLUE home zone
        (default markers 34/35/36).
        """
        return "red" if 1 <= slot <= 3 else "blue"

    def _auto_plan(self, s) -> None:
        """Pick the highest-EV v7 play available this tick and apply it.

        Delegates the play selection to :mod:`strategy.planner` which ranks
        feasibility-checked plays (defensive recap > 10-pt double-strike >
        1-pt singleton) and returns the assignments to apply. We pass the
        latest C2 overview through so the planner can honour the v7
        §1.4.4 home-presence gate + FC-INIT readiness check per drone.
        """
        from . import planner as _planner
        now = time.time()
        with self._states_lock:
            role_states_snap = dict(self._role_states)
        plays = _planner.plan(s, self._markers, self._last_overview,
                              role_states_snap, now, team_phase=self._team_phase)
        for p in plays:
            self.assign_target(p.fc_name, p.slot)
            self._events.add(p.play_kind, p.reason, drone=p.fc_name)

    # ------------------------------------------------------------------
    # Height deconfliction
    # ------------------------------------------------------------------

    def _deconflicted_cruise_alts(self, s) -> Dict[str, float]:
        """Return {fc_name: cruise_altitude_m}, distinct across BOTH teams.

        Every drone gets its own height so no two ever share one — the rule is
        >= MIN_GAP (0.30 m) between ANY two drones, including enemy drones. We
        can't see enemy drones (regs §1.3), so both teams follow one fixed
        shared CONVENTION keyed off team colour: RED takes EVEN rungs, BLUE the
        ODD ones, so the two ladders interleave to >= MIN_GAP without either
        side reading the other's altitudes. Two bands:

        SCOUTS fly LOW — level with the ~1 m box faces — so they actually SEE
        the slot markers and keep the slot-state belief fresh. A high scout is
        useless for this: the box face sits at ~1 m, so from 3.9 m the line to
        a centre slot is sqrt(7.5^2 + 2.9^2) ~ 8.0 m — past the 8 m vision
        range — AND hits the marker at a steep angle. Dropping to ~1 m brings
        it to 7.5 m (in range) and head-on.
            red scout  -> SCOUT_BASE + 0*GAP = 1.0 m
            blue scout -> SCOUT_BASE + 1*GAP = 1.3 m

        MOVERS (attackers + defenders) fly the CRUISE band, above the 1.4 m box
        tops for collision-free transit and >= MIN_GAP above the highest scout:
            red  -> 1.6, 2.2, 2.8, 3.4   blue -> 1.9, 2.5, 3.1, 3.7

        All 10: 1.0 1.3 1.6 1.9 2.2 2.5 2.8 3.1 3.4 3.7 — uniform 0.30 m apart.
        Capped at MAX_ALT_M; ordered by fc_name for a stable mapping.
        """
        MIN_GAP = 0.30        # >= 30 cm between ANY two drones (both teams)
        SCOUT_BASE_M = 1.00   # scouts loiter low, level with the box faces
        CRUISE_BASE_M = 1.60  # movers: >= MIN_GAP above the top scout (1.3 m)
                              # AND clear of the 1.4 m box tops for transit
        MAX_ALT_M = 5.0       # 1 m below the 6 m arena ceiling
        # Team isolation (regs §1.3): only assign rungs to OUR-team drones.
        our_team = (s.markers.our_team or "").lower()
        parity = 1 if our_team == "blue" else 0   # red/unknown -> even rungs
        scouts: list[str] = []
        movers: list[str] = []
        for d in s.drones:
            if not d.enabled:
                continue
            if our_team and d.team and d.team.lower() != our_team:
                continue
            (scouts if d.role == "scout" else movers).append(d.fc_name)
        out: Dict[str, float] = {}
        # Scouts: low band, interleaved by team so the two centre-loitering
        # scouts never share a height.
        for i, fc in enumerate(sorted(scouts)):
            k = 2 * i + parity
            out[fc] = round(SCOUT_BASE_M + k * MIN_GAP, 2)
        # Movers: cruise band above the scouts, interleaved by team.
        for i, fc in enumerate(sorted(movers)):
            k = 2 * i + parity
            out[fc] = min(round(CRUISE_BASE_M + k * MIN_GAP, 2), MAX_ALT_M)
        return out

    def _home_park_lanes(self, s) -> Dict[str, tuple]:
        """Return {fc: (x, y)} home-park points, x spread across the home zone.

        Each OUR-team drone gets its own lane so a bank recall fans them out
        instead of stacking everyone on x=0 (which crossed altitude bands and
        triggered the collision-avoidance storm that stalled re-attacks).
        """
        from .roles import home_park_xy as _hp
        our = (s.markers.our_team or "").lower()
        ours = sorted(d.fc_name for d in s.drones
                      if d.enabled and d.team and d.team.lower() == our)
        n = len(ours)
        team = our or "red"
        return {fc: _hp(team, i, n) for i, fc in enumerate(ours)}

    # ------------------------------------------------------------------
    # Collision avoidance
    # ------------------------------------------------------------------

    def _collision_overrides(self, s, overview: Dict[str, Any]
                             ) -> Dict[str, tuple]:
        """Predict path conflicts among OUR drones; return altitude dodges.

        Returns ``{fc: (target_alt_m, reason)}`` for drones that must change
        altitude this tick to avoid a predicted collision. Emits a one-shot
        ``collision_avoid`` event per new conflict (deduped via ``_avoiding``).
        """
        our_team = (s.markers.our_team or "").lower()
        positions: Dict[str, tuple] = {}
        roles: Dict[str, str] = {}
        for d in s.drones:
            if not d.enabled:
                continue
            if our_team and d.team and d.team.lower() != our_team:
                continue
            ds = DroneState.from_overview(d.fc_name, overview.get(d.fc_name) or {})
            pos = ds.world_position_m
            if not pos or len(pos) < 3:
                continue
            positions[d.fc_name] = (float(pos[0]), float(pos[1]), float(pos[2]))
            roles[d.fc_name] = d.role

        now = time.time()
        vel = self._collision.update_velocities(positions, now)
        tracks = [Track(fc, positions[fc], vel[fc], roles.get(fc, "idle"))
                  for fc in positions]
        overrides = self._collision.resolve(tracks)

        # One-shot event per new/changed conflict; clear when resolved.
        for fc, (target, reason) in overrides.items():
            if abs(self._avoiding.get(fc, -99.0) - target) > 0.05:
                self._events.add("collision_avoid", reason, drone=fc)
            self._avoiding[fc] = target
        for fc in list(self._avoiding):
            if fc not in overrides:
                del self._avoiding[fc]
        return overrides

    # ------------------------------------------------------------------
    # Team scoring coordinator (v7 5-pt play)
    # ------------------------------------------------------------------

    # Safety-net cap on bank duration. The PRIMARY exit is all-5-home (per the
    # operator spec); this only fires if a drone can't get home (lost position,
    # stuck). Sized to the worst-case homeward transit: a committed attacker at
    # y~7.5 is ~13.5 m out, ~7 s at 2 m/s, +turn/brake -> 15 s covers it so the
    # all-home path normally wins first. The operator accepts this return time
    # ("ALL 5 must return... by the moment the last drone arrives").
    BANK_TIMEOUT_S = 15.0
    # Grace window after the FIRST enemy capture before we actually bank, so a
    # near-simultaneous SECOND capture (the 10-pt double-strike, < 1 s apart)
    # can complete first. > 1 s (the double-strike window) + a margin for
    # capture-detection lag (the scout has to SEE the second flip).
    BANK_GRACE_S = 2.5
    # After a bank ends (sortie resumes), suppress the NEXT bank for this long.
    # Without it the team banks every ~30 s, and each bank aborts all in-flight
    # attacks — so attackers thrashed near home and never reached the enemy
    # boxes. A cooldown lets the attackers complete a full out-and-back attack
    # cycle (~45 s) between banks, so most of the match is spent ATTACKING, with
    # the occasional bank to bag a 5-pt bonus rather than constant whipsawing.
    BANK_COOLDOWN_S = 45.0

    def _update_team_phase(self, s, overview: Dict[str, Any]) -> None:
        """Flip team phase sortie<->bank to maximise 5-pt scoring plays.

        v7 5-pt rule: a capture earns 5 (not 1) pts when EVERY one of our
        drones is outside our home zone at the capture, and then ALL of them
        return home before the box is recaptured. So the whole team pulses:

          sortie : everyone OUT (scout centre, defender neutral, attackers on
                   enemy boxes) -> attackers capture.
          bank   : the instant we capture an enemy box, recall EVERY drone
                   home; once all are home the attempt is secured -> sortie.

        Transitions:
          sortie -> bank : a fresh capture event by us on an ENEMY box
                           (home recaptures don't count — those are 0 pts).
          bank   -> sortie: all our drones detected home, or BANK_TIMEOUT_S.
        """
        now = time.time()
        our = (s.markers.our_team or "").lower()

        # Detect a fresh enemy-box capture by us since we last reacted.
        newest = self._last_cap_seen
        captured_enemy_box = False
        for ev in self._markers.capture_events(limit=20):
            u = float(ev.get("unix_s", 0.0))
            if u <= self._last_cap_seen:
                continue
            newest = max(newest, u)
            if ev.get("new") == our and not ev.get("home_recap"):
                captured_enemy_box = True
        self._last_cap_seen = newest

        if self._team_phase == "sortie":
            # First enemy capture arms a short grace timer rather than banking
            # immediately. The 10-pt double-strike needs a SECOND capture < 1 s
            # later; banking the instant the first lands would recall the
            # partner before it fires. We hold sortie for BANK_GRACE_S so a
            # near-simultaneous second capture can complete, THEN bank. Against
            # the ~40 s the enemy needs to fly over and recapture, this <2 s
            # delay costs nothing on the 5-pt "return before recapture" clock.
            # Cooldown gate: don't arm a new bank until the attackers have had
            # an uninterrupted attack cycle since the last one ended. Captures
            # during the cooldown still score (1 pt each) — we just don't pull
            # everyone home to chase the 5-pt upgrade every single time.
            in_cooldown = (self._last_sortie_resumed > 0.0
                           and now - self._last_sortie_resumed < self.BANK_COOLDOWN_S)
            if (captured_enemy_box and self._bank_pending_since <= 0.0
                    and not in_cooldown):
                self._bank_pending_since = now
                self._events.add(
                    "team_bank_arm",
                    f"captured an enemy box — holding sortie {self.BANK_GRACE_S:.1f}s "
                    "for a possible 10-pt double-strike, then banking",
                )
            if (self._bank_pending_since > 0.0
                    and now - self._bank_pending_since >= self.BANK_GRACE_S):
                self._team_phase = "bank"
                self._bank_since = now
                self._bank_pending_since = 0.0
                self._events.add(
                    "team_bank",
                    "recall ALL drones home to secure the scoring attempt "
                    "(no new attacks until everyone is home)",
                )
            return

        # team_phase == "bank": the operator spec is explicit — ALL FIVE drones
        # must return to the home zone, and THE MOMENT THE LAST ONE ARRIVES the
        # whole team instantly redeploys (scout->centre, defender->neutral,
        # attackers->re-attack). So we exit the bank only when every one of our
        # drones is detected home. The redeploy is automatic: flipping back to
        # "sortie" makes each role re-decide on the very next tick (~1 s).
        ours = [d for d in s.drones
                if d.enabled and d.team and d.team.lower() == our]
        known = 0
        homed = 0
        for d in ours:
            ds = DroneState.from_overview(d.fc_name, overview.get(d.fc_name) or {})
            pos = ds.world_position_m
            if not pos:
                continue
            known += 1
            if in_home_zone(d.team, pos[0], pos[1]):
                homed += 1
        all_home = known == len(ours) and known > 0 and homed == known
        if all_home:
            self._team_phase = "sortie"
            self._last_sortie_resumed = now      # start the bank cooldown
            self._events.add(
                "team_sortie",
                f"all {homed} drones home — redeploy NOW "
                "(scout->centre, defender->neutral, attackers re-attack)",
            )
        elif now - self._bank_since > self.BANK_TIMEOUT_S:
            # Safety net: a drone that lost its position / can't get home must
            # not deadlock the team forever. Redeploy anyway.
            self._team_phase = "sortie"
            self._last_sortie_resumed = now
            self._events.add(
                "team_sortie",
                f"bank timeout ({self.BANK_TIMEOUT_S:.0f}s, {homed}/{len(ours)} "
                "home) — redeploy to keep playing",
            )

    # ------------------------------------------------------------------
    # Decision dispatch
    # ------------------------------------------------------------------

    async def _apply_decision(self, fc_name: str, decision: Decision) -> None:
        rs = self.role_state(fc_name)
        if rs is None:
            return

        rs.last_decision_reason = decision.reason or rs.last_decision_reason

        if decision.kind == "noop":
            return

        # Honour arming gate: never command FCs while disarmed. Always log
        # what we would have done, so the operator can see decisions are
        # being made even before arming.
        if not self.is_armed():
            self._events.add(
                "disarmed_skip",
                f"{decision.kind} suppressed (disarmed): {decision.reason}",
                drone=fc_name,
            )
            return

        if decision.kind == "push":
            # Throttle: same script within MIN_PUSH_INTERVAL_S is suppressed.
            now = time.time()
            same_script = decision.script == rs.last_pushed_script
            recently = (now - rs.last_pushed_unix_s) < 2.0
            if same_script and recently:
                return
            ok, _ = await self._c2.start(fc_name, script=decision.script)
            if ok:
                rs.last_pushed_script = decision.script
                rs.last_pushed_unix_s = now
                if decision.new_phase:
                    rs.advance_phase(decision.new_phase, decision.reason)
                # Record the EXACT script we sent so the operator can read it
                # back / copy-paste it onto a live drone (mission-step log).
                self._mission_log.record(fc_name, decision.reason, decision.script)
                self._events.add(
                    "script_push",
                    f"pushed script ({len(decision.script.splitlines())} lines): "
                    f"{decision.reason}",
                    drone=fc_name,
                )
            else:
                self._events.add(
                    "script_push_failed",
                    f"C2 rejected push: {decision.reason}",
                    drone=fc_name,
                )
            return

        if decision.kind == "stop":
            ok, _ = await self._c2.stop(fc_name)
            self._events.add(
                "stop" if ok else "stop_failed",
                decision.reason,
                drone=fc_name,
            )
            rs.advance_phase("idle", decision.reason)
            return
