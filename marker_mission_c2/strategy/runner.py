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
from .markers import MarkerTracker
from .roles import (
    Decision,
    DroneState,
    Role,
    RoleContext,
    RoleState,
    get as get_role,
    in_home_zone,
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
    ) -> None:
        self._settings = settings
        self._c2 = c2
        self._markers = markers
        self._events = events or EventLog()
        self._tick_interval_s = tick_interval_s

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
        self._auto_mode = False
        self._auto_lock = threading.RLock()

        # Loop bookkeeping.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._tick_count = 0
        self._last_tick_unix_s = 0.0
        self._last_overview: Dict[str, Any] = {}
        # Last computed per-drone deconflicted cruise altitudes (for UI).
        self._last_cruise_alts: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public read API (called from Flask handlers, so must be thread-safe)
    # ------------------------------------------------------------------

    @property
    def events(self) -> EventLog:
        return self._events

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

        # 3a) Auto-adopt: any FC the C2 reports that isn't in our roster
        # yet is added automatically — and we PRE-FILL the team + role so
        # the operator only has to Arm + AUTO instead of clicking through
        # every drone. Team is sniffed from the FC name prefix ("red*" →
        # red, "blue*" → blue); anything else stays unassigned. The default
        # role is "attacker" — the planner will only act on attackers that
        # share our_team, and the dashboard lets the operator flip an
        # individual drone to "defender" / "scout" / "idle" if they want.
        known = {d.fc_name for d in s.drones}
        new_fcs = [fc for fc in overview.keys() if fc and fc not in known]
        if new_fcs:
            for fc in sorted(new_fcs):
                name = fc.lower()
                team = ("red" if name.startswith("red")
                        else "blue" if name.startswith("blue")
                        else None)
                self._settings.update_drone(fc, team=team, role="attacker")
                self._events.add(
                    "adopt",
                    f"discovered from C2; pre-set team={team or '?'} role=attacker",
                    drone=fc,
                )
            s = self._settings.snapshot()          # refresh so they dispatch this tick

        # 3b) AUTO mode: auto-assign target slots within operator-set roles
        # based on the live slot states. This is the event-reaction layer
        # (enemy captures our slot -> dispatch a free defender, etc.). It
        # only sets intent; the arming gate still controls actual pushes.
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

            role = get_role(drone.role) or get_role("idle")
            if role is None:
                continue

            # v7 §1.4.4 home-presence: True iff the drone is currently inside
            # its own home zone. False if we don't have a position yet (be
            # conservative — better to hold than push from an unknown spot).
            in_home_now = False
            if drone.team and ds.world_position_m is not None:
                x, y, _z = ds.world_position_m
                in_home_now = in_home_zone(drone.team, x, y)
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
                              role_states_snap, now)
        for p in plays:
            self.assign_target(p.fc_name, p.slot)
            self._events.add(p.play_kind, p.reason, drone=p.fc_name)

    # ------------------------------------------------------------------
    # Height deconfliction
    # ------------------------------------------------------------------

    def _deconflicted_cruise_alts(self, s) -> Dict[str, float]:
        """Return {fc_name: cruise_altitude_m}, each DISTINCT.

        Start from each drone's preferred altitude (scout_alt_m for
        scouts, else attack_alt_m) and nudge upward in MIN_GAP steps so
        no two enabled drones share a cruise height — avoids mid-air
        collisions during loiter/transit. Deterministic by ascending
        preference then fc_name.

        Capped at MAX_ALT_M so a ghost-laden roster (drones lingering in
        settings.json from earlier runs) can't push the stack into the
        ceiling. Once the stack hits the cap we share the cap altitude
        among any overflow drones — operator should clean stale drones
        instead of stacking into the ceiling.
        """
        MIN_GAP = 0.4
        MAX_ALT_M = 5.0     # 1 m below the 6 m arena ceiling
        prefs: list[tuple[float, str]] = []
        for d in s.drones:
            if not d.enabled:
                continue
            pref = float(d.scout_alt_m if d.role == "scout" else d.attack_alt_m)
            prefs.append((pref, d.fc_name))
        prefs.sort()  # ascending altitude, then name
        out: Dict[str, float] = {}
        used: list[float] = []
        # Tiny epsilon protects against float-accumulation in the loop
        # (e.g. 1.0 + 0.4 yields 1.3999999... so the bare ``< MIN_GAP``
        # test trips on the next iteration and we'd over-bump by 0.4).
        EPS = 1e-6
        for pref, fc in prefs:
            a = round(pref, 2)
            # bump up until clear of every already-placed altitude
            while any(abs(a - u) < (MIN_GAP - EPS) for u in used) \
                    and a < MAX_ALT_M:
                a = round(a + MIN_GAP, 2)
            if a > MAX_ALT_M:
                a = MAX_ALT_M
            out[fc] = a
            used.append(a)
        return out

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
