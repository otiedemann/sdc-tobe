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

        # 3b) AUTO mode: auto-assign target slots within operator-set roles
        # based on the live slot states. This is the event-reaction layer
        # (enemy captures our slot -> dispatch a free defender, etc.). It
        # only sets intent; the arming gate still controls actual pushes.
        if self.is_auto():
            try:
                self._auto_plan(s)
            except Exception:
                logger.exception("strategy: auto-plan crashed (continuing)")

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

            ctx = RoleContext(
                drone=drone,
                match=s.match,
                state=ds,
                markers=self._markers,
                role_state=rs,
                our_team=s.markers.our_team,
                active_slots=tuple(s.markers.active_slots),
                cruise_alt_m=cruise_alts.get(drone.fc_name),
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
        """Which team's zone a physical slot sits in (1-3 blue, 4-6 red)."""
        return "blue" if 1 <= slot <= 3 else "red"

    def _auto_plan(self, s) -> None:
        """Assign target slots to FREE attackers/defenders from slot state.

        Roles stay operator-set; AUTO only fills in each drone's
        ``target_slot``. A drone is "free" when it has no target and is
        idle/done (not mid-run), so once assigned it won't be re-tasked
        until it finishes. Reactions:
          * an OWN slot held by the enemy  -> a free DEFENDER re-captures it
          * an ENEMY slot not yet ours     -> a free ATTACKER captures it
        """
        our = s.markers.our_team
        enemy = "blue" if our == "red" else "red"
        active = [int(x) for x in s.markers.active_slots]
        our_slots = sorted(x for x in active if self._slot_home_team(x) == our)
        enemy_slots = sorted(x for x in active if self._slot_home_team(x) == enemy)
        holder = {sl: self._markers.slot_holder(sl) for sl in active}

        # Gather free drones + already-targeted slots under the lock; do the
        # actual assign_target() calls outside it (assign_target re-locks).
        with self._states_lock:
            taken: set[int] = set()
            free_def: List[str] = []
            free_att: List[str] = []
            for d in s.drones:
                if not d.enabled or not d.team:
                    continue
                rs = self._role_states.get(d.fc_name)
                if rs is None:
                    continue
                if rs.target_slot is not None:
                    taken.add(int(rs.target_slot))
                    continue
                if rs.phase not in ("", "idle", "done"):
                    continue  # mid-run; leave it be
                if d.role == "defender":
                    free_def.append(d.fc_name)
                elif d.role == "attacker":
                    free_att.append(d.fc_name)

        assignments: list[tuple[str, int, str]] = []
        # Priority 1: defend our threatened slots (enemy holds them).
        for sl in our_slots:
            if not free_def:
                break
            if holder.get(sl) == enemy and sl not in taken:
                fc = free_def.pop(0)
                taken.add(sl)
                assignments.append((fc, sl, f"AUTO: defend slot {sl} (enemy holds it)"))
        # Priority 2: attack enemy slots we don't already hold.
        for sl in enemy_slots:
            if not free_att:
                break
            if holder.get(sl) != our and sl not in taken:
                fc = free_att.pop(0)
                taken.add(sl)
                assignments.append((fc, sl, f"AUTO: attack slot {sl} (holder={holder.get(sl)})"))

        for fc, sl, reason in assignments:
            self.assign_target(fc, sl)
            self._events.add("auto", reason, drone=fc)

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
        """
        MIN_GAP = 0.4
        prefs: list[tuple[float, str]] = []
        for d in s.drones:
            if not d.enabled:
                continue
            pref = float(d.scout_alt_m if d.role == "scout" else d.attack_alt_m)
            prefs.append((pref, d.fc_name))
        prefs.sort()  # ascending altitude, then name
        out: Dict[str, float] = {}
        used: list[float] = []
        for pref, fc in prefs:
            a = pref
            # bump up until clear of every already-placed altitude
            while any(abs(a - u) < MIN_GAP for u in used):
                a += MIN_GAP
            a = round(a, 2)
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
