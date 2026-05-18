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

# Pull SCOUT/ATTACKER side-effect imports so they register themselves.
from . import scout as _scout  # noqa: F401
from . import attacker as _attacker  # noqa: F401

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

        # Loop bookkeeping.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._tick_count = 0
        self._last_tick_unix_s = 0.0
        self._last_overview: Dict[str, Any] = {}

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
            "tick_count": self._tick_count,
            "last_tick_unix_s": self._last_tick_unix_s,
            "tick_interval_s": self._tick_interval_s,
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
