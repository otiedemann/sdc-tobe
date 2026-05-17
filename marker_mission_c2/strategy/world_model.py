"""SwarmState: a tick-snapshot of the whole fleet, derived from FCPool.

Decouples the rest of the strategy layer from the FCPool's mutable
state. The planner / tasks / safety read this immutable dataclass and
never reach into the pool directly, so unit tests just construct a
``SwarmState`` and drive the strategy with synthetic data.

Build cost is one ``FCPool.snapshot()`` call per tick — already a deep
copy, so reads here are cheap.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

log = logging.getLogger("c2.strategy.world")


@dataclass(frozen=True)
class DroneObservation:
    """Everything the strategy layer needs to know about one drone for
    one tick. ``None`` / sentinel fields mean the FC didn't report.

    Coordinates are in arena frame (metres). ``yaw_deg`` is the drone's
    heading relative to arena +Y.
    """
    name: str                          # FC name from C2 config (flightctrl1, ...)
    online: bool                       # FC reachable via HTTP
    drone_connected: bool              # FC says drone link is up
    flying: bool                       # drone airborne
    battery_pct: Optional[float]       # 0..100
    phase: Optional[str]               # mission phase string from /api/state
    pose: Optional[Tuple[float, float, float]]  # arena (x, y, z)
    yaw_deg: Optional[float]
    last_marker_id: Optional[int]      # most recently detected ArUco id
    serial: Optional[str]
    last_error: Optional[str]
    age_s: Optional[float]             # seconds since last successful FC poll

    @property
    def stale(self) -> bool:
        """True if we haven't heard from this FC recently. Heuristic only —
        callers should use their own threshold for "stale enough to act
        on" (see safety.SafetyConfig.poll_stale_s)."""
        return self.age_s is None or self.age_s > 3.0


@dataclass(frozen=True)
class SwarmState:
    """Read-only snapshot of the fleet at one instant in time.

    Construct via :meth:`SwarmWorldModel.observe`; never mutate.
    """
    t: float                                    # time.monotonic() at snapshot
    drones: Mapping[str, DroneObservation]      # keyed by FC name

    def online(self) -> Mapping[str, DroneObservation]:
        return {n: d for n, d in self.drones.items() if d.online}

    def flying(self) -> Mapping[str, DroneObservation]:
        return {n: d for n, d in self.drones.items() if d.flying}


class SwarmWorldModel:
    """Builds :class:`SwarmState` snapshots from a live :class:`FCPool`.

    The pool already polls each FC's ``/api/state`` at
    ``state_poll_hz`` (default 5 Hz). We just lift its latest values
    into the strategy's preferred shape and field names.

    Schema notes on FC state — these are best-effort lookups; the
    pool's ``state`` dict is whatever the FC returned, so missing
    keys become ``None``:

      state.telemetry.battery        → battery_pct
      state.telemetry.yaw            → yaw_deg
      state.mission.phase            → phase
      state.world_position_m         → pose       (THE source of truth)

    Pose is read from ``state.world_position_m`` — the marker_mission
    arena-aware solver's output, same field the FC's own UI shows as
    "World position". The legacy ``/api/position`` endpoint reports a
    different position (different solver, different coordinate frame
    — gave bogus y=22.7 m values on a 20 m-deep arena) and is
    deliberately NOT used here. ``world_position_age_s`` indicates
    freshness; we treat a missing or stale value as "no fix" and
    leave ``pose=None``.
    """

    # World position is considered stale beyond this — drone may be
    # mid-flight but the solver lost its marker lock; the strategy
    # should treat that the same as "no pose" rather than acting on a
    # multi-second-old fix.
    POSE_FRESHNESS_S = 2.0

    def __init__(self, fc_pool) -> None:
        # Avoid hard import so this module is testable without httpx.
        self._pool = fc_pool

    def observe(self) -> SwarmState:
        t = time.monotonic()
        snap = self._pool.snapshot()
        drones: dict[str, DroneObservation] = {}
        for name, entry in snap.items():
            try:
                drones[name] = _extract(name, entry,
                                        pose_freshness_s=self.POSE_FRESHNESS_S)
            except Exception:
                log.exception("world_model: failed to extract obs for %s", name)
                drones[name] = DroneObservation(
                    name=name, online=False, drone_connected=False,
                    flying=False, battery_pct=None, phase=None,
                    pose=None, yaw_deg=None, last_marker_id=None,
                    serial=None, last_error="world_model extract crashed",
                    age_s=None,
                )
        return SwarmState(t=t, drones=drones)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract(name: str, entry: dict,
             pose_freshness_s: float = 2.0) -> DroneObservation:
    state = entry.get("state") or {}
    tel = (state.get("telemetry") or {}) if isinstance(state, dict) else {}
    mission = (state.get("mission") or {}) if isinstance(state, dict) else {}

    def _f(d: dict, k: str) -> Optional[float]:
        v = d.get(k)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # Pose: read from state.world_position_m (marker_mission's arena
    # solver). Treat stale fixes (> pose_freshness_s) as "no pose" so
    # the strategy doesn't act on a multi-second-old position.
    pose_xyz: Optional[Tuple[float, float, float]] = None
    wp = state.get("world_position_m") if isinstance(state, dict) else None
    wp_age = state.get("world_position_age_s") if isinstance(state, dict) else None
    if (isinstance(wp, (list, tuple)) and len(wp) >= 3
            and (wp_age is None or float(wp_age) <= pose_freshness_s)):
        try:
            pose_xyz = (float(wp[0]), float(wp[1]), float(wp[2]))
        except (TypeError, ValueError):
            pose_xyz = None

    # Phase lives at state.phase (top-level), not state.mission.phase
    # on the current FC. Try both for resilience across FC versions.
    phase = state.get("phase") if isinstance(state.get("phase"), str) else None
    if phase is None:
        phase = (mission.get("phase")
                 if isinstance(mission.get("phase"), str) else None)

    return DroneObservation(
        name=name,
        online=bool(entry.get("connection_ok")),
        drone_connected=bool(entry.get("drone_connected")),
        flying=bool(tel.get("flying")),
        battery_pct=_f(tel, "battery"),
        phase=phase,
        pose=pose_xyz,
        yaw_deg=_f(tel, "yaw"),
        last_marker_id=(int(state.get("active_marker_id"))
                        if isinstance(state.get("active_marker_id"), (int, float))
                        else (int(state.get("last_marker_id"))
                              if isinstance(state.get("last_marker_id"), (int, float))
                              else None)),
        serial=entry.get("drone_serial"),
        last_error=entry.get("last_error"),
        age_s=(float(entry.get("last_state_age_s"))
               if entry.get("last_state_age_s") is not None else None),
    )
