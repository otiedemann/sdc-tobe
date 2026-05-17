"""SwarmState: a tick-snapshot of the whole fleet, derived from FCPool.

Decouples the rest of the strategy layer from the FCPool's mutable
state. The planner / tasks / safety read this immutable dataclass and
never reach into the pool directly, so unit tests just construct a
``SwarmState`` and drive the strategy with synthetic data.

Build cost is one ``FCPool.snapshot()`` call per tick — already a deep
copy, so reads here are cheap.
"""
from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

import requests

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

    Pose is *not* in ``/api/state`` — the FC only publishes it on
    ``/api/position``. We fetch that endpoint per online FC inside
    :meth:`observe` so the strategy sees ``pose`` whenever the FC's
    positioning solver has a fresh fix. Stale / disabled / missing
    position responses leave ``pose=None`` and don't block the tick.
    """

    POSITION_FETCH_TIMEOUT_S = 0.3

    def __init__(self, fc_pool) -> None:
        # Avoid hard import so this module is testable without httpx.
        self._pool = fc_pool
        # One requests.Session keeps an HTTP keepalive per FC, so each
        # /api/position fetch is a sub-10 ms TCP-reused round trip on
        # the same tailnet — fine to call at the strategy tick rate.
        self._http = requests.Session()

    def observe(self) -> SwarmState:
        t = time.monotonic()
        snap = self._pool.snapshot()
        drones: dict[str, DroneObservation] = {}
        for name, entry in snap.items():
            try:
                obs = _extract(name, entry)
                pose = self._fetch_pose(name, entry)
                if pose is not None:
                    obs = dataclasses.replace(obs, pose=pose)
                drones[name] = obs
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

    def _fetch_pose(self, name: str,
                    entry: dict) -> Optional[Tuple[float, float, float]]:
        """Best-effort GET /api/position. Returns ``None`` on any failure
        (FC offline, timeout, stale fix, malformed payload) — the caller
        leaves :attr:`DroneObservation.pose` as ``None`` in that case.
        Pose only appears once the FC's positioning solver has a live
        marker lock (typically after takeoff), so ``None`` is the
        steady-state for a drone sitting on the ground."""
        if not entry.get("connection_ok"):
            return None
        base = entry.get("base_url")
        if not base:
            return None
        try:
            r = self._http.get(f"{base}/api/position",
                               timeout=self.POSITION_FETCH_TIMEOUT_S)
        except requests.RequestException:
            return None
        if r.status_code != 200:
            return None
        try:
            data = r.json()
        except ValueError:
            return None
        if not data or not data.get("ok") or data.get("stale"):
            return None
        pos = data.get("pos")
        if not isinstance(pos, (list, tuple)) or len(pos) < 3:
            return None
        try:
            return (float(pos[0]), float(pos[1]), float(pos[2]))
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract(name: str, entry: dict) -> DroneObservation:
    state = entry.get("state") or {}
    tel = (state.get("telemetry") or {}) if isinstance(state, dict) else {}
    pos = (state.get("position") or {}) if isinstance(state, dict) else {}
    mission = (state.get("mission") or {}) if isinstance(state, dict) else {}

    def _f(d: dict, k: str) -> Optional[float]:
        v = d.get(k)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    pose_xyz = None
    px = _f(pos, "x_m"); py = _f(pos, "y_m"); pz = _f(pos, "z_m")
    if px is not None and py is not None and pz is not None:
        pose_xyz = (px, py, pz)

    return DroneObservation(
        name=name,
        online=bool(entry.get("connection_ok")),
        drone_connected=bool(entry.get("drone_connected")),
        flying=bool(tel.get("flying")),
        battery_pct=_f(tel, "battery"),
        phase=mission.get("phase") if isinstance(mission.get("phase"), str) else None,
        pose=pose_xyz,
        yaw_deg=_f(tel, "yaw"),
        last_marker_id=(int(state.get("last_marker_id"))
                        if isinstance(state.get("last_marker_id"), (int, float))
                        else None),
        serial=entry.get("drone_serial"),
        last_error=entry.get("last_error"),
        age_s=(float(entry.get("last_state_age_s"))
               if entry.get("last_state_age_s") is not None else None),
    )
