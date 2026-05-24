"""World model for the SDC26 C2 strategy server.

The :class:`WorldModel` is the single fused picture of the game. It ingests
each drone's raw ``GET /api/position`` payload (pose + ArUco target
detections) and ``/api/telemetry`` (link/battery), and maintains:

  * the state of all 6 physical target slots (colour, position, freshness),
    fused across every drone that can see them, and
  * the pose / status of every drone in the fleet.

A :class:`WorldSnapshot` (a deep copy) can be taken at any time for the
strategy loop or the web UI.

Coordinate / heading conventions follow ``c2.models`` (arena frame, heading
CW from +Y). All magic marker-ID numbers live in ``cfg.slot_map``.

Threading: ``ingest_*`` is driven from the commander's async loop while
``snapshot()`` may be called from a web-handler thread, so every access to
the mutable slot/drone state is guarded by a single ``threading.Lock``.
"""

from __future__ import annotations

import copy
import math
import threading
from typing import Optional

if __package__ in (None, ""):  # run directly as a script: bootstrap the package
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from c2.config import C2Config
    from c2.models import (
        DronePhase,
        DroneRole,
        DroneState,
        GameMode,
        SlotColor,
        TargetSlot,
        Vec3,
        WorldSnapshot,
        now_s,
    )
else:
    from .config import C2Config
    from .models import (
        DronePhase,
        DroneRole,
        DroneState,
        GameMode,
        SlotColor,
        TargetSlot,
        Vec3,
        WorldSnapshot,
        now_s,
    )


def _vec3(seq, *, default: Optional[Vec3] = None) -> Optional[Vec3]:
    """Build a Vec3 from a [x, y, z] sequence, tolerating None / short lists."""
    if not seq:
        return default
    try:
        x = float(seq[0])
        y = float(seq[1])
        z = float(seq[2]) if len(seq) > 2 and seq[2] is not None else 0.0
    except (TypeError, ValueError, IndexError):
        return default
    return Vec3(x, y, z)


class WorldModel:
    """Fuses per-drone vision into the shared game state."""

    def __init__(self, cfg: C2Config) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._drones: dict[str, DroneState] = {}
        self._slots: dict[int, TargetSlot] = {}

        # Seed all 6 slots so `slots` always carries 1..6, even before any
        # drone has observed them.
        sm = cfg.slot_map
        for i in range(1, sm.n_slots + 1):
            self._slots[i] = TargetSlot(
                index=i,
                home_team=sm.home_team_of(i),
                color=SlotColor.UNKNOWN,
                position=cfg.nominal_slot_positions.get(i),
            )

    # -- fleet registration ------------------------------------------------

    def ensure_drone(
        self, drone_id: str, base_altitude_m: float, role: DroneRole
    ) -> None:
        """Register a drone (idempotent); leaves existing state untouched."""
        with self._lock:
            d = self._drones.get(drone_id)
            if d is None:
                self._drones[drone_id] = DroneState(
                    drone_id=drone_id,
                    role=role,
                    base_altitude_m=base_altitude_m,
                )
            else:
                d.base_altitude_m = base_altitude_m
                d.role = role

    # -- ingestion ---------------------------------------------------------

    def ingest_position(self, drone_id: str, pos_payload: dict, now: float) -> None:
        """Fuse one drone's ``/api/position`` payload into pose + slots."""
        with self._lock:
            d = self._drones.get(drone_id)
            if d is None:
                d = DroneState(drone_id=drone_id)
                self._drones[drone_id] = d

            # --- own pose -------------------------------------------------
            pos = pos_payload.get("pos")
            stale = bool(pos_payload.get("stale", False))
            if pos is None:
                # Keep last known position; just flag staleness.
                d.pos_stale = True
            else:
                new_pos = _vec3(pos)
                if new_pos is not None:
                    d.position = new_pos
                d.pos_stale = stale

            vel = _vec3(pos_payload.get("vel"))
            if vel is not None:
                d.velocity = vel

            direction = pos_payload.get("dir")
            if direction:
                try:
                    dx = float(direction[0])
                    dy = float(direction[1])
                    d.heading_deg = math.degrees(math.atan2(dx, dy))
                except (TypeError, ValueError, IndexError):
                    pass  # leave heading unchanged on malformed dir

            # --- target detections ---------------------------------------
            targets = pos_payload.get("targets") or {}
            for raw_id, obs in targets.items():
                self._fuse_target(raw_id, obs, now)

            self._age_slots(now)

    def _fuse_target(self, raw_id, obs: dict, now: float) -> None:
        """Fold a single ArUco detection into its slot, keeping the freshest.

        Caller must hold ``self._lock``.
        """
        if not isinstance(obs, dict):
            return
        try:
            marker_id = int(raw_id)
        except (TypeError, ValueError):
            return

        slot_idx = self.cfg.slot_map.slot_of(marker_id)
        if slot_idx is None:
            return  # wall / pillar marker, not a target box

        slot = self._slots.get(slot_idx)
        if slot is None:  # should not happen (seeded in __init__) but be safe
            slot = TargetSlot(
                index=slot_idx,
                home_team=self.cfg.slot_map.home_team_of(slot_idx),
            )
            self._slots[slot_idx] = slot

        age = obs.get("age_s")
        try:
            age = float(age) if age is not None else None
        except (TypeError, ValueError):
            age = None

        # Fusion rule: only overwrite if THIS observation is at least as
        # fresh as what we already have. A detection's effective wall-clock
        # time is (now - age); compare those so the newest wins regardless
        # of which drone reported it.
        obs_seen = now - age if age is not None else now
        if slot.last_seen_s is not None and obs_seen < slot.last_seen_s:
            return  # we already hold a fresher fix for this slot

        slot.color = self.cfg.slot_map.color_of(marker_id)
        slot.marker_id_seen = marker_id
        new_pos = _vec3(obs.get("pos"))
        if new_pos is not None:
            slot.position = new_pos
        slot.age_s = age
        slot.fresh = bool(obs.get("fresh", True))
        slot.last_seen_s = obs_seen

    def _age_slots(self, now: float) -> None:
        """Recompute age_s / fresh for every slot. Caller holds the lock.

        A slot keeps its last known colour/position forever once seen; it is
        only marked ``fresh=False`` after ``slot_age_fresh_s`` with no new
        observation. Never-seen slots stay UNKNOWN.
        """
        fresh_window = self.cfg.strategy.slot_age_fresh_s
        for slot in self._slots.values():
            if slot.last_seen_s is None:
                slot.age_s = None
                slot.fresh = False
                continue
            slot.age_s = max(0.0, now - slot.last_seen_s)
            slot.fresh = slot.age_s <= fresh_window

    def ingest_telemetry(self, drone_id: str, tel: dict, now: float) -> None:
        """Update link / battery state from a ``/api/telemetry`` payload."""
        with self._lock:
            d = self._drones.get(drone_id)
            if d is None:
                d = DroneState(drone_id=drone_id)
                self._drones[drone_id] = d
            if "connected" in tel:
                d.connected = bool(tel.get("connected"))
            else:
                # Receiving telemetry at all implies a live link.
                d.connected = True
            if "flying" in tel:
                d.flying = bool(tel.get("flying"))
            batt = tel.get("battery")
            if batt is not None:
                try:
                    d.battery_pct = float(batt)
                except (TypeError, ValueError):
                    pass
            d.last_telemetry_s = now

    # -- mutators (driven by commander) ------------------------------------

    def set_role(self, drone_id: str, role: DroneRole) -> None:
        with self._lock:
            self._drones.setdefault(drone_id, DroneState(drone_id=drone_id)).role = role

    def set_assigned_slot(self, drone_id: str, slot: Optional[int]) -> None:
        with self._lock:
            self._drones.setdefault(
                drone_id, DroneState(drone_id=drone_id)
            ).assigned_slot = slot

    def set_phase(self, drone_id: str, phase: DronePhase) -> None:
        with self._lock:
            self._drones.setdefault(
                drone_id, DroneState(drone_id=drone_id)
            ).phase = phase

    # -- accessors ---------------------------------------------------------

    def get_drone(self, drone_id: str) -> DroneState:
        with self._lock:
            d = self._drones.get(drone_id)
            if d is None:
                d = DroneState(drone_id=drone_id)
                self._drones[drone_id] = d
            return copy.deepcopy(d)

    def snapshot(self, mode: GameMode) -> WorldSnapshot:
        """Deep-copied, consistent view of the whole game state."""
        with self._lock:
            drones = copy.deepcopy(self._drones)
            slots = copy.deepcopy(self._slots)

        our_color = SlotColor.from_team(self.cfg.our_team)
        enemy_color = SlotColor.from_team(self.cfg.enemy_team())
        score_us = sum(1 for s in slots.values() if s.color == our_color)
        score_them = sum(1 for s in slots.values() if s.color == enemy_color)

        return WorldSnapshot(
            t=now_s(),
            mode=mode,
            our_team=self.cfg.our_team,
            drones=drones,
            slots=slots,
            score_us=score_us,
            score_them=score_them,
            events=[],
        )

    @property
    def slots(self) -> dict[int, TargetSlot]:
        with self._lock:
            return copy.deepcopy(self._slots)

    @property
    def drones(self) -> dict[str, DroneState]:
        with self._lock:
            return copy.deepcopy(self._drones)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = C2Config()
    wm = WorldModel(cfg)
    print("seeded slots:", {i: s.color.value for i, s in wm.slots.items()})

    wm.ensure_drone("d1", base_altitude_m=4.0, role=DroneRole.SCOUT)

    t = now_s()
    payload = {
        "ok": True,
        "pos": [1.0, -2.0, 4.0],
        "vel": [0.1, 0.0, 0.0],
        "dir": [1.0, 0.0],  # pointing +x -> heading 90 deg CW from +Y
        "stale": False,
        "targets": {
            "44": {"pos": [3.0, -7.5, 1.0], "age_s": 0.2, "fresh": True},  # slot4 red
            "31": {"pos": [-3.0, 7.5, 1.0], "age_s": 1.0, "fresh": True},  # slot1 blue
            "99": {"pos": [0.0, 0.0, 0.0], "age_s": 0.0, "fresh": True},   # wall -> ignored
        },
    }
    wm.ingest_position("d1", payload, t)

    print("slot colors after ingest:",
          {i: s.color.value for i, s in wm.slots.items()})
    d1 = wm.get_drone("d1")
    print("d1 pos:", d1.position.to_list(),
          "heading_deg:", round(d1.heading_deg, 1),
          "pos_stale:", d1.pos_stale)

    # Fusion: a second, staler observation of slot 4 must NOT overwrite.
    payload2 = {
        "pos": [0.0, 0.0, 3.0], "dir": None, "stale": False,
        "targets": {"41": {"pos": [9.9, 9.9, 9.9], "age_s": 5.0, "fresh": True}},
    }
    wm.ensure_drone("d2", base_altitude_m=2.0, role=DroneRole.ATTACKER)
    wm.ingest_position("d2", payload2, t + 0.1)
    s4 = wm.slots[4]
    print("slot4 after staler obs -> color:", s4.color.value,
          "pos kept:", s4.position.to_list(), "(fresher fix retained)")

    wm.ingest_telemetry("d1", {"connected": True, "flying": True, "battery": 87}, t)
    snap = wm.snapshot(GameMode.AUTO)
    print("snapshot score_us:", snap.score_us, "score_them:", snap.score_them,
          "mode:", snap.mode.value, "our_team:", snap.our_team.value)
    print("d1 battery:", wm.get_drone("d1").battery_pct,
          "connected:", wm.get_drone("d1").connected)
    print("OK world_model smoke test")
