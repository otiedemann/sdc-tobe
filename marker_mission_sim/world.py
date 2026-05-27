"""The simulator world: arena + target boxes + drones, with a thread-safe
tick that advances flight (script_exec) and capture state.

This module owns the shared mutable state and the data model every other
module reads/writes. Keep it dependency-light; script_exec / capture /
vision are imported lazily inside tick()/helpers to avoid import cycles.
"""

from __future__ import annotations

import collections
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import SimConfig, ArenaDef, BoxDef, WallMarkerDef
from .geometry import Vec3


# ---------------------------------------------------------------------------
# Mutable game pieces
# ---------------------------------------------------------------------------

@dataclass
class SimBox:
    slot: int
    pos: Vec3
    holder: str                 # "red" | "blue"
    home_team: str              # immutable: which team's home zone this box sits in
    blue_face_id: int
    red_face_id: int
    last_flip_unix_s: float = 0.0
    # v7 §1.4.3: after a successful capture the box is locked for 5 s — no
    # further captures (in either direction) until this clock-time is reached.
    lock_until: float = 0.0

    @property
    def current_face_id(self) -> int:
        return self.red_face_id if self.holder == "red" else self.blue_face_id

    @classmethod
    def from_def(cls, b: BoxDef) -> "SimBox":
        return cls(slot=b.slot, pos=b.pos.copy(), holder=b.home_team,
                   home_team=b.home_team,
                   blue_face_id=b.blue_face_id, red_face_id=b.red_face_id)

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "pos": self.pos.to_list(),
            "holder": self.holder,
            "home_team": self.home_team,
            "current_face_id": self.current_face_id,
            "blue_face_id": self.blue_face_id,
            "red_face_id": self.red_face_id,
            "lock_until": round(self.lock_until, 2),
            "last_flip_unix_s": round(self.last_flip_unix_s, 2),
        }


@dataclass
class SimDrone:
    id: str
    team: str
    fc_port: int
    pos: Vec3
    heading_deg: float = 0.0
    speed_mps: float = 0.0          # current ground speed (for capture gating)
    phase: str = "init"             # "init" (idle) | "running" | "done"
    flying: bool = False
    connected: bool = True
    battery: int = 100
    # --- mission script execution state (owned by script_exec) ---
    script: list = field(default_factory=list)     # parsed Steps (opaque here)
    script_text: str = ""                          # last script text set/started
    step_idx: int = 0
    step_t0: float = 0.0                            # wall-clock when current step began
    mem: dict = field(default_factory=dict)        # per-step scratch (timers, sub-targets)
    # --- command latency ---
    pending_script_text: Optional[str] = None
    pending_apply_at: float = 0.0
    # --- capture dwell bookkeeping (owned by capture) ---
    capture_slot: Optional[int] = None
    capture_since: float = 0.0

    def to_state_dict(self, visible_marker_ids: list[int]) -> dict:
        """The /api/state payload the C2 reads (marker_mission FC shape)."""
        active = self.script[self.step_idx] if (0 <= self.step_idx < len(self.script)) else None
        active_id = getattr(active, "marker_id", None) if active else None
        return {
            "phase": self.phase,
            "phase_age_s": round(max(0.0, time.time() - self.step_t0), 2),
            "distance_m": None,
            "yaw_to_marker_deg": None,
            "relative_heading_deg": None,
            "visible_marker_ids": list(visible_marker_ids),
            "active_marker_id": active_id,
            "world_position_m": self.pos.to_list(),
            "heading_deg": round(self.heading_deg, 1),
            "mission_step_idx": self.step_idx if self.script else None,
            "mission_script": [getattr(s, "raw", "") for s in self.script],
            "drone_connected": self.connected,
            "telemetry": {
                "battery": int(self.battery),
                "height_cm": int(max(0.0, self.pos.z) * 100),
                "yaw": round(self.heading_deg, 1),
                "flying": self.flying,
            },
        }

    def to_ui_dict(self) -> dict:
        return {
            "id": self.id,
            "team": self.team,
            "fc_port": self.fc_port,
            "pos": self.pos.to_list(),
            "heading_deg": round(self.heading_deg, 1),
            "phase": self.phase,
            "flying": self.flying,
            "speed_mps": round(self.speed_mps, 2),
            "step_idx": self.step_idx,
            "n_steps": len(self.script),
        }


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

class World:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.arena: ArenaDef = cfg.arena
        self.wall_markers: list[WallMarkerDef] = list(cfg.arena.wall_markers)
        self.lock = threading.RLock()
        self.rng = random.Random(cfg.seed)
        self.events: collections.deque = collections.deque(maxlen=200)
        self.start_unix_s = time.time()

        self.boxes: dict[int, SimBox] = {
            b.slot: SimBox.from_def(b) for b in cfg.arena.boxes
        }
        self.drones: dict[str, SimDrone] = {}
        for dc in cfg.drones:
            self.drones[dc.id] = SimDrone(
                id=dc.id, team=dc.team, fc_port=dc.fc_port,
                pos=Vec3(dc.spawn_x, dc.spawn_y, 0.0),
                heading_deg=dc.spawn_heading_deg,
            )

    # -- events ------------------------------------------------------------
    def log(self, kind: str, text: str, drone: Optional[str] = None) -> None:
        self.events.append({"unix_s": time.time(), "kind": kind,
                            "text": text, "drone": drone})

    # -- mission control (called from the FC API) --------------------------
    def queue_script(self, drone_id: str, text: str, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        with self.lock:
            d = self.drones.get(drone_id)
            if d is None:
                return False
            d.pending_script_text = text
            d.pending_apply_at = now + max(0.0, self.cfg.noise.cmd_latency_s)
            self.log("start", f"script queued ({len(text.splitlines())} lines)", drone_id)
            return True

    def stop_script(self, drone_id: str) -> bool:
        with self.lock:
            d = self.drones.get(drone_id)
            if d is None or d.phase in ("init", "done"):
                return False
            d.script = []
            d.step_idx = 0
            d.mem.clear()
            d.phase = "done"
            self.log("stop", "mission stopped", drone_id)
            return True

    # -- tick --------------------------------------------------------------
    def tick(self, dt: float, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        # Lazy imports so the scaffold loads before these modules exist and
        # to avoid import cycles.
        from . import script_exec, capture
        with self.lock:
            for d in self.drones.values():
                try:
                    script_exec.advance(d, self, dt, now)
                except Exception as e:  # never let one drone kill the loop
                    self.log("error", f"script_exec: {e}", d.id)
            try:
                for ev in capture.update(self, dt, now):
                    self.log("capture", ev)
            except Exception as e:
                self.log("error", f"capture: {e}")

    # -- reads -------------------------------------------------------------
    def get_drone(self, drone_id: str) -> Optional[SimDrone]:
        with self.lock:
            return self.drones.get(drone_id)

    def visible_marker_ids(self, drone_id: str, now: Optional[float] = None) -> list[int]:
        from . import vision
        now = now if now is not None else time.time()
        with self.lock:
            d = self.drones.get(drone_id)
            if d is None:
                return []
            return vision.visible_marker_ids(d, self, now)

    def world_snapshot(self) -> dict:
        with self.lock:
            return {
                "t": round(time.time() - self.start_unix_s, 2),
                "arena": {
                    "width_m": self.arena.width_m,
                    "depth_m": self.arena.depth_m,
                    "ceiling_m": self.arena.ceiling_m,
                },
                "wall_markers": [
                    {"id": m.id, "pos": m.pos.to_list(), "wall": m.wall}
                    for m in self.wall_markers
                ],
                "boxes": [self.boxes[s].to_dict() for s in sorted(self.boxes)],
                "drones": [d.to_ui_dict() for d in self.drones.values()],
                "events": list(self.events)[-30:],
            }
