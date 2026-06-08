"""Per-drone live world state, parsed from one FC's ``GET /api/state``.

A ``DroneWorld`` is the strategy's read-model of a single drone: where it is in
the arena (from ArUco vision), its connection + telemetry + battery, the FC
phase, and a human-readable note of what it's doing. Parsed DEFENSIVELY — the FC
may omit fields while booting or when vision is cold, so every accessor tolerates
missing/None values rather than throwing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _f(d: Dict[str, Any], *keys: str) -> Optional[float]:
    """Nested float lookup; None if missing/unparseable."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    try:
        return float(cur)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@dataclass
class DroneWorld:
    name: str                       # fc name == hostname (flightctrlN)
    base_url: str

    # --- connection ---
    connected: bool = False         # last /api/state read succeeded & fresh
    last_state_unix_s: float = 0.0  # when we last got a good /api/state
    last_error: str = ""

    # --- arena position (vision/ArUco; NOT GPS) ---
    position_m: Optional[Tuple[float, float, float]] = None   # (x, y, z)
    position_age_s: Optional[float] = None
    position_conf: Optional[float] = None
    visible_marker_ids: List[int] = field(default_factory=list)

    # --- telemetry ---
    drone_connected: bool = False   # FC<->drone link (telemetry.connected)
    flying: bool = False
    battery_pct: Optional[float] = None
    height_cm: Optional[float] = None
    drone_yaw_deg: Optional[float] = None
    serial: str = ""

    # --- mission / what it's doing ---
    phase: str = ""                 # FC phase: init/takeoff/search/approach/...
    current_step_kind: str = ""     # current mission verb
    mission_step_idx: Optional[int] = None
    mission_len: int = 0
    note: str = ""                  # FC's own human note
    active_marker_id: Optional[int] = None

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.last_state_unix_s) if self.last_state_unix_s else 1e9

    @property
    def battery_low(self) -> bool:
        return self.battery_pct is not None and self.battery_pct <= 15.0

    def plan_text(self) -> str:
        """Short human summary of what the drone is currently doing (Phase 1
        shows the FC's own view; the coordinator will set richer intent later)."""
        if not self.connected:
            return "offline"
        bits = []
        if self.phase:
            bits.append(self.phase)
        if self.current_step_kind:
            step = self.current_step_kind
            if self.mission_step_idx is not None and self.mission_len:
                step += f" [{self.mission_step_idx + 1}/{self.mission_len}]"
            bits.append(step)
        if self.active_marker_id is not None:
            bits.append(f"->{self.active_marker_id}")
        return " · ".join(bits) or "idle"

    def to_client(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "connected": self.connected,
            "age_s": round(self.age_s, 1),
            "last_error": self.last_error,
            "position_m": list(self.position_m) if self.position_m else None,
            "position_age_s": self.position_age_s,
            "position_conf": self.position_conf,
            "visible_marker_ids": list(self.visible_marker_ids),
            "drone_connected": self.drone_connected,
            "flying": self.flying,
            "battery_pct": self.battery_pct,
            "battery_low": self.battery_low,
            "height_cm": self.height_cm,
            "height_m": round(self.height_cm / 100.0, 2) if self.height_cm is not None else None,
            "drone_yaw_deg": self.drone_yaw_deg,
            "serial": self.serial,
            "phase": self.phase,
            "current_step_kind": self.current_step_kind,
            "mission_step_idx": self.mission_step_idx,
            "mission_len": self.mission_len,
            "note": self.note,
            "active_marker_id": self.active_marker_id,
            "plan": self.plan_text(),
        }

    def update_from_state(self, state: Dict[str, Any], now: float) -> None:
        """Merge a successful ``GET /api/state`` JSON body into this world."""
        if not isinstance(state, dict):
            return
        self.connected = True
        self.last_state_unix_s = now
        self.last_error = ""

        wp = state.get("world_position_m")
        if isinstance(wp, (list, tuple)) and len(wp) >= 3:
            try:
                self.position_m = (float(wp[0]), float(wp[1]), float(wp[2]))
            except (TypeError, ValueError):
                self.position_m = None
        else:
            self.position_m = None
        self.position_age_s = _f(state, "world_position_age_s")
        self.position_conf = _f(state, "world_position_confidence")

        vis = state.get("visible_marker_ids") or []
        ids: List[int] = []
        for m in vis:
            try:
                ids.append(int(m))
            except (TypeError, ValueError):
                continue
        self.visible_marker_ids = ids

        tel = state.get("telemetry") or {}
        self.drone_connected = bool(tel.get("connected"))
        self.flying = bool(tel.get("flying"))
        self.battery_pct = _f(tel, "battery") if "battery" in tel else _f(state, "telemetry", "battery")
        self.height_cm = _f(tel, "height_cm")
        self.drone_yaw_deg = _f(tel, "yaw")
        sn = tel.get("serial_number")
        if sn:
            self.serial = str(sn)

        self.phase = str(state.get("phase") or "")
        self.current_step_kind = str(state.get("current_step_kind") or "")
        idx = state.get("mission_step_idx")
        self.mission_step_idx = int(idx) if isinstance(idx, int) else None
        script = state.get("mission_script") or []
        self.mission_len = len(script) if isinstance(script, list) else 0
        self.note = str(state.get("note") or "")
        amid = state.get("active_marker_id")
        self.active_marker_id = int(amid) if isinstance(amid, int) else None

    def mark_disconnected(self, err: str, now: float, stale_after_s: float) -> None:
        """Called when a /api/state read fails OR the last good read is stale."""
        if err:
            self.last_error = err
        if self.age_s > stale_after_s:
            self.connected = False
            self.drone_connected = False
