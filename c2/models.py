"""Shared data models for the SDC26 C2 strategy server.

ALL positions are in marker_mission ARENA coordinates:

    centred origin, +x = right, +y = toward the front (blue) wall, +z = up
    width  (x) in [-5, +5]    (10 m)
    depth  (y) in [-10, +10]  (20 m)
    height (z) in [0, 6]

Headings are degrees clockwise from +Y (the front wall), matching the
marker_mission convention (``heading = atan2(dx, dy)``).

These types are the single contract shared by every C2 module
(fc_client, world_model, events, roles, navigation, commander, server).
Keep them dependency-light: dataclasses + enums only, JSON-friendly via
``to_dict()`` helpers so the web layer can serialise without pydantic.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Team(str, Enum):
    RED = "red"
    BLUE = "blue"

    @property
    def enemy(self) -> "Team":
        return Team.BLUE if self is Team.RED else Team.RED


class SlotColor(str, Enum):
    """Current owner colour of a target box (from the marker ID seen)."""
    RED = "red"
    BLUE = "blue"
    UNKNOWN = "unknown"   # never observed yet (or aged out)

    @classmethod
    def from_team(cls, team: Team) -> "SlotColor":
        return cls.RED if team is Team.RED else cls.BLUE


class DroneRole(str, Enum):
    SCOUT = "scout"
    ATTACKER = "attacker"
    DEFENDER = "defender"
    IDLE = "idle"


class DronePhase(str, Enum):
    IDLE = "idle"
    TAKING_OFF = "taking_off"
    TRANSIT = "transit"        # flying toward a waypoint
    CAPTURING = "capturing"    # hovering over an enemy box to trigger it
    UNCAPPING = "uncapping"    # hovering over our box to revert its colour
    RETURNING = "returning"    # heading home for capture validation
    ORBIT = "orbit"            # scout orbit / defender patrol loop
    HOLDING = "holding"        # hovering in place (manual stop)
    LANDED = "landed"
    LOST = "lost"              # no telemetry / position


class GameMode(str, Enum):
    MANUAL = "manual"   # operator issues every command
    AUTO = "auto"       # commander reacts to events autonomously


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class Vec2:
    x: float
    y: float

    def dist(self, o: "Vec2") -> float:
        return math.hypot(self.x - o.x, self.y - o.y)

    def to_list(self) -> list[float]:
        return [round(self.x, 3), round(self.y, 3)]


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def xy(self) -> Vec2:
        return Vec2(self.x, self.y)

    def dist_xy(self, o: "Vec3") -> float:
        return math.hypot(self.x - o.x, self.y - o.y)

    def to_list(self) -> list[float]:
        return [round(self.x, 3), round(self.y, 3), round(self.z, 3)]


# ---------------------------------------------------------------------------
# Target slots
# ---------------------------------------------------------------------------
#
# There are 6 physical target boxes. Each box's ArUco marker ID encodes
# BOTH its physical slot (1..6) AND its current colour:
#
#     blue-state ID = 30 + slot_index      (e.g. slot 1 blue -> 31)
#     red-state  ID = 40 + slot_index      (e.g. slot 1 red  -> 41)
#
# Slots 1..3 sit in the BLUE home zone (default blue, IDs 31/32/33),
# slots 4..6 in the RED home zone (default red, IDs 44/45/46). When a box
# is captured its marker flips into the other range (e.g. 31 -> 41).
# Helpers for this mapping live in config.SlotMap so the magic numbers
# stay in one place.

@dataclass
class TargetSlot:
    index: int                              # 1..6 physical slot
    home_team: Team                         # whose zone the box sits in
    color: SlotColor = SlotColor.UNKNOWN    # current owner colour
    marker_id_seen: Optional[int] = None    # last ArUco ID observed
    position: Optional[Vec3] = None         # last known arena position
    last_seen_s: Optional[float] = None     # wall-clock of last detection
    age_s: Optional[float] = None           # seconds since last detection
    fresh: bool = False                     # detected in the latest frame

    def owned_by(self, team: Team) -> bool:
        return self.color == SlotColor.from_team(team)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["home_team"] = self.home_team.value
        d["color"] = self.color.value
        d["position"] = self.position.to_list() if self.position else None
        return d


# ---------------------------------------------------------------------------
# Drone state
# ---------------------------------------------------------------------------

@dataclass
class DroneState:
    drone_id: str
    role: DroneRole = DroneRole.IDLE
    phase: DronePhase = DronePhase.IDLE
    position: Optional[Vec3] = None
    heading_deg: Optional[float] = None       # CW from +Y (front wall)
    velocity: Optional[Vec3] = None           # arena-frame m/s
    base_altitude_m: float = 2.0              # assigned cruise altitude (deconfliction)
    connected: bool = False
    flying: bool = False
    battery_pct: Optional[float] = None
    assigned_slot: Optional[int] = None       # target slot this drone is acting on
    pos_stale: bool = True                     # /api/position stale flag
    last_telemetry_s: Optional[float] = None
    note: str = ""                            # human-readable status line

    def to_dict(self) -> dict:
        d = asdict(self)
        d["role"] = self.role.value
        d["phase"] = self.phase.value
        d["position"] = self.position.to_list() if self.position else None
        d["velocity"] = self.velocity.to_list() if self.velocity else None
        return d


# ---------------------------------------------------------------------------
# World snapshot (consumed by strategy + web UI)
# ---------------------------------------------------------------------------

@dataclass
class WorldSnapshot:
    t: float
    mode: GameMode
    our_team: Team
    drones: dict[str, DroneState] = field(default_factory=dict)
    slots: dict[int, TargetSlot] = field(default_factory=dict)
    score_us: int = 0
    score_them: int = 0
    events: list[str] = field(default_factory=list)   # recent human-readable events

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "mode": self.mode.value,
            "our_team": self.our_team.value,
            "drones": {k: v.to_dict() for k, v in self.drones.items()},
            "slots": {str(k): v.to_dict() for k, v in sorted(self.slots.items())},
            "score_us": self.score_us,
            "score_them": self.score_them,
            "events": list(self.events),
        }


# ---------------------------------------------------------------------------
# Manual commands
# ---------------------------------------------------------------------------

class CommandType(str, Enum):
    SET_MODE = "set_mode"          # params: {mode: "manual"|"auto"}
    ASSIGN_ROLE = "assign_role"    # drone_id + role
    ATTACK = "attack"              # drone_id + slot   -> capture enemy slot
    DEFEND = "defend"              # drone_id (+ optional slot) -> patrol / un-cap
    SCOUT = "scout"                # drone_id -> centre orbit
    RETURN_HOME = "return_home"    # drone_id
    TAKEOFF = "takeoff"            # drone_id or "all"
    LAND = "land"                  # drone_id or "all"
    STOP = "stop"                  # drone_id or "all" -> hover in place
    EMERGENCY = "emergency"        # drone_id or "all" -> motor cut


@dataclass
class Command:
    type: CommandType
    drone_id: Optional[str] = None       # specific id, or "all", or None
    slot: Optional[int] = None
    role: Optional[DroneRole] = None
    mode: Optional[GameMode] = None
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Command":
        return cls(
            type=CommandType(d["type"]),
            drone_id=d.get("drone_id"),
            slot=(int(d["slot"]) if d.get("slot") is not None else None),
            role=(DroneRole(d["role"]) if d.get("role") else None),
            mode=(GameMode(d["mode"]) if d.get("mode") else None),
            params=d.get("params", {}) or {},
        )


# ---------------------------------------------------------------------------
# Role actions (roles emit these; navigation executes them)
# ---------------------------------------------------------------------------

class ActionKind(str, Enum):
    GOTO = "goto"            # closed-loop fly to `target` Vec3
    HOVER = "hover"          # hold current position (rc zeros) for `hold_s`
    YAW_SCAN = "yaw_scan"    # rotate in place at `yaw_rate` (deg/s) to scan
    ORBIT = "orbit"          # circle `orbit_center` at `orbit_radius_m`, altitude `target.z`
    LAND = "land"
    TAKEOFF = "takeoff"
    NONE = "none"            # do nothing this tick


@dataclass
class RoleAction:
    kind: ActionKind = ActionKind.NONE
    target: Optional[Vec3] = None       # GOTO / ORBIT altitude reference
    yaw_rate: float = 0.0               # YAW_SCAN deg/s (CW positive)
    orbit_center: Optional[Vec2] = None
    orbit_radius_m: float = 0.0
    hold_s: float = 0.0
    next_phase: DronePhase = DronePhase.IDLE   # phase to report while doing this
    note: str = ""                      # human-readable rationale for the UI

    @staticmethod
    def none(note: str = "") -> "RoleAction":
        return RoleAction(kind=ActionKind.NONE, note=note)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def now_s() -> float:
    """Wall-clock seconds (single definition so all modules agree)."""
    return time.time()


def heading_to_vec(heading_deg: float) -> Vec2:
    """Unit vector for a heading measured CW from +Y (front wall)."""
    r = math.radians(heading_deg)
    return Vec2(x=math.sin(r), y=math.cos(r))
