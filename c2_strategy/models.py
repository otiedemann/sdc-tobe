"""
Data models for the SDC26 strategic C2 server.

All game-related types: drone state, targets, scoring, game phases.
Positions use continuous meters (not grid cells) for ArUco accuracy.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Spatial types
# ---------------------------------------------------------------------------

class Vec2(BaseModel):
    """Continuous 2D position in meters (arena coords: 0-20 x 0-10)."""
    x: float
    y: float

    def distance_to(self, other: "Vec2") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def grid(self) -> tuple[int, int]:
        """Snap to 1m grid cell."""
        return int(round(self.x)), int(round(self.y))


class Vec3(BaseModel):
    """3D position in meters."""
    x: float
    y: float
    z: float


# ---------------------------------------------------------------------------
# Drone roles and phases
# ---------------------------------------------------------------------------

class DroneRole(str, Enum):
    SCOUT = "scout"
    ATTACKER = "attacker"
    DEFENDER = "defender"
    RETURNER = "returner"


class DronePhase(str, Enum):
    IDLE = "idle"
    TAKING_OFF = "taking_off"
    FLYING_TO_TARGET = "flying_to_target"
    HOVERING_ON_TARGET = "hovering_on_target"
    RETURNING_HOME = "returning_home"
    IN_HOME_ZONE = "in_home_zone"
    ORBIT_PATROL = "orbit_patrol"
    LANDED = "landed"
    LOST = "lost"


# ---------------------------------------------------------------------------
# Game phases
# ---------------------------------------------------------------------------

class GamePhase(str, Enum):
    SETUP = "setup"
    SCOUTING = "scouting"
    SIMULTANEOUS_ATTACK = "simultaneous_attack"
    TEAM_RETURN = "team_return"
    DEFEND = "defend"
    INSTANT_WIN_ATTEMPT = "instant_win_attempt"
    GAME_OVER = "game_over"


# ---------------------------------------------------------------------------
# Drone state
# ---------------------------------------------------------------------------

class DroneState(BaseModel):
    drone_id: str
    role: DroneRole = DroneRole.ATTACKER
    phase: DronePhase = DronePhase.IDLE
    position: Optional[Vec2] = None
    heading_deg: float = 0.0
    altitude_m: float = 0.0
    battery_pct: float = 100.0
    assigned_target: Optional[str] = None  # box_id
    home: Vec2 = Vec2(x=1.0, y=1.0)
    last_telemetry_at: float = 0.0
    api_base_url: str = "http://localhost:8080"
    sim_drone_id: Optional[str] = None  # e.g. "B1" for simulator selection
    connected: bool = False
    flying: bool = False
    # ArUco position tracking
    aruco_markers_detected: int = 0
    aruco_ref_markers: list[int] = []
    position_source: str = "none"  # "aruco", "sim", "none"
    position_age_s: float = 999.0  # seconds since last position update
    last_position_at: float = 0.0


# ---------------------------------------------------------------------------
# Target boxes
# ---------------------------------------------------------------------------

class TargetBox(BaseModel):
    box_id: str
    aruco_marker_id: Optional[int] = None
    position: Optional[Vec2] = None
    owner: str = "red"  # "red" or "blue"
    discovered: bool = False
    captured_by: Optional[str] = None  # team that last captured
    capture_start_time: Optional[float] = None
    captured: bool = False


# ---------------------------------------------------------------------------
# Game state (full snapshot)
# ---------------------------------------------------------------------------

class GameState(BaseModel):
    phase: GamePhase = GamePhase.SETUP
    our_team: str = "blue"
    score_us: int = 0
    score_them: int = 0
    drones: dict[str, DroneState] = {}
    target_boxes: dict[str, TargetBox] = {}
    start_time: Optional[float] = None
    captures_log: list[dict] = []
    all_in_home: bool = False


# ---------------------------------------------------------------------------
# Capture event
# ---------------------------------------------------------------------------

class CaptureEvent(BaseModel):
    box_id: str
    drone_id: str
    start_time: float
    end_time: float
    points: int = 1


# ---------------------------------------------------------------------------
# Command models (REST API)
# ---------------------------------------------------------------------------

class SetPhaseCommand(BaseModel):
    phase: str


class AssignTargetCommand(BaseModel):
    drone_id: str
    box_id: str


class ManualOverrideCommand(BaseModel):
    drone_id: str
    action: str  # "takeoff", "land", "move", "goto", "emergency"
    params: dict = {}


class DroneConfigItem(BaseModel):
    drone_id: str
    api_base_url: str = "http://localhost:8080"
    drone_type: str = "simulator"  # "simulator", "tello", "anafi"
    sim_drone_id: Optional[str] = None  # for simulator: which sim drone to control


class ArenaConfigUpdate(BaseModel):
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    our_team: Optional[str] = None
    pillar_markers: Optional[dict[str, dict]] = None
    drone_configs: Optional[list[DroneConfigItem]] = None
