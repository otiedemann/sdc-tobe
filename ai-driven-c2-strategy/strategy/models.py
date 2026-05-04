from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Side = Literal["left", "right"]
TeamColor = Literal["blue", "red"]
DroneTeam = Literal["own", "enemy"]
DroneStatus = Literal["active", "inactive", "landed", "failed"]
Objective = Literal[
    "stage_attack",
    "capture_target",
    "recover_target",
    "defense_patrol",
    "offset_loiter",
    "return_home",
    "hold_home",
    "regroup",
    "land",
]


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def distance_2d(self, other: "Vec3") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def distance_3d(self, other: "Vec3") -> float:
        return (
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        ) ** 0.5

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class Pose(Vec3):
    yaw: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw}


@dataclass
class Zone:
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, point: Vec3) -> bool:
        return self.x_min <= point.x <= self.x_max and self.y_min <= point.y <= self.y_max

    def center(self, z: float) -> Pose:
        return Pose((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0, z, 0.0)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }


@dataclass
class Arena:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    left_zone_x_max: float
    right_zone_x_min: float

    def clamp_pose(self, pose: Pose) -> Pose:
        return Pose(
            min(max(pose.x, self.x_min), self.x_max),
            min(max(pose.y, self.y_min), self.y_max),
            min(max(pose.z, self.z_min), self.z_max),
            pose.yaw,
        )

    def zone_for_side(self, side: Side) -> Zone:
        if side == "left":
            return Zone("left", self.x_min, self.left_zone_x_max, self.y_min, self.y_max)
        return Zone("right", self.right_zone_x_min, self.x_max, self.y_min, self.y_max)

    def neutral_zone(self) -> Zone:
        return Zone("neutral", self.left_zone_x_max, self.right_zone_x_min, self.y_min, self.y_max)

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
            "left_zone_x_max": self.left_zone_x_max,
            "right_zone_x_min": self.right_zone_x_min,
        }


@dataclass
class TeamConfig:
    color: TeamColor
    opponent_color: TeamColor
    home_side: Side

    @property
    def enemy_side(self) -> Side:
        return "right" if self.home_side == "left" else "left"


@dataclass
class GameplayConfig:
    match_duration_seconds: float
    capture_radius_m: float
    capture_hold_seconds: float
    validation_seconds: float
    simultaneous_window_seconds: float
    drone_speed_mps: float
    enemy_speed_mps: float
    capture_z: float
    transit_z: float
    min_drone_separation_m: float


@dataclass
class LlmConfig:
    enabled: bool
    provider: str
    model: str
    temperature: float
    timeout_seconds: float
    max_tokens: int
    decision_interval_seconds: float


@dataclass
class OpenRouterConfig:
    site_url: str
    app_name: str


@dataclass
class SimulationConfig:
    own_drone_count: int
    enemy_drone_count: int
    tick_seconds: float
    llm_on_major_events: bool


@dataclass
class MockEnemyConfig:
    enabled: bool
    style: str


@dataclass
class AppConfig:
    team: TeamConfig
    arena: Arena
    gameplay: GameplayConfig
    llm: LlmConfig
    openrouter: OpenRouterConfig
    simulation: SimulationConfig
    mock_enemy: MockEnemyConfig
    targets: list["TargetState"]

    @property
    def own_home_zone(self) -> Zone:
        return self.arena.zone_for_side(self.team.home_side)

    @property
    def enemy_home_zone(self) -> Zone:
        return self.arena.zone_for_side(self.team.enemy_side)


@dataclass
class DroneState:
    id: str
    team: DroneTeam
    pose: Pose
    status: DroneStatus = "active"
    role: str = "unassigned"
    objective: str = "idle"
    assigned_target_id: str | None = None
    command_position: Pose | None = None
    hold_until: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "team": self.team,
            "pose": self.pose.to_dict(),
            "status": self.status,
            "role": self.role,
            "objective": self.objective,
            "assigned_target_id": self.assigned_target_id,
            "command_position": self.command_position.to_dict() if self.command_position else None,
            "hold_until": self.hold_until,
        }


@dataclass
class TargetState:
    id: str
    side: Side
    position: Vec3
    owner: TeamColor
    last_changed_seconds: float = 0.0
    stable_owner_since_seconds: float = 0.0
    capture_progress: dict[TeamColor, float] = field(default_factory=dict)
    validated_for_owner: TeamColor | None = None

    def relation_to_team(self, team: TeamConfig) -> Literal["own", "enemy"]:
        return "own" if self.side == team.home_side else "enemy"

    def to_dict(self, team: TeamConfig | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "side": self.side,
            "position": self.position.to_dict(),
            "owner": self.owner,
            "last_changed_seconds": self.last_changed_seconds,
            "stable_owner_since_seconds": self.stable_owner_since_seconds,
            "validated_for_owner": self.validated_for_owner,
        }
        if team:
            data["relation"] = self.relation_to_team(team)
        return data


@dataclass
class ScoreState:
    own: int = 0
    opponent: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"own": self.own, "opponent": self.opponent}


@dataclass
class MatchEvent:
    time_seconds: float
    type: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_seconds": self.time_seconds,
            "type": self.type,
            "message": self.message,
            "data": self.data,
        }


@dataclass
class MatchState:
    match_time_seconds: float
    time_remaining_seconds: float
    team: TeamConfig
    arena: Arena
    gameplay: GameplayConfig
    own_home_zone: Zone
    enemy_home_zone: Zone
    neutral_zone: Zone
    score: ScoreState
    own_drones: list[DroneState]
    enemy_drones: list[DroneState]
    targets: list[TargetState]
    recent_events: list[MatchEvent]

    def active_own_drones(self) -> list[DroneState]:
        return [drone for drone in self.own_drones if drone.status == "active"]

    def target_by_id(self, target_id: str) -> TargetState | None:
        return next((target for target in self.targets if target.id == target_id), None)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "match_time_seconds": round(self.match_time_seconds, 2),
            "time_remaining_seconds": round(self.time_remaining_seconds, 2),
            "team": {
                "color": self.team.color,
                "opponent_color": self.team.opponent_color,
                "home_side": self.team.home_side,
            },
            "arena": self.arena.to_dict(),
            "zones": {
                "own_home": self.own_home_zone.to_dict(),
                "enemy_home": self.enemy_home_zone.to_dict(),
                "neutral": self.neutral_zone.to_dict(),
            },
            "score": self.score.to_dict(),
            "own_drones": [drone.to_dict() for drone in self.own_drones],
            "enemy_drones": [drone.to_dict() for drone in self.enemy_drones],
            "targets": [target.to_dict(self.team) for target in self.targets],
            "recent_events": [event.to_dict() for event in self.recent_events[-12:]],
        }


@dataclass
class DroneCommand:
    drone_id: str
    role: str
    objective: Objective
    target_position: Pose
    target_id: str | None = None
    hold_seconds: float = 0.0
    priority: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "drone_id": self.drone_id,
            "role": self.role,
            "objective": self.objective,
            "target_id": self.target_id,
            "target_position": self.target_position.to_dict(),
            "hold_seconds": self.hold_seconds,
            "priority": self.priority,
            "reason": self.reason,
        }


@dataclass
class PlannerResult:
    mode: str
    strategy_summary: str
    commands: list[DroneCommand]
    source: Literal["llm", "fallback"]
    raw_response: dict[str, Any] | None = None


@dataclass
class ValidationResult:
    ok: bool
    commands: list[DroneCommand]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
