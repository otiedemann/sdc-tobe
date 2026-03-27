from enum import Enum
from pydantic import BaseModel


class DroneStatus(str, Enum):
    IDLE = "idle"
    MOVING = "moving"
    SEARCHING = "searching"
    TARGET_HOVER = "target_hover"
    RETURNING_HOME = "returning_home"
    OFFLINE = "offline"


class GridPoint(BaseModel):
    x: int
    y: int


class DroneState(BaseModel):
    drone_id: str
    position: GridPoint
    home: GridPoint
    status: DroneStatus = DroneStatus.IDLE
    target_cell: GridPoint | None = None
    target_seen: bool = False
    video_url: str | None = None


class ArenaCell(BaseModel):
    x: int
    y: int
    zone: str  # red | neutral | blue


class GotoCommand(BaseModel):
    x: int
    y: int


class RegisterDroneCommand(BaseModel):
    drone_id: str
    home_x: int
    home_y: int
    video_url: str | None = None


class TargetSpottedCommand(BaseModel):
    x: int | None = None
    y: int | None = None


class SystemConfigCommand(BaseModel):
    mode: str  # live | simulator | sim_api
    drone_count: int  # 2..5
    sim_api_host: str | None = None  # host:port of sim_api_server (for sim_api mode)


class TargetCell(BaseModel):
    x: int
    y: int
    color: str = "red"  # red | blue


class SetTargetsCommand(BaseModel):
    targets: list[TargetCell]
