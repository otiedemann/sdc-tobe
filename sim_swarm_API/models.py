from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class Vec2:
    x: float
    y: float

    def dist(self, other: "Vec2") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Drone:
    id: str
    pos: Vec2
    speed_mps: float = 1.2
    battery_pct: float = 100.0
    state: str = "IDLE"  # IDLE|SEARCH|TRACK|RETURN|LAND|ABORT|DROPOUT
    target: Optional[Vec2] = None
    active: bool = True
    link_ok: bool = True


@dataclass
class World:
    width_m: float
    height_m: float
    home: Vec2
    target: Vec2
    target_captured: bool = False
    capture_radius_m: float = 0.7
    min_sep_m: float = 0.6
    near_collision_count: int = 0
    link_loss_events: int = 0
