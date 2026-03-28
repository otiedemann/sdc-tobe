"""
Arena and ArUco marker configuration for SDC26.

Default marker positions derived from aruco-position/control-unit/pi_position.py.
The pi_position coordinate system uses centered origin (x: -10..10, y: 0..10).
Arena coordinates use (x: 0..20, y: 0..10), so arena_x = world_x + 10.
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .models import DroneConfigItem, Vec2, Vec3


class PillarMarker(BaseModel):
    """A single ArUco marker on a boundary pillar."""
    marker_id: int
    position: Vec3  # in arena coordinates (0-20, 0-10, 0-6)
    wall: str  # "front", "back", "left", "right"


class ArenaConfig(BaseModel):
    """Full arena and game configuration."""

    # Arena dimensions
    width_m: float = 20.0
    height_m: float = 10.0
    ceiling_m: float = 6.0

    # Team configuration
    our_team: str = "blue"  # "red" or "blue"

    # Home zones (x_min, y_min, x_max, y_max) — using arena coords
    # Red: left 7m, Blue: right 7m (HOME_ZONE_LEN = 7 in simulator)
    red_home_zone: tuple[float, float, float, float] = (0.0, 0.0, 7.0, 10.0)
    blue_home_zone: tuple[float, float, float, float] = (13.0, 0.0, 20.0, 10.0)

    # ArUco marker positions on boundary pillars
    # Default positions from pi_position.py, transformed to arena coords
    pillar_markers: dict[str, PillarMarker] = {}

    # Target box marker ID range
    target_marker_id_min: int = 1
    target_marker_id_max: int = 12

    # Capture rules
    capture_hover_seconds: float = 2.0  # seconds to hover for capture
    special_capture_points: int = 5
    simultaneous_capture_bonus: int = 10
    normal_capture_points: int = 1

    # Drone fleet configuration
    drone_configs: list[DroneConfigItem] = []

    # Battery management
    battery_return_threshold: float = 20.0  # percent

    # Movement parameters
    approach_tolerance_m: float = 0.5  # how close is "at target"
    home_zone_tolerance_m: float = 0.5

    def home_zone(self) -> tuple[float, float, float, float]:
        """Return our team's home zone."""
        return self.blue_home_zone if self.our_team == "blue" else self.red_home_zone

    def enemy_zone(self) -> tuple[float, float, float, float]:
        """Return the enemy's home zone."""
        return self.red_home_zone if self.our_team == "blue" else self.blue_home_zone

    def is_in_home_zone(self, pos: Vec2) -> bool:
        z = self.home_zone()
        return z[0] <= pos.x <= z[2] and z[1] <= pos.y <= z[3]

    def is_in_enemy_zone(self, pos: Vec2) -> bool:
        z = self.enemy_zone()
        return z[0] <= pos.x <= z[2] and z[1] <= pos.y <= z[3]

    def home_center(self) -> Vec2:
        """Center of our home zone."""
        z = self.home_zone()
        return Vec2(x=(z[0] + z[2]) / 2, y=(z[1] + z[3]) / 2)

    def default_home_positions(self, count: int) -> list[Vec2]:
        """Generate evenly-spaced home positions for drones."""
        z = self.home_zone()
        cx = (z[0] + z[2]) / 2
        positions = []
        for i in range(count):
            y = z[1] + 1.0 + i * ((z[3] - z[1] - 2.0) / max(1, count - 1)) if count > 1 else (z[1] + z[3]) / 2
            positions.append(Vec2(x=cx, y=y))
        return positions


def _default_pillar_markers() -> dict[str, PillarMarker]:
    """
    Default ArUco marker positions on boundary pillars.

    Derived from pi_position.py marker_world_positions.
    Transform: arena_x = world_x + 10, arena_y = world_y.
    Heights at 2m and 4m on each pillar.
    """
    markers = {}

    # Pillar positions from the simulator (10 pillars):
    # Top edge (front, y≈0): x = -5, 0, 5 → arena x = 5, 10, 15
    # Bottom edge (back, y≈10): x = -5, 0, 5 → arena x = 5, 10, 15
    # Left edge (x≈-10): z = -arena.y/6, arena.y/6 → y ≈ 3.3, 6.7
    # Right edge (x≈10): z = -arena.y/6, arena.y/6 → y ≈ 3.3, 6.7

    pillar_positions = [
        # Front wall (arena y=0 boundary)
        {"x": 5.0, "y": 0.0, "wall": "front"},
        {"x": 10.0, "y": 0.0, "wall": "front"},
        {"x": 15.0, "y": 0.0, "wall": "front"},
        # Back wall (arena y=10 boundary)
        {"x": 5.0, "y": 10.0, "wall": "back"},
        {"x": 10.0, "y": 10.0, "wall": "back"},
        {"x": 15.0, "y": 10.0, "wall": "back"},
        # Left wall (arena x=0 boundary)
        {"x": 0.0, "y": 3.3, "wall": "left"},
        {"x": 0.0, "y": 6.7, "wall": "left"},
        # Right wall (arena x=20 boundary)
        {"x": 20.0, "y": 3.3, "wall": "right"},
        {"x": 20.0, "y": 6.7, "wall": "right"},
    ]

    marker_id = 21  # Starting ID matches the simulator's addBoundaryPillars
    for pillar in pillar_positions:
        for height in [2.0, 4.0]:
            markers[str(marker_id)] = PillarMarker(
                marker_id=marker_id,
                position=Vec3(x=pillar["x"], y=pillar["y"], z=height),
                wall=pillar["wall"],
            )
            marker_id += 1

    return markers


def default_arena_config() -> ArenaConfig:
    """Create a default arena config with standard SDC26 layout."""
    return ArenaConfig(
        pillar_markers=_default_pillar_markers(),
        drone_configs=[
            DroneConfigItem(
                drone_id="drone-1",
                api_base_url="http://localhost:8080",
                drone_type="simulator",
                sim_drone_id="B1",
            ),
            DroneConfigItem(
                drone_id="drone-2",
                api_base_url="http://localhost:8080",
                drone_type="simulator",
                sim_drone_id="B2",
            ),
            DroneConfigItem(
                drone_id="drone-3",
                api_base_url="http://localhost:8080",
                drone_type="simulator",
                sim_drone_id="B3",
            ),
        ],
    )


def load_arena_config(path: Optional[str] = None) -> ArenaConfig:
    """Load arena config from JSON file, or return defaults."""
    if path is None:
        path = str(Path(__file__).parent / "arena_config.json")
    p = Path(path)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return ArenaConfig(**data)
        except Exception:
            pass
    return default_arena_config()


def save_arena_config(config: ArenaConfig, path: Optional[str] = None) -> None:
    """Save arena config to JSON file."""
    if path is None:
        path = str(Path(__file__).parent / "arena_config.json")
    Path(path).write_text(
        config.model_dump_json(indent=2),
        encoding="utf-8",
    )
