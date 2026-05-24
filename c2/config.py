"""Configuration for the SDC26 C2 strategy server.

Arena geometry is INHERITED from ``marker_mission.arena.default_arena()``
(or an explicit arena JSON) so the C2, the simulator build
(tools/sphinx-arena) and the position estimator all share one source of
truth. Everything team/strategy specific lives here.

Load order:
  1. ``C2Config.load(path)`` reads a YAML file (see config.example.yaml).
  2. Missing keys fall back to the dataclass defaults below.
  3. Arena dims + target geometry come from marker_mission unless the
     YAML overrides ``arena_config_path``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import Team, DroneRole, Vec2, Vec3, SlotColor

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Target-slot <-> marker-ID mapping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlotMap:
    """Maps target marker IDs to physical slots and colours.

    Convention (see models.TargetSlot):
        blue-state ID = blue_base + index   (default blue_base = 30)
        red-state  ID = red_base  + index   (default red_base  = 40)
        index in 1..6 ; slots 1..3 are in the blue zone, 4..6 in red.
    """
    blue_base: int = 30
    red_base: int = 40
    n_slots: int = 6
    blue_zone_slots: tuple[int, ...] = (1, 2, 3)
    red_zone_slots: tuple[int, ...] = (4, 5, 6)

    def slot_of(self, marker_id: int) -> Optional[int]:
        if self.blue_base + 1 <= marker_id <= self.blue_base + self.n_slots:
            return marker_id - self.blue_base
        if self.red_base + 1 <= marker_id <= self.red_base + self.n_slots:
            return marker_id - self.red_base
        return None

    def color_of(self, marker_id: int) -> SlotColor:
        if self.blue_base + 1 <= marker_id <= self.blue_base + self.n_slots:
            return SlotColor.BLUE
        if self.red_base + 1 <= marker_id <= self.red_base + self.n_slots:
            return SlotColor.RED
        return SlotColor.UNKNOWN

    def home_team_of(self, slot_index: int) -> Team:
        return Team.BLUE if slot_index in self.blue_zone_slots else Team.RED

    def is_target_marker(self, marker_id: int) -> bool:
        return self.slot_of(marker_id) is not None

    def blue_id(self, slot_index: int) -> int:
        return self.blue_base + slot_index

    def red_id(self, slot_index: int) -> int:
        return self.red_base + slot_index


# ---------------------------------------------------------------------------
# Per-drone fleet config
# ---------------------------------------------------------------------------

@dataclass
class DroneFcConfig:
    drone_id: str
    fc_base_url: str                       # e.g. http://sphinx3.otconsulting.de:8080
    initial_role: DroneRole = DroneRole.IDLE
    base_altitude_m: float = 2.0           # cruise altitude (collision deconfliction)

    @classmethod
    def from_dict(cls, d: dict) -> "DroneFcConfig":
        return cls(
            drone_id=str(d["drone_id"]),
            fc_base_url=str(d["fc_base_url"]).rstrip("/"),
            initial_role=DroneRole(d.get("initial_role", "idle")),
            base_altitude_m=float(d.get("base_altitude_m", 2.0)),
        )


# ---------------------------------------------------------------------------
# Arena geometry (loaded from marker_mission)
# ---------------------------------------------------------------------------

@dataclass
class ArenaGeometry:
    width_m: float = 10.0       # x extent (total)
    depth_m: float = 20.0       # y extent (total)
    ceiling_m: float = 6.0
    # Team zones split the LONG (y) axis: red at -y end, blue at +y end,
    # 5 m each, neutral 10 m in the middle (regs §1.1).
    team_zone_depth_m: float = 5.0

    @property
    def x_min(self) -> float: return -self.width_m / 2
    @property
    def x_max(self) -> float: return self.width_m / 2
    @property
    def y_min(self) -> float: return -self.depth_m / 2
    @property
    def y_max(self) -> float: return self.depth_m / 2

    def center(self) -> Vec2:
        return Vec2(0.0, 0.0)

    def home_zone_y(self, team: Team) -> tuple[float, float]:
        """(y_lo, y_hi) of a team's home zone. Red = -y end, blue = +y end."""
        if team is Team.RED:
            return (self.y_min, self.y_min + self.team_zone_depth_m)
        return (self.y_max - self.team_zone_depth_m, self.y_max)

    def home_center(self, team: Team) -> Vec2:
        lo, hi = self.home_zone_y(team)
        return Vec2(0.0, (lo + hi) / 2)

    def in_home_zone(self, pos: Vec3, team: Team) -> bool:
        lo, hi = self.home_zone_y(team)
        return (self.x_min <= pos.x <= self.x_max) and (lo <= pos.y <= hi)


def load_arena_geometry(arena_config_path: Optional[str]) -> tuple[ArenaGeometry, dict[int, Vec3]]:
    """Return (geometry, {slot_index: nominal target position}) from
    marker_mission. The nominal target positions seed slot positions
    before the drones have actually seen them."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from marker_mission.arena import default_arena, ArenaConfig as MMArena  # type: ignore

    if arena_config_path:
        arena = MMArena.load(Path(arena_config_path))
    else:
        arena = default_arena()

    geo = ArenaGeometry(
        width_m=float(arena.width_m),
        depth_m=float(arena.depth_m),
    )
    # marker_mission only carries the 16 wall markers; target nominal
    # positions are not in its arena. Seed them from the regs layout:
    # 3 per team at the zone-centre y, spread across x = -3/0/+3.
    nominal: dict[int, Vec3] = {}
    red_lo, red_hi = geo.home_zone_y(Team.RED)
    blue_lo, blue_hi = geo.home_zone_y(Team.BLUE)
    red_y = (red_lo + red_hi) / 2
    blue_y = (blue_lo + blue_hi) / 2
    xs = (-3.0, 0.0, 3.0)
    # slots 1..3 blue zone, 4..6 red zone
    for i, x in enumerate(xs):
        nominal[1 + i] = Vec3(x, blue_y, 1.0)
        nominal[4 + i] = Vec3(x, red_y, 1.0)
    return geo, nominal


# ---------------------------------------------------------------------------
# Strategy tuning
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    tick_hz: float = 4.0                # commander loop rate

    # Capture mechanics (regs §1.3)
    capture_altitude_m: float = 1.5     # inside the 1-2 m box detection band
    capture_hover_s: float = 2.0        # hold to trigger a colour change
    capture_validate_home_s: float = 5.0
    arrival_tol_m: float = 0.6          # "close enough" to a waypoint (xy)
    altitude_tol_m: float = 0.3

    # Defender behaviour
    defender_patrol_altitude_m: float = 2.2   # above the 2 m capture band
    defender_uncap_altitude_m: float = 1.5
    anti_camp_max_s: float = 3.0        # never linger >Ns over an own box

    # Scout behaviour
    scout_altitude_m: float = 4.0       # high, sees the whole arena
    scout_orbit_radius_m: float = 2.5   # orbit the arena centre
    scout_yaw_rate_dps: float = 30.0    # scan rate while orbiting

    # RC closed-loop nav gains (rc range -100..100)
    rc_kp_xy: float = 35.0              # %RC per metre of xy error
    rc_kp_z: float = 40.0
    rc_kp_yaw: float = 1.5             # %RC per degree of heading error
    rc_max: int = 50                   # clamp on any rc axis
    slot_age_fresh_s: float = 3.0      # a slot older than this is "stale"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class C2Config:
    our_team: Team = Team.RED
    mode_start_auto: bool = False       # start in MANUAL by default
    host: str = "0.0.0.0"
    port: int = 8095

    drones: list[DroneFcConfig] = field(default_factory=list)
    arena_config_path: Optional[str] = None

    slot_map: SlotMap = field(default_factory=SlotMap)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    # populated by __post_init__ / load
    arena: ArenaGeometry = field(default_factory=ArenaGeometry)
    nominal_slot_positions: dict[int, Vec3] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.nominal_slot_positions:
            self.arena, self.nominal_slot_positions = load_arena_geometry(self.arena_config_path)

    # -- helpers -----------------------------------------------------------
    def enemy_team(self) -> Team:
        return self.our_team.enemy

    def our_slots(self) -> list[int]:
        """Slot indices physically in OUR home zone (the ones we defend)."""
        return [i for i in range(1, self.slot_map.n_slots + 1)
                if self.slot_map.home_team_of(i) == self.our_team]

    def enemy_slots(self) -> list[int]:
        """Slot indices in the ENEMY home zone (the ones we attack)."""
        return [i for i in range(1, self.slot_map.n_slots + 1)
                if self.slot_map.home_team_of(i) == self.our_team.enemy]

    @classmethod
    def load(cls, path: Optional[str] = None) -> "C2Config":
        import yaml  # local import: only needed when loading from disk
        data: dict = {}
        if path:
            data = yaml.safe_load(Path(path).read_text()) or {}
        drones = [DroneFcConfig.from_dict(d) for d in data.get("drones", [])]
        slot_map = SlotMap(**data["slot_map"]) if data.get("slot_map") else SlotMap()
        strat = StrategyConfig(**data["strategy"]) if data.get("strategy") else StrategyConfig()
        cfg = cls(
            our_team=Team(data.get("our_team", "red")),
            mode_start_auto=bool(data.get("mode_start_auto", False)),
            host=str(data.get("host", "0.0.0.0")),
            port=int(data.get("port", 8095)),
            drones=drones,
            arena_config_path=data.get("arena_config_path"),
            slot_map=slot_map,
            strategy=strat,
        )
        return cfg
