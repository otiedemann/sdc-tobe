"""Simulator configuration + arena/target loading.

Geometry comes from ``marker_mission.arena`` (the same source the C2 and
the real FC use) for the 16 wall markers + arena dimensions, and from
``tools/sphinx-arena/default_target_layout.json`` for the 6 target boxes
(or an explicit path). Everything sim-specific (drones, ports, noise) is
in the sim config JSON — see sim_config.example.json.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .geometry import Vec3

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Static arena description (loaded once, immutable during a run)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WallMarkerDef:
    id: int
    pos: Vec3
    wall: str               # "front" | "back" | "left" | "right"


@dataclass(frozen=True)
class BoxDef:
    slot: int               # 1..6
    pos: Vec3               # arena position of the box (z = box top-ish)
    home_team: str          # "red" | "blue" — default holder
    blue_face_id: int       # 30 + slot
    red_face_id: int        # 40 + slot


@dataclass
class ArenaDef:
    width_m: float
    depth_m: float
    ceiling_m: float
    wall_markers: list[WallMarkerDef]
    boxes: list[BoxDef]

    @property
    def x_min(self) -> float: return -self.width_m / 2
    @property
    def x_max(self) -> float: return self.width_m / 2
    @property
    def y_min(self) -> float: return -self.depth_m / 2
    @property
    def y_max(self) -> float: return self.depth_m / 2


# ---------------------------------------------------------------------------
# Sim config (from JSON)
# ---------------------------------------------------------------------------

@dataclass
class SimDroneConfig:
    id: str
    team: str                       # "red" | "blue"
    fc_port: int                    # HTTP port for THIS drone's FC API
    spawn_x: float = 0.0
    spawn_y: float = 0.0
    spawn_heading_deg: float = 0.0  # CW from +Y

    @classmethod
    def from_dict(cls, d: dict) -> "SimDroneConfig":
        return cls(
            id=str(d["id"]),
            team=str(d.get("team", "red")),
            fc_port=int(d["fc_port"]),
            spawn_x=float(d.get("spawn_x", 0.0)),
            spawn_y=float(d.get("spawn_y", 0.0)),
            spawn_heading_deg=float(d.get("spawn_heading_deg", 0.0)),
        )


@dataclass
class NoiseConfig:
    pos_noise_m: float = 0.05         # stddev of position noise added to reported pose/markers
    marker_dropout_prob: float = 0.10  # per-marker chance a visible marker is dropped this frame
    cmd_latency_s: float = 0.20        # delay before a pushed script starts executing
    vision_range_m: float = 8.0        # max marker detection distance
    vision_fov_deg: float = 70.0       # horizontal camera FOV (full angle)

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseConfig":
        d = d or {}
        return cls(
            pos_noise_m=float(d.get("pos_noise_m", 0.05)),
            marker_dropout_prob=float(d.get("marker_dropout_prob", 0.10)),
            cmd_latency_s=float(d.get("cmd_latency_s", 0.20)),
            vision_range_m=float(d.get("vision_range_m", 8.0)),
            vision_fov_deg=float(d.get("vision_fov_deg", 70.0)),
        )


@dataclass
class CaptureConfig:
    radius_m: float = 0.8        # xy distance to a box to count as "over" it
    z_min_m: float = 1.0         # capture altitude band (box detect band)
    z_max_m: float = 2.0
    hold_s: float = 2.0          # dwell needed to flip a box
    max_speed_mps: float = 0.5   # must be slower than this to capture

    @classmethod
    def from_dict(cls, d: dict) -> "CaptureConfig":
        d = d or {}
        return cls(
            radius_m=float(d.get("radius_m", 0.8)),
            z_min_m=float(d.get("z_min_m", 1.0)),
            z_max_m=float(d.get("z_max_m", 2.0)),
            hold_s=float(d.get("hold_s", 2.0)),
            max_speed_mps=float(d.get("max_speed_mps", 0.5)),
        )


@dataclass
class SimConfig:
    drones: list[SimDroneConfig] = field(default_factory=list)
    ui_host: str = "127.0.0.1"
    ui_port: int = 9100
    fc_host: str = "127.0.0.1"
    sim_hz: float = 30.0
    arena_config_path: Optional[str] = None
    target_layout_path: Optional[str] = None
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    seed: int = 0                # RNG seed for reproducible noise

    # filled in __post_init__
    arena: ArenaDef = field(default=None)  # type: ignore

    def __post_init__(self) -> None:
        if self.arena is None:
            self.arena = load_arena(self.arena_config_path, self.target_layout_path)

    @classmethod
    def load(cls, path: Optional[str]) -> "SimConfig":
        data: dict = {}
        if path:
            data = json.loads(Path(path).read_text())
        drones = [SimDroneConfig.from_dict(d) for d in data.get("drones", [])]
        return cls(
            drones=drones,
            ui_host=str(data.get("ui_host", "127.0.0.1")),
            ui_port=int(data.get("ui_port", 9100)),
            fc_host=str(data.get("fc_host", "127.0.0.1")),
            sim_hz=float(data.get("sim_hz", 30.0)),
            arena_config_path=data.get("arena_config_path"),
            target_layout_path=data.get("target_layout_path"),
            noise=NoiseConfig.from_dict(data.get("noise", {})),
            capture=CaptureConfig.from_dict(data.get("capture", {})),
            seed=int(data.get("seed", 0)),
        )


# ---------------------------------------------------------------------------
# Arena + target loading
# ---------------------------------------------------------------------------

def _slot_of_face(marker_id: int) -> Optional[int]:
    if 31 <= marker_id <= 36:
        return marker_id - 30
    if 41 <= marker_id <= 46:
        return marker_id - 40
    return None


def load_arena(arena_config_path: Optional[str],
               target_layout_path: Optional[str]) -> ArenaDef:
    """Build the immutable ArenaDef from marker_mission + the target layout."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from marker_mission.arena import default_arena, ArenaConfig as MMArena  # type: ignore

    arena = (MMArena.load(Path(arena_config_path))
             if arena_config_path else default_arena())

    wall_markers: list[WallMarkerDef] = []
    for m in sorted(arena.markers.values(), key=lambda x: x.id):
        p = m.position_m
        wall_markers.append(WallMarkerDef(
            id=int(m.id),
            pos=Vec3(float(p[0]), float(p[1]), float(p[2])),
            wall=str(getattr(m, "wall", "") or ""),
        ))

    # Target boxes from the layout JSON (enabled boxes only).
    tl_path = Path(target_layout_path) if target_layout_path else (
        REPO_ROOT / "tools" / "sphinx-arena" / "default_target_layout.json")
    boxes: list[BoxDef] = []
    try:
        tl = json.loads(tl_path.read_text())
        for b in tl.get("boxes", []):
            if not b.get("enabled", True):
                continue
            fid = int(b["id"])
            slot = _slot_of_face(fid)
            if slot is None:
                continue
            boxes.append(BoxDef(
                slot=slot,
                pos=Vec3(float(b["x"]), float(b["y"]), float(b.get("z", 1.0))),
                home_team=str(b.get("team", "red")),
                blue_face_id=30 + slot,
                red_face_id=40 + slot,
            ))
    except (OSError, ValueError, KeyError):
        boxes = []
    boxes.sort(key=lambda x: x.slot)

    return ArenaDef(
        width_m=float(arena.width_m),
        depth_m=float(arena.depth_m),
        ceiling_m=6.0,
        wall_markers=wall_markers,
        boxes=boxes,
    )
