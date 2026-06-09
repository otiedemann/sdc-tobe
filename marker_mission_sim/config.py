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
    # v7 §1.4.3 capture mechanics — sim mirrors the official rules.
    radius_m: float = 0.8        # xy distance to a box to count as "over" it
    z_min_m: float = 1.0         # capture altitude band (box detect band, 1-2 m)
    z_max_m: float = 2.0
    hold_s: float = 2.0          # >=2 s hover required to flip an enemy box
    max_speed_mps: float = 0.5   # must be slow enough to count as hovering
    # After a successful capture the box is locked for this long — no further
    # captures (in either direction) until the lock expires (v7 §1.4.3).
    post_capture_lock_s: float = 5.0

    @classmethod
    def from_dict(cls, d: dict) -> "CaptureConfig":
        d = d or {}
        return cls(
            radius_m=float(d.get("radius_m", 0.8)),
            z_min_m=float(d.get("z_min_m", 1.0)),
            z_max_m=float(d.get("z_max_m", 2.0)),
            hold_s=float(d.get("hold_s", 2.0)),
            max_speed_mps=float(d.get("max_speed_mps", 0.5)),
            post_capture_lock_s=float(d.get("post_capture_lock_s", 5.0)),
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
    arena_name: Optional[str] = None   # registry name (arenas.json); wins over paths
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    seed: int = 0                # RNG seed for reproducible noise

    # filled in __post_init__
    arena: ArenaDef = field(default=None)  # type: ignore

    def __post_init__(self) -> None:
        if self.arena is None:
            if self.arena_name:
                self.arena = load_arena_by_name(self.arena_name)
            elif self.arena_config_path or self.target_layout_path:
                self.arena = load_arena(self.arena_config_path,
                                        self.target_layout_path)
            else:
                # Nothing specified -> load the registry's default arena BY NAME
                # so the sim's startup geometry is byte-identical to a live
                # switch back to that arena (not the built-in default_arena()).
                self.arena_name = default_arena_name()
                self.arena = load_arena_by_name(self.arena_name)
        # Always resolve a human-facing arena name so the World can report it.
        if not self.arena_name:
            self.arena_name = default_arena_name()

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
            arena_name=data.get("arena_name"),
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

    # Target boxes: prefer the arena config's own `targets` block (id = slot
    # 1..6, team-derived facing); fall back to the legacy target-layout file
    # (face-id `boxes`). Keeps the sim's ground truth identical to the strategy.
    boxes: list[BoxDef] = []
    cfg_targets = None
    if arena_config_path:
        try:
            acfg = json.loads(Path(arena_config_path).read_text())
            if isinstance(acfg.get("targets"), list):
                cfg_targets = acfg["targets"]
        except (OSError, ValueError):
            cfg_targets = None

    if cfg_targets:
        for t in cfg_targets:
            if not isinstance(t, dict) or not t.get("enabled", True):
                continue
            try:
                slot = int(t["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (1 <= slot <= 6):
                continue
            boxes.append(BoxDef(
                slot=slot,
                pos=Vec3(float(t["x"]), float(t["y"]), float(t.get("z", 1.0))),
                home_team=str(t.get("team", "red")),
                blue_face_id=30 + slot,
                red_face_id=40 + slot,
            ))
    else:
        tl_path = Path(target_layout_path) if target_layout_path else (
            REPO_ROOT / "tools" / "sphinx-arena" / "default_target_layout.json")
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


# ---------------------------------------------------------------------------
# Named-arena registry (shared with the strategy) — enables a live switch
# ---------------------------------------------------------------------------

ARENA_REGISTRY_PATH = REPO_ROOT / "marker_mission_c2" / "arena" / "arenas.json"


def _read_registry() -> dict:
    try:
        return json.loads(ARENA_REGISTRY_PATH.read_text())
    except (OSError, ValueError):
        return {"default": "real", "arenas": {}}


def arena_names() -> list[str]:
    """Registered arena names, e.g. ['real', 'gvz']."""
    return list(_read_registry().get("arenas", {}).keys())


def default_arena_name() -> str:
    """The registry's default arena name (falls back to 'real')."""
    return str(_read_registry().get("default", "real"))


def load_arena_by_name(name: str) -> ArenaDef:
    """Build an :class:`ArenaDef` for a *registered* arena name (see
    ``marker_mission_c2/arena/arenas.json``). Unknown names fall back to the
    built-in default arena + default target layout."""
    entry = _read_registry().get("arenas", {}).get(str(name))
    if not entry:
        return load_arena(None, None)
    acp = entry.get("arena_config_path")
    tlp = entry.get("target_layout_path")
    acp = str(REPO_ROOT / acp) if acp else None
    tlp = str(REPO_ROOT / tlp) if tlp else None
    return load_arena(acp, tlp)
