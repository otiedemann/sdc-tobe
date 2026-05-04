#!/usr/bin/env python3
"""Convert the SDC arena layout into a Parrot Sphinx ``--config-file`` YAML.

Reads:
  - ``controller_unified/arena_config.json`` for wall-marker positions
    (16 markers, 4 walls × 2 positions × 2 heights).
  - ``default_target_layout.json`` for the 6 SDC26 target boxes plus
    spares (only ``enabled=true`` boxes are emitted).

Writes:
  - A single YAML file Sphinx loads at startup. Each entry references
    one FBX (built earlier by ``build_marker_fbx.py``) and places it at
    a converted UE4 coordinate.

Coordinate-system conversion:
  Arena frame  (from ``arena_config.json``): metres, right-handed,
                +X = arena right, +Y = arena forward (away from origin
                wall), +Z = up. Origin is the arena floor centre.
  Sphinx/UE4 frame: centimetres, left-handed, +X = forward, +Y = right,
                +Z = up. (Default UE convention: ``X-forward, Y-right,
                Z-up`` with metres × 100.)

  The ``--axis-map`` flag controls which arena axis becomes which UE
  axis, since groundtruth axis orientation depends on how you place the
  arena's "front" wall in the UE world. Default ``y2x,x2y_neg,z2z`` puts
  the arena +Y direction along UE +X (forward) and arena +X along UE -Y
  (right→left), which matches a UE world set up "looking down the long
  axis from origin". Verify visually after first launch and adjust if
  markers land on the wrong wall.

Marker rotation (``Rotation`` field):
  Each FBX plane is built to face -Y in its local frame (see
  ``build_marker_fbx.py``). Rotation here aims the plane so its visible
  side faces inward toward the arena centre. ``yaw_deg`` per wall:
    front: 0°    (FBX -Y already points -Y in arena → into arena)
    back:  180°
    left:  +90°
    right: -90°
  These are derived from the ``wall`` field for each marker.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARENA_CONFIG = REPO_ROOT / "controller_unified" / "arena_config.json"
DEFAULT_TARGET_LAYOUT = Path(__file__).parent / "default_target_layout.json"
DEFAULT_FBX_DIR = Path(__file__).parent / "out" / "fbx"
DEFAULT_OUT_YAML = Path(__file__).parent / "out" / "arena.yml"


# Marker orientation in YAML. The Sphinx YAML's per-mesh ``Rotation``
# field is parsed by UE4's FRotator parser, which uses ``(Pitch, Yaw,
# Roll)`` order — NOT the Blender-ish ``(Roll, Pitch, Yaw)`` we tried
# first. With the wrong order, the live render showed markers as tall
# thin rectangles with the ArUco pattern rotated 90° (yaw was being
# applied as a pre-tilt rotation around Z and roll was tilting the
# plane sideways).
#
# Correct order, applied to a flat XY plane (normal +Z) coming out
# of Blender:
#   - Pitch (around UE Y) tilts the plane upright. After Pitch=+90°
#     the plane is in YZ with normal pointing +X (forward).
#   - Yaw (around UE Z) then rotates that +X normal to face the
#     inward direction of each wall.
#   - Roll stays 0.
#
# Inward direction per wall (in UE coords, default axis_map):
#   front (arena y=0,    inward arena +Y → UE +X)  yaw   0°
#   back  (arena y=10.8, inward arena -Y → UE -X)  yaw 180°
#   left  (arena x=-10,  inward arena +X → UE -Y)  yaw -90°
#   right (arena x=+10,  inward arena -X → UE +Y)  yaw +90°
MARKER_PITCH_DEG: float = 90.0
# UE4's FRotator pitch is "nose up": positive pitch rotates the actor's
# +X (forward) toward +Z (up). Applied to a flat plane whose local
# normal is +Z, that means after Pitch=+90° the normal lands at UE -X
# (NOT +X, NOT +Y — the previous comment got this wrong, which is why
# the live render kept showing markers facing the wrong direction).
# Yaw then rotates that -X normal around UE +Z; UE positive yaw is
# clockwise viewed from above, so +X → +Y → -X → -Y → +X.
#
# Required inward direction per wall (in UE coords, axis_map=y2x,x2y_neg):
#   front (arena y=0,    inward arena +Y → UE +X)  : -X → +X needs yaw 180°
#   back  (arena y=10.8, inward arena -Y → UE -X)  : -X → -X needs yaw   0°
#   left  (arena x=-10,  inward arena +X → UE -Y)  : -X → -Y needs yaw +90°
#   right (arena x=+10,  inward arena -X → UE +Y)  : -X → +Y needs yaw -90°
#
# If markers come out facing OUTWARD after this fix, every value here
# should be flipped 180°. The white back plate added in
# build_marker_fbx.py makes this trivial to spot: from inside the
# arena you should see the ArUco face; from outside, plain white.
WALL_YAW_DEG: dict[str, float] = {
    "front": 180.0,   # arena y=0    → faces UE +X
    "back":    0.0,   # arena y=10.8 → faces UE -X
    "left":   90.0,   # arena x=-10  → faces UE -Y
    "right": -90.0,   # arena x=+10  → faces UE +Y
}


# Paintings: one logo per wall, mounted at mid-height between the
# low (z=2) and high (z=4) markers so they don't occlude either.
# On long walls (front/back) markers sit at x = -5, 0, +5 — so we
# offset paintings along the wall to fit between markers without
# overlap.
_PAINTING_SIZE_M: tuple[float, float] = (1.5, 1.0)  # smaller, fits between markers
_PAINTING_Z_M: float = 3.0     # mid-height between the marker rings

# (logo_filename_stem, wall_id, offset_along_wall_m).
# offset_along_wall_m: how far from wall centre to position the
# painting along the wall's long axis. For long walls (front/back),
# centre marker sits at x=0, so we offset to x=±2.5 (between the
# centre marker at 0 and side marker at ±5). For short walls
# (left/right), there's a single marker pair at y=5.4, so the
# painting goes off-centre along Y to clear it.
PAINTINGS: list[tuple[str, str, float]] = [
    ("mbda",  "front", -2.5),  # front wall, between centre and -X marker
    ("brigk", "back",  +2.5),  # back wall, between centre and +X marker
    ("sdc",   "left",  -2.5),  # left wall, offset from centre marker
    ("team",  "right", +2.5),  # right wall, offset from centre marker
]


@dataclass(frozen=True)
class AxisMap:
    """How arena (x, y, z) maps to UE (x, y, z). Each entry is a tuple
    ``(arena_axis, sign)`` where ``arena_axis ∈ {0,1,2}`` and ``sign ∈ {+1,-1}``.
    Default: arena +Y → UE +X, arena +X → UE -Y, arena +Z → UE +Z."""

    ue_x: tuple[int, int]  # default: (1, +1) — arena Y becomes UE X
    ue_y: tuple[int, int]  # default: (0, -1) — arena X becomes UE -Y
    ue_z: tuple[int, int]  # default: (2, +1) — arena Z becomes UE Z

    def apply(self, arena_xyz_m: tuple[float, float, float]) -> tuple[float, float, float]:
        ax, ay, az = arena_xyz_m
        a = (ax, ay, az)
        ux = a[self.ue_x[0]] * self.ue_x[1]
        uy = a[self.ue_y[0]] * self.ue_y[1]
        uz = a[self.ue_z[0]] * self.ue_z[1]
        return (ux, uy, uz)


_AXIS_MAP_PRESETS: dict[str, AxisMap] = {
    "y2x,x2y_neg,z2z": AxisMap(ue_x=(1, +1), ue_y=(0, -1), ue_z=(2, +1)),
    "x2x,y2y,z2z":     AxisMap(ue_x=(0, +1), ue_y=(1, +1), ue_z=(2, +1)),
    "x2x_neg,y2y,z2z": AxisMap(ue_x=(0, -1), ue_y=(1, +1), ue_z=(2, +1)),
    "y2x_neg,x2y,z2z": AxisMap(ue_x=(1, -1), ue_y=(0, +1), ue_z=(2, +1)),
}


def fmt3(x: float, y: float, z: float) -> str:
    """Sphinx wants space-separated floats inside a quoted string."""
    return f"{x:.3f} {y:.3f} {z:.3f}"


def emit_yaml_block(
    name: str,
    fbx_path: Path,
    location_cm: tuple[float, float, float],
    rotation_deg: tuple[float, float, float],
    scale: float | tuple[float, float, float],
    snap_to_ground: bool = False,
) -> str:
    """Render one Sphinx ``Meshes:`` entry in the documented YAML form.

    ``rotation_deg`` is **(Pitch, Yaw, Roll)** — UE4's FRotator order
    when serialised as space-separated floats. Earlier code emitted
    (Roll, Pitch, Yaw) which the live render showed Sphinx
    misinterprets (markers came out tall-and-thin with patterns
    rotated 90°).

    ``scale`` may be a uniform float OR a per-axis tuple. Per-axis is
    needed for the arena floor (large in X/Y, unit in Z) and pillars
    (thin in X/Y, tall in Z)."""
    if isinstance(scale, (int, float)):
        sx = sy = sz = float(scale)
    else:
        sx, sy, sz = (float(v) for v in scale)
    lines = [
        f"  - Name: {name}",
        f"    FbxPath: \"{fbx_path}\"",
        f"    Location: \"{fmt3(*location_cm)}\"",
        f"    Rotation: \"{fmt3(*rotation_deg)}\"",
        f"    Scale: \"{fmt3(sx, sy, sz)}\"",
        f"    SnapToGround: {'true' if snap_to_ground else 'false'}",
    ]
    return "\n".join(lines)


# Per-wall outward offset applied to pillar centerlines. Markers in
# arena_config.json sit at the *inward* face of the wall; the pillar
# body extends outward (away from the arena interior) by half its
# width so the marker plane is flush with the inner surface.
_PILLAR_WIDTH_M = 0.20
_HALF_PILLAR_M = _PILLAR_WIDTH_M / 2.0
_PILLAR_HEIGHT_M = 6.0

# Wall → (dx, dy) outward offset for the pillar relative to the marker (x,y).
WALL_OUTWARD_OFFSET_M: dict[str, tuple[float, float]] = {
    "left":  (-_HALF_PILLAR_M, 0.0),   # x = -10  → pillar pushed to -10.1
    "right": (+_HALF_PILLAR_M, 0.0),   # x = +10  → +10.1
    "back":  (0.0, +_HALF_PILLAR_M),   # y = 10.8 → 10.9
    "front": (0.0, -_HALF_PILLAR_M),   # y = 0    → -0.1
}


def collect_pillars(markers: list[dict]) -> list[dict]:
    """Group markers into pillars by their (x, y, wall). Each unique
    (x, y, wall) becomes one pillar; markers on it are the high+low
    pair (e.g. arena_config has 8 high + 8 low → 8 pillars)."""
    seen: dict[tuple[float, float, str], list[int]] = {}
    for m in markers:
        key = (float(m["x"]), float(m["y"]), str(m.get("wall", "front")))
        seen.setdefault(key, []).append(int(m["id"]))
    pillars: list[dict] = []
    for (x, y, wall), ids in seen.items():
        dx, dy = WALL_OUTWARD_OFFSET_M.get(wall, (0.0, 0.0))
        pillars.append({
            "x": x + dx,
            "y": y + dy,
            "z": _PILLAR_HEIGHT_M / 2.0,  # centred at half height
            "wall": wall,
            "marker_ids": sorted(ids),
        })
    return pillars


# Net-wall settings. Nets stretch between pillars along each arena
# perimeter so the drone can't fly out. Sphinx's mesh-injection has
# no opacity/transparency support (only BaseColor/Roughness/Specular/
# Metallic) — so we can't make truly see-through walls without
# building a custom UE4 app (Path B) or sparse-grid net geometry.
# Compromise: keep them low (1 m, like a fence) so the camera view
# above them is unobstructed, and use a light colour so they read
# as a barrier rather than an occlusion.
_NET_THICKNESS_M = 0.05      # 5 cm — visually thin, physically solid
_NET_HEIGHT_M = 1.0          # short fence, not a tall wall
_NET_END_OVERLAP_M = 0.30    # extend past the corner pillars so the
                             # joint isn't a visible gap


def collect_nets(markers: list[dict]) -> list[dict]:
    """Return 4 net wall descriptors (in arena coords).

    Each net is a thin vertical slab placed at the arena perimeter
    along one wall. The thickness is in the wall's *outward* axis;
    the long axis runs parallel to the wall.

    Coordinates are in arena metres. The YAML emitter applies the
    ``axis_map`` to translate to UE space and emits the entry with
    a per-axis ``Scale`` so the unit cube becomes a long thin wall.
    """
    if not markers:
        return []
    xs = sorted({float(m["x"]) for m in markers})
    ys = sorted({float(m["y"]) for m in markers})
    x_min, x_max = xs[0], xs[-1]
    y_min, y_max = ys[0], ys[-1]
    width = x_max - x_min
    depth = y_max - y_min
    z_center = _NET_HEIGHT_M / 2.0
    long_x = width + _NET_END_OVERLAP_M
    long_y = depth + _NET_END_OVERLAP_M
    return [
        # Left wall — at arena x=x_min, full depth.
        {"name": "net_left",
         "x": x_min, "y": (y_min + y_max) / 2, "z": z_center,
         "sx": _NET_THICKNESS_M, "sy": long_y, "sz": _NET_HEIGHT_M},
        # Right wall — at arena x=x_max.
        {"name": "net_right",
         "x": x_max, "y": (y_min + y_max) / 2, "z": z_center,
         "sx": _NET_THICKNESS_M, "sy": long_y, "sz": _NET_HEIGHT_M},
        # Front wall — at arena y=y_min, full width.
        {"name": "net_front",
         "x": (x_min + x_max) / 2, "y": y_min, "z": z_center,
         "sx": long_x, "sy": _NET_THICKNESS_M, "sz": _NET_HEIGHT_M},
        # Back wall — at arena y=y_max.
        {"name": "net_back",
         "x": (x_min + x_max) / 2, "y": y_max, "z": z_center,
         "sx": long_x, "sy": _NET_THICKNESS_M, "sz": _NET_HEIGHT_M},
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arena-config", type=Path, default=DEFAULT_ARENA_CONFIG)
    p.add_argument("--target-layout", type=Path, default=DEFAULT_TARGET_LAYOUT)
    p.add_argument("--fbx-dir", type=Path, default=DEFAULT_FBX_DIR,
                   help="Directory holding the FBX files built by build_marker_fbx.py.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_YAML,
                   help="Output Sphinx YAML config file.")
    p.add_argument("--fbx-dir-on-host", type=Path, default=None,
                   help="Path the Sphinx host will see the FBX files at. "
                        "If omitted, the local --fbx-dir is used as-is. Useful "
                        "when generating the YAML on a Mac but Sphinx runs on "
                        "an Ubuntu box where the path is different.")
    p.add_argument(
        "--axis-map", choices=sorted(_AXIS_MAP_PRESETS.keys()),
        default="y2x,x2y_neg,z2z",
        help="Arena→UE axis mapping. Default puts arena +Y along UE +X.",
    )
    p.add_argument(
        "--wall-marker-size-m-override", type=float, default=None,
        help="Override the wall-marker physical size (otherwise read from "
             "arena_config.marker_size_m).",
    )
    p.add_argument(
        "--target-marker-size-m-override", type=float, default=None,
        help="Override the target-sticker physical size (otherwise read from "
             "target_layout.marker_size_m).",
    )
    p.add_argument(
        "--include-floor-box", action="store_true",
        help="Deprecated alias for --emit-floor (retained for back-compat).",
    )
    p.add_argument(
        "--emit-floor", action="store_true",
        help="Emit the arena floor (uses floor.fbx from --fbx-dir).",
    )
    p.add_argument(
        "--emit-pillars", action="store_true",
        help="Emit one pillar per unique marker (x, y, wall) "
             "(uses pillar.fbx from --fbx-dir).",
    )
    p.add_argument(
        "--emit-nets", action="store_true",
        help="Emit 4 net walls along the arena perimeter so drones "
             "can't fly out (uses pillar.fbx scaled into thin slabs).",
    )
    p.add_argument(
        "--emit-arena-static", action="store_true",
        help="Emit a single arena_static.fbx entry instead of separate "
             "floor/walls/pillars. The combined FBX (built by "
             "build_full_arena.py) already has all dimensions and "
             "positions baked, so the YAML places it at world origin "
             "with Scale=1 and Rotation=0. This is the recommended "
             "path — eliminates the per-mesh scale-and-rotate quirks "
             "that were producing tilted walls and lost-material "
             "floors.",
    )
    p.add_argument(
        "--emit-paintings", action="store_true",
        help="Emit 4 wall paintings — one per wall — referencing "
             "painting_<name>.fbx files in --fbx-dir. Run "
             "build_painting_fbxs.py first to produce those.",
    )
    p.add_argument(
        "--floor-size-m", type=str, default="50x50",
        help="Floor footprint as <width>x<depth> in metres. Default 50x50, "
             "which completely covers the arena and hides UE4's default "
             "ground plane (which otherwise shows through as a checker).",
    )
    p.add_argument(
        "--center-arena", action="store_true", default=True,
        help="Shift arena geometry so its centre lies at UE world origin "
             "(0,0,0). Sphinx spawns the drone at world origin, so this "
             "puts the drone inside the arena instead of having to fly in "
             "from outside. Default: ON.",
    )
    p.add_argument(
        "--no-center-arena", dest="center_arena", action="store_false",
        help="Disable the arena recentering (geometry stays at the literal "
             "arena_config.json positions in UE coords).",
    )
    args = p.parse_args()

    axis_map = _AXIS_MAP_PRESETS[args.axis_map]

    with args.arena_config.open() as f:
        arena = json.load(f)
    with args.target_layout.open() as f:
        targets = json.load(f)

    wall_size = float(args.wall_marker_size_m_override
                      or arena.get("marker_size_m", 0.5))
    target_size = float(args.target_marker_size_m_override
                        or targets.get("marker_size_m", 0.19))

    fbx_dir = args.fbx_dir_on_host or args.fbx_dir.resolve()

    # Arena recentering offset (in arena coords). Sphinx spawns the
    # drone at UE world origin; we want the drone to start *inside*
    # the arena, not at one corner. Compute the centre of the marker
    # bounding box in arena XY and use that as the offset (z stays
    # 0 — the arena floor is at z=0 and we want the drone to lift
    # off from the floor).
    if args.center_arena and arena.get("markers"):
        xs = [float(m["x"]) for m in arena["markers"]]
        ys = [float(m["y"]) for m in arena["markers"]]
        center_offset = (
            (min(xs) + max(xs)) / 2.0,
            (min(ys) + max(ys)) / 2.0,
            0.0,
        )
    else:
        center_offset = (0.0, 0.0, 0.0)

    def shifted_apply(arena_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        """Apply --center-arena shift then axis_map."""
        ox, oy, oz = center_offset
        return axis_map.apply(
            (arena_xyz[0] - ox, arena_xyz[1] - oy, arena_xyz[2] - oz)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_lines: list[str] = []
    out_lines.append("# Auto-generated by tools/sphinx-arena/arena_to_sphinx_yaml.py")
    out_lines.append("# Do not edit by hand — re-run with `make yaml` after changing")
    out_lines.append("# arena_config.json or default_target_layout.json.")
    out_lines.append(f"# Axis map: {args.axis_map}  (arena → UE)")
    out_lines.append(f"# Center arena at UE origin: {args.center_arena} "
                     f"(offset arena {center_offset})")
    out_lines.append(f"# Wall marker size: {wall_size} m")
    out_lines.append(f"# Target marker size: {target_size} m")
    out_lines.append("")
    out_lines.append("Meshes:")

    n_wall = 0
    for m in arena.get("markers", []):
        mid = int(m["id"])
        wall = str(m.get("wall", "front"))
        yaw = WALL_YAW_DEG.get(wall, 0.0)
        # arena → UE conversion (metres) → centimetres
        ux_m, uy_m, uz_m = shifted_apply((float(m["x"]), float(m["y"]), float(m["z"])))
        loc_cm = (ux_m * 100.0, uy_m * 100.0, uz_m * 100.0)
        # Pitch stands the flat-plane FBX upright; yaw aims it at the
        # arena interior. UE4 Rotator order is (Pitch, Yaw, Roll).
        rot = (MARKER_PITCH_DEG, yaw, 0.0)
        fbx_path = fbx_dir / f"aruco_{mid:03d}.fbx"
        out_lines.append(emit_yaml_block(
            name=f"wall_{wall}_id_{mid:03d}_{m.get('label','').lower().replace(' ', '_') or 'marker'}",
            fbx_path=fbx_path,
            location_cm=loc_cm,
            rotation_deg=rot,
            scale=wall_size,
            snap_to_ground=False,
        ))
        out_lines.append("")
        n_wall += 1

    n_target = 0
    for b in targets.get("boxes", []):
        if not b.get("enabled", True):
            continue
        mid = int(b["id"])
        ux_m, uy_m, uz_m = shifted_apply((float(b["x"]), float(b["y"]), float(b["z"])))
        loc_cm = (ux_m * 100.0, uy_m * 100.0, uz_m * 100.0)
        # Target stickers face UP (mounted on top of the box). The
        # default flat-plane FBX (normal +Z) already points up, so no
        # pitch — only the box's yaw offset. UE4 Rotator order is
        # (Pitch, Yaw, Roll).
        rot = (0.0, float(b.get("yaw_deg", 0.0)), 0.0)
        fbx_path = fbx_dir / f"aruco_{mid:03d}.fbx"
        out_lines.append(emit_yaml_block(
            name=f"target_{b.get('team','tgt')}_id_{mid:03d}",
            fbx_path=fbx_path,
            location_cm=loc_cm,
            rotation_deg=rot,
            scale=target_size,
            snap_to_ground=False,
        ))
        out_lines.append("")
        n_target += 1

    n_floor = 0
    n_pillars = 0
    n_nets = 0
    n_arena_static = 0
    n_paintings = 0
    emit_floor = args.emit_floor or args.include_floor_box
    emit_pillars = args.emit_pillars
    emit_nets = args.emit_nets
    emit_arena_static = args.emit_arena_static

    if emit_arena_static:
        # Combined arena: floor + 4 walls + 8 pillars in one FBX with
        # everything baked in Blender vertex space. Place at world
        # origin with Scale=1 and Rotation=0; the FBX already encodes
        # the right positions.
        out_lines.append(
            "  # Combined static arena — floor + 4 walls + 8 pillars."
        )
        out_lines.append(emit_yaml_block(
            name="arena_static",
            fbx_path=fbx_dir / "arena_static.fbx",
            location_cm=(0.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            snap_to_ground=False,
        ))
        out_lines.append("")
        n_arena_static = 1
        # Skip the per-mesh emits since the combined FBX has them.
        emit_floor = False
        emit_pillars = False
        emit_nets = False

    if args.emit_paintings:
        # One painting per wall, mounted at mid-height between the low
        # (z=2) and high (z=4) marker rings so the logos don't occlude
        # either marker. Position is the wall's CENTRE in arena coords;
        # rotation matches the wall's marker yaw so the painting faces
        # arena interior.
        markers_for_bbox = arena.get("markers", [])
        if markers_for_bbox:
            xs = [float(m["x"]) for m in markers_for_bbox]
            ys = [float(m["y"]) for m in markers_for_bbox]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            cx_arena = (x_min + x_max) / 2.0
            cy_arena = (y_min + y_max) / 2.0
            # For each wall, the painting is offset along the wall's
            # long axis (so it doesn't overlap the centre marker on
            # long walls or the lone marker pair on short walls).
            wall_centres = {
                # name → (centre_x, centre_y, long_axis_is_x?)
                "left":  (x_min, cy_arena, False),  # runs along Y
                "right": (x_max, cy_arena, False),
                "front": (cx_arena, y_min, True),   # runs along X
                "back":  (cx_arena, y_max, True),
            }
            pw, ph = _PAINTING_SIZE_M
            for logo_name, wall, offset in PAINTINGS:
                cwx, cwy, long_is_x = wall_centres[wall]
                # Apply offset along the wall's long axis
                if long_is_x:
                    wx, wy = cwx + offset, cwy
                else:
                    wx, wy = cwx, cwy + offset
                ux_m, uy_m, uz_m = shifted_apply((wx, wy, _PAINTING_Z_M))
                loc_cm = (ux_m * 100.0, uy_m * 100.0, uz_m * 100.0)
                rot = (MARKER_PITCH_DEG, WALL_YAW_DEG[wall], 0.0)
                out_lines.append(
                    f"  # Painting — {logo_name} on {wall} wall"
                )
                out_lines.append(emit_yaml_block(
                    name=f"painting_{wall}_{logo_name}",
                    fbx_path=fbx_dir / f"painting_{logo_name}.fbx",
                    location_cm=loc_cm,
                    rotation_deg=rot,
                    scale=(pw, ph, 1.0),
                    snap_to_ground=False,
                ))
                out_lines.append("")
                n_paintings += 1
    if emit_nets:
        # Net walls along the perimeter. Reuses pillar.fbx (the unit
        # cube) and scales it into a thin tall slab per wall. Position
        # is at the wall's marker plane in arena coords; the thickness
        # is in the outward-facing axis so the slab "sits" in front of
        # the markers from outside the arena.
        ax_for_ue_x = axis_map.ue_x[0]
        ax_for_ue_y = axis_map.ue_y[0]
        for net in collect_nets(arena.get("markers", [])):
            ux_m, uy_m, uz_m = shifted_apply((net["x"], net["y"], net["z"]))
            loc_cm = (ux_m * 100.0, uy_m * 100.0, uz_m * 100.0)
            arena_scales = {0: net["sx"], 1: net["sy"], 2: net["sz"]}
            ue_x_scale = arena_scales[ax_for_ue_x]
            ue_y_scale = arena_scales[ax_for_ue_y]
            ue_z_scale = arena_scales[2]
            out_lines.append(
                f"  # Perimeter net — {net['name']}, "
                f"keeps the drone inside the arena."
            )
            out_lines.append(emit_yaml_block(
                name=net["name"],
                fbx_path=fbx_dir / "net.fbx",
                location_cm=loc_cm,
                rotation_deg=(0.0, 0.0, 0.0),
                scale=(ue_x_scale, ue_y_scale, ue_z_scale),
                snap_to_ground=False,
            ))
            out_lines.append("")
            n_nets += 1

    if emit_pillars:
        # Place 8 vertical pillars at the unique (x, y, wall) positions
        # from arena_config.json, pushed outward by half the pillar
        # width so the marker plane sits on the pillar's inward face at
        # exactly the (x, y) recorded in arena_config.json. ctrl_position
        # uses those coordinates as ground truth — never offset them.
        for pillar in collect_pillars(arena.get("markers", [])):
            ux_m, uy_m, uz_m = shifted_apply(
                (pillar["x"], pillar["y"], pillar["z"])
            )
            loc_cm = (ux_m * 100.0, uy_m * 100.0, uz_m * 100.0)
            ids_label = "_".join(f"{i:02d}" for i in pillar["marker_ids"])
            out_lines.append("  # Pillar carrying markers " + ", ".join(
                str(i) for i in pillar["marker_ids"]) + f" ({pillar['wall']} wall).")
            out_lines.append(emit_yaml_block(
                name=f"pillar_{pillar['wall']}_{ids_label}",
                fbx_path=fbx_dir / "pillar.fbx",
                location_cm=loc_cm,
                rotation_deg=(0.0, 0.0, 0.0),
                scale=(_PILLAR_WIDTH_M, _PILLAR_WIDTH_M, _PILLAR_HEIGHT_M),
                snap_to_ground=False,
            ))
            out_lines.append("")
            n_pillars += 1

    if emit_floor:
        # Floor: a unit cube scaled to a generous footprint (default
        # 50×50 m, 1 m thick). Centre it at the arena's marker bbox
        # centre so it covers symmetrically. Built oversized on
        # purpose: in earlier iterations the floor exactly matched
        # the arena footprint (22×12 m) and UE4's default ground
        # showed through as a debug checker on the edges. A 50×50 m
        # floor is impossible to miss and physically hides whatever
        # is below.
        try:
            fw_str, fd_str = args.floor_size_m.lower().split("x", 1)
            floor_w = float(fw_str)
            floor_d = float(fd_str)
        except ValueError:
            print(f"warning: --floor-size-m {args.floor_size_m!r} unparseable; "
                  f"using 50x50", file=sys.stderr)
            floor_w, floor_d = 50.0, 50.0
        floor_thickness = 1.0
        # Floor centre: top of slab at z=0, so centre is at z=-0.5.
        # The drone spawns at z=0 (UE world origin). Putting the
        # slab centre at z=-0.5 means the slab's top face is exactly
        # at z=0 — drone lifts off cleanly without intersecting the
        # floor mesh.
        ys = [float(m["y"]) for m in arena.get("markers", [])] or [0.0, 10.8]
        cx_arena, cy_arena, cz_arena = 0.0, (min(ys) + max(ys)) / 2.0, -floor_thickness / 2.0
        ux_m, uy_m, uz_m = shifted_apply((cx_arena, cy_arena, cz_arena))
        loc_cm = (ux_m * 100.0, uy_m * 100.0, uz_m * 100.0)
        # When the axis_map swaps X and Y, the floor "width" along
        # arena +X may end up along UE Y (and vice versa). For uniform
        # rectangles the practical answer is to scale the FBX
        # symmetrically around the swap by checking which arena axis
        # ends up where. Simplest correct version: figure out which
        # UE axis maps from arena X vs Y and apply the right side.
        ax_for_ue_x = axis_map.ue_x[0]
        ax_for_ue_y = axis_map.ue_y[0]
        size_per_arena_axis = {0: floor_w, 1: floor_d}
        ue_x_size = size_per_arena_axis[ax_for_ue_x]
        ue_y_size = size_per_arena_axis[ax_for_ue_y]
        out_lines.append(
            f"  # Arena floor — {floor_w} × {floor_d} × {floor_thickness} m. "
            f"Top face at z=0; oversized so UE4's default ground plane "
            f"can't show through at the edges."
        )
        out_lines.append(emit_yaml_block(
            name="arena_floor",
            fbx_path=fbx_dir / "floor.fbx",
            location_cm=loc_cm,
            rotation_deg=(0.0, 0.0, 0.0),
            scale=(ue_x_size, ue_y_size, floor_thickness),
            snap_to_ground=False,
        ))
        out_lines.append("")
        n_floor = 1

    # Legacy include-floor-box-only branch retained for back-compat.
    # Skipped because emit_floor above already handled it.
    if False:
        out_lines.append("  # Optional 20m × 10m floor placeholder. Replace with a")
        out_lines.append("  # real concrete-textured FBX once you have one.")
        out_lines.append(emit_yaml_block(
            name="floor_placeholder",
            fbx_path=fbx_dir / "floor_20x10.fbx",
            location_cm=(0.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            snap_to_ground=False,
        ))
        out_lines.append("")

    args.out.write_text("\n".join(out_lines))
    print(f"Wrote {args.out}")
    print(f"  wall markers: {n_wall}")
    print(f"  target markers: {n_target}")
    print(f"  pillars: {n_pillars}")
    print(f"  nets: {n_nets}")
    print(f"  floor: {n_floor}")
    print(f"  arena_static: {n_arena_static}")
    print(f"  paintings: {n_paintings}")
    print(f"  axis map: {args.axis_map}")
    print(f"  fbx dir (in YAML): {fbx_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
