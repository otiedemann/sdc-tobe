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
# Marker pitch is ZERO — the marker FBX itself bakes a 90° X-rotation
# in Blender so the plane is already vertical with normal +X in UE
# (after the Sphinx -Y/-Z axis conversion). YAML only needs YAW to
# spin the +X normal around UE +Z to face the desired direction.
MARKER_PITCH_DEG: float = 0.0

# Per-marker direction is now read directly from arena_config.json's
# new ``direction_deg`` field (in arena coords). The mapping below is
# only a fallback for old configs missing direction_deg, and uses the
# wall name to derive a sensible default.  Strongly preferred: set
# direction_deg explicitly on every marker in arena_config.json.
WALL_FALLBACK_DIRECTION_DEG: dict[str, float] = {
    "left":   0.0,    # x=-10, inward = arena +X (0° in arena polar)
    "right":  180.0,  # x=+10, inward = arena -X
    "front":  90.0,   # y=0,   inward = arena +Y
    "back":  -90.0,   # y=max, inward = arena -Y
}


def arena_dir_to_ue_yaw(direction_deg_arena: float) -> float:
    """Convert a marker's facing direction (in arena coord polar) to a
    UE4 FRotator yaw value, given the default axis_map y2x,x2y_neg.

    Derivation: an arena unit vector at angle θ (measured CCW from
    arena +X in the arena XY plane) is (cos θ, sin θ, 0). Under
    axis_map y2x,x2y_neg this maps to UE (sin θ, -cos θ, 0). Our
    marker FBX exports with normal at UE +X (= (1, 0, 0)) post-import.
    UE positive yaw rotates +X clockwise viewed from above:
       yaw α takes +X to (cos α, sin α).
    Setting (cos α, sin α) = (sin θ, -cos θ) gives α = θ - 90°.

    So: ue_yaw = arena_direction_deg - 90°.
    """
    return direction_deg_arena - 90.0


# Legacy alias kept so older code that imports WALL_YAW_DEG still works.
# Computed from WALL_FALLBACK_DIRECTION_DEG via the same conversion.
WALL_YAW_DEG: dict[str, float] = {
    wall: arena_dir_to_ue_yaw(d)
    for wall, d in WALL_FALLBACK_DIRECTION_DEG.items()
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


# Wall settings — visible solid walls (emitted via wall.fbx scaled
# per-axis). Walls are pushed OUTWARD of the pillars by
# (_PILLAR_WIDTH_M + _WALL_THICKNESS_M/2) so they sit BEHIND the
# pillars (relative to the arena interior) instead of being co-planar
# with the markers — otherwise the wall surface obscures the markers
# from inside the arena.
_WALL_THICKNESS_M = 0.20
_WALL_HEIGHT_M = 4.0
_WALL_END_OVERLAP_M = 0.40   # extend past corners so corner joints don't gap
_WALL_OUTWARD_OFFSET_M = _PILLAR_WIDTH_M + _WALL_THICKNESS_M / 2.0


def collect_walls(markers: list[dict],
                  thickness_m: float = _WALL_THICKNESS_M,
                  height_m: float = _WALL_HEIGHT_M,
                  overlap_m: float = _WALL_END_OVERLAP_M,
                  outward_offset_m: float = _WALL_OUTWARD_OFFSET_M) -> list[dict]:
    """Return 4 wall descriptors (in arena coords) for the perimeter.

    Each wall is a tall thin slab placed OUTWARD of the corresponding
    pillar centerline by ``outward_offset_m`` so it doesn't visually
    overlap with the markers (which sit on the pillars' inward faces).
    Coordinates are in arena metres. The YAML emitter applies the
    ``axis_map`` to translate to UE space and emits the entry with
    a per-axis ``Scale`` so the unit-cube wall.fbx becomes a long
    thin wall at the right size.
    """
    if not markers:
        return []
    xs = sorted({float(m["x"]) for m in markers})
    ys = sorted({float(m["y"]) for m in markers})
    x_min, x_max = xs[0], xs[-1]
    y_min, y_max = ys[0], ys[-1]
    width = x_max - x_min
    depth = y_max - y_min
    z_center = height_m / 2.0
    long_x = width + overlap_m
    long_y = depth + overlap_m
    o = outward_offset_m
    return [
        {"name": "wall_left",
         "x": x_min - o, "y": (y_min + y_max) / 2, "z": z_center,
         "sx": thickness_m, "sy": long_y, "sz": height_m},
        {"name": "wall_right",
         "x": x_max + o, "y": (y_min + y_max) / 2, "z": z_center,
         "sx": thickness_m, "sy": long_y, "sz": height_m},
        {"name": "wall_front",
         "x": (x_min + x_max) / 2, "y": y_min - o, "z": z_center,
         "sx": long_x, "sy": thickness_m, "sz": height_m},
        {"name": "wall_back",
         "x": (x_min + x_max) / 2, "y": y_max + o, "z": z_center,
         "sx": long_x, "sy": thickness_m, "sz": height_m},
    ]


# Backward-compat alias — older code paths used "nets" terminology.
collect_nets = collect_walls


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
        "--emit-walls", action="store_true",
        help="Emit 4 perimeter walls along the arena edges (uses "
             "wall.fbx scaled into long thin slabs per side).",
    )
    p.add_argument(
        "--emit-nets", action="store_true",
        help="Deprecated alias for --emit-walls (kept for back-compat).",
    )
    p.add_argument(
        "--emit-arena-static", action="store_true",
        help="Emit a single arena_static.fbx (combined floor+walls+"
             "pillars baked in vertex space). Sphinx mesh-injection "
             "doesn't render multi-mesh + multi-material FBXs "
             "reliably — the per-element --emit-floor/--emit-walls/"
             "--emit-pillars path is preferred and is what the "
             "Makefile uses by default.",
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
    # Per-wall yaw overrides for empirical iteration.  The math-derived
    # defaults are 0/180/-90/+90 (face inward) but axis conventions
    # between Blender, FBX, and Sphinx's mesh-injection have produced
    # repeated off-by-90° and off-by-180° issues; these flags let the
    # operator iterate in seconds without rebuilding any FBXs:
    #
    #   make yaml YAML_EXTRA="--yaw-front 180 --yaw-back 0 --yaw-left 90 --yaw-right -90"
    #
    p.add_argument("--yaw-front", type=float, default=None,
                   help="Override YAML yaw for 'front'-wall markers (deg).")
    p.add_argument("--yaw-back", type=float, default=None,
                   help="Override YAML yaw for 'back'-wall markers (deg).")
    p.add_argument("--yaw-left", type=float, default=None,
                   help="Override YAML yaw for 'left'-wall markers (deg).")
    p.add_argument("--yaw-right", type=float, default=None,
                   help="Override YAML yaw for 'right'-wall markers (deg).")
    p.add_argument("--marker-pitch", type=float, default=None,
                   help="Override MARKER_PITCH_DEG for wall markers. "
                        "Default 0 (FBX is already vertical).")
    p.add_argument(
        "--no-center-arena", dest="center_arena", action="store_false",
        help="Disable the arena recentering (geometry stays at the literal "
             "arena_config.json positions in UE coords).",
    )
    args = p.parse_args()

    axis_map = _AXIS_MAP_PRESETS[args.axis_map]

    # Apply CLI yaw overrides on top of the module-level defaults.
    wall_yaws = dict(WALL_YAW_DEG)
    if args.yaw_front is not None:
        wall_yaws["front"] = args.yaw_front
    if args.yaw_back is not None:
        wall_yaws["back"] = args.yaw_back
    if args.yaw_left is not None:
        wall_yaws["left"] = args.yaw_left
    if args.yaw_right is not None:
        wall_yaws["right"] = args.yaw_right
    marker_pitch = (args.marker_pitch if args.marker_pitch is not None
                    else MARKER_PITCH_DEG)

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
    cli_overrides = {
        "front": args.yaw_front,
        "back":  args.yaw_back,
        "left":  args.yaw_left,
        "right": args.yaw_right,
    }
    for m in arena.get("markers", []):
        mid = int(m["id"])
        wall = str(m.get("wall", "front"))
        # Yaw resolution priority:
        #   1. CLI --yaw-<wall> flag (for empirical iteration)
        #   2. Per-marker direction_deg from arena_config.json
        #      (the cleanest source of truth — explicitly states which
        #       way the marker faces in arena coords)
        #   3. WALL_FALLBACK_DIRECTION_DEG keyed by wall name
        if cli_overrides.get(wall) is not None:
            yaw = cli_overrides[wall]
        elif "direction_deg" in m:
            yaw = arena_dir_to_ue_yaw(float(m["direction_deg"]))
        else:
            yaw = wall_yaws.get(wall, 0.0)
        # arena → UE conversion (metres) → centimetres
        ux_m, uy_m, uz_m = shifted_apply((float(m["x"]), float(m["y"]), float(m["z"])))
        loc_cm = (ux_m * 100.0, uy_m * 100.0, uz_m * 100.0)
        # Marker FBX is already vertical (+X normal in UE) per
        # build_marker_fbx.py; yaw spins it around UE +Z to face the
        # direction specified by m["direction_deg"] (or the per-wall
        # fallback). UE4 Rotator order is (Pitch, Yaw, Roll).
        rot = (marker_pitch, yaw, 0.0)
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
        # Target markers are mounted on a 45°-tilted plate, 1 m above
        # the floor, facing upward and toward the arena centre. The
        # default flat-plane FBX has normal +Z; UE4 pitch=-45° rotates
        # that to (+0.707, 0, +0.707) — facing UE +X tilted up — and
        # yaw spins the horizontal component to whichever direction
        # the per-target JSON requests. UE4 Rotator order is
        # (Pitch, Yaw, Roll). pitch_deg defaults to 0 (flat) so old
        # box-top stickers still work without modification.
        pitch = float(b.get("pitch_deg", 0.0))
        yaw = float(b.get("yaw_deg", 0.0))
        rot = (pitch, yaw, 0.0)
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
    n_walls = 0
    n_arena_static = 0
    n_paintings = 0
    emit_floor = args.emit_floor or args.include_floor_box
    emit_pillars = args.emit_pillars
    emit_walls = args.emit_walls or args.emit_nets  # nets is the legacy alias
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
        emit_walls = False

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
                rot = (marker_pitch, wall_yaws[wall], 0.0)
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
    if emit_walls:
        # 4 perimeter walls. Reuses wall.fbx (a unit cube — the
        # single-mesh-single-material approach Sphinx renders
        # reliably) and scales it per-axis into a long thin tall slab.
        # Position is at each wall's plane in arena coords.
        ax_for_ue_x = axis_map.ue_x[0]
        ax_for_ue_y = axis_map.ue_y[0]
        for wall in collect_walls(arena.get("markers", [])):
            ux_m, uy_m, uz_m = shifted_apply((wall["x"], wall["y"], wall["z"]))
            loc_cm = (ux_m * 100.0, uy_m * 100.0, uz_m * 100.0)
            arena_scales = {0: wall["sx"], 1: wall["sy"], 2: wall["sz"]}
            ue_x_scale = arena_scales[ax_for_ue_x]
            ue_y_scale = arena_scales[ax_for_ue_y]
            ue_z_scale = arena_scales[2]
            out_lines.append(
                f"  # Perimeter wall — {wall['name']}."
            )
            out_lines.append(emit_yaml_block(
                name=wall["name"],
                fbx_path=fbx_dir / "wall.fbx",
                location_cm=loc_cm,
                rotation_deg=(0.0, 0.0, 0.0),
                scale=(ue_x_scale, ue_y_scale, ue_z_scale),
                snap_to_ground=False,
            ))
            out_lines.append("")
            n_walls += 1

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
    print(f"  walls: {n_walls}")
    print(f"  floor: {n_floor}")
    print(f"  arena_static: {n_arena_static}")
    print(f"  paintings: {n_paintings}")
    print(f"  axis map: {args.axis_map}")
    print(f"  fbx dir (in YAML): {fbx_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
