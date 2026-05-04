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


# Rotations applied to each wall so the FBX faces inward.
# Sphinx ``Rotation`` is "Roll Pitch Yaw" in degrees per Parrot's docs;
# we leave roll/pitch at 0 and only spin around Z (yaw).
WALL_YAW_DEG: dict[str, float] = {
    "front": 0.0,
    "back": 180.0,
    "left": 90.0,
    "right": -90.0,
}


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
        "--floor-size-m", type=str, default="22x12",
        help="Floor footprint as <width>x<depth> in metres. Default 22x12.",
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_lines: list[str] = []
    out_lines.append("# Auto-generated by tools/sphinx-arena/arena_to_sphinx_yaml.py")
    out_lines.append("# Do not edit by hand — re-run with `make yaml` after changing")
    out_lines.append("# arena_config.json or default_target_layout.json.")
    out_lines.append(f"# Axis map: {args.axis_map}  (arena → UE)")
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
        ux_m, uy_m, uz_m = axis_map.apply((float(m["x"]), float(m["y"]), float(m["z"])))
        loc_cm = (ux_m * 100.0, uy_m * 100.0, uz_m * 100.0)
        rot = (0.0, 0.0, yaw)
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
        ux_m, uy_m, uz_m = axis_map.apply((float(b["x"]), float(b["y"]), float(b["z"])))
        loc_cm = (ux_m * 100.0, uy_m * 100.0, uz_m * 100.0)
        rot = (0.0, 0.0, float(b.get("yaw_deg", 0.0)))
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
    emit_floor = args.emit_floor or args.include_floor_box
    emit_pillars = args.emit_pillars
    if emit_pillars:
        # Place 8 vertical pillars at the unique (x, y, wall) positions
        # from arena_config.json, pushed outward by half the pillar
        # width so the marker plane sits on the pillar's inward face at
        # exactly the (x, y) recorded in arena_config.json. ctrl_position
        # uses those coordinates as ground truth — never offset them.
        for pillar in collect_pillars(arena.get("markers", [])):
            ux_m, uy_m, uz_m = axis_map.apply(
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
        # Floor: a unit FBX plane scaled to the configured footprint.
        # Centre it at the arena's marker bbox centre so it covers
        # symmetrically (markers span x ∈ [-10, +10], y ∈ [0, 10.8]).
        try:
            fw_str, fd_str = args.floor_size_m.lower().split("x", 1)
            floor_w = float(fw_str)
            floor_d = float(fd_str)
        except ValueError:
            print(f"warning: --floor-size-m {args.floor_size_m!r} unparseable; "
                  f"using 22x12", file=sys.stderr)
            floor_w, floor_d = 22.0, 12.0
        # Floor centre in arena frame: x=0, y=mean of y range, z=0.
        ys = [float(m["y"]) for m in arena.get("markers", [])] or [0.0, 10.8]
        cx_arena, cy_arena, cz_arena = 0.0, (min(ys) + max(ys)) / 2.0, 0.0
        ux_m, uy_m, uz_m = axis_map.apply((cx_arena, cy_arena, cz_arena))
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
            f"  # Arena floor — {floor_w} × {floor_d} m at z=0."
        )
        out_lines.append(emit_yaml_block(
            name="arena_floor",
            fbx_path=fbx_dir / "floor.fbx",
            location_cm=loc_cm,
            rotation_deg=(0.0, 0.0, 0.0),
            scale=(ue_x_size, ue_y_size, 1.0),
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
    print(f"  floor: {n_floor}")
    print(f"  axis map: {args.axis_map}")
    print(f"  fbx dir (in YAML): {fbx_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
