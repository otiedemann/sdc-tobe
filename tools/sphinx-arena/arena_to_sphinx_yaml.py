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


def emit_yaml_block(name: str, fbx_path: Path, location_cm: tuple[float, float, float],
                    rotation_deg: tuple[float, float, float], scale: float,
                    snap_to_ground: bool = False) -> str:
    """Render one Sphinx ``Meshes:`` entry in the documented YAML form."""
    lines = [
        f"  - Name: {name}",
        f"    FbxPath: \"{fbx_path}\"",
        f"    Location: \"{fmt3(*location_cm)}\"",
        f"    Rotation: \"{fmt3(*rotation_deg)}\"",
        f"    Scale: \"{fmt3(scale, scale, scale)}\"",
        f"    SnapToGround: {'true' if snap_to_ground else 'false'}",
    ]
    return "\n".join(lines)


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
        help="Also emit a 20m × 10m floor plane at z=0 (placeholder asset).",
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

    if args.include_floor_box:
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
    print(f"  axis map: {args.axis_map}")
    print(f"  fbx dir (in YAML): {fbx_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
