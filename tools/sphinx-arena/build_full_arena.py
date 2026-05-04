#!/usr/bin/env python3
"""Build the entire static arena (floor + 4 walls + 8 pillars) as a
SINGLE FBX with all dimensions and positions baked in Blender vertex
space.

The Sphinx YAML places this at UE world origin with ``Scale: "1 1 1"``
and ``Rotation: "0 0 0"`` — no per-mesh transform math. This is the
fix for the persistent issue where unit cubes scaled non-uniformly in
YAML (long thin walls especially) imported into UE4 with garbled
orientation.

Coordinates: positions and dimensions are baked in **UE-final
coordinates** (after axis_map + center-arena shift). The transform
matches what ``arena_to_sphinx_yaml.py`` does for markers, so the
combined arena FBX and the per-marker FBXs end up in the same world
frame.

Usage:
    blender --background --python build_full_arena.py -- \\
        --arena-config /path/to/arena_config.json \\
        --out /path/to/out/fbx/arena_static.fbx
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

try:
    import bpy  # type: ignore
except ModuleNotFoundError:
    bpy = None


# ─── helpers ─────────────────────────────────────────────────────


def _slice_blender_argv() -> list[str]:
    try:
        idx = sys.argv.index("--")
    except ValueError:
        return []
    return sys.argv[idx + 1:]


def _clear_scene() -> None:
    assert bpy is not None
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def _write_png_solid(path: Path, color_rgb: tuple[int, int, int],
                     side_px: int = 256) -> None:
    """Write a uniform-colour PNG without a Pillow dependency."""
    r, g, b = color_rgb
    ihdr = struct.pack(">IIBBBBB", side_px, side_px, 8, 2, 0, 0, 0)
    row = bytes([0]) + bytes([r, g, b]) * side_px
    raw = row * side_px
    idat = zlib.compress(raw, 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    blob = (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", idat)
            + chunk(b"IEND", b""))
    path.write_bytes(blob)


def _make_textured_material(name: str, color_rgb: tuple[int, int, int],
                             roughness: float, out_dir: Path):
    """Principled BSDF with a 256×256 PNG sidecar as BaseColor.

    Uses the same texture-driven approach the markers use (proven to
    survive Blender → FBX → UE4 import). Vector ``Base Color`` did
    not survive — UE4 fell back to its debug-checker default for
    every mesh that used it."""
    assert bpy is not None
    png_path = out_dir / f"{name}.png"
    _write_png_solid(png_path, color_rgb)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Specular"].default_value = 0.2

    tex_node = nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(str(png_path))
    for cs in ("Non-Color", "Linear", "Raw"):
        try:
            img.colorspace_settings.name = cs
            break
        except TypeError:
            continue
    tex_node.image = img
    tex_node.interpolation = "Closest"
    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _add_box(name: str, location_m: tuple[float, float, float],
             dimensions_m: tuple[float, float, float], material) -> None:
    """Add a cube at the given UE-frame position and dimensions, then
    bake the scale into the mesh data via transform_apply (so the
    object's scale is reset to 1, 1, 1 and the dimensions live in the
    vertex positions)."""
    assert bpy is not None
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location_m)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = dimensions_m
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)


# ─── main ─────────────────────────────────────────────────────────


def main() -> int:
    if bpy is None:
        print(
            "ERROR: must run inside Blender:\n"
            "  blender --background --python build_full_arena.py -- ...",
            file=sys.stderr,
        )
        return 2

    args = _slice_blender_argv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arena-config", type=Path, required=True,
                   help="Path to arena_config.json")
    p.add_argument("--out", type=Path, required=True,
                   help="Output FBX path (e.g. out/fbx/arena_static.fbx)")
    p.add_argument("--floor-size-m", type=float, default=50.0,
                   help="Floor side length (square). Default 50.0 m to "
                        "hide UE4's default ground.")
    p.add_argument("--floor-thickness-m", type=float, default=1.0)
    p.add_argument("--floor-top-z-m", type=float, default=0.1,
                   help="Z of the floor's top face. Default 0.1 m so it "
                        "sits clearly above UE4's default ground (which "
                        "is at z=0 and otherwise z-fights, leaving the "
                        "checker pattern visible through ours).")
    p.add_argument("--wall-thickness-m", type=float, default=0.10)
    p.add_argument("--wall-height-m", type=float, default=4.0,
                   help="Walls are tall enough to clearly enclose the "
                        "arena (4 m default). Sphinx YAML can't honour "
                        "real opacity, so we accept solid walls.")
    p.add_argument("--pillar-thickness-m", type=float, default=0.40)
    p.add_argument("--pillar-height-m", type=float, default=6.0)
    p.add_argument("--floor-color-rgb", type=str, default="180,180,180")
    p.add_argument("--wall-color-rgb", type=str, default="220,225,235")
    p.add_argument("--pillar-color-rgb", type=str, default="46,46,52")
    ns = p.parse_args(args)

    def parse_rgb(s: str) -> tuple[int, int, int]:
        parts = [int(x) for x in s.split(",")]
        assert len(parts) == 3
        return tuple(parts)  # type: ignore

    with open(ns.arena_config) as f:
        arena = json.load(f)

    markers = arena.get("markers", [])
    if not markers:
        raise RuntimeError("arena_config has no markers; nothing to build")

    xs = [float(m["x"]) for m in markers]
    ys = [float(m["y"]) for m in markers]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    arena_w = x_max - x_min
    arena_d = y_max - y_min

    # axis_map y2x,x2y_neg,z2z combined with center-on-origin shift:
    #   arena (x, y, z) → UE (y - cy, cx - x, z)
    def arena_to_ue(p: tuple[float, float, float]) -> tuple[float, float, float]:
        ax, ay, az = p
        return (ay - cy, cx - ax, az)

    out_dir = ns.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    _clear_scene()

    # Materials
    mat_floor = _make_textured_material(
        "mat_floor", parse_rgb(ns.floor_color_rgb), 0.85, out_dir)
    mat_wall = _make_textured_material(
        "mat_wall", parse_rgb(ns.wall_color_rgb), 0.4, out_dir)
    mat_pillar = _make_textured_material(
        "mat_pillar", parse_rgb(ns.pillar_color_rgb), 0.6, out_dir)

    # ── FLOOR ──
    # Centered at UE origin; top face at the configured z (default 0.1m
    # to clear UE4's default ground), bottom at top - thickness.
    # Oversized (50×50 m default) so UE4's default ground plane is
    # completely hidden underneath.
    floor_z_center = ns.floor_top_z_m - ns.floor_thickness_m / 2.0
    _add_box(
        name="arena_floor",
        location_m=(0.0, 0.0, floor_z_center),
        dimensions_m=(ns.floor_size_m, ns.floor_size_m, ns.floor_thickness_m),
        material=mat_floor,
    )

    # ── WALLS ── 4 perimeter walls, low fence height by default.
    # Dimensions are in arena coords; converted to UE via axis_map
    # (X and Y swap, sign on Y irrelevant for non-negative scale).
    wall_z_center = ns.wall_height_m / 2.0
    overlap = 0.4

    walls = [
        # (name, arena centre x, arena centre y, arena dimensions sx,sy,sz)
        ("wall_left",  x_min, cy, ns.wall_thickness_m, arena_d + overlap, ns.wall_height_m),
        ("wall_right", x_max, cy, ns.wall_thickness_m, arena_d + overlap, ns.wall_height_m),
        ("wall_front", cx, y_min, arena_w + overlap, ns.wall_thickness_m, ns.wall_height_m),
        ("wall_back",  cx, y_max, arena_w + overlap, ns.wall_thickness_m, ns.wall_height_m),
    ]
    for name, ax, ay, sx_arena, sy_arena, sz_arena in walls:
        ux, uy, _ = arena_to_ue((ax, ay, 0.0))
        # axis_map y2x,x2y_neg: arena Y dimension → UE X, arena X → UE Y
        ue_dims = (sy_arena, sx_arena, sz_arena)
        _add_box(
            name=name,
            location_m=(ux, uy, wall_z_center),
            dimensions_m=ue_dims,
            material=mat_wall,
        )

    # ── PILLARS ── at unique marker (x, y) positions with outward offset.
    half_pillar = ns.pillar_thickness_m / 2.0
    wall_offset_map = {
        "left":  (-half_pillar, 0.0),
        "right": (+half_pillar, 0.0),
        "back":  (0.0, +half_pillar),
        "front": (0.0, -half_pillar),
    }
    pillar_z_center = ns.pillar_height_m / 2.0
    seen: set[tuple[float, float]] = set()
    for m in markers:
        wall = str(m.get("wall", "front"))
        dx, dy = wall_offset_map.get(wall, (0.0, 0.0))
        key = (float(m["x"]) + dx, float(m["y"]) + dy)
        if key in seen:
            continue
        seen.add(key)
        ux, uy, _ = arena_to_ue((key[0], key[1], 0.0))
        _add_box(
            name=f"pillar_{wall}_{int(m['id']):02d}",
            location_m=(ux, uy, pillar_z_center),
            dimensions_m=(ns.pillar_thickness_m, ns.pillar_thickness_m,
                          ns.pillar_height_m),
            material=mat_pillar,
        )

    # ── EXPORT ── single FBX with all meshes selected.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath=str(ns.out),
        use_selection=False,
        # Sidecar PNGs (same approach as the marker FBXs, which works).
        path_mode="AUTO",
        embed_textures=False,
        apply_unit_scale=True,
        global_scale=1.0,
        bake_space_transform=False,
        axis_forward="X",
        axis_up="Z",
        object_types={"MESH"},
        mesh_smooth_type="FACE",
    )

    print(f"  ✓ {ns.out}")
    print(f"  arena bbox: x=[{x_min},{x_max}], y=[{y_min},{y_max}] "
          f"→ centre arena ({cx:.2f}, {cy:.2f})")
    print(f"  pillars: {len(seen)}")
    print(f"  walls: {len(walls)}")
    print(f"  floor: {ns.floor_size_m} × {ns.floor_size_m} × "
          f"{ns.floor_thickness_m} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
