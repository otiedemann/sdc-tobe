#!/usr/bin/env python3
"""Generate the Sphinx arena's static geometry (floor + reusable pillar)
as FBX files, headless via Blender.

Usage:
    blender --background --python build_arena_geometry.py -- --out-dir <dir>

Produces two FBX files:

* ``floor.fbx``  — unit 1 m × 1 m plane lying flat in the XY plane,
                    normal +Z. Scaled by ``arena_to_sphinx_yaml.py`` to
                    cover the arena footprint (default 22 × 12 m).
                    Material: light concrete grey.
* ``pillar.fbx`` — unit cube centred at the origin (1 m × 1 m × 1 m).
                    Scaled by the YAML emitter to ``0.2 × 0.2 × 6`` m
                    per pillar. Material: dark grey.

The YAML emitter places one floor and N pillars at the marker (x, y)
positions read from ``arena_config.json``, then layers the existing
marker FBXs on top.

Why one canonical pillar FBX instead of 8 separate ones: Sphinx's mesh
loader supports per-instance Location/Rotation/Scale, so the same
``pillar.fbx`` can be referenced 8 times at 8 different positions.
Saves Blender time, disk, and Sphinx load time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import bpy  # type: ignore
except ModuleNotFoundError:
    bpy = None


def _slice_blender_argv() -> list[str]:
    try:
        idx = sys.argv.index("--")
    except ValueError:
        return []
    return sys.argv[idx + 1 :]


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


def _make_pbr_material(name: str, base_color_rgba: tuple[float, float, float, float],
                       roughness: float) -> "bpy.types.Material":
    """Build a Principled-BSDF material with a flat colour. Sphinx
    only honours BaseColor / Roughness / Specular / Metallic — the rest
    of the Principled inputs are ignored on FBX import."""
    assert bpy is not None
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = base_color_rgba
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Specular"].default_value = 0.2
    return mat


def _export_active(out_path: Path) -> None:
    assert bpy is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(out_path),
        use_selection=False,
        path_mode="COPY",
        embed_textures=True,
        apply_unit_scale=True,
        global_scale=1.0,
        bake_space_transform=True,
        object_types={"MESH"},
        mesh_smooth_type="FACE",
    )


def build_floor(out_path: Path) -> None:
    """1 m × 1 m unit plane, normal +Z, light-concrete material.

    The YAML emitter scales this to the arena footprint (default
    22 × 12 m). We bake the unit dimension here rather than baking
    the final size, so a single FBX serves any arena layout."""
    assert bpy is not None
    _clear_scene()
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "arena_floor"
    mat = _make_pbr_material(
        "mat_floor",
        base_color_rgba=(0.55, 0.55, 0.58, 1.0),  # neutral concrete
        roughness=0.85,
    )
    obj.data.materials.append(mat)
    _export_active(out_path)


def build_pillar(out_path: Path) -> None:
    """1 m × 1 m × 1 m unit cube centred at origin, dark-grey material.

    Centred so the YAML emitter can place it at the desired (x, y, z)
    without any Blender-side offset arithmetic."""
    assert bpy is not None
    _clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "arena_pillar"
    mat = _make_pbr_material(
        "mat_pillar",
        base_color_rgba=(0.18, 0.18, 0.20, 1.0),  # near-black
        roughness=0.6,
    )
    obj.data.materials.append(mat)
    _export_active(out_path)


def main() -> int:
    if bpy is None:
        print(
            "ERROR: this script must be run inside Blender:\n"
            "  blender --background --python build_arena_geometry.py -- --out-dir <dir>",
            file=sys.stderr,
        )
        return 2

    args = _slice_blender_argv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for floor.fbx and pillar.fbx.")
    ns = p.parse_args(args)

    floor_path = ns.out_dir / "floor.fbx"
    pillar_path = ns.out_dir / "pillar.fbx"

    print(f"  building {floor_path.name}…")
    build_floor(floor_path)
    print(f"  ✓ {floor_path}")
    print(f"  building {pillar_path.name}…")
    build_pillar(pillar_path)
    print(f"  ✓ {pillar_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
