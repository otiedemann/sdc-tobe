#!/usr/bin/env python3
"""Build a painting FBX per logo PNG — flat textured plane.

Run inside Blender:
    blender --background --python build_painting_fbxs.py -- \\
        --in-dir out/logos --out-dir out/fbx

For each ``out/logos/<name>.png`` produces ``out/fbx/painting_<name>.fbx``,
a 1×1 unit plane with the PNG as ``BaseColor``. The Sphinx YAML emitter
scales each painting to the desired size and places it on a wall.
Same colocated-PNG pipeline that the marker FBXs use (proven to
survive the Blender 3.0 → FBX → UE4 import).
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


def _build_one(png_path: Path, out_path: Path) -> None:
    """Build a single painting FBX. Mirrors build_marker_fbx.py's flow:
    flat XY plane + Principled-BSDF material with PNG sidecar."""
    assert bpy is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy PNG next to FBX so the FBX can reference it by relative
    # filename (UE4's FBX importer reliably reads sidecars this way).
    sibling_png = out_path.with_suffix(".png")
    sibling_png.write_bytes(png_path.read_bytes())

    _clear_scene()

    # Flat plane. Pitch=90 in YAML stands it upright (same handling as
    # marker FBXs).
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = f"painting_{png_path.stem}"

    # Material
    mat = bpy.data.materials.new(name=f"mat_painting_{png_path.stem}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.7
    bsdf.inputs["Specular"].default_value = 0.1
    bsdf.inputs["Metallic"].default_value = 0.0

    tex_node = nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(str(sibling_png))
    # Logos are sRGB-encoded display imagery (unlike ArUco markers which
    # we treat as Non-Color/linear). Use the default sRGB colorspace
    # so the colour reproduction is correct.
    for cs_name in ("sRGB", "Linear", "Raw"):
        try:
            img.colorspace_settings.name = cs_name
            break
        except TypeError:
            continue
    tex_node.image = img
    tex_node.interpolation = "Linear"  # logos look better with smoothing

    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.append(mat)

    bpy.ops.export_scene.fbx(
        filepath=str(out_path),
        use_selection=False,
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


def main() -> int:
    if bpy is None:
        print(
            "ERROR: must run inside Blender:\n"
            "  blender --background --python build_painting_fbxs.py -- ...",
            file=sys.stderr,
        )
        return 2

    args = _slice_blender_argv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, required=True,
                   help="Directory of logo PNGs.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to write painting_*.fbx files.")
    ns = p.parse_args(args)

    pngs = sorted(ns.in_dir.glob("*.png"))
    if not pngs:
        print(f"no PNGs in {ns.in_dir}", file=sys.stderr)
        return 1

    for png in pngs:
        out_fbx = ns.out_dir / f"painting_{png.stem}.fbx"
        _build_one(png, out_fbx)
        print(f"  ✓ {out_fbx.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
