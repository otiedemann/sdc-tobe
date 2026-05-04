#!/usr/bin/env python3
"""Wrap one or more ArUco PNGs into Sphinx-compatible FBX planes.

This script is meant to be run inside Blender's bundled Python:

    blender --background --python build_marker_fbx.py -- <args...>

The double-dash is mandatory — Blender consumes everything before it; the
script's argparse takes everything after.

Each output FBX contains a single 1 m × 1 m flat plane mesh with a basic
material whose ``BaseColor`` is the marker PNG. Sphinx scales the mesh
later via the YAML config's ``Scale`` field — we keep the canonical mesh
at 1 m so a 0.5 m wall marker becomes ``Scale: "0.5 0.5 0.5"`` and a
0.19 m target sticker becomes ``Scale: "0.19 0.19 0.19"``.

The plane is built FLAT in the XY plane with normal +Z. Sphinx YAML
rotation then aims it at the appropriate wall (see
``arena_to_sphinx_yaml.py``). Only Material parameters Sphinx actually
honors are set: BaseColor (the texture), Roughness, Specular, Metallic
— per the Sphinx ``customize_the_environment`` docs.

Axis convention on export — IMPORTANT:
    Sphinx's documentation specifies for blender users:
      "forward vector should be set to -Y and up vector should be set
       to -Z to match the Unreal Engine's coordinate system."
    We use those exact values. An earlier revision used
    axis_forward="X", axis_up="Z" — geometry imported into Sphinx with
    a 90° rotation offset on every wall, AND walls/pillars from the
    combined arena_static.fbx came out either invisible or in
    unexpected positions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Blender exposes its API via the ``bpy`` module only when this script is
# run inside Blender. Outside (e.g. linting on macOS) we still want the
# file to import cleanly so editors and CI can syntax-check it.
try:
    import bpy  # type: ignore
except ModuleNotFoundError:  # running outside Blender
    bpy = None


def _slice_blender_argv() -> list[str]:
    """Return CLI args after the lone ``--`` Blender uses as separator."""
    try:
        idx = sys.argv.index("--")
    except ValueError:
        return []
    return sys.argv[idx + 1 :]


def _clear_scene() -> None:
    """Wipe Blender's default cube/light/camera so each FBX is minimal."""
    assert bpy is not None
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Also purge orphan data so repeated invocations stay clean.
    for img in list(bpy.data.images):
        bpy.data.images.remove(img)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def _build_plane(name: str) -> "bpy.types.Object":
    """Create a 1 m × 1 m flat plane (XY plane, normal +Z)."""
    assert bpy is not None
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    return obj


def _make_texture_material(name: str, png_path: Path) -> "bpy.types.Material":
    """Create a Principled-BSDF material with the PNG plugged into BaseColor."""
    assert bpy is not None
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.9
    bsdf.inputs["Specular"].default_value = 0.0
    bsdf.inputs["Metallic"].default_value = 0.0

    tex_node = nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(str(png_path))
    # Blender renamed the linear-no-gamma colorspace from "Linear"
    # (3.x and earlier) to "Non-Color" (4.x). The host might be either.
    # Try the modern name first; fall back so the build doesn't crash
    # on older Blender. We just need *some* non-sRGB space so the
    # ArUco bitmap stays high-contrast through the render pipeline.
    for cs_name in ("Non-Color", "Linear", "Raw"):
        try:
            img.colorspace_settings.name = cs_name
            break
        except TypeError:
            continue
    tex_node.image = img
    tex_node.interpolation = "Closest"  # keep ArUco corners sharp

    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def build_one(png_path: Path, out_path: Path) -> None:
    assert bpy is not None
    if not png_path.is_file():
        raise FileNotFoundError(png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy the source PNG next to the FBX so the FBX can reference it
    # by filename (not a path that depends on the build host). UE4's
    # FBX importer in Sphinx mesh-injection consistently picks up
    # textures via co-located PNGs; embedded textures via
    # ``embed_textures=True`` were silently dropped on the SDC host
    # (Blender 3.0.1 + Sphinx 2.15.1), leaving the marker plane with
    # the default debug material.
    sibling_png = out_path.with_suffix(".png")
    sibling_png.write_bytes(png_path.read_bytes())

    _clear_scene()
    obj = _build_plane(name=f"aruco_{png_path.stem}")
    mat = _make_texture_material(f"mat_{png_path.stem}", sibling_png)
    obj.data.materials.append(mat)

    bpy.ops.export_scene.fbx(
        filepath=str(out_path),
        use_selection=False,
        # path_mode=AUTO writes the texture path as a filename relative
        # to the .fbx (since we co-located the PNG). UE4 reads it back
        # by that relative path on import.
        path_mode="AUTO",
        embed_textures=False,
        apply_unit_scale=True,
        global_scale=1.0,
        bake_space_transform=False,
        # Per Sphinx 2.15 docs (customize_the_environment): "forward
        # vector should be set to -Y and up vector should be set to -Z
        # to match the Unreal Engine's coordinate system." This is
        # NON-STANDARD but it's what Sphinx's mesh-injection actually
        # consumes — using "X"/"Z" produced rotated and missing
        # geometry on import.
        axis_forward="-Y",
        axis_up="-Z",
        object_types={"MESH"},
        mesh_smooth_type="FACE",
    )


def main() -> int:
    if bpy is None:
        print(
            "ERROR: this script must be run inside Blender:\n"
            "  blender --background --python build_marker_fbx.py -- <args>",
            file=sys.stderr,
        )
        return 2

    args = _slice_blender_argv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in-dir", type=Path, required=False,
        help="Directory of input PNGs. Each *.png becomes one FBX.",
    )
    p.add_argument(
        "--out-dir", type=Path, required=False,
        help="Directory for output FBX files.",
    )
    p.add_argument(
        "--single-in", type=Path, required=False,
        help="Single input PNG (alternative to --in-dir).",
    )
    p.add_argument(
        "--single-out", type=Path, required=False,
        help="Single output FBX (alternative to --out-dir).",
    )
    ns = p.parse_args(args)

    if ns.single_in and ns.single_out:
        build_one(ns.single_in, ns.single_out)
        print(f"  ✓ {ns.single_out}")
        return 0

    if ns.in_dir and ns.out_dir:
        pngs = sorted(ns.in_dir.glob("*.png"))
        if not pngs:
            print(f"No PNGs found in {ns.in_dir}", file=sys.stderr)
            return 1
        # Skip the white back-plate PNGs from the previous experiment;
        # they aren't ArUco markers themselves.
        pngs = [p for p in pngs if not p.stem.endswith("_white")]
        for png in pngs:
            out_path = ns.out_dir / f"{png.stem}.fbx"
            build_one(png, out_path)
            print(f"  ✓ {out_path.name}")
        return 0

    print(
        "ERROR: pass either --in-dir/--out-dir or --single-in/--single-out.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
