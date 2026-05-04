#!/usr/bin/env python3
"""Wrap one or more ArUco PNGs into Sphinx-compatible double-sided FBX
plates.

This script is meant to be run inside Blender's bundled Python:

    blender --background --python build_marker_fbx.py -- <args...>

The double-dash is mandatory — Blender consumes everything before it; the
script's argparse takes everything after.

Each output FBX contains TWO co-located 1 m × 1 m planes:

  * Front plane — UV-textured with the ArUco PNG, normal +Z.
  * Back plane  — uniform-white textured, normal -Z (built by rotating
                  a unit plane 180° around X so the visible face lands
                  at -Z).

This was added because — even with the per-wall yaw computed in
``arena_to_sphinx_yaml.py`` — the live UE4 render kept showing markers
facing partly outward. Without a back plate, the underside of the plane
is transparent in UE4's default mesh-injection material, so an
incorrectly-oriented marker looks plain wrong from inside the arena.
With the back plate, the worst case is "I see a white plate where I
expected an ArUco" — which is much easier to diagnose visually and far
less misleading. From inside the arena, the ArUco face should always
be visible; from outside, the white plate.

Sphinx scales the mesh later via the YAML config's ``Scale`` field — we
keep the canonical mesh at 1 m so a 0.5 m wall marker becomes
``Scale: "0.5 0.5 0.5"``.

Both planes are flat in the local XY plane (normal ±Z). Sphinx YAML
rotation then aims them at the appropriate wall (see
``arena_to_sphinx_yaml.py``). Only Material parameters Sphinx actually
honors are set: BaseColor (the texture), Roughness, Specular, Metallic.
Anything fancier than that gets stripped at FBX import time anyway.
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
import zlib
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


def _write_white_png(path: Path, side_px: int = 64) -> None:
    """Hand-craft a small uniform-white PNG via stdlib zlib + struct.

    Used as the BaseColor for the back plate. Tiny (64×64) because it's
    a flat colour — no detail to preserve. Same hand-crafted approach
    ``build_full_arena.py`` uses for the floor/wall/pillar materials,
    which we know survives Blender → FBX → UE4 (vector ``Base Color``
    alone does not).
    """
    ihdr = struct.pack(">IIBBBBB", side_px, side_px, 8, 2, 0, 0, 0)
    row = bytes([0]) + bytes([255, 255, 255]) * side_px
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


def _make_texture_material(name: str, png_path: Path) -> "bpy.types.Material":
    """Create a Principled-BSDF material with the PNG plugged into BaseColor.

    ``Non-Color`` colorspace keeps the ArUco bitmap high-contrast through
    the render pipeline (no sRGB → linear gamma curve eating the black
    borders).
    """
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


def _make_white_material(name: str, png_path: Path) -> "bpy.types.Material":
    """Create a Principled-BSDF material backed by the white PNG.

    Same approach as the ArUco material — texture-driven so it survives
    the FBX → UE4 import. Plain vector ``Base Color`` was previously
    silently dropped, leaving the back plate showing UE's debug
    checker (which defeats the whole point of the back plate)."""
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
    for cs_name in ("Non-Color", "Linear", "Raw"):
        try:
            img.colorspace_settings.name = cs_name
            break
        except TypeError:
            continue
    tex_node.image = img
    tex_node.interpolation = "Closest"

    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def build_one(png_path: Path, out_path: Path) -> None:
    """Build one double-sided marker FBX.

    Geometry: two co-located 1×1 m planes, separated by 2 mm along Z so
    they don't z-fight. Front plane (normal +Z) at z=+0.001 with the
    ArUco texture; back plane (normal -Z, made by rotating a +Z plane
    180° around X) at z=-0.001 with the white texture. Both are baked
    via ``transform_apply`` so the FBX exports clean planes — no
    leftover object-level rotation that UE4's importer might
    misinterpret.
    """
    assert bpy is not None
    if not png_path.is_file():
        raise FileNotFoundError(png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Co-locate the source PNG next to the FBX so the FBX can reference
    # it by filename (not a path that depends on the build host). UE4's
    # FBX importer in Sphinx mesh-injection consistently picks up
    # textures via co-located PNGs; embedded textures via
    # ``embed_textures=True`` were silently dropped on the SDC host
    # (Blender 3.0.1 + Sphinx 2.15.1), leaving the marker plane with
    # the default debug material. Verified live: nets and pillars
    # showed material differences only after switching to this
    # external-texture approach.
    sibling_png = out_path.with_suffix(".png")
    sibling_png.write_bytes(png_path.read_bytes())

    # Co-located white PNG for the back plate. One-per-FBX so the
    # texture path stays relative.
    white_png = out_path.parent / f"{png_path.stem}_white.png"
    _write_white_png(white_png)

    _clear_scene()

    # ── Front plane: ArUco, normal +Z, slight +Z offset ──
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0.001))
    front = bpy.context.active_object
    front.name = f"aruco_{png_path.stem}_front"
    front_mat = _make_texture_material(f"mat_{png_path.stem}", sibling_png)
    front.data.materials.append(front_mat)

    # ── Back plane: white, normal -Z (180° X-rotation), slight -Z offset ──
    # Default plane normal is +Z. Rotating 180° around X flips the
    # normal to -Z. Then transform_apply bakes the rotation into the
    # mesh data so the exported FBX has a clean back-facing plane (no
    # object-level rotation_euler for UE4 to misread).
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, -0.001))
    back = bpy.context.active_object
    back.name = f"aruco_{png_path.stem}_back"
    back.rotation_euler = (math.pi, 0.0, 0.0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    back_mat = _make_white_material(
        f"mat_{png_path.stem}_white", white_png
    )
    back.data.materials.append(back_mat)

    # Select both meshes for export.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath=str(out_path),
        # use_selection=False with both objects in the scene exports
        # both. Keeps things simple — we cleared the scene above so
        # there's nothing else to accidentally include.
        use_selection=False,
        # path_mode=AUTO writes the texture path as a filename relative
        # to the .fbx (since we co-located the PNG). UE4 reads it back
        # by that relative path on import.
        path_mode="AUTO",
        embed_textures=False,
        apply_unit_scale=True,
        global_scale=1.0,
        # Use UE4's coord convention so the imported plane lands flat
        # in UE's XY (normal +Z) rather than getting an FBX axis
        # conversion that left it half-tilted. bake_space_transform
        # interacts badly with this — keep it off so transforms aren't
        # double-baked.
        bake_space_transform=False,
        axis_forward="X",
        axis_up="Z",
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
