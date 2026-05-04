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
    of the Principled inputs are ignored on FBX import.

    NOTE: vector BaseColor (default_value RGBA) does NOT survive
    Blender → FBX → UE4 cleanly — the SDC host's render fell back to
    UE4's default debug checker for the floor. The texture-driven
    materials (markers) DID survive when the texture was a colocated
    PNG. See ``_make_solid_color_texture_material`` below — used for
    the floor + pillar instead of this vector-only material."""
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


def _make_solid_color_texture_material(
    name: str,
    color_rgb_0_255: tuple[int, int, int],
    roughness: float,
    out_dir: Path,
    alpha_0_255: int | None = None,
) -> "bpy.types.Material":
    """Build a material whose BaseColor is a 64×64 PNG of the given
    colour (optionally RGBA). Texture lives next to the FBX as a
    sidecar — UE4's FBX import reliably picks it up that way.

    Why a real texture instead of a vector ``Base Color``: live
    evidence on the SDC host showed vector PBR colours were dropped
    during Blender 3.0 → FBX → UE4 import; the floor came out with
    UE4's default debug checker. Marker materials, which use a real
    texture, survived. So we generate a small PNG and reference it
    as a colocated sidecar — same pipeline the markers use.

    When ``alpha_0_255`` is given, the PNG is RGBA. UE4's auto-
    generated material from FBX import sometimes treats this as a
    masked/translucent BaseColor (giving a see-through look on the
    nets); when it doesn't, the wall stays opaque but at the lighter
    colour, which is still visibly less imposing than the pillars.
    """
    assert bpy is not None
    png_path = out_dir / f"{name}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_png_solid(png_path, color_rgb_0_255, side_px=64,
                     alpha_0_255=alpha_0_255)

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


def _write_png_solid(
    path: Path,
    color_rgb_0_255: tuple[int, int, int],
    side_px: int = 64,
    alpha_0_255: int | None = None,
) -> None:
    """Write a solid-colour PNG. RGB by default, RGBA when ``alpha`` is
    given. Standalone — no Pillow dependency.

    ``side_px`` defaults to 64 instead of 1 because some UE4 FBX
    import paths (verified live on the SDC host with Blender 3.0.1)
    silently fail to apply 1×1 textures, falling back to UE4's debug
    checker. 64×64 is still tiny on disk (~few hundred bytes after
    zlib) and works reliably.
    """
    import struct
    import zlib
    r, g, b = color_rgb_0_255
    if alpha_0_255 is None:
        # RGB
        color_type = 2  # truecolor
        bytes_per_pixel = 3
        pixel = bytes([r, g, b])
    else:
        # RGBA
        color_type = 6  # truecolor + alpha
        bytes_per_pixel = 4
        pixel = bytes([r, g, b, int(alpha_0_255) & 0xFF])
    ihdr = struct.pack(">IIBBBBB", side_px, side_px, 8, color_type, 0, 0, 0)
    # Each scanline starts with a filter byte (0 = none) followed by
    # the pixel row. Repeat ``side_px`` times.
    row = bytes([0]) + pixel * side_px
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


def _export_active(out_path: Path) -> None:
    assert bpy is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(out_path),
        use_selection=False,
        # path_mode=AUTO + embed_textures=False makes the FBX point
        # at the PNG sidecar by filename — verified live: this is the
        # only path UE4's Sphinx mesh-importer reliably honours.
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


def build_floor(out_path: Path) -> None:
    """A 1 m × 1 m × 0.1 m flat slab with neutral concrete material.

    NOTE: built as a CUBE (not a plane). The plane approach left UE4
    falling back to its debug-checker default material in the live
    render — UE4's FBX import on a flat plane sometimes drops the
    material entirely. A flat slab with thickness has unambiguously
    3D geometry and reliably picks up the texture-driven material.

    The YAML emitter scales this to the arena footprint (default
    22 × 12 m × 0.1 m)."""
    assert bpy is not None
    _clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "arena_floor"
    # Make it a flat slab: 1×1 in XY, 0.1 in Z. Centred at origin.
    obj.scale = (1.0, 1.0, 0.1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    mat = _make_solid_color_texture_material(
        name="mat_floor",
        color_rgb_0_255=(140, 140, 145),
        roughness=0.85,
        out_dir=out_path.parent,
    )
    obj.data.materials.append(mat)
    _export_active(out_path)


def build_pillar(out_path: Path) -> None:
    """1 m × 1 m × 1 m unit cube centred at origin, dark-grey
    PNG-textured material.

    Centred so the YAML emitter can place it at the desired (x, y, z)
    without any Blender-side offset arithmetic."""
    assert bpy is not None
    _clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "arena_pillar"
    mat = _make_solid_color_texture_material(
        name="mat_pillar",
        color_rgb_0_255=(46, 46, 52),  # near-black
        roughness=0.6,
        out_dir=out_path.parent,
    )
    obj.data.materials.append(mat)
    _export_active(out_path)


def build_net(out_path: Path) -> None:
    """1 m × 1 m × 1 m unit cube centred at origin, light-grey RGBA-
    textured material — the perimeter "net" walls.

    Distinct FBX from pillar.fbx so the wall material can be lighter
    (and use alpha) without affecting pillars. Sphinx mesh injection
    technically only honours four PBR params, but UE4's auto-generated
    material from an RGBA-textured FBX often picks up a masked-
    translucent blend mode anyway. If that succeeds, the walls render
    truly semi-transparent; if not, the lighter colour still makes
    them visually less imposing than the dark pillars (less wall-
    looking)."""
    assert bpy is not None
    _clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "arena_net"
    mat = _make_solid_color_texture_material(
        name="mat_net",
        color_rgb_0_255=(220, 225, 235),  # very light blue-grey
        roughness=0.4,
        out_dir=out_path.parent,
        alpha_0_255=80,  # ~30 % opaque if UE4 honours the alpha channel
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
    net_path = ns.out_dir / "net.fbx"

    print(f"  building {floor_path.name}…")
    build_floor(floor_path)
    print(f"  ✓ {floor_path}")
    print(f"  building {pillar_path.name}…")
    build_pillar(pillar_path)
    print(f"  ✓ {pillar_path}")
    print(f"  building {net_path.name}…")
    build_net(net_path)
    print(f"  ✓ {net_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
