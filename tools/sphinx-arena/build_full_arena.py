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


def _build_unit_cube_fbx(out_fbx: Path, name: str,
                          color_rgb: tuple[int, int, int],
                          roughness: float) -> None:
    """Build a SINGLE-MESH unit cube FBX — one mesh, one material — and
    export it. The combined arena_static.fbx with multiple meshes and
    materials kept rendering invisible/blank in Sphinx; individual
    single-mesh FBXs (the same approach that markers and paintings
    use successfully) load reliably. The YAML emitter scales these
    unit cubes per-axis to the right size at the right position.

    Takes color_rgb + roughness rather than a Blender material object
    because _clear_scene() purges orphan materials, so reusing a
    material across exports raises 'StructRNA has been removed'.
    Recreating the material from primitives inside this function
    guarantees a fresh, valid material per export.
    """
    assert bpy is not None
    _clear_scene()
    out_dir = out_fbx.parent
    mat = _make_textured_material(
        f"mat_{out_fbx.stem}", color_rgb, roughness, out_dir
    )
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(mat)
    bpy.ops.export_scene.fbx(
        filepath=str(out_fbx),
        use_selection=False,
        path_mode="AUTO",
        embed_textures=False,
        apply_unit_scale=True,
        global_scale=1.0,
        bake_space_transform=False,
        # Per Sphinx 2.15 customize_the_environment docs.
        axis_forward="-Y",
        axis_up="-Z",
        object_types={"MESH"},
        mesh_smooth_type="FACE",
    )


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
    p.add_argument("--wall-thickness-m", type=float, default=0.20,
                   help="Walls are 20 cm thick by default — thick enough "
                        "to be visible from any approach angle, thin "
                        "enough not to dominate the scene.")
    p.add_argument("--wall-height-m", type=float, default=4.0,
                   help="Walls are tall enough to clearly enclose the "
                        "arena (4 m default). Sphinx YAML can't honour "
                        "real opacity, so we accept solid walls.")
    p.add_argument("--pillar-thickness-m", type=float, default=0.50,
                   help="50 cm pillars — match the SDC arena spec and "
                        "are visually obvious from across the arena.")
    p.add_argument("--pillar-height-m", type=float, default=6.0)
    # Default colours pick HIGH-CONTRAST values so the arena structure
    # is unmistakable against UE4's default sky and ground:
    #   floor:  warm warehouse grey (clearly not default-ground checker)
    #   walls:  dark blue-grey (industrial, doesn't blend with sky)
    #   pillars: bright cream (jumps out, easy to spot from anywhere)
    p.add_argument("--floor-color-rgb", type=str, default="120,120,120")
    p.add_argument("--wall-color-rgb", type=str, default="50,60,80")
    p.add_argument("--pillar-color-rgb", type=str, default="240,220,160")
    p.add_argument("--save-blend", type=Path, default=None,
                   help="Also save a Blender .blend file at this path. "
                        "Default: same dir as --out, with .blend suffix. "
                        "Open it via 'blender path/to/arena_static.blend' "
                        "to inspect or tweak the geometry interactively.")
    p.add_argument("--no-blend", action="store_true",
                   help="Skip the .blend save step entirely.")
    p.add_argument("--gui", action="store_true",
                   help="After building, leave Blender's GUI open with "
                        "the scene loaded so you can inspect / edit / "
                        "re-export interactively. Only useful when this "
                        "script is launched WITHOUT --background.")
    p.add_argument("--no-fbx", action="store_true",
                   help="Skip the FBX export step. Useful with --gui "
                        "when you just want to view the arena without "
                        "overwriting the existing FBX on disk.")
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

    # Blender FBX export with axis_forward="-Y", axis_up="-Z" (Sphinx
    # convention) maps Blender axes to UE as:
    #   Blender +X → UE -Y     (because FBX right = Blender -X under -Y fwd)
    #   Blender +Y → UE -X     (because Blender -Y → FBX +X = UE +X)
    #   Blender +Z → UE -Z     (because Blender -Z → FBX +Z = UE +Z)
    # Inverse — to place geometry at UE (ux, uy, uz), build it in
    # Blender at (-uy, -ux, -uz). For dimensions, X and Y swap because
    # the X-Y axes are exchanged by the rotation; Z magnitude stays.
    def ue_to_blender_pos(ue: tuple[float, float, float]) -> tuple[float, float, float]:
        ux, uy, uz = ue
        return (-uy, -ux, -uz)

    def ue_to_blender_dims(ue_dims: tuple[float, float, float]) -> tuple[float, float, float]:
        dx, dy, dz = ue_dims
        return (dy, dx, dz)

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
    floor_z_center_ue = ns.floor_top_z_m - ns.floor_thickness_m / 2.0
    _add_box(
        name="arena_floor",
        location_m=ue_to_blender_pos((0.0, 0.0, floor_z_center_ue)),
        dimensions_m=ue_to_blender_dims(
            (ns.floor_size_m, ns.floor_size_m, ns.floor_thickness_m)
        ),
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
            location_m=ue_to_blender_pos((ux, uy, wall_z_center)),
            dimensions_m=ue_to_blender_dims(ue_dims),
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
            location_m=ue_to_blender_pos((ux, uy, pillar_z_center)),
            dimensions_m=ue_to_blender_dims(
                (ns.pillar_thickness_m, ns.pillar_thickness_m,
                 ns.pillar_height_m)
            ),
            material=mat_pillar,
        )

    # ── SAVE .BLEND ── always (small, lets the user inspect/tweak).
    # Stored next to the FBX with the same stem unless overridden.
    if not ns.no_blend:
        blend_path = (ns.save_blend if ns.save_blend
                      else ns.out.with_suffix(".blend"))
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        print(f"  ✓ {blend_path}  (open in Blender to inspect/edit)")

    # ── EXPORT COMBINED FBX ── single file with all meshes (kept for
    # the .blend inspection workflow, but the YAML actually consumes
    # the per-element FBXs built below — Sphinx doesn't render this
    # multi-mesh + multi-material combined file reliably).
    if not ns.no_fbx:
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
            # Sphinx 2.15's customize_the_environment docs: "forward
            # vector should be set to -Y and up vector should be set
            # to -Z to match the Unreal Engine's coordinate system."
            # NOT the standard X/Z used by most UE4 FBX exports —
            # Sphinx's mesh-injection has its own convention.
            axis_forward="-Y",
            axis_up="-Z",
            object_types={"MESH"},
            mesh_smooth_type="FACE",
        )
        print(f"  ✓ {ns.out}")

        # ── ALSO EXPORT INDIVIDUAL UNIT-CUBE FBXs ──
        # Sphinx mesh-injection apparently has trouble rendering the
        # multi-mesh + multi-material combined FBX above; markers and
        # paintings (single mesh + single material) load fine. So we
        # also produce simple per-element unit-cube FBXs that the YAML
        # emitter places via --emit-floor / --emit-pillars / --emit-walls
        # with per-axis Scale to size them at runtime. Each call below
        # clears Blender's scene and rebuilds a single object — that's
        # why this MUST come after the combined export above (which
        # otherwise would lose its meshes).
        _build_unit_cube_fbx(
            out_dir / "floor.fbx", "arena_floor",
            parse_rgb(ns.floor_color_rgb), 0.85,
        )
        print(f"  ✓ {out_dir / 'floor.fbx'}")
        # Coloured floor zones for the SDC arena: red 7m on left side,
        # blue 7m on right side, neutral middle handled by the base
        # floor.fbx. YAML emitter scales these into 7×10.8m strips.
        _build_unit_cube_fbx(
            out_dir / "floor_red.fbx", "arena_floor_red",
            (200, 60, 60), 0.85,
        )
        print(f"  ✓ {out_dir / 'floor_red.fbx'}")
        _build_unit_cube_fbx(
            out_dir / "floor_blue.fbx", "arena_floor_blue",
            (60, 80, 200), 0.85,
        )
        print(f"  ✓ {out_dir / 'floor_blue.fbx'}")
        _build_unit_cube_fbx(
            out_dir / "pillar.fbx", "arena_pillar",
            parse_rgb(ns.pillar_color_rgb), 0.6,
        )
        print(f"  ✓ {out_dir / 'pillar.fbx'}")
        _build_unit_cube_fbx(
            out_dir / "wall.fbx", "arena_wall",
            parse_rgb(ns.wall_color_rgb), 0.4,
        )
        print(f"  ✓ {out_dir / 'wall.fbx'}")

    print(f"  arena bbox: x=[{x_min},{x_max}], y=[{y_min},{y_max}] "
          f"→ centre arena ({cx:.2f}, {cy:.2f})")
    print(f"  pillars: {len(seen)} ({ns.pillar_thickness_m}×"
          f"{ns.pillar_thickness_m}×{ns.pillar_height_m} m, "
          f"colour rgb={ns.pillar_color_rgb})")
    print(f"  walls: {len(walls)} ({ns.wall_thickness_m} m thick, "
          f"{ns.wall_height_m} m tall, colour rgb={ns.wall_color_rgb})")
    print(f"  floor: {ns.floor_size_m} × {ns.floor_size_m} × "
          f"{ns.floor_thickness_m} m, colour rgb={ns.floor_color_rgb}")

    # ── INTERACTIVE GUI MODE ── if --gui was passed AND Blender is
    # running with a GUI (i.e., NOT --background), keep the window
    # open so the user can poke at the scene. In --background mode
    # this flag is silently a no-op (there's no UI to keep open).
    if ns.gui:
        # Detect headless: bpy.app.background is True under --background.
        if getattr(bpy.app, "background", False):
            print("  (note: --gui ignored under --background; relaunch "
                  "Blender without --background to enter interactive mode)")
        else:
            print("  GUI mode: arena scene loaded, Blender will stay open.")
            # In a non-background launch, returning from this function
            # hands control back to Blender's main loop.
    return 0


if __name__ == "__main__":
    sys.exit(main())
