# Sphinx arena scaffolding (Path A)

This directory builds a Parrot Sphinx scene of the SDC arena by injecting
ArUco-marker-textured FBX planes into a stock Sphinx world via the
`--config-file` YAML option. **No custom UE editor build is required**;
the only heavy dependency is Blender (used headlessly to wrap each PNG
into an FBX). Total time from a clean Ubuntu host to a flying simulated
Anafi inside a marker grid: maybe one afternoon.

If you eventually decide you need higher fidelity (real lighting,
scaffolding clutter, etc.), there's a "Path B" full-UE-editor workflow
documented in the SDC notes — but start here. Most algorithm-level
testing only needs the marker geometry to be right.

## What this builds

```
arena_config.json (16 wall markers)        ┐
default_target_layout.json (6 boxes)       │── arena_to_sphinx_yaml.py ──▶ out/arena.yml
PNG bitmaps (28 ArUco IDs)  ◀── generate_aruco_pngs.py
                            │── build_marker_fbx.py (Blender) ──▶ out/fbx/aruco_<id>.fbx
```

The end product is a single `out/arena.yml` plus 28 FBX files. You copy
those to a Linux Sphinx host and pass the YAML to Sphinx at launch.

## Prerequisites

Local (anywhere — Mac or Linux):
- Python 3.9+ with `opencv-python` and `numpy` (see `requirements.txt`)
- Blender 3.x or 4.x (`apt install blender` on Ubuntu, or download from
  blender.org). Used headlessly via `--background --python`.

Linux Sphinx host (Ubuntu 22.04 confirmed by Parrot):
- `sudo apt install parrot-sphinx parrot-ue-sdk`
- A GitHub account linked to your Unreal Engine account (only needed if
  you ever escalate to Path B; not required for Path A).

## Quick start

From this directory:

```bash
# Once: install Python deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build everything (runs Blender headlessly):
make all

# Or self-test without Blender (skips FBX generation):
make check
```

After `make all` you'll have:

```
out/markers/aruco_001.png ... aruco_046.png
out/fbx/aruco_001.fbx     ... aruco_046.fbx
out/arena.yml
```

Copy `out/` to the Sphinx host. If the path on the host differs from
your local checkout, regenerate the YAML with the host path baked in:

```bash
python3 arena_to_sphinx_yaml.py \
    --fbx-dir-on-host /home/sdc/sphinx-arena/fbx \
    --out out/arena_for_host.yml
```

## Launching Sphinx with the arena

On the Linux Sphinx host:

```bash
sudo systemctl start firmwared.service

sphinx \
    /opt/parrot-sphinx/usr/share/sphinx/drones/anafi2.drone::firmware="https://firmware.parrot.com/Versions/anafi2/pc/%23latest/images/anafi2-pc.ext2.zip" \
    --config-file /home/sdc/sphinx-arena/arena.yml \
    & parrot-ue4-empty
```

`parrot-ue4-empty` is the stock empty world; the YAML adds our markers
on top of it. The Anafi spawns at the world origin; you can fly with
the Olympe API or your existing `controller_unified` stack — the
simulated drone presents the same interface as the real Anafi.

## Verifying it works

1. **Visual check.** The empty world should show 22 floating squares
   (16 wall markers in two horizontal rings + 6 target stickers near
   the floor). If they're on the wrong walls, see "Coordinate-system
   gotchas" below.
2. **Detection check.** Connect your `controller_unified` to the
   simulated drone and confirm `/api/position` reports a sane
   position when you fly the drone toward a marker.
3. **Calibration check.** The Anafi sim's camera intrinsics are not
   guaranteed to match your real-world `position_calib.npz`. If
   distances are off by a constant factor, re-calibrate inside the sim
   (fly the drone at a known distance from a marker, log
   `/api/position`, compare).

## Coordinate-system gotchas

Arena frame (the JSON inputs): metres, right-handed, +X = arena right,
+Y = into the arena (away from origin wall), +Z = up.

UE4 / Sphinx frame: centimetres, left-handed, default UE convention is
+X-forward, +Y-right, +Z-up.

The `--axis-map` flag picks how arena axes map to UE axes. Default
(`y2x,x2y_neg,z2z`) puts arena +Y along UE +X. If after the first
launch you see markers on the wrong walls (e.g. left and right swapped,
or front and back swapped), re-run the YAML emitter with a different
preset:

| Preset | Effect |
|---|---|
| `y2x,x2y_neg,z2z` | default — arena Y→UE X, arena X→UE -Y |
| `x2x,y2y,z2z` | identity (use if Sphinx world matches arena coords) |
| `x2x_neg,y2y,z2z` | flip X only (mirror across Y axis) |
| `y2x_neg,x2y,z2z` | swap X/Y and flip X |

If none of those match your setup, add a new entry to
`_AXIS_MAP_PRESETS` in `arena_to_sphinx_yaml.py` — it's a 4-tuple per
axis: `(arena_axis_index, sign)` for each of the three UE axes.

Wall yaw rotations (`front=0, back=180, left=90, right=-90`) assume the
default axis map. If you change the axis map, you may also need to
change `WALL_YAW_DEG` so the marker faces inward instead of outward.

## File-by-file reference

- **`generate_aruco_pngs.py`** — reads `arena_config.json` and
  `default_target_layout.json`, calls
  `cv2.aruco.generateImageMarker()` for each ID, writes
  `aruco_<id>.png`. Configurable resolution (`--side-px`, default 800)
  and quiet-zone border (`--border-px`, default 40).
- **`build_marker_fbx.py`** — Blender headless script. For each PNG,
  builds a 1 m × 1 m plane with the PNG as `BaseColor`, exports as
  FBX with the texture embedded. Called from the Makefile via
  `blender --background --python ...`.
- **`arena_to_sphinx_yaml.py`** — emits the Sphinx YAML config. Reads
  arena config + target layout, applies the axis map, emits one
  `Meshes:` entry per enabled marker. Pure stdlib + Python — no cv2 or
  Blender needed.
- **`default_target_layout.json`** — six SDC26 target boxes (3 blue,
  3 red) in their starting home-zone positions, plus six disabled
  spares. Edit this to change the box layout for your scenarios.
- **`Makefile`** — orchestrates the pipeline. `make all` does
  PNGs → FBXs → YAML.
- **`requirements.txt`** — Python deps for the local-side scripts (cv2
  and numpy). Blender is *not* pip-installable.

## Customizing

- **Different arena layout?** Edit `controller_unified/arena_config.json`
  (the source of truth — `ctrl_position.py` already reads from it). Then
  `make all` regenerates everything from that.
- **Different target box positions?** Edit `default_target_layout.json`.
  Or copy it and pass `--target-layout your_layout.json`.
- **Bigger/smaller markers?** Override `--wall-marker-size-m-override`
  or `--target-marker-size-m-override` on `arena_to_sphinx_yaml.py`.
  (The FBX is always 1 m and gets scaled per-mesh by Sphinx.)
- **Adding a floor or walls?** Sphinx's stock `empty` world is just a
  flat ground plane — that may be enough. If you want walls or
  scaffolding, build them as additional FBXs, drop them in `out/fbx/`,
  add `Meshes:` entries by hand (or extend
  `arena_to_sphinx_yaml.py` with a `--include-floor-box` analogue).

## Limits of Path A

- **Materials are limited to BaseColor + Roughness + Specular +
  Metallic.** No normal maps, no emissive, no transparency. Fine for
  flat ArUco markers; not fine for replicating the qualifying arena's
  bright skylights.
- **Lighting comes from `parrot-ue4-empty`.** No backlit-window
  modeling, no shadow geometry, no realistic exposure. If your tracker
  failures are exposure-driven (and the field-log analysis suggests
  they partly are), the sim won't reproduce them.
- **Camera intrinsics drift.** Already noted above. Plan for separate
  sim and real `position_calib.npz` files if accuracy matters.

When you outgrow these, that's the signal to start Path B. Until then,
Path A is enough to drive most algorithm work.
