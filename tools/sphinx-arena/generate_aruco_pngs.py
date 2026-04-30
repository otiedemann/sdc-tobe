#!/usr/bin/env python3
"""Generate ArUco marker PNG bitmaps for the Sphinx arena scene.

Produces one PNG per marker ID at a fixed pixel size; the physical size
(0.5 m wall markers, 0.19 m target-box stickers) is set later when the
PNGs get wrapped into FBX planes by ``build_marker_fbx.py``.

The arena uses ``cv2.aruco.DICT_4X4_100`` everywhere — same dictionary as
``controller_unified/ctrl_position.py``. The output PNGs include a
configurable white border so the marker stays detectable when the
texture sampling slightly bleeds past the edge in the simulator render.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARENA_CONFIG = REPO_ROOT / "controller_unified" / "arena_config.json"
DEFAULT_TARGET_LAYOUT = Path(__file__).parent / "default_target_layout.json"
DEFAULT_OUT_DIR = Path(__file__).parent / "out" / "markers"

# Wall-marker IDs we need from the arena config; target IDs are read from
# the SDC26 strict set (Blue 31-36, Red 41-46) in the target layout JSON.
ARUCO_DICT_NAME = "DICT_4X4_100"


def load_arena_marker_ids(arena_path: Path) -> list[int]:
    with arena_path.open() as f:
        cfg = json.load(f)
    return sorted({int(m["id"]) for m in cfg.get("markers", [])})


def load_target_marker_ids(layout_path: Path) -> list[int]:
    with layout_path.open() as f:
        layout = json.load(f)
    return sorted({int(b["id"]) for b in layout.get("boxes", [])})


def render_marker(
    aruco_dict, marker_id: int, side_px: int, border_px: int
) -> np.ndarray:
    """Return an HxW grayscale image with a white quiet-zone border.

    OpenCV's marker generator already draws a 1-cell black quiet zone;
    the extra ``border_px`` we paint on top keeps the marker detectable
    even when the simulator's texture sampler bleeds slightly outside
    the planar mesh (a recurring problem with FBX UVs at marker edges).
    """
    inner = side_px - 2 * border_px
    if inner < 16:
        raise ValueError(
            f"side_px={side_px} too small after {border_px}-px border (inner={inner})"
        )
    inner_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, inner)
    canvas = np.full((side_px, side_px), 255, dtype=np.uint8)
    canvas[border_px : border_px + inner, border_px : border_px + inner] = inner_img
    return canvas


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arena-config", type=Path, default=DEFAULT_ARENA_CONFIG,
        help="Path to controller_unified/arena_config.json (or compatible).",
    )
    p.add_argument(
        "--target-layout", type=Path, default=DEFAULT_TARGET_LAYOUT,
        help="Path to default_target_layout.json (SDC26 box positions).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Output directory for PNG bitmaps.",
    )
    p.add_argument(
        "--side-px", type=int, default=800,
        help="Output PNG side length in pixels. 800 is plenty even for "
             "1080p sim renders; cheap to bump if you need finer detail.",
    )
    p.add_argument(
        "--border-px", type=int, default=40,
        help="White quiet-zone border around the OpenCV-rendered marker.",
    )
    p.add_argument(
        "--ids", type=int, nargs="*", default=None,
        help="Override the ID set entirely (skips arena_config + target_layout).",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.ids is not None:
        ids = sorted(set(args.ids))
        source = "--ids override"
    else:
        arena_ids = load_arena_marker_ids(args.arena_config)
        target_ids = load_target_marker_ids(args.target_layout)
        ids = sorted(set(arena_ids) | set(target_ids))
        source = (
            f"{len(arena_ids)} arena + {len(target_ids)} target "
            f"= {len(ids)} unique IDs"
        )

    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, ARUCO_DICT_NAME)
    )

    print(f"Generating {len(ids)} ArUco PNGs ({source})")
    print(f"  dictionary: {ARUCO_DICT_NAME}")
    print(f"  side: {args.side_px} px (border {args.border_px} px)")
    print(f"  out: {args.out_dir}")

    written = 0
    for mid in ids:
        img = render_marker(aruco_dict, mid, args.side_px, args.border_px)
        out_path = args.out_dir / f"aruco_{mid:03d}.png"
        if not cv2.imwrite(str(out_path), img):
            print(f"  ✗ write failed: {out_path}", file=sys.stderr)
            return 2
        written += 1
        print(f"  ✓ {out_path.name}")
    print(f"Done: {written} PNGs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
