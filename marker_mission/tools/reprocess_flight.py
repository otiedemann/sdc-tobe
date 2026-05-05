"""
Offline re-processor: re-run the marker detector + arena estimator
on a recorded flight's raw.mp4, producing two annotated mp4 outputs
that surface the world-position calculation under two configurations
(typically "OLD" without magnetometer-aided IPPE branch picking, and
"NEW" with the offset applied).

Usage:

    python -m marker_mission.tools.reprocess_flight \\
        --flight-dir /home/sdirbach/2026-05-05_14-45-09_PI040416BA8G061033 \\
        --mag-offset -37.0 \\
        [--arena ~/.marker_mission/active_arena_config.json] \\
        [--output-dir <flight-dir>]

The script reads:

* ``raw.mp4`` -- the input video.
* ``flight_log.csv`` -- maps frame timestamps to telemetry (``tel_yaw``).
* ``mission_meta.json`` -- camera intrinsics.
* ``cfg_start.json`` -- marker_size_m + aruco_dict.

Side-by-side it produces:

* ``reprocessed_old.mp4`` -- ``estimate_position(..., tel_yaw_deg=None)``
  with the magnetic offset stripped from the arena. This isolates the
  pre-magnetometer behaviour while still benefiting from the mirror
  collapse / OOB / prev-anchor fixes that landed before the magnetometer
  feature.
* ``reprocessed_new.mp4`` -- ``estimate_position(..., tel_yaw_deg=tel_yaw)``
  with the magnetic offset injected (CLI override, or from the arena
  config). When the arena's offset isn't set and ``--mag-offset`` is
  omitted, the new branch is identical to the old branch (warned).

Both outputs carry the standard marker overlay plus an arena mini-map
in the upper-right corner showing the computed position.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import math

from ..arena import ArenaConfig, estimate_position, load_priority_arena
from ..aruco_detector import (ArucoDetector, annotate_frame,
                                draw_arena_minimap)
from ..calibration_store import Calibration, CalibrationStore
from ..config import CALIB_DIR


def _arena_yaw_from(world_pos: Optional[tuple],
                     poses: list,
                     arena: ArenaConfig) -> Optional[float]:
    """Drone arena yaw derived from one of the visible reference
    markers + the supplied world position. Returns None when no
    reference marker is visible or the world position isn't set
    (we can't compute the bearing-to-marker without it).

    Used by the reprocessor's mini-map so the yaw arrow tracks each
    pipeline's (OLD vs NEW) chosen-branch interpretation -- they can
    diverge by 20-60 deg whenever the magnetometer pick swaps the
    branch in NEW but not in OLD.
    """
    if world_pos is None or not poses or arena is None:
        return None
    candidate = None
    for p in poses:
        if int(p.marker_id) in arena.markers:
            candidate = p
            break
    if candidate is None:
        return None
    m = arena.markers[int(candidate.marker_id)]
    dx = float(m.position_m[0]) - float(world_pos[0])
    dy = float(m.position_m[1]) - float(world_pos[1])
    if dx == 0.0 and dy == 0.0:
        return None
    bearing = math.degrees(math.atan2(dx, dy))
    return ((bearing - float(candidate.yaw_deg) + 540.0) % 360.0) - 180.0


def _load_calibration(meta: dict, image_size: tuple) -> Calibration:
    """Reconstitute the per-flight Calibration.

    Prefer the on-disk CalibrationStore entry (matches what the live
    mission used at flight time, including distortion coefficients).
    Fall back to mission_meta.json's intrinsics if the store doesn't
    have a matching serial+resolution -- in that case D = zeros and
    corner positions diverge by a fraction of a pixel from the live
    detector, which can flip the branch on borderline frames.
    """
    c = meta.get("calibration") or {}
    serial = str(c.get("serial", "unknown"))
    resolution = str(c.get("resolution", "720p"))
    store = CalibrationStore(CALIB_DIR)
    if store.has(serial, resolution):
        cal = store.load(serial, resolution, allow_default=False)
        print(f"[reprocess] using stored calibration "
              f"{serial}/{resolution} (fx={cal.fx:.1f} "
              f"fy={cal.fy:.1f} dist[:3]={cal.dist_coeffs[:3]})")
        return cal
    print(f"[reprocess] WARNING: no stored calibration for "
          f"{serial}/{resolution}; falling back to mission_meta.json "
          f"intrinsics with zero distortion. Branch picks may differ "
          f"slightly from the live recording.")
    fx = float(c.get("fx", 940.0))
    fy = float(c.get("fy", 940.0))
    cx = float(c.get("cx", image_size[0] / 2))
    cy = float(c.get("cy", image_size[1] / 2))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.zeros(5)
    return Calibration(
        camera_matrix=K, dist_coeffs=D,
        image_size=tuple(int(s) for s in image_size),
        serial=serial, resolution=resolution)


def _load_csv_index(csv_path: Path) -> list:
    """Return list of dicts indexed by row, with monotonic + tel_yaw."""
    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            try:
                t = float(r["monotonic"])
            except (KeyError, ValueError, TypeError):
                continue
            tel_yaw = None
            try:
                v = r.get("tel_yaw")
                if v not in (None, ""):
                    tel_yaw = float(v)
            except (TypeError, ValueError):
                tel_yaw = None
            rows.append({"t": t, "tel_yaw": tel_yaw})
    return rows


def _tel_yaw_at(rows: list, t: float) -> Optional[float]:
    """Look up the closest CSV row's tel_yaw at flight time t."""
    if not rows:
        return None
    # rows are time-ordered; binary search would be faster but the
    # row count (~hundreds) doesn't justify the complexity.
    best = min(rows, key=lambda r: abs(r["t"] - t))
    return best["tel_yaw"]


def _open_writer(path: Path, fps: float, size: tuple) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, float(fps), size)
    if not w.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed for {path}")
    return w


def reprocess(flight_dir: Path,
              arena_cfg: ArenaConfig,
              mag_offset: Optional[float],
              output_dir: Path) -> dict:
    raw = flight_dir / "raw.mp4"
    if not raw.exists():
        raise FileNotFoundError(raw)
    csv_path = flight_dir / "flight_log.csv"
    meta = json.loads((flight_dir / "mission_meta.json").read_text())

    cap = cv2.VideoCapture(str(raw))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {raw}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[reprocess] {raw.name}: {width}x{height} @ {fps:.2f} fps, "
          f"{n_frames} frames")

    cal = _load_calibration(meta, image_size=(width, height))
    cfg = meta.get("config") or {}
    marker_size_m = float(cfg.get("marker_size_m", 0.18))
    aruco_dict = str(cfg.get("aruco_dict", "DICT_4X4_50"))
    detector = ArucoDetector(cal, marker_size_m=marker_size_m,
                              dict_name=aruco_dict)

    rows = _load_csv_index(csv_path)
    if rows:
        t0_csv = rows[0]["t"]
        print(f"[reprocess] csv: {len(rows)} rows, span "
              f"{rows[-1]['t'] - t0_csv:.2f}s, t0_mono={t0_csv:.3f}")
    else:
        t0_csv = 0.0
        print("[reprocess] csv: empty (no tel_yaw lookups)")

    # Two arena configs: one without the offset (OLD path), one with
    # it set (NEW path). When no mag_offset is supplied, NEW falls
    # back to the arena config's existing value (which may be None
    # for legacy arenas, in which case the two videos are identical).
    arena_old = ArenaConfig.from_dict({
        **arena_cfg.to_json_dict(),
        "magnetic_north_arena_yaw_deg": None,
    })
    new_offset = (mag_offset
                  if mag_offset is not None
                  else arena_cfg.magnetic_north_arena_yaw_deg)
    if new_offset is None:
        print("[reprocess] WARNING: no magnetic offset set "
              "(arena config has none and --mag-offset omitted). "
              "OLD and NEW will be identical.")
    arena_new = ArenaConfig.from_dict({
        **arena_cfg.to_json_dict(),
        "magnetic_north_arena_yaw_deg":
            (None if new_offset is None else float(new_offset)),
    })

    out_old_path = output_dir / "reprocessed_old.mp4"
    out_new_path = output_dir / "reprocessed_new.mp4"
    writer_old = _open_writer(out_old_path, fps, (width, height))
    writer_new = _open_writer(out_new_path, fps, (width, height))

    # Per-stream prev-anchor state (independent for OLD and NEW so we
    # don't leak between them).
    import time
    prev_old = None; prev_old_t = 0.0
    prev_new = None; prev_new_t = 0.0
    sim_t = 0.0
    sim_step = 1.0 / max(1.0, fps)

    counts = {"frames_processed": 0,
              "old_with_fix": 0, "new_with_fix": 0,
              "new_used_mag_swap": 0}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        # Map frame index -> CSV time. The CSV is recorded by
        # log_worker at 5 Hz, the video at ~14 fps; we use frame
        # index / fps as the local time and look up the closest
        # CSV row by absolute monotonic.
        frame_t = t0_csv + frame_idx * sim_step
        tel_yaw = _tel_yaw_at(rows, frame_t)

        poses = detector.detect(frame, wanted_id=None)

        # OLD: no magnetometer (arena has no offset, no tel_yaw passed).
        prev_age_old = (sim_t - prev_old_t) if prev_old_t > 0 else None
        est_old = estimate_position(
            arena_old, poses,
            prev_position_m=prev_old,
            prev_age_s=prev_age_old,
            tel_yaw_deg=None)
        # NEW: full magnetometer logic.
        prev_age_new = (sim_t - prev_new_t) if prev_new_t > 0 else None
        est_new = estimate_position(
            arena_new, poses,
            prev_position_m=prev_new,
            prev_age_s=prev_age_new,
            tel_yaw_deg=tel_yaw)

        # Sticky carry-forward: if est is None, prev stays the same
        # (matches vision_worker's sticky behaviour).
        if est_old is not None:
            prev_old = tuple(float(c) for c in est_old.position_m)
            prev_old_t = sim_t
            counts["old_with_fix"] += 1
        if est_new is not None:
            prev_new = tuple(float(c) for c in est_new.position_m)
            prev_new_t = sim_t
            counts["new_with_fix"] += 1
            for v in est_new.per_marker_method.values():
                if v == "ippe_mag_swap":
                    counts["new_used_mag_swap"] += 1
                    break

        # Compute per-pipeline drone yaw from the corresponding
        # world position so the mini-map arrow tracks each branch's
        # interpretation. If the magnetometer pick swaps the branch
        # in NEW but not OLD the two arrows can disagree by 20-60
        # deg -- which is the visual we want to compare.
        ay_old = _arena_yaw_from(prev_old, poses, arena_old)
        ay_new = _arena_yaw_from(prev_new, poses, arena_new)

        seen_ids = [int(p.marker_id) for p in poses]

        # OLD frame.
        ann_old = annotate_frame(frame, poses)
        draw_arena_minimap(
            ann_old,
            arena_width_m=arena_cfg.width_m,
            arena_depth_m=arena_cfg.depth_m,
            world_pos=prev_old,
            arena_yaw_deg=ay_old,
            markers=arena_cfg.markers,
            visible_marker_ids=seen_ids,
            title="OLD")
        writer_old.write(ann_old)

        # NEW frame.
        ann_new = annotate_frame(frame, poses)
        draw_arena_minimap(
            ann_new,
            arena_width_m=arena_cfg.width_m,
            arena_depth_m=arena_cfg.depth_m,
            world_pos=prev_new,
            arena_yaw_deg=ay_new,
            markers=arena_cfg.markers,
            visible_marker_ids=seen_ids,
            title=("NEW" if new_offset is not None else "NEW (no offset)"))
        writer_new.write(ann_new)

        counts["frames_processed"] += 1
        sim_t += sim_step
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  ... {frame_idx} / {n_frames}")

    cap.release()
    writer_old.release()
    writer_new.release()
    return {
        "out_old": out_old_path, "out_new": out_new_path,
        **counts,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Re-run marker_mission's arena estimator over a "
                    "recorded flight's raw.mp4 and produce two annotated "
                    "comparison videos (OLD vs NEW position calc).")
    p.add_argument("--flight-dir", required=True, type=Path)
    p.add_argument("--arena", type=Path, default=None,
                   help="Arena config JSON. Defaults to the active "
                        "arena under ~/.marker_mission/.")
    p.add_argument("--mag-offset", type=float, default=None,
                   help="Magnetic-north arena yaw in degrees, "
                        "overriding whatever the arena config has.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write reprocessed_old.mp4 / "
                        "reprocessed_new.mp4. Defaults to FLIGHT_DIR.")
    args = p.parse_args()

    if not args.flight_dir.exists():
        print(f"flight dir not found: {args.flight_dir}", file=sys.stderr)
        return 2
    if args.arena is not None:
        arena_cfg = ArenaConfig.load(args.arena)
    else:
        arena_cfg = load_priority_arena()
    output_dir = args.output_dir or args.flight_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    info = reprocess(args.flight_dir, arena_cfg, args.mag_offset,
                      output_dir)
    print()
    print("[reprocess] DONE")
    print(f"  frames processed:  {info['frames_processed']}")
    print(f"  OLD frames w/ fix: {info['old_with_fix']}")
    print(f"  NEW frames w/ fix: {info['new_with_fix']}")
    print(f"  NEW frames using ippe_mag_swap: {info['new_used_mag_swap']}")
    print(f"  -> {info['out_old']}")
    print(f"  -> {info['out_new']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
