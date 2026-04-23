"""Offline smoke test: replay a recorded flight video through the
HeadlessAruCoPositioning processor under the "Claude" / Auto Positioning
config, and compare the resulting pose trace to the pose recorded in the
matching .jsonl flight log.

Usage:
    python tools/smoke_test_claude_positioner.py <video.mp4> <log.jsonl>
    python tools/smoke_test_claude_positioner.py --all  # process all three known logs

The script:
  * loads camera calibration from controller_unified/position_calib.npz
  * loads arena markers from controller_unified/arena_config.json
  * constructs a HeadlessAruCoPositioning with the same params as the
    live Auto Positioning mode (CLAUDE_AUTO_CONFIG) including the new
    max_pose_jump_m=3.0 gate
  * walks every frame in the video, calls process_frame(), and records:
      - number of markers seen
      - ref_markers (from fusion)
      - fresh/stale pose
      - camera position in arena frame
  * prints summary stats and writes a CSV trace next to the video

No side-effects on the running FC. Pure offline analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "controller_unified"))

import ctrl_position as _cp  # noqa: E402 — path tweak above

CLAUDE_AUTO_CONFIG: dict = {
    "detect_profile":       "balanced",
    "imu_weight":           0.30,
    "enable_kalman_filter": True,
    "marker_size_m":        0.50,
    "top_k_markers":        4,
    "outlier_reject_m":     1.5,
    "distance_scale":       1.0,
    "pose_hold_sec":        0.5,
    "min_ref_count":        1,
    "min_ref_weight":       0.15,
    "meas_blend_min":       0.20,
    "meas_blend_max":       0.70,
    "vel_blend":            0.30,
    "max_state_dt":         0.5,
    "kalman_process_var":   5e-4,
    "kalman_meas_var":      0.15,
    "imu_lowpass_hz":       5.0,
    "seen_hold_s":          0.6,
    "max_pose_jump_m":      3.0,
}

CALIB_PATH = REPO / "controller_unified" / "position_calib.npz"
ARENA_PATH = REPO / "controller_unified" / "arena_config.json"


def _apply_claude_module_globals(cfg: dict) -> None:
    """Patch the ctrl_position module globals the same way the FC does."""
    _cp.POSE_HOLD_SEC = float(cfg["pose_hold_sec"])
    _cp.MIN_REF_COUNT = int(cfg["min_ref_count"])
    _cp.MIN_REF_WEIGHT = float(cfg["min_ref_weight"])
    _cp.MEAS_BLEND_MIN = float(cfg["meas_blend_min"])
    _cp.MEAS_BLEND_MAX = float(cfg["meas_blend_max"])
    _cp.VEL_BLEND = float(cfg["vel_blend"])
    _cp.MAX_STATE_DT = float(cfg["max_state_dt"])


def _build_processor(cfg: dict):
    data = np.load(str(CALIB_PATH))
    cam_mat = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    _apply_claude_module_globals(cfg)
    proc = _cp.HeadlessAruCoPositioning(
        cam_mat, dist,
        detect_profile=cfg["detect_profile"],
        marker_size=cfg["marker_size_m"],
        enable_kalman_filter=cfg["enable_kalman_filter"],
        imu_weight=cfg["imu_weight"],
    )
    # Apply the arena override (marker positions + wall types)
    arena = json.loads(ARENA_PATH.read_text())
    positions = {}
    walls = {}
    for m in arena.get("markers", []):
        mid = int(m["id"])
        positions[mid] = np.array([float(m["x"]), float(m["y"]), float(m["z"])])
        walls[mid] = m.get("wall", "front")
    proc.marker_positions = positions
    proc.marker_wall_type = walls
    proc.marker_size = float(cfg["marker_size_m"])
    half = proc.marker_size / 2.0
    proc.MARKER_3D_POINTS = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)
    # Runtime-tunable processor fields
    proc.top_k_markers = int(cfg["top_k_markers"]) or 4
    proc.outlier_reject_m = float(cfg["outlier_reject_m"])
    proc.distance_scale = float(cfg["distance_scale"])
    proc.max_pose_jump_m = float(cfg["max_pose_jump_m"])
    for kf in proc.kf_pos:
        kf.process_variance = float(cfg["kalman_process_var"])
        kf.measurement_variance = float(cfg["kalman_meas_var"])
    return proc


def _load_log_positions(log_path: Path) -> list[dict]:
    """Return [{ts, pos, stale, visible_markers, ref_markers, seen_markers}, ...]
    from the flight log (ticks only)."""
    out = []
    for line in log_path.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") != "tick":
            continue
        out.append({
            "ts": r.get("ts"),
            "pos": r.get("position"),
            "stale": r.get("pos_stale"),
            "visible": r.get("visible_markers") or [],
            "ref": r.get("ref_markers") or [],
            "seen": r.get("seen_markers") or [],
        })
    return out


def smoke_test(video_path: Path, log_path: Path | None) -> None:
    cfg = dict(CLAUDE_AUTO_CONFIG)
    proc = _build_processor(cfg)

    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    print(f"\n=== SMOKE TEST: {video_path.name} ===")
    print(f"    {n} frames @ {fps:.2f} fps  (Claude preset)")

    results = []   # per-frame dict of {ts, seen_count, n_refs, fresh, pos}
    t0 = time.time()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Synthetic monotonic ts at video fps (relative to arbitrary epoch)
        ts_rel = frame_idx / fps
        # processor.process_frame expects wall-clock-like timestamps; give
        # it a monotonic-like base so Kalman state advances sensibly.
        result = proc.process_frame(
            frame,
            frame_ts=ts_rel,
            latency_s=0.20,  # default latency_comp_s
            now_ts=ts_rel,
        )
        if result is None:
            row = {"frame": frame_idx, "ts": ts_rel,
                   "seen_count": 0, "n_refs": 0, "fresh": False,
                   "pos": None, "stale": True}
        else:
            row = {
                "frame": frame_idx,
                "ts": ts_rel,
                "seen_count": int(result.get("seen_count", 0)),
                "n_refs": len(result.get("ref_markers", [])),
                "fresh": not bool(result.get("stale", True)),
                "pos": result.get("cam"),
                "stale": bool(result.get("stale", True)),
            }
        results.append(row)
        frame_idx += 1

    cap.release()
    wall = time.time() - t0
    print(f"    processed {frame_idx} frames in {wall:.1f}s "
          f"({frame_idx/max(wall,0.001):.1f} fps effective)")

    n_fresh = sum(1 for r in results if r["fresh"])
    n_stale = sum(1 for r in results if not r["fresh"])
    n_markers = sum(1 for r in results if r["seen_count"] > 0)
    avg_markers = (sum(r["seen_count"] for r in results) /
                   max(1, len(results)))
    print(f"    frames with markers detected: {n_markers} / {frame_idx}  "
          f"(avg {avg_markers:.2f}/frame)")
    print(f"    fresh fixes: {n_fresh} ({n_fresh/max(1,len(results))*100:.1f}%)"
          f"  stale: {n_stale}")

    # Jump analysis on fresh-to-fresh transitions
    jumps = []
    prev = None
    for r in results:
        pos = r["pos"]
        if pos is not None and r["fresh"]:
            if prev is not None:
                d = sum((pos[i]-prev[i])**2 for i in range(3))**0.5
                jumps.append(d)
            prev = pos
        elif not r["fresh"]:
            prev = None
    if jumps:
        jumps_sorted = sorted(jumps)
        big_jumps = [d for d in jumps if d > 1.0]
        print(f"    fresh-to-fresh 3D jumps: n={len(jumps)} "
              f"mean={statistics.mean(jumps):.3f}m "
              f"median={statistics.median(jumps):.3f}m "
              f"p95={jumps_sorted[int(len(jumps)*0.95)]:.3f}m "
              f"max={max(jumps):.3f}m")
        print(f"    jumps > 1m: {len(big_jumps)} "
              f"({len(big_jumps)/len(jumps)*100:.2f}%)")

    # Compare to log if present
    if log_path is not None and log_path.exists():
        logs = _load_log_positions(log_path)
        log_fresh = sum(1 for r in logs if r["stale"] is False)
        log_markers = sum(1 for r in logs if r["seen"])
        print(f"    log baseline: {len(logs)} ticks, "
              f"fresh={log_fresh} ({log_fresh/max(1,len(logs))*100:.1f}%), "
              f"non-empty seen={log_markers}")

    # Dump CSV next to the video
    csv_path = video_path.with_suffix(".claude_trace.csv")
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["frame", "ts", "seen_count", "n_refs", "fresh", "x", "y", "z"])
        for r in results:
            pos = r["pos"] or [None, None, None]
            w.writerow([r["frame"], f"{r['ts']:.3f}",
                        r["seen_count"], r["n_refs"],
                        int(r["fresh"]),
                        pos[0] if pos[0] is None else f"{pos[0]:.4f}",
                        pos[1] if pos[1] is None else f"{pos[1]:.4f}",
                        pos[2] if pos[2] is None else f"{pos[2]:.4f}"])
    print(f"    trace → {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("log", nargs="?")
    ap.add_argument("--all", action="store_true",
                    help="Process the three 19:11/19:32/19:39 flight pairs in Downloads")
    args = ap.parse_args()

    if args.all:
        downloads = Path.home() / "Downloads"
        stems = [
            "flight_2026-04-23_19-11-40_drone-2_Anafi_2",
            "flight_2026-04-23_19-32-29_drone-2_Anafi_2",
            "flight_2026-04-23_19-39-59_drone-2_Anafi_2",
        ]
        for stem in stems:
            mp4 = downloads / f"{stem}.mp4"
            log = downloads / f"{stem}.jsonl"
            if not mp4.exists():
                print(f"skip {stem}: mp4 missing")
                continue
            smoke_test(mp4, log if log.exists() else None)
        return

    if args.video is None:
        ap.error("pass a video path or --all")

    smoke_test(Path(args.video).expanduser(),
               Path(args.log).expanduser() if args.log else None)


if __name__ == "__main__":
    main()
