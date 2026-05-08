"""
Offline Kalman-filter replay over a recorded flight log.

Reads ``flight_log.csv`` row-by-row, runs a 6-state CV Kalman filter
fed by:

* the per-tick aggregate ArUco position (``world_x/y/z``) -- treated
  as a position measurement when fresh (a different value than the
  previous row), skipped on stale carry-forward ticks;
* the Anafi-reported NED velocity (``tel_vgx/vgy/vgz``) rotated into
  the arena frame using ``magnetic_north_arena_yaw_deg`` from the
  per-flight ``arena_config.json`` -- treated as a velocity
  measurement on every tick where all three components are present.

Outputs (next to the input flight log, or in ``--output-dir``):

* ``flight_log_kf.csv`` -- the original CSV plus six new columns:
  ``world_x_kf, world_y_kf, world_z_kf, world_vx_kf, world_vy_kf,
  world_vz_kf``.
* ``kalman_ab.png`` (if matplotlib importable) -- six stacked subplots
  comparing raw measurement vs KF output per axis.
* stdout summary -- per-axis RMS residuals raw-vs-KF, fresh-vs-predict
  tick counts, longest marker-loss bridge.

Usage::

    python -m marker_mission.tools.replay_kalman \\
        --flight-dir <path>               # required
        [--accel-var 0.1]                 # m^2/s^4 process noise
        [--pos-meas-var 0.0025]           # ArUco R, sigma~5cm
        [--vel-meas-var 0.0025]           # IMU R, sigma~5cm/s
        [--reset-gap-s 0.5]               # KF reset on >gap dt
        [--output-dir <path>]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from ..kalman import PositionKalman, ned_velocity_to_arena


def _f(s) -> Optional[float]:
    """Lenient float coerce -- empty / None -> None."""
    if s is None or s == "" or s == "None":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _load_arena_offset(flight_dir: Path) -> Optional[float]:
    """Per-flight magnetometer offset, or None if not configured."""
    p = flight_dir / "arena_config.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    raw = d.get("magnetic_north_arena_yaw_deg")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def replay(flight_dir: Path,
           accel_var: float,
           pos_meas_var: float,
           vel_meas_var: float,
           reset_gap_s: float,
           output_dir: Path) -> dict:
    csv_path = flight_dir / "flight_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    arena_offset = _load_arena_offset(flight_dir)
    if arena_offset is None:
        print("[replay_kalman] WARNING: no magnetic_north_arena_yaw_deg "
              "in arena_config.json; IMU velocity will be skipped "
              "(KF runs as a pure position smoother).")

    kf = PositionKalman(accel_var=accel_var,
                         pos_meas_var=pos_meas_var,
                         vel_meas_var=vel_meas_var)

    # Output rows + summary stats.
    out_rows: list[dict] = []
    fresh_pos_ticks = 0
    pos_meas_ticks = 0
    vel_meas_ticks = 0
    predict_only_ticks = 0
    longest_predict_run = 0.0
    cur_predict_run_start: Optional[float] = None

    # Residual buffers for RMS reporting.
    pos_resid: dict[str, list[float]] = {"x": [], "y": [], "z": []}
    vel_resid: dict[str, list[float]] = {"x": [], "y": [], "z": []}

    # For "fresh measurement" detection.
    prev_meas: Optional[tuple[float, float, float]] = None
    prev_t: Optional[float] = None

    # Plot buffers.
    t_log: list[float] = []
    raw_p: list[tuple[Optional[float], Optional[float], Optional[float]]] = []
    raw_v: list[tuple[Optional[float], Optional[float], Optional[float]]] = []
    kf_p: list[tuple[Optional[float], Optional[float], Optional[float]]] = []
    kf_v: list[tuple[Optional[float], Optional[float], Optional[float]]] = []

    # Read the input CSV and stream rows.
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        new_cols = ["world_x_kf", "world_y_kf", "world_z_kf",
                    "world_vx_kf", "world_vy_kf", "world_vz_kf"]
        for n in new_cols:
            if n not in fieldnames:
                fieldnames.append(n)

        for row in reader:
            t = _f(row.get("monotonic"))
            if t is None:
                # Pass through with empty KF cols if monotonic missing.
                for n in new_cols:
                    row.setdefault(n, "")
                out_rows.append(row)
                continue

            # dt and reset.
            dt = (t - prev_t) if prev_t is not None else 0.0
            if dt > reset_gap_s and prev_t is not None:
                kf.reset()
                prev_meas = None
            prev_t = t

            # Predict step.
            if dt > 0.0 and kf.initialised:
                kf.predict(dt)

            # Position measurement (fresh-detection by value change).
            wx = _f(row.get("world_x"))
            wy = _f(row.get("world_y"))
            wz = _f(row.get("world_z"))
            cur_meas = ((wx, wy, wz)
                        if wx is not None and wy is not None and wz is not None
                        else None)
            fresh = (cur_meas is not None
                     and (prev_meas is None or cur_meas != prev_meas))

            if cur_meas is not None and fresh:
                kf.update_position(np.array(cur_meas, dtype=float))
                fresh_pos_ticks += 1
                pos_meas_ticks += 1
                # Reset the predict-only run tracker.
                if cur_predict_run_start is not None:
                    cur_predict_run_start = None
            elif cur_meas is not None and kf.initialised:
                # Stale carry-forward; KF predicts without measurement.
                if cur_predict_run_start is None:
                    cur_predict_run_start = t
                else:
                    longest_predict_run = max(
                        longest_predict_run, t - cur_predict_run_start)
                predict_only_ticks += 1
            prev_meas = cur_meas

            # Velocity measurement -- IMU NED -> arena frame.
            # Anafi's vgx/vgy/vgz come out of Olympe SpeedChanged in cm/s
            # (verified by magnitude: indoor flights peak at ~50cm/s in
            # the log; m/s would mean 50 m/s = 180 km/h, absurd).
            arena_vel: Optional[np.ndarray] = None
            vN = _f(row.get("tel_vgx"))
            vE = _f(row.get("tel_vgy"))
            vD = _f(row.get("tel_vgz"))
            if (arena_offset is not None
                    and vN is not None and vE is not None and vD is not None):
                arena_vel = ned_velocity_to_arena(
                    vN / 100.0, vE / 100.0, vD / 100.0, arena_offset)
                if kf.initialised:
                    kf.update_velocity(arena_vel)
                    vel_meas_ticks += 1

            # Residuals (after the updates -- innovations would be
            # pre-update, but post-update residuals are what the
            # operator sees as "raw vs KF").
            kf_pos = kf.position()
            kf_vel = kf.velocity()
            if kf_pos is not None and cur_meas is not None:
                for i, axis in enumerate("xyz"):
                    pos_resid[axis].append(float(cur_meas[i]) - float(kf_pos[i]))
            if kf_vel is not None and arena_vel is not None:
                for i, axis in enumerate("xyz"):
                    vel_resid[axis].append(float(arena_vel[i]) - float(kf_vel[i]))

            # Write output row.
            if kf_pos is not None:
                row["world_x_kf"] = f"{kf_pos[0]:.4f}"
                row["world_y_kf"] = f"{kf_pos[1]:.4f}"
                row["world_z_kf"] = f"{kf_pos[2]:.4f}"
            else:
                row["world_x_kf"] = ""
                row["world_y_kf"] = ""
                row["world_z_kf"] = ""
            if kf_vel is not None:
                row["world_vx_kf"] = f"{kf_vel[0]:.4f}"
                row["world_vy_kf"] = f"{kf_vel[1]:.4f}"
                row["world_vz_kf"] = f"{kf_vel[2]:.4f}"
            else:
                row["world_vx_kf"] = ""
                row["world_vy_kf"] = ""
                row["world_vz_kf"] = ""
            out_rows.append(row)

            # Plot buffers (relative time).
            t_log.append(t)
            raw_p.append(cur_meas if fresh else (None, None, None))
            raw_v.append(tuple(arena_vel.tolist())
                         if arena_vel is not None else (None, None, None))
            kf_p.append(tuple(kf_pos.tolist()) if kf_pos is not None
                         else (None, None, None))
            kf_v.append(tuple(kf_vel.tolist()) if kf_vel is not None
                         else (None, None, None))

    # Write output CSV.
    out_csv = output_dir / "flight_log_kf.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    # Plot.
    plot_path = output_dir / "kalman_ab.png"
    try:
        _plot_ab(t_log, raw_p, kf_p, raw_v, kf_v, plot_path)
        plot_ok = True
    except Exception as e:
        print(f"[replay_kalman] WARNING: plot generation failed ({e}); "
              f"skipping {plot_path.name}.")
        plot_ok = False

    # Summary stats.
    def _rms(xs: list[float]) -> float:
        if not xs:
            return float("nan")
        return float(math.sqrt(sum(x * x for x in xs) / len(xs)))

    info = {
        "rows": len(out_rows),
        "fresh_position_meas": fresh_pos_ticks,
        "velocity_meas": vel_meas_ticks,
        "predict_only_ticks": predict_only_ticks,
        "longest_predict_run_s": longest_predict_run,
        "out_csv": str(out_csv),
        "out_plot": str(plot_path) if plot_ok else None,
        "pos_rms_x": _rms(pos_resid["x"]),
        "pos_rms_y": _rms(pos_resid["y"]),
        "pos_rms_z": _rms(pos_resid["z"]),
        "vel_rms_x": _rms(vel_resid["x"]),
        "vel_rms_y": _rms(vel_resid["y"]),
        "vel_rms_z": _rms(vel_resid["z"]),
    }
    return info


def _plot_ab(t_log, raw_p, kf_p, raw_v, kf_v, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not t_log:
        return
    t0 = t_log[0]
    ts = [t - t0 for t in t_log]

    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    labels_p = ["world x (m)", "world y (m)", "world z (m)"]
    labels_v = ["vx arena (m/s)", "vy arena (m/s)", "vz arena (m/s)"]

    for i in range(3):
        # Position: raw as dots (only at fresh ticks), KF as line.
        raw = [p[i] if p[i] is not None else float("nan") for p in raw_p]
        kf = [p[i] if p[i] is not None else float("nan") for p in kf_p]
        axes[i].plot(ts, raw, "o", color="tab:orange", markersize=2,
                      label="raw (fresh)", alpha=0.6)
        axes[i].plot(ts, kf, "-", color="tab:blue", linewidth=1,
                      label="KF")
        axes[i].set_ylabel(labels_p[i])
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc="upper right", fontsize=8)

    for i in range(3):
        raw = [v[i] if v[i] is not None else float("nan") for v in raw_v]
        kf = [v[i] if v[i] is not None else float("nan") for v in kf_v]
        axes[3 + i].plot(ts, raw, ".", color="tab:orange", markersize=2,
                          label="IMU", alpha=0.6)
        axes[3 + i].plot(ts, kf, "-", color="tab:blue", linewidth=1,
                          label="KF")
        axes[3 + i].set_ylabel(labels_v[i])
        axes[3 + i].grid(True, alpha=0.3)
        if i == 0:
            axes[3 + i].legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("flight time (s)")
    fig.suptitle("Kalman A/B  --  raw measurements vs filtered output",
                 y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Offline Kalman-filter replay over a recorded "
                    "flight log; produces an OLD/NEW comparison "
                    "without re-detecting markers.")
    p.add_argument("--flight-dir", required=True, type=Path)
    p.add_argument("--accel-var", type=float, default=0.1,
                   help="Process noise variance (m^2/s^4).")
    p.add_argument("--pos-meas-var", type=float, default=0.0025,
                   help="Position measurement variance (m^2). "
                        "Default 0.0025 -> sigma=5cm.")
    p.add_argument("--vel-meas-var", type=float, default=0.0025,
                   help="Velocity measurement variance (m^2/s^2). "
                        "Default 0.0025 -> sigma=5cm/s.")
    p.add_argument("--reset-gap-s", type=float, default=0.5,
                   help="KF reset on dt larger than this (s).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Defaults to --flight-dir.")
    args = p.parse_args()

    if not args.flight_dir.exists():
        print(f"flight dir not found: {args.flight_dir}", file=sys.stderr)
        return 2
    output_dir = args.output_dir or args.flight_dir

    info = replay(flight_dir=args.flight_dir,
                  accel_var=args.accel_var,
                  pos_meas_var=args.pos_meas_var,
                  vel_meas_var=args.vel_meas_var,
                  reset_gap_s=args.reset_gap_s,
                  output_dir=output_dir)

    print("")
    print("[replay_kalman] DONE")
    print(f"  rows                : {info['rows']}")
    print(f"  fresh pos meas      : {info['fresh_position_meas']}")
    print(f"  velocity meas       : {info['velocity_meas']}")
    print(f"  predict-only ticks  : {info['predict_only_ticks']}")
    print(f"  longest predict run : {info['longest_predict_run_s']:.2f} s")
    print(f"  pos residual RMS    : x={info['pos_rms_x']:.3f}  "
          f"y={info['pos_rms_y']:.3f}  z={info['pos_rms_z']:.3f} m")
    print(f"  vel residual RMS    : x={info['vel_rms_x']:.3f}  "
          f"y={info['vel_rms_y']:.3f}  z={info['vel_rms_z']:.3f} m/s")
    print(f"  -> {info['out_csv']}")
    if info["out_plot"]:
        print(f"  -> {info['out_plot']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
