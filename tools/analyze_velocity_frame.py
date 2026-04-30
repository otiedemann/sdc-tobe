#!/usr/bin/env python3
"""
Compare two interpretations of vgx/vgy against marker-derived ground
truth, to settle the open frame-convention question:

    NED:        vgx = north-velocity, vgy = east-velocity
    BODY+yaw:   vgx = body-forward,   vgy = body-right;
                rotate by tel_yaw to get earth-frame velocity.

For each flight log we have, we (a) build a "ground-truth" trajectory
from the marker pose (drone position relative to marker, transformed
through tel_yaw into a compass-aligned world frame), and (b) integrate
the velocity log under each interpretation. Whichever integration's
trajectory matches the ground-truth shape wins.

Usage:
    python -m tools.analyze_velocity_frame <flight_dir>
or
    python tools/analyze_velocity_frame.py <flight_dir>
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def _f(row: dict, key: str) -> Optional[float]:
    v = row.get(key, "")
    if v in ("", "nan", "NaN", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def marker_relative_world_pos(distance_m: float,
                               yaw_to_marker_deg: float,
                               tel_yaw_deg: float) -> Tuple[float, float]:
    """Drone position in a compass-aligned world frame, with the marker
    fixed at origin. (east, north) in metres."""
    drone_to_marker_compass = (tel_yaw_deg + yaw_to_marker_deg) % 360.0
    marker_to_drone_compass = (drone_to_marker_compass + 180.0) % 360.0
    rad = math.radians(marker_to_drone_compass)
    east  = distance_m * math.sin(rad)
    north = distance_m * math.cos(rad)
    return (east, north)


def integrate(rows: List[dict], mode: str) -> List[Tuple[float, float, float]]:
    """Integrate vgx/vgy under one interpretation. Returns
    [(monotonic, x_east, y_north), ...] in metres."""
    out: List[Tuple[float, float, float]] = []
    x_east = 0.0
    y_north = 0.0
    prev_t: Optional[float] = None
    for r in rows:
        t = _f(r, "monotonic")
        vgx = _f(r, "tel_vgx")
        vgy = _f(r, "tel_vgy")
        yaw = _f(r, "tel_yaw")
        if t is None:
            continue
        if vgx is None or vgy is None or yaw is None:
            out.append((t, x_east, y_north))
            prev_t = t
            continue
        if prev_t is not None:
            dt = t - prev_t
            if 0.0 < dt < 1.0:
                if mode == "NED":
                    v_east, v_north = vgy, vgx
                elif mode == "BODY":
                    th = math.radians(yaw)
                    v_east  = vgx * math.sin(th) + vgy * math.cos(th)
                    v_north = vgx * math.cos(th) - vgy * math.sin(th)
                else:
                    raise ValueError(f"unknown mode {mode}")
                x_east  += (v_east  / 100.0) * dt
                y_north += (v_north / 100.0) * dt
        out.append((t, x_east, y_north))
        prev_t = t
    return out


def analyze(flight_dir: Path) -> int:
    csv_path = flight_dir / "flight_log.csv"
    if not csv_path.exists():
        print(f"error: {csv_path} not found", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        print(f"error: {csv_path} is empty", file=sys.stderr)
        return 1

    # Ground truth: marker-relative world position, computed only when
    # the marker was actually seen this tick.
    truth: List[Tuple[float, float, float]] = []
    for r in rows:
        if r.get("marker_seen") != "1":
            continue
        t = _f(r, "monotonic")
        d = _f(r, "distance_m")
        ytm = _f(r, "yaw_to_marker_deg")
        ty  = _f(r, "tel_yaw")
        if None in (t, d, ytm, ty):
            continue
        e, n = marker_relative_world_pos(d, ytm, ty)
        truth.append((t, e, n))
    if len(truth) < 5:
        print(f"warning: only {len(truth)} marker-seen samples; results "
              f"will be noisy", file=sys.stderr)
    if not truth:
        print(f"error: no marker-seen samples; can't establish ground truth",
              file=sys.stderr)
        return 1

    int_ned  = integrate(rows, "NED")
    int_body = integrate(rows, "BODY")

    # Index integrated arrays by monotonic for nearest-neighbour lookup.
    def at(arr: List[Tuple[float, float, float]], t: float
           ) -> Tuple[float, float]:
        # Binary search for nearest t.
        lo, hi = 0, len(arr) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid][0] < t:
                lo = mid + 1
            else:
                hi = mid
        i = lo
        if i > 0 and abs(arr[i-1][0] - t) < abs(arr[i][0] - t):
            i -= 1
        return (arr[i][1], arr[i][2])

    # Both ground-truth and integrated trajectories have arbitrary
    # origins; align by subtracting each's first sample at the start
    # of the comparison range.
    t0 = truth[0][0]
    gt0 = (truth[0][1], truth[0][2])
    ned0  = at(int_ned,  t0)
    body0 = at(int_body, t0)

    rms_ned_sq = 0.0
    rms_body_sq = 0.0
    samples = 0
    final_truth = (0.0, 0.0)
    final_ned   = (0.0, 0.0)
    final_body  = (0.0, 0.0)
    for (t, e_truth, n_truth) in truth:
        e_truth_d = e_truth - gt0[0]
        n_truth_d = n_truth - gt0[1]
        ne = at(int_ned, t);  ne_d = (ne[0] - ned0[0],  ne[1] - ned0[1])
        bo = at(int_body, t); bo_d = (bo[0] - body0[0], bo[1] - body0[1])
        rms_ned_sq  += (ne_d[0] - e_truth_d) ** 2 + (ne_d[1] - n_truth_d) ** 2
        rms_body_sq += (bo_d[0] - e_truth_d) ** 2 + (bo_d[1] - n_truth_d) ** 2
        samples += 1
        final_truth = (e_truth_d, n_truth_d)
        final_ned   = ne_d
        final_body  = bo_d
    rms_ned  = math.sqrt(rms_ned_sq  / samples) if samples else 0.0
    rms_body = math.sqrt(rms_body_sq / samples) if samples else 0.0
    final_err_ned  = math.hypot(final_ned[0]  - final_truth[0],
                                 final_ned[1]  - final_truth[1])
    final_err_body = math.hypot(final_body[0] - final_truth[0],
                                 final_body[1] - final_truth[1])

    print(f"flight: {flight_dir.name}")
    print(f"  duration: {rows[-1].get('monotonic', '?')}s, "
          f"rows: {len(rows)}, marker-seen: {len(truth)}")
    print(f"  ground-truth net displacement: "
          f"east={final_truth[0]:+.2f}m  north={final_truth[1]:+.2f}m  "
          f"|d|={math.hypot(*final_truth):.2f}m")
    print(f"  --- NED interpretation (vgx=vN, vgy=vE) ---")
    print(f"      final pos: east={final_ned[0]:+.2f}m  north={final_ned[1]:+.2f}m")
    print(f"      RMS error vs ground truth: {rms_ned:.2f}m")
    print(f"      final-position error:      {final_err_ned:.2f}m")
    print(f"  --- BODY interpretation (vgx=fwd, vgy=right, rotate by yaw) ---")
    print(f"      final pos: east={final_body[0]:+.2f}m  north={final_body[1]:+.2f}m")
    print(f"      RMS error vs ground truth: {rms_body:.2f}m")
    print(f"      final-position error:      {final_err_body:.2f}m")
    if rms_ned < rms_body * 0.7:
        print(f"  ===> NED interpretation matches {rms_body/max(rms_ned,1e-9):.1f}x better")
    elif rms_body < rms_ned * 0.7:
        print(f"  ===> BODY interpretation matches {rms_ned/max(rms_body,1e-9):.1f}x better")
    else:
        print(f"  ===> roughly tied; not enough signal to discriminate")
    return 0


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    rc = 0
    for arg in argv[1:]:
        rc |= analyze(Path(arg))
        print()
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
