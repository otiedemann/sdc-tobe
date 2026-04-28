"""
Central configuration for the marker-approach mission.

Every tuning constant has a `# TUNE` comment so that a `grep TUNE config.py`
returns the full list of values that you should verify on the actual
hardware before flying.

Most parameters can be overridden via environment variables (prefix
``MM_``) or via a JSON file at ``$MM_CONFIG_PATH`` (default
``~/.marker_mission/config.json``). CLI flags in ``mission.py`` override
both.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(os.environ.get("MM_DATA_DIR",
                                       Path.home() / ".marker_mission"))
CALIB_DIR = DEFAULT_DATA_DIR / "calibrations"
FLIGHTS_DIR = DEFAULT_DATA_DIR / "flights"


# ---------------------------------------------------------------------------
# Mission parameters
# ---------------------------------------------------------------------------

@dataclass
class MissionConfig:
    # --- Marker geometry -----------------------------------------------------
    target_marker_id: int = 4              # ArUco ID of the central marker
    marker_size_m: float = 0.18            # physical side length [m]
    aruco_dict: str = "DICT_4X4_50"        # OpenCV dict name

    # --- Mission goals -------------------------------------------------------
    target_distance_m: float = 1.0         # final standoff distance from marker [m]
    target_relative_heading_deg: float = 90.0  # final bearing around marker [deg]
    hold_time_s: float = 60.0              # hover duration after arriving

    # --- Control loop --------------------------------------------------------
    control_rate_hz: float = 10.0          # PD update rate
    rc_command_duration_ms: int = 300      # >= 1 / control_rate_hz to avoid gaps

    # Dead-bands -- inside these, the controller commands zero ----------------
    yaw_deadband_deg: float = 1.5          # TUNE
    distance_deadband_m: float = 0.05      # TUNE
    heading_deadband_deg: float = 2.0      # TUNE
    lateral_deadband_m: float = 0.05       # TUNE

    # PD gains ---------------------------------------------------------------
    # Output is the RC channel value in [-100, +100]. Inputs are in
    # SI / degrees. Start small; raise until the drone tracks crisply
    # without overshoot.
    yaw_kp: float = 2.0                    # TUNE  RC counts per deg
    yaw_kd: float = 0.5                    # TUNE  RC counts per (deg/s)
    # NOTE: fwd / lat gains were lowered after the first-flight wall crash.
    # The drone saturated rc_fb=+30 for the entire long-range approach,
    # accelerated to ~120 cm/s, and started braking too late. The new caps
    # limit cruise to roughly 40 cm/s and the new k_v terms damp against
    # measured body velocity (vgy / vgx in cm/s) so the controller hits the
    # brakes before distance error closes.
    fwd_kp: float = 30.0                   # TUNE  RC counts per m   (was 60)
    fwd_kd: float = 30.0                   # TUNE  RC counts per (m/s)
    fwd_kv: float = 0.4                    # TUNE  RC counts per (cm/s); damps vgy
    lat_kp: float = 30.0                   # TUNE  RC counts per m   (was 60)
    lat_kd: float = 30.0                   # TUNE  RC counts per (m/s)
    lat_kv: float = 0.4                    # TUNE  RC counts per (cm/s); damps vgx

    # Output clamps for safety. Lower = slower & safer. -----------------------
    yaw_rc_max: int = 40                   # TUNE
    fwd_rc_max: int = 10                   # TUNE  (was 30; caps fwd speed ~40 cm/s)
    lat_rc_max: int = 10                   # TUNE  (was 30; matches fwd cap)
    ud_rc_max: int = 25                    # TUNE  (we do not actively control altitude here)

    # Hard distance floor: refuse forward commands inside this multiple of the
    # target standoff. Last-line guard against marker-pose noise driving the
    # drone closer than intended.
    distance_floor_factor: float = 0.7     # TUNE  e.g. 0.7 * 1.0m = 0.7 m

    # --- Search behaviour ----------------------------------------------------
    # When the marker is not visible, we yaw in place looking for it. After
    # one full rotation without a sighting, we give up and land.
    search_yaw_rc: int = 25                # TUNE  RC value while sweeping
    search_total_deg: float = 360.0        # how far to sweep before giving up
    search_marker_lost_grace_s: float = 1.5  # TUNE  hold last command this long
                                              # before declaring the marker
                                              # lost (handles brief occlusions)

    # --- Phase transition criteria -------------------------------------------
    approach_settle_time_s: float = 1.0    # how long all errors must be inside
                                              # dead-bands before we consider
                                              # the phase complete
    orbit_step_deg: float = 5.0            # TUNE  bearing change per control tick
                                              # (limits how fast we orbit)

    # --- Pose smoothing ------------------------------------------------------
    pose_smoothing_alpha: float = 0.6      # TUNE  EMA factor for measurements
                                              # (1.0 = no smoothing,
                                              #  0.0 = freeze at first sample)
    pose_max_age_s: float = 0.8            # if no detection in this window we
                                              # consider the marker lost

    # --- Network -------------------------------------------------------------
    api_base_url: str = "http://127.0.0.1:5050"
    request_timeout_s: float = 2.0

    # --- UI ------------------------------------------------------------------
    ui_host: str = "0.0.0.0"
    ui_port: int = 8080
    ui_telemetry_history_s: float = 60.0   # how long to keep on the chart screen

    # --- Recording -----------------------------------------------------------
    record_fps: int = 25                   # MJPEG frames are decoded -> we re-mux
    record_jpeg_quality: int = 90          # for any direct JPEG saves

    # ------------------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "MissionConfig":
        cfg = cls()
        path = path or Path(os.environ.get("MM_CONFIG_PATH",
                                           DEFAULT_DATA_DIR / "config.json"))
        if path.exists():
            try:
                blob = json.loads(path.read_text())
                for k, v in blob.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
            except Exception as e:
                print(f"[config] could not load {path}: {e}")
        # Environment overrides win
        for f in cfg.__dataclass_fields__:
            ev = os.environ.get(f"MM_{f.upper()}")
            if ev is None:
                continue
            try:
                cur = getattr(cfg, f)
                # cast based on current type
                if isinstance(cur, bool):
                    setattr(cfg, f, ev.lower() in ("1", "true", "yes", "on"))
                elif isinstance(cur, int):
                    setattr(cfg, f, int(ev))
                elif isinstance(cur, float):
                    setattr(cfg, f, float(ev))
                else:
                    setattr(cfg, f, ev)
            except Exception:
                pass
        return cfg

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or Path(os.environ.get("MM_CONFIG_PATH",
                                           DEFAULT_DATA_DIR / "config.json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path
