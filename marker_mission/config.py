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
    # Yaw gains lowered after first tethered test showed a saturation-driven
    # limit cycle: with kp=2.0 and rc_max=40, P-saturated at |error|>20°
    # while the drone was repeatedly seeing 20-26° errors, pinning the
    # command at +/-40 and bang-banging at ~1.7 Hz (loop delay ~300 ms).
    # New values move the saturation threshold to 40° (past observed peak)
    # and slow the maximum yaw rate, giving the D-term room to act.
    yaw_kp: float = 1.0                    # TUNE  RC counts per deg   (was 2.0)
    yaw_kd: float = 0.5                    # TUNE  RC counts per (deg/s)
    # NOTE: fwd / lat gains were lowered after the first-flight wall crash.
    # The drone saturated rc_fb=+30 for the entire long-range approach,
    # accelerated to ~120 cm/s, and started braking too late. The new caps
    # limit cruise to roughly 40 cm/s and the new k_v terms damp against
    # measured body velocity (vgy / vgx in cm/s) so the controller hits the
    # brakes before distance error closes.
    fwd_kp: float = 30.0                   # TUNE  RC counts per m   (was 60)
    fwd_kd: float = 30.0                   # TUNE  RC counts per (m/s)
    # The damping helpers in controller.py project the world-NED
    # velocities (vgx/vgy from Olympe SpeedChanged) onto the body
    # frame using the live drone yaw, so these gains operate on
    # body-forward / body-right velocity in cm/s. Verified against
    # the steady-forward approach in flight 21-22-03 (vgx=-2.2,
    # vgy=+11.7, yaw=109.9 -> v_body_fwd=+11.8, positive as expected
    # while distance was closing). Without damping that flight
    # built up ~40 cm/s on the cruise and crashed into the marker
    # because PD alone only brakes inside the deadband -- too late.
    fwd_kv: float = 0.4                    # TUNE  RC counts per (cm/s)
    lat_kp: float = 30.0                   # TUNE  RC counts per m   (was 60)
    lat_kd: float = 30.0                   # TUNE  RC counts per (m/s)
    lat_kv: float = 0.4                    # TUNE  RC counts per (cm/s)

    # Output clamps for safety. Lower = slower & safer. -----------------------
    yaw_rc_max: int = 25                   # TUNE  (was 40; caps yaw rate ~25°/s)
    fwd_rc_max: int = 10                   # TUNE  (was 30; caps fwd speed ~40 cm/s)
    # Orbit needs lateral motion slow enough for yaw to keep tracking the
    # marker -- otherwise yaw lags, the marker drifts off-axis, and the
    # commanded "sideways" leaks into radial drift. First orbit attempt at
    # rc=10 spiraled outward until the marker fell out of FOV. With rc=4
    # the required yaw rate (~v/d) drops well below yaw_rc_max and v^2/r
    # centrifugal drift is ~6x smaller.
    lat_rc_max: int = 4                    # TUNE  (was 10; orbit-stability limited)
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
    align_settle_time_s: float = 1.0       # TUNE same idea, for the ALIGN phase
                                              # (orbit-to-heading-0 before
                                              # closing distance) -- the marker
                                              # tilts past the detector's limit
                                              # if approach starts at high
                                              # heading
    # ALIGN's job is just to keep the marker out of the detector's
    # oblique-angle dead zone before APPROACH closes distance -- it
    # does NOT need to centre heading precisely (the ORBIT phase
    # later does that with its tighter 2 deg deadband at close
    # range). At ~3 m range the per-frame heading estimate jitters
    # +/-15-20 deg, so a tight ALIGN threshold makes the phase spin
    # forever without improving anything; +/-30 deg lets ALIGN
    # settle as soon as the drone is roughly facing the marker and
    # hands off to APPROACH.
    align_heading_deadband_deg: float = 30.0  # TUNE
    # APPROACH uses its own, *much* tighter heading deadband. Reusing
    # ALIGN's +/-30 deg here turned out to give the drone 60 deg of
    # "free space" inside which the lateral PD never fires; the
    # relative-heading then drifts CCW (driven by an arcing
    # yaw-plus-forward coupling) until it exits the band already
    # carrying tangential momentum, and overshoots. Flight 21-42-17:
    # heading went +15 -> -30 over 6 s with rc_lr=0 the whole time,
    # and only when |e_hdg|>30 did PD wake up -- by then the drone
    # was at -30 with momentum and ended at -45. A small deadband
    # here lets PD fight the drift early, while ALIGN keeps its
    # looser threshold for noise tolerance at long range.
    approach_heading_deadband_deg: float = 5.0  # TUNE
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
