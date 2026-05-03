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
SNAPSHOTS_DIR = DEFAULT_DATA_DIR / "snapshots"
MISSION_SCRIPTS_DIR = DEFAULT_DATA_DIR / "mission_scripts"
ACTIVE_MISSION_SCRIPT_PATH = DEFAULT_DATA_DIR / "active_mission_script.txt"
PER_FLIGHT_SCRIPT_FILENAME = "mission_script.txt"
ARENA_CONFIGS_DIR = DEFAULT_DATA_DIR / "arenas"
ACTIVE_ARENA_CONFIG_PATH = DEFAULT_DATA_DIR / "active_arena_config.json"


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
    # HOLD's heading setpoint. Default 0 means "stay where APPROACH put us"
    # (drone facing the marker straight on). Set non-zero only if you want
    # HOLD to actively slide the drone around the marker after approach.
    target_relative_heading_deg: float = 0.0  # final bearing around marker [deg]
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
    # Integral term. 0 = pure PD (default; matches all flights up to
    # 2026-04-30). Bump above zero to eliminate steady-state error
    # during HOLD; back-calculation anti-windup + i_clip protect
    # against runaway during saturation / marker loss.
    yaw_ki: float = 0.0                    # TUNE  RC counts per (deg*s)
    yaw_i_clip: float = 50.0               # TUNE  hard bound on |integral|
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
    fwd_ki: float = 0.0                    # TUNE  RC counts per (m*s)
    fwd_i_clip: float = 1.0                # TUNE  hard bound on |integral|
    lat_kp: float = 30.0                   # TUNE  RC counts per m   (was 60)
    lat_kd: float = 30.0                   # TUNE  RC counts per (m/s)
    lat_kv: float = 0.4                    # TUNE  RC counts per (cm/s)
    lat_ki: float = 0.0                    # TUNE  RC counts per (m*s)
    lat_i_clip: float = 1.0                # TUNE  hard bound on |integral|

    # Output clamps for safety. Lower = slower & safer. -----------------------
    yaw_rc_max: int = 25                   # TUNE  (was 40; caps yaw rate ~25°/s)
    # Cap the forward speed low enough that PD's D-term can brake before
    # the drone closes inside the distance floor. Operator reported
    # rc_max=5 was still approaching too fast for indoor walls; bumped
    # down to 3 for ~12 cm/s top forward speed. Approach is slower but
    # gives PD ample headroom to decelerate before the floor.
    fwd_rc_max: int = 3                    # TUNE  caps fwd speed ~12 cm/s
    # Lateral motion has to be slow enough that yaw can keep tracking
    # the marker -- otherwise yaw lags, the marker drifts off-axis, and
    # the commanded "sideways" leaks into radial drift. An early try at
    # rc=10 spiraled outward until the marker fell out of FOV. With
    # rc=4 the required yaw rate (~v/d) drops well below yaw_rc_max
    # and v^2/r centrifugal drift is ~6x smaller.
    lat_rc_max: int = 4                    # TUNE  (was 10; tangential-stability limited)
    ud_rc_max: int = 25                    # TUNE  (we do not actively control altitude here)

    # Hard distance floor: refuse forward commands inside this multiple of the
    # target standoff. Last-line guard against marker-pose noise driving the
    # drone closer than intended.
    distance_floor_factor: float = 0.7     # TUNE  e.g. 0.7 * 1.0m = 0.7 m

    # --- Altitude / height control ------------------------------------------
    # The drone climbs to default_height_m after takeoff before searching,
    # and re-aligns to the marker's altitude in the HEIGHT_ALIGN phase.
    # min/max bound the height the drone is allowed to fly at -- _send_rc
    # clamps the ud channel against these so wild PD output, residual
    # control commands, or wind can't push the drone outside the safe
    # envelope. min also caps the HEIGHT_ALIGN target for low markers
    # (drone refuses to descend below min_height even if the marker is
    # mounted on the floor).
    default_height_m: float = 1.5          # TUNE  climb-to height after takeoff
    min_height_m: float = 0.5              # TUNE  ud >= 0 below this
    max_height_m: float = 2.0              # TUNE  ud <= 0 above this
    height_kp: float = 30.0                # TUNE  RC counts per m of height error
    height_kd: float = 10.0                # TUNE  RC counts per (m/s)
    height_ki: float = 0.0                 # TUNE  RC counts per (m*s)
    height_i_clip: float = 1.0             # TUNE  hard bound on |integral|
    height_deadband_m: float = 0.10        # TUNE
    height_settle_time_s: float = 1.0      # TUNE  before HEIGHT_ALIGN -> ALIGN

    # --- Mission script -----------------------------------------------------
    # Defaults filled in when a script step omits its argument.
    default_dance_seconds_s: float = 5.0   # TUNE  DANCE step default duration
    dance_radius_m: float = 0.5            # TUNE  bound dance excursion within
                                            #       this radius of dance origin

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
    # does NOT need to centre heading precisely (APPROACH and HOLD
    # tighten it later). At ~3 m range the per-frame heading estimate
    # jitters +/-15-20 deg, so a tight ALIGN threshold makes the phase
    # spin forever without improving anything; +/-30 deg lets ALIGN
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
    # Operator-facing emergency-land hotkey. Pressing this key anywhere on
    # the web UI (including inside text fields) triggers /api/stop. Single
    # character; comparison is case-insensitive.
    killswitch_key: str = "Q"              # TUNE  global emergency-land hotkey

    # --- Recording -----------------------------------------------------------
    record_fps: int = 25                   # MJPEG frames are decoded -> we re-mux
    record_jpeg_quality: int = 90          # for any direct JPEG saves

    # ------------------------------------------------------------------------
    def update_from_dict(self, values: dict) -> dict:
        """Apply a {field: value} dict to this cfg in-place. Returns a
        {field: error_message} dict for any fields that failed to parse
        or aren't valid configuration keys. Values are coerced to the
        existing field's type so the JSON-from-the-browser road-trip
        (which delivers everything as strings) just works."""
        errors: dict = {}
        for k, v in values.items():
            if k not in self.__dataclass_fields__:
                errors[k] = "unknown field"
                continue
            try:
                cur = getattr(self, k)
                if isinstance(cur, bool):
                    new = str(v).lower() in ("1", "true", "yes", "on")
                elif isinstance(cur, int):
                    new = int(float(v))
                elif isinstance(cur, float):
                    new = float(v)
                else:
                    new = v
                setattr(self, k, new)
            except (ValueError, TypeError) as e:
                errors[k] = str(e)
        return errors

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


# ---------------------------------------------------------------------------
# UI tuning schema: drives the /tune page. Each entry maps a config field
# to a label + numeric step + optional unit. Grouped for layout. Anything
# NOT listed here is intentionally hidden from the live-tuning UI (e.g.,
# the network/UI/recording fields whose changes need a restart anyway).
# ---------------------------------------------------------------------------

TUNING_FIELDS = {
    "target_distance_m": {
        "label": "Target distance", "kind": "float", "unit": "m", "step": 0.01,
        "desc": "Final standoff from the marker. APPROACH closes to this distance; HOLD station-keeps here.",
    },
    "target_relative_heading_deg": {
        "label": "Target rel. heading", "kind": "float", "unit": "deg", "step": 0.01,
        "desc": "Bearing around the marker that HOLD maintains (CW from the marker normal). 0 = drone directly in front of the marker. Default 0 means HOLD just stays where APPROACH put it; non-zero makes HOLD slide laterally to the chosen bearing.",
    },
    "hold_time_s": {
        "label": "HOLD duration", "kind": "float", "unit": "s", "step": 0.01,
        "desc": "How long HOLD hovers (station-keeping) before transitioning to LAND.",
    },

    "fwd_kp": {
        "label": "fwd kp", "kind": "float", "step": 0.01,
        "desc": "Proportional gain on distance error, in RC counts per metre. e.g. kp=30 means a 1 m distance error tries to command rc_fb=30 (clamped at fwd_rc_max). Higher = more aggressive distance correction; too high = overshoot.",
    },
    "fwd_kd": {
        "label": "fwd kd", "kind": "float", "step": 0.01,
        "desc": "Derivative gain on distance, in RC counts per (m/s) of d/dt(distance). Brakes when the drone is closing fast, even if still far from target. Helps prevent overshoot, but amplifies distance noise.",
    },
    "fwd_kv": {
        "label": "fwd kv", "kind": "float", "step": 0.01,
        "desc": "Velocity feedforward on body-forward velocity, in RC counts per (cm/s). Subtracts an active brake proportional to current speed (independent of position error). 0 = disabled. Wrapped in a no-sign-flip guard so a wrong-frame velocity reading can't push the drone the wrong way.",
    },
    "fwd_ki": {
        "label": "fwd ki", "kind": "float", "step": 0.01,
        "desc": "Integral gain on distance error (RC per m*s). 0 = pure PD (default). Bump above 0 to eliminate steady-state distance error during HOLD; back-calculation anti-windup + fwd_i_clip protect against runaway during saturation or marker loss.",
    },
    "fwd_i_clip": {
        "label": "fwd i clip", "kind": "float", "unit": "m*s", "step": 0.01,
        "desc": "Hard cap on the magnitude of the fwd integrator. With ki=30 and i_clip=1.0, the I term can contribute at most 30 RC counts. Last-line guard against integrator runaway.",
    },
    "fwd_rc_max": {
        "label": "fwd RC max", "kind": "int", "step": 1,
        "desc": "Hard cap on the rc_fb command magnitude (Anafi PCMD range -100..+100). Roughly each unit ≈ 4 cm/s body-forward at level flight, so rc_max=3 caps approach to ~12 cm/s. Lower = slower + safer.",
    },

    "lat_kp": {
        "label": "lat kp", "kind": "float", "step": 0.01,
        "desc": "Proportional gain on heading-arc error (RC per metre of tangential offset). PD computes arc_err = -d * radians(e_hdg) so the gain is dimensionally consistent across distances. Higher = more aggressive heading correction.",
    },
    "lat_kd": {
        "label": "lat kd", "kind": "float", "step": 0.01,
        "desc": "Derivative gain on heading-arc rate. Damps lateral correction motion to prevent oscillation around the heading target.",
    },
    "lat_kv": {
        "label": "lat kv", "kind": "float", "step": 0.01,
        "desc": "Velocity feedforward on body-right velocity. Same idea as fwd_kv: brakes lateral momentum so the drone doesn't sail past the heading target. No-sign-flip-guarded.",
    },
    "lat_ki": {
        "label": "lat ki", "kind": "float", "step": 0.01,
        "desc": "Integral gain on heading-arc error (RC per m*s). 0 = pure PD. Useful for HOLD if the drone has a steady tangential bias (gimbal offset, mass imbalance). Anti-windup paired with lat_i_clip.",
    },
    "lat_i_clip": {
        "label": "lat i clip", "kind": "float", "unit": "m*s", "step": 0.01,
        "desc": "Hard cap on the magnitude of the lat integrator. Last-line guard against integrator runaway during saturation or marker loss.",
    },
    "lat_rc_max": {
        "label": "lat RC max", "kind": "int", "step": 1,
        "desc": "Cap on rc_lr. Lower = slower lateral motion + better yaw tracking (yaw can keep up with the apparent-marker drift). 4 ≈ 16 cm/s. Going much higher tends to spiral the drone outward because yaw lags.",
    },

    "height_kp": {
        "label": "height kp", "kind": "float", "step": 0.01,
        "desc": "Proportional gain on height error (m), in RC counts per metre. Default 30 means a 1 m error tries to command rc_ud=30 (clamped at ud_rc_max).",
    },
    "height_kd": {
        "label": "height kd", "kind": "float", "step": 0.01,
        "desc": "Derivative gain on height-rate (RC per m/s). Damps oscillation around the height target.",
    },
    "height_ki": {
        "label": "height ki", "kind": "float", "step": 0.01,
        "desc": "Integral gain on height error (RC per m*s). 0 = pure PD. Eliminates steady altitude offset (gimbal drift, residual lift bias). Anti-windup paired with height_i_clip.",
    },
    "height_i_clip": {
        "label": "height i clip", "kind": "float", "unit": "m*s", "step": 0.01,
        "desc": "Hard cap on the magnitude of the height integrator. Last-line guard against integrator runaway during saturation or marker loss.",
    },

    "yaw_kp": {
        "label": "yaw kp", "kind": "float", "step": 0.01,
        "desc": "Proportional gain on yaw_to_marker, in RC counts per degree. Saturates at yaw_rc_max for |error| > yaw_rc_max / yaw_kp. Too high → bang-bang at the controller's loop period.",
    },
    "yaw_kd": {
        "label": "yaw kd", "kind": "float", "step": 0.01,
        "desc": "Derivative gain on yaw_to_marker (RC per °/s). Damps the yaw oscillation that pure-P tends to produce at high gains.",
    },
    "yaw_ki": {
        "label": "yaw ki", "kind": "float", "step": 0.01,
        "desc": "Integral gain on yaw_to_marker (RC per deg*s). 0 = pure PD. Eliminates steady yaw bias (e.g., camera mounting offset). Anti-windup paired with yaw_i_clip.",
    },
    "yaw_i_clip": {
        "label": "yaw i clip", "kind": "float", "unit": "deg*s", "step": 0.01,
        "desc": "Hard cap on the magnitude of the yaw integrator. Last-line guard against integrator runaway during saturation or marker loss.",
    },
    "yaw_rc_max": {
        "label": "yaw RC max", "kind": "int", "step": 1,
        "desc": "Cap on rc_yaw. Roughly 1 unit ≈ 1°/s of yaw rate, so 25 ≈ 25°/s.",
    },

    "default_height_m": {
        "label": "Default height (post-takeoff)", "kind": "float", "unit": "m", "step": 0.01,
        "desc": "Height the drone climbs to immediately after takeoff before searching. PD-driven via height_kp/kd; clamped between min_height_m and max_height_m.",
    },
    "min_height_m": {
        "label": "Min height", "kind": "float", "unit": "m", "step": 0.01,
        "desc": "Floor on the drone's flight altitude. _send_rc forces rc_ud >= 0 below this height (drone can climb but not descend). HEIGHT_ALIGN also caps its target at this -- a marker mounted lower than min_height does not pull the drone down to ground level.",
    },
    "max_height_m": {
        "label": "Max height", "kind": "float", "unit": "m", "step": 0.01,
        "desc": "Ceiling on the drone's flight altitude. _send_rc forces rc_ud <= 0 above this height. HEIGHT_ALIGN also caps its target here.",
    },
    "height_deadband_m": {
        "label": "height deadband", "kind": "float", "unit": "m", "step": 0.01,
        "desc": "|drone_height - target_height| below this → rc_ud=0. Defines what 'at target altitude' means.",
    },
    "height_settle_time_s": {
        "label": "HEIGHT_ALIGN settle time", "kind": "float", "unit": "s", "step": 0.01,
        "desc": "How long height + yaw must both stay inside their deadbands before HEIGHT_ALIGN transitions to ALIGN.",
    },
    "default_dance_seconds_s": {
        "label": "DANCE default duration", "kind": "float", "unit": "s", "step": 0.01,
        "desc": "Default seconds for a DANCE step in the mission script when the operator omits the argument.",
    },
    "dance_radius_m": {
        "label": "DANCE radius bound", "kind": "float", "unit": "m", "step": 0.01,
        "desc": "DANCE position-bounded routine. If the drone drifts more than this distance from its dance-entry position, the offending RC channel is zeroed and an inward correction is applied. Same bound applies vertically.",
    },

    "yaw_deadband_deg": {
        "label": "yaw deadband", "kind": "float", "unit": "deg", "step": 0.01,
        "desc": "|yaw_to_marker| below this → rc_yaw=0. Prevents twitchy yaw command when the marker is essentially centred.",
    },
    "distance_deadband_m": {
        "label": "distance deadband", "kind": "float", "unit": "m", "step": 0.01,
        "desc": "|distance - target| below this → rc_fb=0. Defines what 'at target distance' means for phase-settle and PD.",
    },
    "heading_deadband_deg": {
        "label": "heading deadband (HOLD)", "kind": "float", "unit": "deg", "step": 0.01,
        "desc": "HOLD's deadband on heading error. PD produces no lateral output when |hdg - target| < this. Tighter than ALIGN/APPROACH because at HOLD distance the marker is large and pose is reliable.",
    },
    "approach_heading_deadband_deg": {
        "label": "heading deadband (APPROACH)", "kind": "float", "unit": "deg", "step": 0.01,
        "desc": "APPROACH's deadband. Tighter than ALIGN's so the drone fights the yaw+forward arc-coupling drift while closing distance. Too loose → drift accumulates inside the band; too tight → chatter on pose noise.",
    },
    "align_heading_deadband_deg": {
        "label": "heading deadband (ALIGN)", "kind": "float", "unit": "deg", "step": 0.01,
        "desc": "ALIGN's deadband. Looser than APPROACH because long-range pose is noisier and ALIGN only needs to roughly face the marker before APPROACH closes distance.",
    },

    "approach_settle_time_s": {
        "label": "APPROACH settle time", "kind": "float", "unit": "s", "step": 0.01,
        "desc": "How long all errors must stay inside their deadbands before APPROACH transitions to HOLD. Longer = stricter.",
    },
    "align_settle_time_s": {
        "label": "ALIGN settle time", "kind": "float", "unit": "s", "step": 0.01,
        "desc": "Same as above but for ALIGN → APPROACH.",
    },
    "search_marker_lost_grace_s": {
        "label": "marker-lost grace", "kind": "float", "unit": "s", "step": 0.01,
        "desc": "After the marker disappears, hold zero RC for this long before escalating to SEARCH. Tolerates brief occlusions without bouncing the state machine.",
    },

    "distance_floor_factor": {
        "label": "distance floor (× target)", "kind": "float", "step": 0.01,
        "desc": "Minimum allowed distance, expressed as a multiple of target_distance_m. Inside the floor the forward command is clamped to ≤0 (drone can back out, never fly closer). Yaw and lateral PDs keep running so the drone can still recenter while inside.",
    },
    "search_yaw_rc": {
        "label": "search yaw RC", "kind": "int", "step": 1,
        "desc": "rc_yaw value used during SEARCH while sweeping for the marker. ≈ degrees-per-second yaw rate.",
    },
    "ud_rc_max": {
        "label": "ud RC max", "kind": "int", "step": 1,
        "desc": "Cap on rc_ud (altitude). The mission doesn't actively command altitude (Anafi self-stabilises), but the cap protects the channel from accidental large values.",
    },

    "pose_smoothing_alpha": {
        "label": "pose EMA alpha", "kind": "float", "step": 0.01,
        "desc": "EMA factor on (distance, yaw_to_marker, relative_heading). 1.0 = no smoothing (raw values); 0.0 = frozen at first sample. Higher = more responsive but noisier control.",
    },
    "pose_max_age_s": {
        "label": "pose max age", "kind": "float", "unit": "s", "step": 0.01,
        "desc": "If no marker detection in this window, the smoother reports None to the controller (= 'marker lost'). Pairs with search_marker_lost_grace_s.",
    },

    "killswitch_key": {
        "label": "killswitch key", "kind": "str",
        "desc": "Single character. Pressing this key anywhere on the web UI (including inside textareas / number inputs) immediately triggers /api/stop and lands the drone. Default 'Q'. Comparison is case-insensitive; modifier keys (Ctrl/Cmd/Alt) are excluded so browser shortcuts still work.",
    },
}

TUNING_GROUPS = [
    ("Mission goals", ["target_distance_m", "target_relative_heading_deg", "hold_time_s"]),
    ("Forward PD (closing distance)",
        ["fwd_kp", "fwd_kd", "fwd_kv", "fwd_ki", "fwd_i_clip", "fwd_rc_max"]),
    ("Lateral PD (orbiting / heading correction)",
        ["lat_kp", "lat_kd", "lat_kv", "lat_ki", "lat_i_clip", "lat_rc_max"]),
    ("Yaw PD (centring marker)",
        ["yaw_kp", "yaw_kd", "yaw_ki", "yaw_i_clip", "yaw_rc_max"]),
    ("Height control",
        ["default_height_m", "min_height_m", "max_height_m",
         "height_kp", "height_kd", "height_ki", "height_i_clip",
         "height_deadband_m", "height_settle_time_s", "ud_rc_max"]),
    ("Mission script defaults",
        ["default_dance_seconds_s", "dance_radius_m"]),
    ("Deadbands",
        ["yaw_deadband_deg", "distance_deadband_m",
         "heading_deadband_deg", "approach_heading_deadband_deg",
         "align_heading_deadband_deg"]),
    ("Phase timing",
        ["approach_settle_time_s", "align_settle_time_s",
         "search_marker_lost_grace_s"]),
    ("Safety / search",
        ["distance_floor_factor", "search_yaw_rc"]),
    ("Pose smoothing",
        ["pose_smoothing_alpha", "pose_max_age_s"]),
    ("Operator UX",
        ["killswitch_key"]),
]


def tuning_view(cfg: MissionConfig) -> dict:
    """Build the JSON payload the /tune page renders against. Includes
    each field's current value, the dataclass default, and the field
    metadata (label/kind/unit/step). Returned in the order that
    TUNING_GROUPS dictates so the UI layout is stable."""
    defaults = MissionConfig()
    groups = []
    for group_name, fields in TUNING_GROUPS:
        items = []
        for f in fields:
            meta = TUNING_FIELDS.get(f, {})
            items.append({
                "name":    f,
                "label":   meta.get("label", f),
                "kind":    meta.get("kind", "float"),
                "unit":    meta.get("unit", ""),
                "step":    meta.get("step", 0.01),
                "desc":    meta.get("desc", ""),
                "value":   getattr(cfg, f),
                "default": getattr(defaults, f),
            })
        groups.append({"name": group_name, "items": items})
    return {"groups": groups}
