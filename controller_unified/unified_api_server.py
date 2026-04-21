"""
Unified Pi API Server — auto-detects Tello or Anafi drone by IP.

  192.168.10.1  → Tello  (djitellopy SDK)
  192.168.42.1  → Anafi  (Olympe SDK)

Override with env: DRONE_TYPE=tello|anafi  DRONE_IP=<ip>

Usage:
  python3 unified_pi_api_server.py
"""

from __future__ import annotations

import abc
import atexit
from collections import deque
import json
import logging
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from flask import Flask, Response, jsonify, request, send_file

# ---------------------------------------------------------------------------
# Architecture detection — skip Olympe on Raspberry Pi (ARM)
# ---------------------------------------------------------------------------
import platform

_machine = platform.machine().lower()
IS_ARM = _machine.startswith("arm") or _machine.startswith("aarch")
if IS_ARM:
    print(f"[UNIFIED] ARM architecture detected ({_machine}) — Olympe SDK disabled, defaulting to Tello")

# ---------------------------------------------------------------------------
# Conditional SDK imports
# ---------------------------------------------------------------------------

HAS_TELLO_SDK = False
try:
    from djitellopy import Tello
    HAS_TELLO_SDK = True
except ImportError:
    pass

HAS_OLYMPE_SDK = False
HAS_EMERGENCY = False
HAS_CAMERA = False
HAS_GIMBAL = False
HAS_GPS_STATE = False
HAS_RTH = False
HAS_MOVE_TO = False
HAS_MAX_HORIZ_SPEED = False
HAS_GEOFENCE = False
HAS_CV2 = False

if not IS_ARM:
    try:
        import olympe
        from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveBy, PCMD
        from olympe.messages.ardrone3.PilotingState import (
            FlyingStateChanged, AttitudeChanged, SpeedChanged, AltitudeChanged,
        )
        from olympe.messages.common.CommonState import BatteryStateChanged
        from olympe.messages.ardrone3.Animations import Flip
        from olympe.messages.ardrone3.SpeedSettings import MaxRotationSpeed, MaxVerticalSpeed
        from olympe.messages.ardrone3.PilotingSettings import MaxAltitude, MaxTilt
        HAS_OLYMPE_SDK = True
    except (ImportError, KeyError):
        pass

if HAS_OLYMPE_SDK:
    try:
        from olympe.messages.ardrone3.Piloting import Emergency as EmergencyCmd
        HAS_EMERGENCY = True
    except (ImportError, KeyError):
        pass
    # Diagnostics: why did the Anafi refuse takeoff?
    # AlertStateChanged: low_battery/too_much_angle/magneto_*/motor_error/etc.
    # MotorErrorStateChanged: which motor failed and why.
    try:
        from olympe.messages.ardrone3.PilotingState import AlertStateChanged
        HAS_ALERT_STATE = True
    except (ImportError, KeyError):
        HAS_ALERT_STATE = False
    try:
        from olympe.messages.ardrone3.SettingsState import MotorErrorStateChanged
        HAS_MOTOR_ERROR = True
    except (ImportError, KeyError):
        HAS_MOTOR_ERROR = False
    try:
        from olympe.messages.common.CommonState import SensorsStatesListChanged
        HAS_SENSOR_STATE = True
    except (ImportError, KeyError):
        HAS_SENSOR_STATE = False
    # Magnetometer calibration: if the Anafi has been moved between locations
    # or re-powered, it often needs a figure-8 calibration before it will
    # arm. These state messages tell us:
    #   MagnetoCalibrationRequiredState: bool — "does it need re-calibration?"
    #   MagnetoCalibrationStateChanged: per-axis calibration done (xAxisCalibration,
    #                                    yAxisCalibration, zAxisCalibration, calibrationFailed)
    #   MagnetoCalibrationStartedChanged: whether the drone is currently in
    #                                    calibration mode
    #   StartCalibration: command to trigger calibration
    try:
        from olympe.messages.common.CalibrationState import (
            MagnetoCalibrationRequiredState,
            MagnetoCalibrationStateChanged,
            MagnetoCalibrationStartedChanged,
        )
        HAS_MAGNETO_CALIB = True
    except (ImportError, KeyError):
        HAS_MAGNETO_CALIB = False
    try:
        from olympe.messages.common.Calibration import MagnetoCalibration as StartMagnetoCalibration
        HAS_MAGNETO_CALIB_CMD = True
    except (ImportError, KeyError):
        HAS_MAGNETO_CALIB_CMD = False


# ─── Optional Wi-Fi control messages ────────────────────────────────────────
# The Anafi is dual-band 2.4/5 GHz 802.11n. The drone exposes AP-channel
# config via Olympe. Only imported on systems where Olympe is available;
# everything that uses these is guarded by HAS_WIFI_CTRL.
HAS_WIFI_CTRL = False
if HAS_OLYMPE_SDK:
    try:
        from olympe.messages.wifi import (
            set_ap_channel as WifiSetApChannel,
            scan as WifiScan,
            rssi_changed as WifiRssiChanged,
            authorized_channel as WifiAuthorizedChannel,
            ap_channel_changed as WifiApChannelChanged,
            country_changed as WifiCountryChanged,
            environement_changed as WifiEnvironmentChanged,
        )
        from olympe.enums.wifi import (
            SelectionType as WifiSelectionType,
            Band as WifiBand,
        )
        HAS_WIFI_CTRL = True
    except (ImportError, KeyError, AttributeError) as _e:
        # Older Olympe versions have slightly different paths. Try a
        # fallback via the .Command submodule.
        try:
            from olympe.messages.wifi import Command as _WifiCmd
            WifiSetApChannel = getattr(_WifiCmd, "SetApChannel", None)
            WifiScan = getattr(_WifiCmd, "ScanChannels", None)
            from olympe.messages import wifi as _wifi_events
            WifiApChannelChanged = getattr(_wifi_events, "ApChannelChanged", None)
            WifiAuthorizedChannel = getattr(_wifi_events, "AuthorizedChannel", None)
            WifiRssiChanged = getattr(_wifi_events, "Rssi_changed", None)
            HAS_WIFI_CTRL = bool(WifiSetApChannel and WifiScan)
        except Exception:
            HAS_WIFI_CTRL = False


def _magneto_needs_calibration(status: Optional[str]) -> bool:
    """Return True only when the magnetometer really needs recalibration.

    The status string from _read_magnetometer_state() is a comma-separated
    list of tokens like "REQUIRED", "not-required", "all-axes-ok", "FAILED",
    "axes=x1y1z1", "in-progress". Our earlier substring check matched
    "REQUIRED" inside "not-required" and produced a false warning — fix
    by splitting on commas and comparing token-by-token.
    """
    if not status:
        return False
    tokens = [t.strip().lower() for t in status.split(",")]
    if "required" in tokens:      # exact token, not a substring
        return True
    for t in tokens:
        # Accepts "failed", "calibrationfailed", "fail", etc.
        if t.startswith("fail") or t == "failed":
            return True
    return False
    try:
        from olympe.messages.camera import start_recording, stop_recording, take_photo
        HAS_CAMERA = True
    except (ImportError, KeyError):
        pass
    try:
        from olympe.messages.gimbal import set_target, attitude as gimbal_attitude
        HAS_GIMBAL = True
    except (ImportError, KeyError):
        pass
    try:
        from olympe.messages.ardrone3.PilotingState import GpsLocationChanged
        HAS_GPS_STATE = True
    except (ImportError, KeyError):
        try:
            from olympe.messages.ardrone3.GPSState import GpsLocationChanged
            HAS_GPS_STATE = True
        except (ImportError, KeyError):
            pass
    try:
        from olympe.messages.rth import return_to_home, cancel_auto_trigger, state as rth_state
        HAS_RTH = True
    except (ImportError, KeyError):
        pass
    try:
        from olympe.messages.move import extended_move_to
        HAS_MOVE_TO = True
    except (ImportError, KeyError):
        pass
    try:
        from olympe.messages.ardrone3.SpeedSettings import MaxHorizontalSpeed
        HAS_MAX_HORIZ_SPEED = True
    except (ImportError, KeyError):
        pass
    try:
        from olympe.messages.ardrone3.PilotingSettings import MaxDistance, NoFlyOverMaxDistance
        HAS_GEOFENCE = True
    except (ImportError, KeyError):
        pass

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    pass

# Try to import HeadlessAruCoPositioning from sibling aruco-position directory
HAS_POSITIONING = False
_HeadlessAruCo = None
if HAS_CV2 and not IS_ARM:
    try:
        # _aruco_path = str(Path(__file__).parent.parent / "aruco-position" / "control-unit")
        # if _aruco_path not in sys.path:
        #     sys.path.insert(0, _aruco_path)
        from ctrl_position import HeadlessAruCoPositioning as _HeadlessAruCo
        HAS_POSITIONING = True
        print("[UNIFIED] HeadlessAruCoPositioning loaded")
    except ImportError as _e:
        print(f"[UNIFIED] ArUco positioning unavailable: {_e}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HTTP_HOST = "0.0.0.0"
HTTP_PORT = int(os.getenv("HTTP_PORT", "8080"))
RC_HZ = 20
STICK = 60
YAW_STICK_ANAFI = 90
MAX_YAW_SPEED = 150
MAX_ALTITUDE_M = float(os.getenv("MAX_ALTITUDE_M", "2.0"))
MAX_VERTICAL_SPEED = float(os.getenv("MAX_VERTICAL_SPEED", "0.5"))
MAX_TILT = float(os.getenv("MAX_TILT", "15"))
CONNECT_RETRY_S = 3.0
RECONNECT_AFTER_S = 3.0
VERIFY_RETRY_S = 2.5
WIFI_RETRY_S = 3.0
TELEMETRY_HZ = float(os.getenv("TELEMETRY_HZ", "2.0"))
KEY_STALE_S = float(os.getenv("KEY_STALE_S", "1.0"))
SAFE_TAKEOFF_S = float(os.getenv("SAFE_TAKEOFF_S", "3.0"))
SAFE_TAKEOFF_DEFAULT = os.getenv("SAFE_TAKEOFF_DEFAULT", "0") in {"1", "true", "True"}
REMOTE_TIMEOUT_S = float(os.getenv("REMOTE_TIMEOUT_S", "2.0"))

# Video
VIDEO_JPEG_QUALITY = int(os.getenv("VIDEO_JPEG_QUALITY", "70"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "15"))
VIDEO_UDP_FORWARD_PORT = int(os.getenv("VIDEO_UDP_FORWARD_PORT", "55004"))

# Logging
TELEMETRY_LOG_DEFAULT = False
TELEMETRY_LOG_PATH_DEFAULT = Path(__file__).with_name("telemetry_log.jsonl")
COMMAND_LOG_ENABLED = os.getenv("API_COMMAND_LOG", "1") in {"1", "true", "True"}
COMMAND_LOG_PATH = Path(
    os.getenv("API_COMMAND_LOG_PATH", str(Path(__file__).with_name("api_command_log.jsonl")))
)

# Tello Wi-Fi config
WIFI_CFG_PATH = Path(__file__).with_name("tello_wifi_config.json")

# Default IPs
TELLO_DEFAULT_IP = "192.168.10.1"
ANAFI_DEFAULT_IP = "192.168.42.1"

# Positioning / arena / flight config paths
POSITION_CONFIG_PATH = Path(__file__).with_name("position_config.json")
POSITION_CALIB_PATH  = Path(__file__).with_name("position_calib.npz")
ARENA_CONFIG_PATH    = Path(__file__).with_name("arena_config.json")
FLIGHT_CONFIG_PATH   = Path(__file__).with_name("flight_config.json")


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def _host_reachable(host: str) -> bool:
    r = subprocess.run(["ping", "-c", "1", "-W", "1", host], capture_output=True, text=True)
    return r.returncode == 0


def detect_drone_type() -> Tuple[str, str]:
    """Returns (drone_type, drone_ip). drone_type is 'tello' or 'anafi'."""
    forced_type = os.getenv("DRONE_TYPE", "").lower().strip()
    forced_ip = os.getenv("DRONE_IP", "").strip()

    if forced_type in ("tello", "anafi"):
        if forced_type == "tello":
            ip = forced_ip or os.getenv("TELLO_HOST", TELLO_DEFAULT_IP)
        else:
            ip = forced_ip or os.getenv("ANAFI_IP", ANAFI_DEFAULT_IP)
        return forced_type, ip

    # Auto-detect: try both IPs
    tello_ip = forced_ip or os.getenv("TELLO_HOST", TELLO_DEFAULT_IP)
    anafi_ip = forced_ip or os.getenv("ANAFI_IP", ANAFI_DEFAULT_IP)

    # If a specific DRONE_IP is given, guess by IP
    if forced_ip:
        if forced_ip == TELLO_DEFAULT_IP or "192.168.10." in forced_ip:
            return "tello", forced_ip
        if forced_ip == ANAFI_DEFAULT_IP or "192.168.42." in forced_ip:
            return "anafi", forced_ip

    print("[UNIFIED] Auto-detecting drone type...")

    # On ARM (Raspberry Pi), only Tello is supported (no Olympe SDK)
    if IS_ARM:
        if _host_reachable(tello_ip):
            print(f"[UNIFIED] Tello detected at {tello_ip}")
        else:
            print(f"[UNIFIED] ARM platform — defaulting to Tello @ {tello_ip} (will retry)")
        return "tello", tello_ip

    if _host_reachable(anafi_ip):
        print(f"[UNIFIED] Anafi detected at {anafi_ip}")
        return "anafi", anafi_ip
    if _host_reachable(tello_ip):
        print(f"[UNIFIED] Tello detected at {tello_ip}")
        return "tello", tello_ip

    # Default: try anafi first since it's the primary competition drone
    print(f"[UNIFIED] No drone reachable — defaulting to anafi @ {anafi_ip} (will retry)")
    return "anafi", anafi_ip


# ---------------------------------------------------------------------------
# Global state (shared across backends)
# ---------------------------------------------------------------------------
app = Flask(__name__)
running = True
flying = False
drone_type = "unknown"
drone_ip = ""

pressed_web: Set[str] = set()
key_last_seen: Dict[str, float] = {}
pressed_lock = threading.Lock()

last_state_seen = 0.0
conn_state = {"connected": False, "last_reconnect": 0.0, "last_verify": 0.0,
              "consecutive_failures": 0, "last_failure_ts": 0.0, "last_error": ""}
conn_lock = threading.Lock()
last_conn_print = None
# Exponential backoff cap — after this many consecutive failures, we force a
# hard reset (fully destroy + recreate the Olympe Drone object). Past this,
# sleep time grows up to RECONNECT_BACKOFF_MAX_S between attempts.
RECONNECT_HARD_RESET_AFTER = 2
RECONNECT_BACKOFF_MAX_S = 15.0

rc_override: Optional[Tuple[int, int, int, int]] = None
rc_override_until = 0.0
rc_lock = threading.Lock()

telemetry: Dict = {
    "battery": None, "temperature": None, "height_cm": None,
    "tof_cm": None, "barometer_cm": None, "flight_time_s": None,
    "pitch": None, "roll": None, "yaw": None,
    "vgx": None, "vgy": None, "vgz": None,
    "agx": None, "agy": None, "agz": None,
    "speed": None, "flying": False, "connected": False, "updated_at": 0.0,
    "updated_at_mono": 0.0,
}
telemetry_lock = threading.Lock()
telemetry_log_enabled = TELEMETRY_LOG_DEFAULT
telemetry_log_path = TELEMETRY_LOG_PATH_DEFAULT
telemetry_log_lock = threading.Lock()
command_log_lock = threading.Lock()
_telemetry_sync_buf = deque(maxlen=3000)
_telemetry_sync_lock = threading.Lock()

command_lock = threading.Lock()
discrete_until = 0.0
takeoff_cooldown_until = 0.0
safe_takeoff_enabled = SAFE_TAKEOFF_DEFAULT

last_remote_request = 0.0
_watchdog_landed = False

# Video state
_video_mode = "off"
_video_last_jpeg = b""
_video_jpeg_lock = threading.Lock()
_video_streaming = False
_video_forward_proc = None
_video_forward_target = ""
_video_frame_count = 0

# ---------------------------------------------------------------------------
# Positioning state
# ---------------------------------------------------------------------------
_pos_frame_q: "queue.Queue" = queue.Queue(maxsize=3)   # BGR frames tapped from video cb
_pos_processor = None   # live HeadlessAruCoPositioning instance (set by positioning_loop)
_pos_sse_queues: list = []
_pos_sse_lock = threading.Lock()
_pos_st: dict = {
    "x": None, "y": None, "z": None,
    "dx": None, "dy": None,
    "vx": 0.0, "vy": 0.0, "vz": 0.0,
    "markers": {}, "ref_markers": [],
    "seen_markers": [], "seen_count": 0,
    "stale": True, "ts": 0.0, "fps": None,
    "tel_vgx": 0.0, "tel_vgy": 0.0, "tel_vgz": 0.0,
    "sync_quality": "none", "sync_age_ms": None,
}
# FPS tracking for positioning loop
_pos_fps_counter = 0
_pos_fps_last_reset = 0.0
_pos_fps_current: float = 0.0
_pos_st_lock = threading.Lock()
_pos_annotated_jpeg = b""
_pos_annotated_lock = threading.Lock()
_pos_cfg_lock = threading.Lock()

# Video recording state
_rec_lock = threading.Lock()
_rec_enabled = False
_rec_raw = False          # True = record raw frame, False = record annotated frame
_rec_writer = None        # cv2.VideoWriter — lazy-created on first frame
_rec_path: str = ""
_rec_frame_count = 0
_rec_fname: str = ""      # requested filename (used when deferring writer creation)
_rec_fps_target: float = 0.0  # target fps captured at start (fallback 5)
_rec_writer_size: tuple = (0, 0)  # (w, h) the writer was opened at
_pos_cfg: dict = {
    "enabled": False,
    "fps": 5,
    "detect_profile": "default",
    "latency_comp_s": 0.05,
    "sync_max_gap_s": 0.20,
    "telemetry_buffer_s": 5.0,
    "camera_matrix": None,
    "dist_coeffs": None,
    # Live-tunable filter params (applied to _pos_processor on POST)
    "enable_kalman_filter": True,
    "marker_size_m": 0.5,   # overrides arena_config.json marker_size_m when set
    "top_k_markers": 0,     # 0 = auto (code picks best 4); set to limit further
    "outlier_reject_m": 2.5,  # max distance from centroid before an estimate is dropped
    "imu_weight": 0.3,      # 0.0 = pure ArUco, 1.0 = pure IMU dead-reckoning
}

# Arena config — SDC challenge default layout.
# Coordinates: X = left(-10) / right(+10), Y = near(0) / far(10), Z = low(-1) / high(+1).
# Markers come in vertical pairs (low/high) except ID 0 (origin).
# Wall types: 'front'=Y=0 wall, 'back'=Y=10 wall, 'right'=X=+10, 'left'=X=-10.
_ARENA_CONFIG_DEFAULT: dict = {
    "arena_width_m": 20.0,
    "arena_height_m": 10.0,
    "arena_origin_x": -10.0,
    "arena_origin_y": 0.0,
    "marker_size_m": 0.5,
    "markers": [
        {"id":  0, "label": "Origin",       "x":   0.0, "y":  0.0,   "z":  0.0, "wall": "front"},
        {"id":  1, "label": "Right A low",  "x":  10.0, "y":  6.667, "z":  2.0, "wall": "right"},
        {"id":  2, "label": "Right A high", "x":  10.0, "y":  6.667, "z":  4.0, "wall": "right"},
        {"id":  3, "label": "Right B low",  "x":  10.0, "y":  3.333, "z":  2.0, "wall": "right"},
        {"id":  4, "label": "Right B high", "x":  10.0, "y":  3.333, "z":  4.0, "wall": "right"},
        {"id":  5, "label": "Front A low",  "x":   6.0, "y":  0.0,   "z":  2.0, "wall": "front"},
        {"id":  6, "label": "Front A high", "x":   6.0, "y":  0.0,   "z":  4.0, "wall": "front"},
        {"id":  7, "label": "Front B low",  "x":   2.0, "y":  0.0,   "z":  2.0, "wall": "front"},
        {"id":  8, "label": "Front B high", "x":   2.0, "y":  0.0,   "z":  4.0, "wall": "front"},
        {"id":  9, "label": "Front C low",  "x":  -2.0, "y":  0.0,   "z":  2.0, "wall": "front"},
        {"id": 10, "label": "Front C high", "x":  -2.0, "y":  0.0,   "z":  4.0, "wall": "front"},
        {"id": 11, "label": "Front D low",  "x":  -6.0, "y":  0.0,   "z":  2.0, "wall": "front"},
        {"id": 12, "label": "Front D high", "x":  -6.0, "y":  0.0,   "z":  4.0, "wall": "front"},
        {"id": 13, "label": "Left A low",   "x": -10.0, "y":  3.333, "z":  2.0, "wall": "left"},
        {"id": 14, "label": "Left A high",  "x": -10.0, "y":  3.333, "z":  4.0, "wall": "left"},
        {"id": 15, "label": "Left B low",   "x": -10.0, "y":  6.667, "z":  2.0, "wall": "left"},
        {"id": 16, "label": "Left B high",  "x": -10.0, "y":  6.667, "z":  4.0, "wall": "left"},
        {"id": 17, "label": "Back A low",   "x":  -6.0, "y": 10.0,   "z":  2.0, "wall": "back"},
        {"id": 18, "label": "Back A high",  "x":  -6.0, "y": 10.0,   "z":  4.0, "wall": "back"},
        {"id": 19, "label": "Back B low",   "x":  -2.0, "y": 10.0,   "z":  2.0, "wall": "back"},
        {"id": 20, "label": "Back B high",  "x":  -2.0, "y": 10.0,   "z":  4.0, "wall": "back"},
        {"id": 21, "label": "Back C low",   "x":   2.0, "y": 10.0,   "z":  2.0, "wall": "back"},
        {"id": 22, "label": "Back C high",  "x":   2.0, "y": 10.0,   "z":  4.0, "wall": "back"},
        {"id": 23, "label": "Back D low",   "x":   6.0, "y": 10.0,   "z":  2.0, "wall": "back"},
        {"id": 24, "label": "Back D high",  "x":   6.0, "y": 10.0,   "z":  4.0, "wall": "back"},
    ],
}
_arena_cfg: dict = {}
_arena_cfg_lock = threading.Lock()


def _load_arena_config() -> dict:
    try:
        if ARENA_CONFIG_PATH.exists():
            data = json.loads(ARENA_CONFIG_PATH.read_text())
            base = dict(_ARENA_CONFIG_DEFAULT)
            base.update(data)
            return base
    except Exception as e:
        print(f"[POS] arena_config load error: {e}")
    return dict(_ARENA_CONFIG_DEFAULT)


def _save_arena_config(cfg: dict):
    try:
        ARENA_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ARENA_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception as e:
        print(f"[POS] arena_config save error: {e}")


def _load_flight_config():
    """Apply persisted flight limits to globals on startup."""
    global MAX_ALTITUDE_M, MAX_VERTICAL_SPEED, MAX_TILT, MAX_YAW_SPEED
    try:
        if FLIGHT_CONFIG_PATH.exists():
            data = json.loads(FLIGHT_CONFIG_PATH.read_text())
            if "max_altitude_m" in data:
                MAX_ALTITUDE_M = float(data["max_altitude_m"])
            if "max_vertical_speed" in data:
                MAX_VERTICAL_SPEED = float(data["max_vertical_speed"])
            if "max_tilt" in data:
                MAX_TILT = float(data["max_tilt"])
            if "max_yaw_speed" in data:
                MAX_YAW_SPEED = float(data["max_yaw_speed"])
            print(f"[UNIFIED] Loaded flight config: alt={MAX_ALTITUDE_M} vs={MAX_VERTICAL_SPEED} tilt={MAX_TILT} yaw={MAX_YAW_SPEED}")
    except Exception as e:
        print(f"[UNIFIED] flight_config load error: {e}")


def _save_flight_config(results: dict):
    """Persist any successfully-updated flight limit to flight_config.json."""
    keys = {"max_altitude_m", "max_vertical_speed", "max_tilt", "max_yaw_speed"}
    if not any(k in results for k in keys):
        return
    try:
        existing: dict = {}
        if FLIGHT_CONFIG_PATH.exists():
            existing = json.loads(FLIGHT_CONFIG_PATH.read_text())
        for k in keys:
            if k in results:
                existing[k] = results[k]
        FLIGHT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        FLIGHT_CONFIG_PATH.write_text(json.dumps(existing, indent=2))
    except Exception as e:
        print(f"[UNIFIED] flight_config save error: {e}")


# Initialise arena config at import time
_arena_cfg = _load_arena_config()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def normalize_key(k: str) -> str:
    return (k or "").lower()

def add_key(k: str):
    k = normalize_key(k)
    if not k:
        return
    with pressed_lock:
        pressed_web.add(k)
        key_last_seen[k] = time.time()

def remove_key(k: str):
    k = normalize_key(k)
    if not k:
        return
    with pressed_lock:
        pressed_web.discard(k)
        key_last_seen.pop(k, None)

def reap_stale_keys(now: float):
    timeout = KEY_STALE_S if KEY_STALE_S > 0 else 1.0
    with pressed_lock:
        stale = [k for k, ts in key_last_seen.items() if (now - ts) > timeout]
        for k in stale:
            pressed_web.discard(k)
            key_last_seen.pop(k, None)

def has_key(k: str) -> bool:
    k = normalize_key(k)
    timeout = KEY_STALE_S if KEY_STALE_S > 0 else 1.0
    with pressed_lock:
        ts = key_last_seen.get(k)
        if ts is not None and (time.time() - ts) > timeout:
            pressed_web.discard(k)
            key_last_seen.pop(k, None)
            return False
        return k in pressed_web

def axis(pos: bool, neg: bool) -> int:
    return (1 if pos else 0) + (-1 if neg else 0)

def _as_int(v):
    try:
        return int(float(v))
    except Exception:
        return None

def append_telemetry_log(payload: dict):
    with telemetry_log_lock:
        if not telemetry_log_enabled:
            return
        p = telemetry_log_path
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

def append_command_log(event: str, payload: dict | None = None):
    if not COMMAND_LOG_ENABLED:
        return
    try:
        entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, "payload": payload or {}}
        with command_log_lock:
            COMMAND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with COMMAND_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def start_discrete_window(seconds: float):
    global discrete_until
    with command_lock:
        discrete_until = max(discrete_until, time.time() + max(0.0, seconds))


def _telemetry_buffer_retention_s() -> float:
    with _pos_cfg_lock:
        v = float(_pos_cfg.get("telemetry_buffer_s", 5.0))
    return max(1.0, min(30.0, v))


def _append_telemetry_sync_sample(sample: dict):
    ts_mono = float(sample.get("ts_mono", 0.0))
    if ts_mono <= 0.0:
        return
    retention = _telemetry_buffer_retention_s()
    with _telemetry_sync_lock:
        _telemetry_sync_buf.append(sample)
        cutoff = ts_mono - retention
        while _telemetry_sync_buf and _telemetry_sync_buf[0].get("ts_mono", 0.0) < cutoff:
            _telemetry_sync_buf.popleft()


def _interp_angle_deg(a0: float, a1: float, alpha: float) -> float:
    d = ((a1 - a0 + 180.0) % 360.0) - 180.0
    return a0 + alpha * d


def _telemetry_at(ts_mono: float, max_gap_s: float | None = None) -> dict:
    if ts_mono <= 0.0:
        return {"sample": None, "quality": "none", "age_s": None}
    if max_gap_s is None:
        with _pos_cfg_lock:
            max_gap_s = float(_pos_cfg.get("sync_max_gap_s", 0.20))
    max_gap_s = max(0.01, min(2.0, max_gap_s))

    with _telemetry_sync_lock:
        items = list(_telemetry_sync_buf)
    if not items:
        return {"sample": None, "quality": "none", "age_s": None}

    prev = None
    nxt = None
    for s in items:
        s_ts = float(s.get("ts_mono", 0.0))
        if s_ts <= ts_mono:
            prev = s
            continue
        nxt = s
        break

    if prev is None and nxt is None:
        return {"sample": None, "quality": "none", "age_s": None}

    if prev is not None and nxt is not None:
        t0 = float(prev.get("ts_mono", 0.0))
        t1 = float(nxt.get("ts_mono", 0.0))
        if t1 <= t0:
            nearest = prev
            quality = "nearest"
        else:
            alpha = max(0.0, min(1.0, (ts_mono - t0) / (t1 - t0)))
            interp = {"ts_mono": ts_mono}
            for key in ("vgx", "vgy", "vgz", "height_cm"):
                v0 = prev.get(key)
                v1 = nxt.get(key)
                if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                    interp[key] = float(v0) + alpha * (float(v1) - float(v0))
                elif isinstance(v0, (int, float)):
                    interp[key] = float(v0)
                elif isinstance(v1, (int, float)):
                    interp[key] = float(v1)
            for key in ("yaw", "pitch", "roll"):
                v0 = prev.get(key)
                v1 = nxt.get(key)
                if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                    interp[key] = _interp_angle_deg(float(v0), float(v1), alpha)
                elif isinstance(v0, (int, float)):
                    interp[key] = float(v0)
                elif isinstance(v1, (int, float)):
                    interp[key] = float(v1)
            interp["connected"] = bool(prev.get("connected", True) and nxt.get("connected", True))
            nearest = interp
            quality = "interpolated"
    else:
        nearest = prev if prev is not None else nxt
        quality = "nearest"

    nearest_ts = float(nearest.get("ts_mono", 0.0))
    age_s = abs(ts_mono - nearest_ts)
    if age_s > max_gap_s:
        return {"sample": nearest, "quality": "stale", "age_s": age_s}
    return {"sample": nearest, "quality": quality, "age_s": age_s}


# ═══════════════════════════════════════════════════════════════════════════
# DroneBackend ABC
# ═══════════════════════════════════════════════════════════════════════════

class DroneBackend(abc.ABC):
    """Abstract interface for drone-specific operations."""

    @abc.abstractmethod
    def connect(self) -> bool: ...
    @abc.abstractmethod
    def disconnect(self): ...
    @abc.abstractmethod
    def is_connected(self) -> bool: ...
    @abc.abstractmethod
    def verify_connection(self) -> bool: ...
    @abc.abstractmethod
    def on_connect(self): ...

    # Flight
    @abc.abstractmethod
    def takeoff(self) -> Tuple[bool, str]: ...
    @abc.abstractmethod
    def land(self) -> Tuple[bool, str]: ...
    @abc.abstractmethod
    def emergency(self) -> Tuple[bool, str]: ...
    @abc.abstractmethod
    def flip(self, direction: str) -> Tuple[bool, str]: ...
    @abc.abstractmethod
    def move(self, direction: str, cm: int) -> Tuple[bool, str]: ...
    @abc.abstractmethod
    def rotate(self, direction: str, degrees: int) -> Tuple[bool, str]: ...
    @abc.abstractmethod
    def go_xyz(self, x: int, y: int, z: int, speed: int) -> Tuple[bool, str]: ...

    # RC
    @abc.abstractmethod
    def send_rc(self, lr: int, fb: int, ud: int, yaw: int): ...
    @abc.abstractmethod
    def send_rc_zero(self): ...
    @abc.abstractmethod
    def before_discrete_command(self): ...
    @abc.abstractmethod
    def after_discrete_command(self): ...
    @abc.abstractmethod
    def rc_during_discrete(self) -> str:
        """Return 'zeros' (send zeros) or 'skip' (send nothing)."""
        ...

    # Telemetry
    @abc.abstractmethod
    def poll_telemetry(self) -> dict: ...

    # Recovery
    @abc.abstractmethod
    def recover(self) -> Tuple[bool, str]: ...

    # Config
    @abc.abstractmethod
    def yaw_stick(self) -> int: ...
    @abc.abstractmethod
    def has_altitude_fence(self) -> bool: ...
    @abc.abstractmethod
    def shutdown(self): ...

    # Capabilities — override in subclass, default = not supported
    def capabilities(self) -> dict:
        return {}

    def set_speed(self, speed: int) -> Tuple[bool, str]:
        return False, "not_supported"
    def curve(self, x1, y1, z1, x2, y2, z2, speed) -> Tuple[bool, str]:
        return False, "not_supported"
    def stream_control(self, action: str) -> Tuple[bool, str]:
        return False, "not_supported"
    def sdk_passthrough(self, command: str) -> Tuple[bool, str]:
        return False, "not_supported"
    def camera_photo(self) -> Tuple[bool, str]:
        return False, "not_supported"
    def camera_record_start(self) -> Tuple[bool, str]:
        return False, "not_supported"
    def camera_record_stop(self) -> Tuple[bool, str]:
        return False, "not_supported"
    def gimbal_set(self, tilt: float, pan: float) -> Tuple[bool, str]:
        return False, "not_supported"
    def rth(self, action: str) -> Tuple[bool, str]:
        return False, "not_supported"
    def moveto(self, lat, lon, alt, heading) -> Tuple[bool, str]:
        return False, "not_supported"
    def get_settings(self) -> dict:
        return {}
    def set_settings(self, data: dict) -> dict:
        return {}
    def video_start_mjpeg(self) -> Tuple[bool, str]:
        return False, "not_supported"
    def video_stop_mjpeg(self): pass
    def video_start_forward(self, host: str, port: int) -> Tuple[bool, str]:
        return False, "not_supported"
    def video_stop_forward(self): pass
    def video_stop_all(self): pass
    def video_status(self) -> dict:
        return {"mode": "off"}
    def get_video_jpeg(self) -> bytes:
        return b""


# ═══════════════════════════════════════════════════════════════════════════
# TelloBackend
# ═══════════════════════════════════════════════════════════════════════════

class TelloBackend(DroneBackend):

    def __init__(self, ip: str):
        self.ip = ip
        self.tello: Optional[Tello] = None
        self.sdk_version = None
        self.serial_number = None

    def _t(self) -> Optional[Tello]:
        return self.tello

    # --- Connection ---
    def connect(self) -> bool:
        if self.tello is None:
            self.tello = Tello(host=self.ip)
        self.tello.connect()
        # Don't call streamon() if video forward is active (it has port 11111)
        if not self._video_forward_active:
            try:
                self.tello.streamon()
            except Exception:
                pass
        self._refresh_info()
        # Verify with round-trip
        try:
            resp = str(self.tello.send_command_with_return("battery?")).strip()
            return resp.isdigit()
        except Exception:
            return False

    def disconnect(self):
        t = self.tello
        if t is None:
            return
        try:
            t.end()
        except Exception:
            pass

    def is_connected(self) -> bool:
        return _host_reachable(self.ip)

    def verify_connection(self) -> bool:
        t = self.tello
        if t is None:
            return False
        try:
            resp = str(t.send_command_with_return("battery?")).strip()
            if resp.isdigit():
                return True
        except Exception:
            pass
        # Fallback: just ping the drone
        return _host_reachable(self.ip)

    def on_connect(self):
        pass  # Tello doesn't need post-connect setup

    # --- Flight ---
    def takeoff(self) -> Tuple[bool, str]:
        t = self._t()
        if t is None:
            return False, "not_ready"
        with command_lock:
            t.send_rc_control(0, 0, 0, 0)
            t.takeoff()
        return True, "ok"

    def land(self) -> Tuple[bool, str]:
        t = self._t()
        if t is None:
            return False, "not_ready"
        last_err = None
        for _ in range(3):
            try:
                with command_lock:
                    t.send_rc_control(0, 0, 0, 0)
                    time.sleep(0.15)
                    t.land()
                return True, "ok"
            except Exception as e:
                last_err = e
                time.sleep(0.25)
        return False, str(last_err)

    def emergency(self) -> Tuple[bool, str]:
        t = self._t()
        if t is None:
            return False, "not_ready"
        with command_lock:
            t.emergency()
        return True, "ok"

    def flip(self, direction: str) -> Tuple[bool, str]:
        t = self._t()
        if t is None:
            return False, "not_ready"
        with command_lock:
            t.send_rc_control(0, 0, 0, 0)
            time.sleep(0.25)
            try:
                t.flip(direction)
            except Exception:
                time.sleep(0.25)
                t.flip(direction)
        return True, "ok"

    def move(self, direction: str, cm: int) -> Tuple[bool, str]:
        t = self._t()
        if t is None:
            return False, "not_ready"
        with command_lock:
            t.send_rc_control(0, 0, 0, 0)
            fn = {
                "up": t.move_up, "down": t.move_down,
                "left": t.move_left, "right": t.move_right,
                "forward": t.move_forward, "back": t.move_back,
            }[direction]
            fn(cm)
        return True, "ok"

    def rotate(self, direction: str, degrees: int) -> Tuple[bool, str]:
        t = self._t()
        if t is None:
            return False, "not_ready"
        with command_lock:
            t.send_rc_control(0, 0, 0, 0)
            if direction == "cw":
                t.rotate_clockwise(degrees)
            else:
                t.rotate_counter_clockwise(degrees)
        return True, "ok"

    def go_xyz(self, x: int, y: int, z: int, speed: int) -> Tuple[bool, str]:
        t = self._t()
        if t is None:
            return False, "not_ready"
        with command_lock:
            t.send_rc_control(0, 0, 0, 0)
            t.go_xyz_speed(x, y, z, speed)
        return True, "ok"

    # --- RC ---
    def send_rc(self, lr: int, fb: int, ud: int, yaw: int):
        t = self._t()
        if t is not None:
            t.send_rc_control(lr, fb, ud, yaw)

    def send_rc_zero(self):
        t = self._t()
        if t is not None:
            try:
                t.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass

    def before_discrete_command(self):
        self.send_rc_zero()

    def after_discrete_command(self):
        pass

    def rc_during_discrete(self) -> str:
        return "zeros"

    # --- Telemetry ---
    def poll_telemetry(self) -> dict:
        t = self._t()
        st = {}
        if t is not None:
            try:
                st = t.get_current_state() or {}
            except Exception:
                st = {}
        if not st:
            return {}
        data = {}
        for k, src in (
            ("battery", "bat"), ("height_cm", "h"), ("tof_cm", "tof"),
            ("barometer_cm", "baro"), ("flight_time_s", "time"), ("wifi_snr", "wifi"),
            ("pitch", "pitch"), ("roll", "roll"), ("yaw", "yaw"),
            ("vgx", "vgx"), ("vgy", "vgy"), ("vgz", "vgz"),
            ("agx", "agx"), ("agy", "agy"), ("agz", "agz"),
            ("mid", "mid"), ("pad_x", "x"), ("pad_y", "y"), ("pad_z", "z"),
        ):
            v = _as_int(st.get(src))
            if v is not None:
                data[k] = v
        mpry = st.get("mpry")
        if mpry is not None:
            data["pad_mpry"] = str(mpry)
        tl = _as_int(st.get("templ"))
        th = _as_int(st.get("temph"))
        temp = int((tl + th) / 2) if tl is not None and th is not None else (tl if tl is not None else th)
        if temp is not None:
            data["temperature"] = temp
        data["sdk_version"] = self.sdk_version
        data["serial_number"] = self.serial_number
        # Internal flags for telemetry_loop
        data["_got_any"] = bool(st)  # True if we got any state from Tello
        h = _as_int(st.get("h"))
        data["_sdk_flying"] = h is not None and h > 5
        return data

    # --- Recovery ---
    def recover(self) -> Tuple[bool, str]:
        old = self.tello
        if old is not None:
            try:
                old.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass
            try:
                old.land()
            except Exception:
                pass
            try:
                old.streamoff()
            except Exception:
                pass
            try:
                old.end()
            except Exception:
                pass
        t = Tello(host=self.ip)
        self.tello = t
        time.sleep(1.0)
        t.connect()
        t.streamon()
        self._refresh_info()
        return True, "recovered_waiting_state"

    # --- Config ---
    def yaw_stick(self) -> int:
        return STICK

    def has_altitude_fence(self) -> bool:
        return False

    def shutdown(self):
        t = self._t()
        if t is None:
            return
        try:
            t.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass
        if flying:
            try:
                t.land()
            except Exception:
                pass
        try:
            t.end()
        except Exception:
            pass

    # --- Tello-specific ---
    def set_speed(self, speed: int) -> Tuple[bool, str]:
        t = self._t()
        if t is None:
            return False, "not_ready"
        with command_lock:
            t.set_speed(speed)
        return True, "ok"

    def curve(self, x1, y1, z1, x2, y2, z2, speed) -> Tuple[bool, str]:
        t = self._t()
        if t is None:
            return False, "not_ready"
        with command_lock:
            t.send_rc_control(0, 0, 0, 0)
            t.curve_xyz_speed(x1, y1, z1, x2, y2, z2, speed)
        return True, "ok"

    def stream_control(self, action: str) -> Tuple[bool, str]:
        t = self._t()
        if t is None:
            return False, "not_ready"
        with command_lock:
            if action == "on":
                t.streamon()
            elif action == "off":
                t.streamoff()
        return True, "ok"

    def sdk_passthrough(self, command: str) -> Tuple[bool, str]:
        t = self._t()
        if t is None:
            return False, "not_ready"
        with command_lock:
            t.send_rc_control(0, 0, 0, 0)
            resp = t.send_command_with_return(command)
        return True, str(resp)

    # --- Video (Tello MJPEG via get_frame_read) ---

    _video_mjpeg_active = False
    _video_frame_lock = threading.Lock()
    _video_last_jpeg: bytes = b""

    def video_start_mjpeg(self) -> Tuple[bool, str]:
        t = self._t()
        if t is None:
            return False, "not_ready"
        if not HAS_CV2:
            return False, "opencv not installed"
        if self._video_mjpeg_active:
            return True, "already_running"
        self._video_mjpeg_active = True
        th = threading.Thread(target=self._video_mjpeg_loop, daemon=True, name="tello-mjpeg")
        th.start()
        print("[TELLO] MJPEG video started")
        return True, "ok"

    def _video_mjpeg_loop(self):
        t = self._t()
        if t is None:
            self._video_mjpeg_active = False
            return
        try:
            frame_read = t.get_frame_read()
        except Exception as e:
            print(f"[TELLO] get_frame_read failed: {e}")
            self._video_mjpeg_active = False
            return
        period = 1.0 / VIDEO_FPS
        while self._video_mjpeg_active and running:
            try:
                frame = frame_read.frame
                if frame is not None:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, VIDEO_JPEG_QUALITY])
                    if ok:
                        with self._video_frame_lock:
                            self._video_last_jpeg = buf.tobytes()
            except Exception:
                pass
            time.sleep(period)

    def video_stop_mjpeg(self):
        self._video_mjpeg_active = False
        with self._video_frame_lock:
            self._video_last_jpeg = b""
        print("[TELLO] MJPEG video stopped")

    # --- Video (Tello UDP forward — relay port 11111 to C2) ---

    _video_forward_active = False
    _video_forward_proc = None

    def video_start_forward(self, host: str, port: int) -> Tuple[bool, str]:
        t = self._t()
        if t is None:
            return False, "not_ready"
        if self._video_forward_active:
            return True, "already_running"
        # Ensure stream is on (djitellopy keeps its own receiver on 11111)
        try:
            t.streamon()
        except Exception:
            pass
        self._video_forward_active = True
        self._fwd_host = host
        self._fwd_port = port
        th = threading.Thread(target=self._video_forward_loop, daemon=True, name="tello-fwd")
        th.start()
        return True, "ok"

    def _video_forward_loop(self):
        """Forward video by sniffing UDP packets on port 11111 alongside djitellopy."""
        import socket as _socket
        host, port = self._fwd_host, self._fwd_port

        # Create a second UDP socket on port 11111 with SO_REUSEADDR + SO_REUSEPORT
        # This allows us to receive the same packets djitellopy receives
        recv_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        recv_sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        # SO_REUSEPORT allows multiple sockets to bind same port (Linux 3.9+)
        if hasattr(_socket, "SO_REUSEPORT"):
            recv_sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEPORT, 1)
        try:
            recv_sock.bind(("0.0.0.0", 11111))
        except OSError as e:
            print(f"[TELLO] Cannot bind port 11111 for forwarding: {e}")
            self._video_forward_active = False
            return
        recv_sock.settimeout(2.0)

        fwd_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        print(f"[TELLO] Raw UDP video forward started → {host}:{port}")
        pkt_count = 0
        while self._video_forward_active and running:
            try:
                data, addr = recv_sock.recvfrom(65535)
                fwd_sock.sendto(data, (host, port))
                pkt_count += 1
                if pkt_count == 1:
                    print(f"[TELLO] First video packet received ({len(data)} bytes), forwarding")
            except _socket.timeout:
                # Re-send streamon in case Tello stopped
                t = self._t()
                if t is not None:
                    try:
                        t.streamon()
                    except Exception:
                        pass
            except Exception as e:
                if self._video_forward_active:
                    print(f"[TELLO] Forward relay error: {e}")
                break
        recv_sock.close()
        fwd_sock.close()
        print(f"[TELLO] Raw UDP forward stopped ({pkt_count} packets sent)")

    def video_stop_forward(self):
        self._video_forward_active = False
        print("[TELLO] UDP video forward stopped")

    def video_stop_all(self):
        self.video_stop_mjpeg()
        self.video_stop_forward()

    def video_status(self) -> dict:
        if self._video_forward_active:
            mode = "forward"
        elif self._video_mjpeg_active:
            mode = "mjpeg"
        else:
            mode = "off"
        with self._video_frame_lock:
            has_frame = len(self._video_last_jpeg) > 0
        return {"mode": mode, "has_frame": has_frame, "has_cv2": HAS_CV2}

    def get_video_jpeg(self) -> bytes:
        with self._video_frame_lock:
            return self._video_last_jpeg

    def capabilities(self) -> dict:
        return {
            "drone_type": "tello",
            "speed": True, "curve": True, "stream": True, "sdk": True,
            "camera": False, "gimbal": False, "rth": False, "moveto": False,
            "video_mjpeg": HAS_CV2, "video_forward": True,
            "settings": False, "altitude_fence": False,
        }

    def _refresh_info(self):
        t = self._t()
        if t is None:
            return
        try:
            if self.sdk_version is None:
                self.sdk_version = str(t.send_command_with_return("sdk?")).strip()
        except Exception:
            pass
        try:
            if self.serial_number is None:
                self.serial_number = str(t.send_command_with_return("sn?")).strip()
        except Exception:
            pass

    # --- Wi-Fi management (Tello-specific thread) ---
    def wifi_connect_loop(self):
        while running:
            cfg = self._load_wifi_config()
            if not cfg:
                time.sleep(WIFI_RETRY_S)
                continue
            ssid = cfg["ssid"]
            if self._wifi_connected_to(ssid):
                time.sleep(WIFI_RETRY_S)
                continue
            password = cfg["password"]
            ifname = cfg["ifname"]
            # Force a Wi-Fi rescan so the target SSID appears in nmcli's list;
            # without this, nmcli often fails with "No network with SSID ... found"
            subprocess.run(["nmcli", "dev", "wifi", "rescan", "ifname", ifname],
                           capture_output=True, text=True)
            time.sleep(2)  # give scan time to complete

            cmd_base = ["nmcli", "dev", "wifi", "connect", ssid, "password", password, "ifname", ifname]
            if cfg.get("sudo"):
                cmd_base = ["sudo"] + cmd_base
            subprocess.run(cmd_base, capture_output=True, text=True)
            time.sleep(WIFI_RETRY_S)

    def _load_wifi_config(self):
        try:
            data = json.loads(WIFI_CFG_PATH.read_text())
            ssid = str(data.get("ssid", "")).strip()
            password = str(data.get("password", "")).strip()
            ifname = str(data.get("ifname", "")).strip()
            sudo = data.get("sudo", False)
            if not ssid or not password:
                return None
            return {"ssid": ssid, "password": password, "ifname": ifname, "sudo": sudo}
        except Exception:
            return None

    def _wifi_connected_to(self, ssid: str) -> bool:
        r = subprocess.run(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"], capture_output=True, text=True)
        if r.returncode != 0:
            return False
        for line in r.stdout.splitlines():
            if line.startswith("yes:") and line[4:] == ssid:
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# OlympeBackend
# ═══════════════════════════════════════════════════════════════════════════

class OlympeBackend(DroneBackend):

    def __init__(self, ip: str):
        self.ip = ip
        self.drone: Optional[olympe.Drone] = None
        self._pcmd_seq = 0
        self._pcmd_seq_lock = threading.Lock()
        self._has_piloting_api: Optional[bool] = None

    def _d(self) -> Optional[olympe.Drone]:
        return self.drone

    # --- PCMD / piloting helpers ---
    def _detect_piloting_api(self) -> bool:
        if self._has_piloting_api is not None:
            return self._has_piloting_api
        d = self.drone
        if d is None:
            return False
        if not callable(getattr(d, "start_piloting", None)):
            self._has_piloting_api = False
            print("[ANAFI] Native piloting API not found — using raw d(PCMD(...))")
            return False
        try:
            d.start_piloting()
            self._has_piloting_api = True
            print("[ANAFI] Native piloting API works")
            return True
        except Exception as e:
            self._has_piloting_api = False
            print(f"[ANAFI] Native piloting API FAILED ({e}) — using raw d(PCMD(...))")
            return False

    _pcmd_debug_count = 0

    def _send_pcmd(self, roll, pitch, yaw, gaz, flag=1):
        d = self.drone
        if d is None:
            return
        if self.__class__._pcmd_debug_count < 5 and (roll or pitch or yaw or gaz):
            print(f"[ANAFI] PCMD: roll={roll} pitch={pitch} yaw={yaw} gaz={gaz} flag={flag} api={self._has_piloting_api}")
            self.__class__._pcmd_debug_count += 1
        if self._detect_piloting_api():
            try:
                if flag == 0:
                    d.piloting_pcmd(0, 0, 0, 0, 0)
                else:
                    d.piloting_pcmd(roll, pitch, yaw, gaz, 0)
                return
            except Exception as e:
                if self.__class__._pcmd_debug_count < 10:
                    print(f"[ANAFI] piloting_pcmd failed: {e}")
        with self._pcmd_seq_lock:
            self._pcmd_seq = (self._pcmd_seq + 1) & 0x7FFFFFFF
            seq = self._pcmd_seq
        try:
            d(PCMD(flag, roll, pitch, yaw, gaz, seq))
        except Exception:
            pass

    def _start_piloting(self):
        d = self.drone
        if d is None:
            return
        if self._detect_piloting_api():
            try:
                d.start_piloting()
            except Exception:
                pass

    def _stop_piloting(self):
        d = self.drone
        if d is None:
            return
        if self._detect_piloting_api():
            try:
                d.stop_piloting()
                return
            except Exception:
                pass
        with self._pcmd_seq_lock:
            self._pcmd_seq = (self._pcmd_seq + 1) & 0x7FFFFFFF
            seq = self._pcmd_seq
        try:
            d(PCMD(0, 0, 0, 0, 0, seq))
        except Exception:
            pass

    def _get_state(self, msg_type):
        d = self.drone
        if d is None:
            return None
        try:
            return d.get_state(msg_type)
        except Exception:
            return None

    def _is_flying_state(self, state_dict) -> bool:
        if not state_dict:
            return False
        try:
            s = state_dict.get("state")
            if s is None:
                return False
            name = str(s).lower()
            if "." in name:
                name = name.rsplit(".", 1)[-1]
            try:
                name = s.name.lower()
            except AttributeError:
                pass
            return name in ("hovering", "flying", "takingoff")
        except Exception:
            return False

    def _apply_flight_limits(self, d):
        for cmd, label in [
            (MaxAltitude(MAX_ALTITUDE_M), f"MaxAltitude={MAX_ALTITUDE_M}m"),
            (MaxVerticalSpeed(MAX_VERTICAL_SPEED), f"MaxVerticalSpeed={MAX_VERTICAL_SPEED}m/s"),
            (MaxTilt(MAX_TILT), f"MaxTilt={MAX_TILT}°"),
            (MaxRotationSpeed(MAX_YAW_SPEED), f"MaxRotationSpeed={MAX_YAW_SPEED}°/s"),
        ]:
            try:
                d(cmd).wait(_timeout=2)
                print(f"[ANAFI] Set {label}")
            except Exception as e:
                print(f"[ANAFI] Failed to set {label}: {e}")

    def _check_connected(self, d) -> bool:
        if d is None:
            return False
        try:
            st = d.get_state(BatteryStateChanged)
            if st and "percent" in st:
                return True
        except Exception:
            pass
        try:
            st = d.get_state(FlyingStateChanged)
            if st and "state" in st:
                return True
        except Exception:
            pass
        return False

    # --- Connection ---
    def connect(self) -> bool:
        if self.drone is None:
            self.drone = olympe.Drone(self.ip)
        self.drone.connect()
        return self._check_connected(self.drone)

    def disconnect(self):
        d = self.drone
        if d is not None:
            try:
                d.disconnect()
            except Exception:
                pass

    def hard_reset(self):
        """Fully destroy the current Drone object so the next connect() creates
        a fresh one. Needed when Olympe's internal state is corrupted (video
        pipeline timed out, piloting interface won't launch, etc.) — reusing
        the same Drone instance after such failures never recovers."""
        d = self.drone
        self.drone = None
        self._has_piloting_api = None   # re-probe after fresh connect
        if d is None:
            return
        # Best-effort cleanup. Any of these may hang or raise; we catch
        # everything because the goal is to drop references and let GC/
        # finalisers handle the rest.
        try: d.disconnect()
        except Exception: pass
        try: d.destroy()
        except Exception: pass
        # Small pause to let Olympe's background threads finish cancelling
        # any pending futures before we recreate the Drone next cycle.
        time.sleep(0.5)

    def is_connected(self) -> bool:
        return self._check_connected(self.drone)

    def verify_connection(self) -> bool:
        """Check if we received telemetry recently — avoids racing with get_state calls."""
        if self.drone is None:
            return False
        # Trust the telemetry loop: if we got data recently, we're connected.
        # last_state_seen is updated by telemetry_loop whenever get_state succeeds.
        return (time.time() - last_state_seen) < 5.0

    def on_connect(self):
        d = self.drone
        if d:
            self._apply_flight_limits(d)
            self._start_piloting()
            # Log magnetometer calibration state at connect time so the
            # operator knows BEFORE attempting takeoff whether a figure-8
            # calibration is needed. The single most common cause of
            # "takeoff refused" on the Anafi is a stale magnetometer.
            mag = self._read_magnetometer_state()
            if mag:
                if _magneto_needs_calibration(mag):
                    print(f"[ANAFI] ⚠ MAGNETOMETER CALIBRATION NEEDED: {mag}")
                    print("[ANAFI]   → use FreeFlight 7 app to run the figure-8")
                    print("[ANAFI]     calibration dance, or POST /api/magneto/calibrate")
                else:
                    print(f"[ANAFI] Magnetometer: {mag}")
            else:
                print("[ANAFI] Magnetometer: state not yet reported by drone")

    # --- Flight ---
    def takeoff(self) -> Tuple[bool, str]:
        d = self._d()
        if d is None:
            return False, "not_ready"
        fs = self._get_state(FlyingStateChanged)
        print(f"[ANAFI] /api/takeoff — flying={flying}, state={fs}")

        # Pre-takeoff diagnostics — ALWAYS print the current state of every
        # relevant subsystem so we know WHY when a takeoff refuses. Previous
        # version only printed problems, but some states return None when
        # they're fine, which left the log ambiguous.
        pre_alert    = self._read_alert_state()
        pre_motor    = self._read_motor_error_state()
        pre_sensors  = self._read_sensors_state()
        pre_magneto  = self._read_magnetometer_state()
        print(f"[ANAFI] Pre-takeoff diagnostics:")
        print(f"[ANAFI]   alert      = {pre_alert if pre_alert is not None else '(not available)'}")
        print(f"[ANAFI]   motor      = {pre_motor  if pre_motor  is not None else '(no issue)'}")
        print(f"[ANAFI]   sensors    = {pre_sensors if pre_sensors else '(all OK)'}")
        print(f"[ANAFI]   magneto    = {pre_magneto if pre_magneto else '(not available)'}")

        self._stop_piloting()
        time.sleep(0.2)
        print("[ANAFI] Sending TakeOff command...")
        with command_lock:
            result = d(TakeOff()).wait(_timeout=10)
        ok = False
        try:
            ok = result.success() if result is not None else False
        except Exception:
            pass
        print(f"[ANAFI] TakeOff result: ok={ok}")

        if ok:
            self._start_piloting()
            return True, "ok"

        # Poll for state change
        for i in range(6):
            time.sleep(0.5)
            fs = self._get_state(FlyingStateChanged)
            if self._is_flying_state(fs):
                self._start_piloting()
                return True, "ok_polled"
        self._start_piloting()

        # Post-failure diagnostics: read alert/motor/sensor/magneto state again
        # in case it changed as a result of the attempt.
        post_alert   = self._read_alert_state()
        post_motor   = self._read_motor_error_state()
        post_sensors = self._read_sensors_state()
        post_magneto = self._read_magnetometer_state()
        reason_parts = []
        if post_alert and post_alert != "none":
            reason_parts.append(f"alert={post_alert}")
        if post_motor:
            reason_parts.append(f"motor={post_motor}")
        if post_sensors:
            reason_parts.append(f"sensors={post_sensors}")
        # The magnetometer is the top recurring cause of Anafi takeoff refusals
        # in this project — surface it prominently.
        if _magneto_needs_calibration(post_magneto):
            reason_parts.append(f"magneto={post_magneto}")
        reason = "takeoff_failed" + (f" ({', '.join(reason_parts)})" if reason_parts else "")
        print(f"[ANAFI] takeoff refused — {reason}")
        if post_magneto:
            print(f"[ANAFI] Post-takeoff magneto status: {post_magneto}")
        if not reason_parts:
            # No single state flagged — print a generic tip so operators know
            # the recommended recovery path.
            print("[ANAFI] No specific fault flagged by Olympe. Common causes: "
                  "magnetometer needs figure-8 calibration via FreeFlight app; "
                  "drone not level; battery too low; motors recently stalled.")
        return False, reason

    def _read_alert_state(self):
        """Return the current AlertStateChanged value as a string, or None."""
        if not HAS_ALERT_STATE:
            return None
        try:
            s = self._get_state(AlertStateChanged)
            if s is None:
                return None
            # Olympe returns an OrderedDict like {'state': <AlertStateChanged_State.low_battery: 3>}
            v = s.get("state") if hasattr(s, "get") else s
            if v is None:
                return None
            # Extract the enum's name
            name = getattr(v, "name", None) or str(v)
            return name
        except Exception as e:
            return f"<err:{e}>"

    def _read_motor_error_state(self):
        """Return the motor error code if any, else None."""
        if not HAS_MOTOR_ERROR:
            return None
        try:
            s = self._get_state(MotorErrorStateChanged)
            if s is None:
                return None
            # {'motorIds': 0, 'motorError': <MotorError.noError: 0>}
            err = s.get("motorError") if hasattr(s, "get") else None
            ids = s.get("motorIds") if hasattr(s, "get") else None
            if err is None:
                return None
            name = getattr(err, "name", None) or str(err)
            if name in ("noError", "noerror"):
                return None
            return f"{name} (motor_mask={ids})"
        except Exception as e:
            return f"<err:{e}>"

    def _read_sensors_state(self):
        """Return a concise string of which sensors report KO, or None."""
        if not HAS_SENSOR_STATE:
            return None
        try:
            s = self._get_state(SensorsStatesListChanged)
            if s is None:
                return None
            # state is OrderedDict with 'sensorName' -> enum, 'sensorState' -> int (1=OK, 0=KO)
            name_enum = s.get("sensorName") if hasattr(s, "get") else None
            state_val = s.get("sensorState") if hasattr(s, "get") else None
            if name_enum is None:
                return None
            name = getattr(name_enum, "name", None) or str(name_enum)
            ok = (state_val == 1)
            return None if ok else f"{name}=KO"
        except Exception as e:
            return f"<err:{e}>"

    def _read_magnetometer_state(self):
        """Return a human-readable magnetometer calibration status, or None.

        Parrot exposes three separate messages for magnetometer state:
          - MagnetoCalibrationRequiredState: {required: 0|1}
          - MagnetoCalibrationStateChanged: per-axis calibration done bits
          - MagnetoCalibrationStartedChanged: {started: 0|1}
        We aggregate them into one string like:
          "REQUIRED" | "in-progress" | "axes=X/Y/Z, no-fail" | "calibrated"
        so a single log line tells the operator whether the figure-8 dance
        is needed before this drone will arm."""
        if not HAS_MAGNETO_CALIB or self.drone is None:
            return None
        parts = []
        try:
            req_state = self._get_state(MagnetoCalibrationRequiredState)
            if req_state is not None:
                # {'required': 0 | 1}
                req = req_state.get("required") if hasattr(req_state, "get") else None
                if req == 1 or req is True:
                    parts.append("REQUIRED")
                elif req == 0 or req is False:
                    parts.append("not-required")
        except Exception as e:
            parts.append(f"req_err:{e}")
        try:
            st_state = self._get_state(MagnetoCalibrationStateChanged)
            if st_state is not None:
                # {'xAxisCalibration': 1, 'yAxisCalibration': 1,
                #  'zAxisCalibration': 1, 'calibrationFailed': 0}
                x = st_state.get("xAxisCalibration", 0)
                y = st_state.get("yAxisCalibration", 0)
                z = st_state.get("zAxisCalibration", 0)
                failed = st_state.get("calibrationFailed", 0)
                parts.append(f"axes=x{int(x)}y{int(y)}z{int(z)}")
                if failed:
                    parts.append("FAILED")
                elif x and y and z:
                    parts.append("all-axes-ok")
        except Exception as e:
            parts.append(f"state_err:{e}")
        try:
            started_state = self._get_state(MagnetoCalibrationStartedChanged)
            if started_state is not None:
                started = started_state.get("started", 0)
                if started:
                    parts.append("in-progress")
        except Exception as e:
            parts.append(f"started_err:{e}")
        return ", ".join(parts) if parts else None

    def land(self) -> Tuple[bool, str]:
        d = self._d()
        if d is None:
            return False, "not_ready"
        self._stop_piloting()
        time.sleep(0.2)
        print("[ANAFI] Sending Landing command...")
        last_err = None
        for attempt in range(3):
            try:
                with command_lock:
                    result = d(Landing()).wait(_timeout=10)
                ok = False
                try:
                    ok = result.success() if result is not None else False
                except Exception:
                    pass
                print(f"[ANAFI] Landing attempt {attempt}: ok={ok}")
                if ok:
                    return True, "ok"
                time.sleep(0.5)
                fs = self._get_state(FlyingStateChanged)
                if not self._is_flying_state(fs):
                    return True, "ok_polled"
            except Exception as e:
                last_err = e
                time.sleep(0.25)
        return False, str(last_err) if last_err else "land_failed"

    def emergency(self) -> Tuple[bool, str]:
        d = self._d()
        if d is None:
            return False, "not_ready"
        self._stop_piloting()
        if HAS_EMERGENCY:
            with command_lock:
                d(EmergencyCmd()).wait(_timeout=3)
        else:
            with command_lock:
                d(Landing()).wait(_timeout=5)
        return True, "ok"

    def flip(self, direction: str) -> Tuple[bool, str]:
        d = self._d()
        if d is None:
            return False, "not_ready"
        dir_map = {"f": "front", "front": "front", "b": "back", "back": "back",
                    "l": "left", "left": "left", "r": "right", "right": "right"}
        flip_enum = {
            "front": Flip.direction.front, "back": Flip.direction.back,
            "left": Flip.direction.left, "right": Flip.direction.right,
        }
        olympe_dir = flip_enum[dir_map.get(direction, direction)]
        self._stop_piloting()
        time.sleep(0.1)
        with command_lock:
            result = d(Flip(olympe_dir)).wait(_timeout=5)
        self._start_piloting()
        if result and result.success():
            return True, "ok"
        return False, "flip_failed"

    def move(self, direction: str, cm: int) -> Tuple[bool, str]:
        d = self._d()
        if d is None:
            return False, "not_ready"
        dist_m = cm / 100.0
        # Olympe moveBy: dX=forward(+), dY=right(+), dZ=down(+), dPsi=yaw(rad)
        move_args = {
            "forward": (dist_m, 0, 0, 0), "back": (-dist_m, 0, 0, 0),
            "right": (0, dist_m, 0, 0), "left": (0, -dist_m, 0, 0),
            "up": (0, 0, -dist_m, 0), "down": (0, 0, dist_m, 0),
        }[direction]
        print(f"[ANAFI] move({direction}, {cm}cm) -> moveBy{move_args}")
        self._stop_piloting()
        time.sleep(0.1)
        with command_lock:
            result = d(moveBy(*move_args)).wait(_timeout=30)
        ok = False
        try:
            ok = result.success() if result is not None else False
        except Exception:
            pass
        print(f"[ANAFI] moveBy result: ok={ok}")
        self._start_piloting()
        if ok:
            return True, "ok"
        # Even if moveBy "failed", check if the drone actually moved
        return False, "move_failed"

    def rotate(self, direction: str, degrees: int) -> Tuple[bool, str]:
        d = self._d()
        if d is None:
            return False, "not_ready"
        rad = math.radians(degrees)
        d_psi = rad if direction == "cw" else -rad
        self._stop_piloting()
        time.sleep(0.1)
        with command_lock:
            result = d(moveBy(0, 0, 0, d_psi)).wait(_timeout=15)
        self._start_piloting()
        if result and result.success():
            return True, "ok"
        return False, "rotate_failed"

    def go_xyz(self, x: int, y: int, z: int, speed: int) -> Tuple[bool, str]:
        d = self._d()
        if d is None:
            return False, "not_ready"
        dx, dy, dz = x / 100.0, y / 100.0, -z / 100.0
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        self._stop_piloting()
        time.sleep(0.1)
        with command_lock:
            result = d(moveBy(dx, dy, dz, 0)).wait(_timeout=30)
        self._start_piloting()
        if result and result.success():
            return True, "ok"
        return False, "go_failed"

    # --- RC ---
    def send_rc(self, lr: int, fb: int, ud: int, yaw: int):
        self._send_pcmd(lr, fb, yaw, ud)

    def send_rc_zero(self):
        self._send_pcmd(0, 0, 0, 0, flag=0)

    def before_discrete_command(self):
        self._stop_piloting()
        time.sleep(0.2)

    def after_discrete_command(self):
        self._start_piloting()

    def rc_during_discrete(self) -> str:
        return "skip"

    # --- Telemetry ---
    def poll_telemetry(self) -> dict:
        data = {}
        got_any = False

        bat_state = self._get_state(BatteryStateChanged)
        if bat_state:
            got_any = True
            try:
                data["battery"] = int(bat_state["percent"])
            except Exception:
                pass

        att_state = self._get_state(AttitudeChanged)
        if att_state:
            got_any = True
            try:
                data["pitch"] = round(math.degrees(float(att_state["pitch"])), 2)
                data["roll"] = round(math.degrees(float(att_state["roll"])), 2)
                data["yaw"] = round(math.degrees(float(att_state["yaw"])), 2)
            except Exception:
                pass

        spd_state = self._get_state(SpeedChanged)
        if spd_state:
            got_any = True
            try:
                data["vgx"] = round(float(spd_state["speedX"]) * 100, 1)
                data["vgy"] = round(float(spd_state["speedY"]) * 100, 1)
                data["vgz"] = round(float(spd_state["speedZ"]) * 100, 1)
            except Exception:
                pass

        alt_state = self._get_state(AltitudeChanged)
        if alt_state:
            got_any = True
            try:
                data["height_cm"] = round(float(alt_state["altitude"]) * 100, 1)
            except Exception:
                pass

        if HAS_GPS_STATE:
            gps_state = self._get_state(GpsLocationChanged)
            if gps_state:
                got_any = True
                try:
                    lat = round(float(gps_state.get("latitude", 500)), 7)
                    lon = round(float(gps_state.get("longitude", 500)), 7)
                    alt = round(float(gps_state.get("altitude", 0)), 2)
                    if lat <= 400:
                        data["gps_lat"] = lat
                    if lon <= 400:
                        data["gps_lon"] = lon
                    data["gps_alt"] = alt
                except Exception:
                    pass

        if HAS_GIMBAL:
            gim_state = self._get_state(gimbal_attitude)
            if gim_state:
                try:
                    data["gimbal_pitch"] = round(float(gim_state.get("pitch_absolute", 0)), 2)
                    data["gimbal_roll"] = round(float(gim_state.get("roll_absolute", 0)), 2)
                    data["gimbal_yaw"] = round(float(gim_state.get("yaw_absolute", 0)), 2)
                except Exception:
                    pass

        # Flying state
        fly_state = self._get_state(FlyingStateChanged)
        data["_sdk_flying"] = self._is_flying_state(fly_state)
        data["_got_any"] = got_any
        return data

    # --- Recovery ---
    def recover(self) -> Tuple[bool, str]:
        d = self.drone
        if d is not None:
            try:
                d.disconnect()
            except Exception:
                pass
        d = olympe.Drone(self.ip)
        self.drone = d
        d.connect()
        ok = self._check_connected(d)
        if ok:
            self._apply_flight_limits(d)
            self._start_piloting()
        return ok, "recovered" if ok else "reconnect_failed"

    # --- Config ---
    def yaw_stick(self) -> int:
        return YAW_STICK_ANAFI

    def has_altitude_fence(self) -> bool:
        return True

    def shutdown(self):
        self.video_stop_all()
        d = self.drone
        if d is None:
            return
        try:
            self._stop_piloting()
        except Exception:
            pass
        if flying:
            try:
                d(Landing()).wait(_timeout=5)
            except Exception:
                pass
        try:
            d.disconnect()
            print("[ANAFI] Drone disconnected.")
        except Exception:
            pass

    # --- Anafi-specific capabilities ---
    def capabilities(self) -> dict:
        return {
            "drone_type": "anafi",
            "speed": False, "curve": False, "stream": False, "sdk": False,
            "camera": HAS_CAMERA, "gimbal": HAS_GIMBAL,
            "rth": HAS_RTH, "moveto": HAS_MOVE_TO,
            "video_mjpeg": HAS_CV2, "video_forward": True,
            "settings": True, "altitude_fence": True,
            "gps": HAS_GPS_STATE, "geofence": HAS_GEOFENCE,
        }

    def camera_photo(self) -> Tuple[bool, str]:
        d = self._d()
        if d is None:
            return False, "not_ready"
        if not HAS_CAMERA:
            return False, "not_supported"
        with command_lock:
            result = d(take_photo(cam_id=0)).wait(_timeout=5)
        return (True, "ok") if result and result.success() else (False, "photo_failed")

    def camera_record_start(self) -> Tuple[bool, str]:
        d = self._d()
        if d is None or not HAS_CAMERA:
            return False, "not_supported"
        with command_lock:
            result = d(start_recording(cam_id=0)).wait(_timeout=5)
        return (True, "ok") if result and result.success() else (False, "record_start_failed")

    def camera_record_stop(self) -> Tuple[bool, str]:
        d = self._d()
        if d is None or not HAS_CAMERA:
            return False, "not_supported"
        with command_lock:
            result = d(stop_recording(cam_id=0)).wait(_timeout=5)
        return (True, "ok") if result and result.success() else (False, "record_stop_failed")

    def gimbal_set(self, tilt: float, pan: float) -> Tuple[bool, str]:
        d = self._d()
        if d is None or not HAS_GIMBAL:
            return False, "not_supported"
        tilt = max(-90, min(30, tilt))
        pan = max(-180, min(180, pan))
        with command_lock:
            result = d(set_target(
                gimbal_id=0, control_mode="position",
                yaw_frame_of_reference="absolute", yaw=pan,
                pitch_frame_of_reference="absolute", pitch=tilt,
                roll_frame_of_reference="absolute", roll=0,
            )).wait(_timeout=5)
        return (True, "ok") if result and result.success() else (False, "gimbal_failed")

    def rth(self, action: str) -> Tuple[bool, str]:
        d = self._d()
        if d is None or not HAS_RTH:
            return False, "not_supported"
        if action == "start":
            with command_lock:
                result = d(return_to_home()).wait(_timeout=5)
        elif action == "cancel":
            with command_lock:
                result = d(cancel_auto_trigger()).wait(_timeout=5)
        else:
            return False, "action must be start|cancel"
        return (True, "ok") if result and result.success() else (False, f"rth_{action}_failed")

    def moveto(self, lat, lon, alt, heading) -> Tuple[bool, str]:
        d = self._d()
        if d is None or not HAS_MOVE_TO:
            return False, "not_supported"
        self._stop_piloting()
        time.sleep(0.1)
        with command_lock:
            result = d(extended_move_to(
                latitude=float(lat), longitude=float(lon), altitude=alt,
                orientation_mode="heading_start", heading=heading,
                max_horizontal_speed=MAX_TILT / 5.0,
                max_vertical_speed=MAX_VERTICAL_SPEED,
                max_yaw_rotation_speed=MAX_YAW_SPEED,
            )).wait(_timeout=60)
        self._start_piloting()
        return (True, "ok") if result and result.success() else (False, "moveto_failed")

    def get_settings(self) -> dict:
        return {
            "max_altitude_m": MAX_ALTITUDE_M,
            "max_vertical_speed": MAX_VERTICAL_SPEED,
            "max_tilt": MAX_TILT,
            "max_yaw_speed": MAX_YAW_SPEED,
            "geofence_available": HAS_GEOFENCE,
            "camera_available": HAS_CAMERA,
            "gimbal_available": HAS_GIMBAL,
            "gps_available": HAS_GPS_STATE,
            "rth_available": HAS_RTH,
            "moveto_available": HAS_MOVE_TO,
            "video_mjpeg_available": HAS_CV2,
            "video_forward_available": True,
        }

    def set_settings(self, data: dict) -> dict:
        global MAX_ALTITUDE_M, MAX_VERTICAL_SPEED, MAX_TILT, MAX_YAW_SPEED
        d = self._d()
        if d is None:
            return {"error": "not_ready"}
        results = {}
        if "max_altitude_m" in data:
            v = max(0.5, min(150, float(data["max_altitude_m"])))
            try:
                d(MaxAltitude(v)).wait(_timeout=2)
                MAX_ALTITUDE_M = v
                results["max_altitude_m"] = v
            except Exception as e:
                results["max_altitude_m_error"] = str(e)
        if "max_vertical_speed" in data:
            v = max(0.1, min(4.0, float(data["max_vertical_speed"])))
            try:
                d(MaxVerticalSpeed(v)).wait(_timeout=2)
                MAX_VERTICAL_SPEED = v
                results["max_vertical_speed"] = v
            except Exception as e:
                results["max_vertical_speed_error"] = str(e)
        if "max_tilt" in data:
            v = max(1, min(35, float(data["max_tilt"])))
            try:
                d(MaxTilt(v)).wait(_timeout=2)
                MAX_TILT = v
                results["max_tilt"] = v
            except Exception as e:
                results["max_tilt_error"] = str(e)
        if "max_yaw_speed" in data:
            v = max(1, min(200, float(data["max_yaw_speed"])))
            try:
                d(MaxRotationSpeed(v)).wait(_timeout=2)
                MAX_YAW_SPEED = v
                results["max_yaw_speed"] = v
            except Exception as e:
                results["max_yaw_speed_error"] = str(e)
        if "geofence_distance" in data and HAS_GEOFENCE:
            v = max(10, min(4000, float(data["geofence_distance"])))
            try:
                d(MaxDistance(v)).wait(_timeout=2)
                results["geofence_distance"] = v
            except Exception as e:
                results["geofence_distance_error"] = str(e)
        if "geofence_enabled" in data and HAS_GEOFENCE:
            enabled = bool(data["geofence_enabled"])
            try:
                d(NoFlyOverMaxDistance(int(enabled))).wait(_timeout=2)
                results["geofence_enabled"] = enabled
            except Exception as e:
                results["geofence_enabled_error"] = str(e)
        return results

    # --- Video (Anafi MJPEG / UDP forward) ---
    def video_start_mjpeg(self) -> Tuple[bool, str]:
        global _video_mode, _video_last_jpeg, _video_frame_count, _video_streaming
        d = self._d()
        if d is None:
            return False, "drone not connected"
        if not HAS_CV2:
            return False, "cv2 not installed — pip install opencv-python-headless"

        # Detect API
        api = "none"
        if hasattr(d, "streaming") and hasattr(d.streaming, "set_callbacks"):
            api = "modern"
        elif hasattr(d, "set_streaming_callbacks"):
            api = "legacy"
        print(f"[ANAFI] Olympe streaming API: {api}")

        _video_mode = "mjpeg"
        _video_last_jpeg = b""
        _video_frame_count = 0

        if api == "modern":
            try:
                d.streaming.set_callbacks(
                    raw_cb=self._video_frame_cb,
                    flush_raw_cb=self._video_flush_cb,
                )
                d.streaming.start()
                _video_streaming = True
                return True, "mjpeg started (modern)"
            except Exception as e:
                print(f"[ANAFI] Modern streaming failed: {e}")
                if hasattr(d, "set_streaming_callbacks"):
                    api = "legacy"

        if api == "legacy":
            try:
                d.set_streaming_callbacks(raw_cb=self._video_frame_cb)
                d.start_video_streaming()
                _video_streaming = True
                return True, "mjpeg started (legacy)"
            except Exception as e:
                return False, f"legacy streaming failed: {e}"

        return False, "no streaming API in this Olympe version"

    def video_stop_mjpeg(self):
        global _video_streaming, _video_last_jpeg
        d = self._d()
        _video_streaming = False
        _video_last_jpeg = b""
        if d is None:
            return
        try:
            d.streaming.stop()
        except Exception:
            try:
                d.stop_video_streaming()
            except Exception:
                pass

    def video_start_forward(self, host: str, port: int) -> Tuple[bool, str]:
        global _video_forward_proc, _video_forward_target, _video_mode
        self.video_stop_forward()
        _video_mode = "forward"
        _video_forward_target = f"{host}:{port}"

        socat = shutil.which("socat")
        if socat:
            cmd = [socat, "-u", f"UDP-RECV:{VIDEO_UDP_FORWARD_PORT},reuseaddr", f"UDP-SENDTO:{host}:{port}"]
            try:
                _video_forward_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True, f"socat → {host}:{port}"
            except Exception:
                pass

        gst = shutil.which("gst-launch-1.0")
        if gst:
            cmd = [gst, "-q", "udpsrc", f"port={VIDEO_UDP_FORWARD_PORT}", "!", "udpsink", f"host={host}", f"port={port}"]
            try:
                _video_forward_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True, f"gstreamer → {host}:{port}"
            except Exception:
                pass

        # Python fallback
        return self._python_udp_relay(host, port)

    def _python_udp_relay(self, host: str, port: int) -> Tuple[bool, str]:
        global _video_streaming
        import socket
        _video_streaming = True

        def relay():
            sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock_in.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock_in.bind(("0.0.0.0", VIDEO_UDP_FORWARD_PORT))
            sock_in.settimeout(1.0)
            sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            n = 0
            while _video_streaming and running:
                try:
                    data, _ = sock_in.recvfrom(65536)
                    sock_out.sendto(data, (host, port))
                    n += 1
                except socket.timeout:
                    continue
                except Exception:
                    break
            sock_in.close()
            sock_out.close()

        threading.Thread(target=relay, daemon=True).start()
        return True, f"python relay → {host}:{port}"

    def video_stop_forward(self):
        global _video_forward_proc, _video_forward_target, _video_streaming
        _video_streaming = False
        if _video_forward_proc is not None:
            try:
                _video_forward_proc.terminate()
                _video_forward_proc.wait(timeout=3)
            except Exception:
                try:
                    _video_forward_proc.kill()
                except Exception:
                    pass
            _video_forward_proc = None
        _video_forward_target = ""

    def video_stop_all(self):
        global _video_mode
        self.video_stop_mjpeg()
        self.video_stop_forward()
        _video_mode = "off"

    def video_status(self) -> dict:
        status = {
            "mode": _video_mode,
            "cv2_available": HAS_CV2,
            "socat_available": shutil.which("socat") is not None,
            "gstreamer_available": shutil.which("gst-launch-1.0") is not None,
        }
        if _video_mode == "mjpeg":
            status["frames_decoded"] = _video_frame_count
            status["has_frame"] = len(_video_last_jpeg) > 0
        elif _video_mode == "forward":
            status["target"] = _video_forward_target
            status["process_alive"] = _video_forward_proc is not None and _video_forward_proc.poll() is None
        return status

    def get_video_jpeg(self) -> bytes:
        with _video_jpeg_lock:
            return _video_last_jpeg

    def _video_frame_cb(self, yuv_frame):
        global _video_last_jpeg, _video_frame_count
        if not HAS_CV2:
            return
        try:
            yuv_frame.ref()
        except Exception:
            pass
        try:
            cv_cvt = cv2.COLOR_YUV2BGR_I420
            try:
                info = yuv_frame.info()
                yuv_fmt = None
                if isinstance(info, dict):
                    if "yuv" in info and isinstance(info["yuv"], dict):
                        yuv_fmt = info["yuv"].get("format")
                    elif "format" in info:
                        yuv_fmt = info["format"]
                    elif "raw" in info and isinstance(info["raw"], dict):
                        yuv_fmt = info["raw"].get("format")
                if yuv_fmt is not None:
                    for attr, flag in [("VDEF_I420", cv2.COLOR_YUV2BGR_I420),
                                       ("VDEF_NV12", cv2.COLOR_YUV2BGR_NV12)]:
                        c = getattr(olympe, attr, None)
                        if c is not None and c == yuv_fmt:
                            cv_cvt = flag
                            break
                if _video_frame_count == 0:
                    print(f"[ANAFI] Frame info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}")
            except Exception as ie:
                if _video_frame_count == 0:
                    print(f"[ANAFI] Frame info() failed ({ie}), using default I420")

            cv_frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv_cvt)
            ok, jpg = cv2.imencode(".jpg", cv_frame, [int(cv2.IMWRITE_JPEG_QUALITY), VIDEO_JPEG_QUALITY])
            if ok:
                with _video_jpeg_lock:
                    _video_last_jpeg = jpg.tobytes()
                _video_frame_count += 1
                if _video_frame_count == 1:
                    h, w = cv_frame.shape[:2]
                    print(f"[ANAFI] First video frame: {w}x{h}")
            # Tap frame for positioning and/or recording (non-blocking — drop if queue full).
            # Feeding the queue when ONLY recording is active ensures the recorder
            # still gets frames even if Position Tracker is disabled.
            with _pos_cfg_lock:
                _pos_enabled = _pos_cfg.get("enabled", False)
            with _rec_lock:
                _rec_active = _rec_enabled
            if _pos_enabled or _rec_active:
                try:
                    _pos_frame_q.put_nowait((cv_frame.copy(), time.monotonic()))
                except queue.Full:
                    pass
        except Exception as e:
            if _video_frame_count == 0:
                print(f"[ANAFI] Video frame error: {e}")
        finally:
            try:
                yuv_frame.unref()
            except Exception:
                pass

    def _video_flush_cb(self, stream):
        try:
            if hasattr(stream, "empty"):
                while not stream.empty():
                    try:
                        stream.get(timeout=0.005).unref()
                    except Exception:
                        break
            elif hasattr(stream, "get"):
                while True:
                    try:
                        stream.get(timeout=0.005).unref()
                    except Exception:
                        break
        except Exception:
            pass
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Backend instance (set at startup)
# ═══════════════════════════════════════════════════════════════════════════
backend: Optional[DroneBackend] = None


# ═══════════════════════════════════════════════════════════════════════════
# Background threads
# ═══════════════════════════════════════════════════════════════════════════

def telemetry_loop():
    global flying, last_state_seen
    while running:
        loop_mono = time.monotonic()
        b = backend
        if b is None:
            time.sleep(0.5)
            continue

        data = b.poll_telemetry()

        # Extract internal flags before processing
        sdk_flying = data.pop("_sdk_flying", None)
        got_any = data.pop("_got_any", False)

        # Only mark as connected if we got REAL sensor data from the drone
        if got_any:
            last_state_seen = time.time()
            with conn_lock:
                if not conn_state["connected"]:
                    conn_state["connected"] = True
                    print(f"[{drone_type.upper()}] Connection detected via telemetry")
        if sdk_flying is not None:
            with command_lock:
                in_discrete = time.time() < discrete_until
            if not in_discrete:
                flying = sdk_flying

        # Compute speed
        vgx = data.get("vgx") or 0
        vgy = data.get("vgy") or 0
        vgz = data.get("vgz") or 0
        if vgx or vgy or vgz:
            data["speed"] = round((vgx**2 + vgy**2 + vgz**2) ** 0.5, 1)

        with conn_lock:
            connected_now = conn_state["connected"]

        with telemetry_lock:
            for k, v in data.items():
                if v is not None:
                    telemetry[k] = v
            telemetry["flying"] = flying
            telemetry["connected"] = connected_now
            telemetry["updated_at"] = time.time()
            telemetry["updated_at_mono"] = loop_mono
            snapshot = dict(telemetry)

        _append_telemetry_sync_sample(
            {
                "ts_mono": loop_mono,
                "vgx": snapshot.get("vgx"),
                "vgy": snapshot.get("vgy"),
                "vgz": snapshot.get("vgz"),
                "yaw": snapshot.get("yaw"),
                "pitch": snapshot.get("pitch"),
                "roll": snapshot.get("roll"),
                "height_cm": snapshot.get("height_cm"),
                "connected": snapshot.get("connected", False),
            }
        )

        append_telemetry_log(snapshot)
        hz = TELEMETRY_HZ if TELEMETRY_HZ > 0 else 2.0
        time.sleep(max(0.05, 1.0 / hz))


def reconnect_loop():
    """Watchdog: maintain the drone connection. When disconnected, retry with
    exponential backoff. After a few consecutive failures (or if we detect a
    corrupted Olympe state), do a HARD RESET — fully destroy and recreate
    the Drone object so the next connect() starts from a clean slate.
    Without this, 'Connection reset by peer' errors can leave Olympe stuck
    forever in 'Unable to launch piloting interface' because pdraw cleanup
    timed out and the old Drone instance is unusable."""
    global last_conn_print, last_state_seen
    while running:
        b = backend
        if b is None:
            time.sleep(1.0)
            continue

        now = time.time()
        with conn_lock:
            last_try = conn_state["last_reconnect"]
            last_verify = conn_state.get("last_verify", 0.0)
            connected_now = conn_state["connected"]
            n_fail = conn_state.get("consecutive_failures", 0)

        if connected_now != last_conn_print:
            tag = drone_type.upper()
            print(f"[{tag}] Drone connected" if connected_now else f"[{tag}] Drone disconnected (retrying...)")
            last_conn_print = connected_now
            if connected_now:
                # Success — reset failure counter
                with conn_lock:
                    conn_state["consecutive_failures"] = 0
                    conn_state["last_error"] = ""

        # Health check while connected
        if connected_now:
            if drone_type == "tello":
                # Tello: periodic verify + ping
                if (now - last_verify) >= VERIFY_RETRY_S:
                    if not _host_reachable(drone_ip) or not b.verify_connection():
                        with conn_lock:
                            conn_state["connected"] = False
                    with conn_lock:
                        conn_state["last_verify"] = now
            else:
                # Anafi: trust telemetry — if we got state data recently, stay connected.
                # Only declare disconnect after 6s of no telemetry (3 missed cycles).
                if not b.verify_connection():
                    # Double-check with a ping before declaring disconnect
                    if not _host_reachable(drone_ip):
                        print("[ANAFI] Connection lost (no telemetry + ping failed)")
                        with conn_lock:
                            conn_state["connected"] = False
                            conn_state["last_failure_ts"] = now
                            # Count the drop so the reconnect path can escalate to hard-reset
                            conn_state["consecutive_failures"] = n_fail + 1
                            conn_state["last_error"] = "telemetry stale + ping failed"
            time.sleep(2.0)
            continue

        # Backoff: grows from CONNECT_RETRY_S up to RECONNECT_BACKOFF_MAX_S
        backoff = min(CONNECT_RETRY_S * (2 ** max(0, n_fail - 1)),
                      RECONNECT_BACKOFF_MAX_S)
        should_retry = (now - last_try) >= backoff
        if drone_type == "tello":
            should_retry = should_retry and _host_reachable(drone_ip)

        if should_retry:
            with conn_lock:
                conn_state["last_reconnect"] = now
                conn_state["connected"] = False

            # HARD RESET: after 2+ consecutive failures, destroy the Drone
            # object so we don't keep retrying on a corrupted Olympe instance.
            if drone_type != "tello" and n_fail >= RECONNECT_HARD_RESET_AFTER:
                hr = getattr(b, "hard_reset", None)
                if callable(hr):
                    print(f"[{drone_type.upper()}] Hard-resetting Olympe Drone object "
                          f"(failure #{n_fail}, backoff={backoff:.1f}s)")
                    try:
                        hr()
                    except Exception as e:
                        print(f"[{drone_type.upper()}] hard_reset raised: {e}")

            try:
                print(f"[{drone_type.upper()}] Attempting reconnect "
                      f"(failure #{n_fail}, backoff={backoff:.1f}s)...")
                ok = b.connect()
                if ok:
                    b.on_connect()
                    last_state_seen = time.time()  # seed for health check
                    print(f"[{drone_type.upper()}] Connection verified")
                    with conn_lock:
                        conn_state["consecutive_failures"] = 0
                        conn_state["last_error"] = ""
                else:
                    with conn_lock:
                        conn_state["consecutive_failures"] = n_fail + 1
                        conn_state["last_error"] = "connect() returned False"
                with conn_lock:
                    conn_state["connected"] = ok
                    conn_state["last_verify"] = now
                if ok and drone_type == "tello":
                    last_state_seen = 0.0
            except Exception as e:
                print(f"[{drone_type.upper()}] Connect failed: {e}")
                with conn_lock:
                    conn_state["connected"] = False
                    conn_state["consecutive_failures"] = n_fail + 1
                    conn_state["last_error"] = str(e)[:120]

        time.sleep(1.0)


def rc_loop():
    global running, flying, rc_override, rc_override_until, takeoff_cooldown_until
    period = 1.0 / RC_HZ
    _was_connected = False
    while running:
        t0 = time.time()
        reap_stale_keys(t0)
        b = backend

        with conn_lock:
            connected = conn_state["connected"]

        if not connected or b is None:
            _was_connected = False
            time.sleep(period)
            continue

        # On (re)connect
        if not _was_connected:
            b.after_discrete_command()  # start_piloting for Anafi, no-op for Tello
            _was_connected = True

        # Takeoff via key
        if has_key("t") and not flying:
            try:
                hold_s = SAFE_TAKEOFF_S if safe_takeoff_enabled else 3.0
                start_discrete_window(hold_s)
                b.before_discrete_command()
                ok, msg = b.takeoff()
                if ok:
                    flying = True
                    takeoff_cooldown_until = time.time() + hold_s
                b.after_discrete_command()
            except Exception as e:
                print(f"[{drone_type.upper()}] Takeoff error: {e}")
                b.after_discrete_command()
            remove_key("t")

        # Land via key
        if has_key("l") and flying:
            try:
                start_discrete_window(3.0)
                with pressed_lock:
                    pressed_web.clear()
                    key_last_seen.clear()
                with rc_lock:
                    rc_override = None
                    rc_override_until = 0.0
                ok, msg = b.land()
                if ok:
                    flying = False
            except Exception as e:
                print(f"[{drone_type.upper()}] Land error: {e}")
            remove_key("l")

        # Build RC axes
        lr = axis(has_key("d"), has_key("a")) * STICK
        fb = axis(has_key("w"), has_key("s")) * STICK
        ud = axis(has_key("r"), has_key("f")) * STICK
        yaw = axis(has_key("e"), has_key("q")) * b.yaw_stick()

        if has_key("space") or has_key("x"):
            lr = fb = ud = yaw = 0

        now = time.time()
        with rc_lock:
            if rc_override is not None and now < rc_override_until:
                lr, fb, ud, yaw = rc_override
            elif rc_override is not None:
                rc_override = None

        # Altitude fence (Anafi)
        if b.has_altitude_fence():
            with telemetry_lock:
                cur_h = telemetry.get("height_cm")
            if cur_h is not None and cur_h >= MAX_ALTITUDE_M * 100 and ud > 0:
                ud = 0

        with command_lock:
            in_discrete = now < discrete_until

        if in_discrete:
            mode = b.rc_during_discrete()
            if mode == "zeros":
                b.send_rc(0, 0, 0, 0)
            # else "skip": send nothing
        else:
            b.send_rc(lr, fb, ud, yaw)

        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def watchdog_loop():
    global flying, _watchdog_landed
    while running:
        time.sleep(1.0)
        if not flying or _watchdog_landed:
            continue
        if last_remote_request <= 0:
            continue
        silence = time.time() - last_remote_request
        if silence >= REMOTE_TIMEOUT_S:
            tag = drone_type.upper()
            print(f"[{tag}] WATCHDOG: No remote request for {silence:.1f}s — AUTO-LANDING")
            _watchdog_landed = True
            b = backend
            if b is None:
                continue
            try:
                with pressed_lock:
                    pressed_web.clear()
                    key_last_seen.clear()
                b.before_discrete_command()
                ok, msg = b.land()
                if ok:
                    flying = False
                print(f"[{tag}] WATCHDOG: Auto-land sent (ok={ok})")
            except Exception as e:
                print(f"[{tag}] WATCHDOG: Auto-land failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Flask middleware
# ═══════════════════════════════════════════════════════════════════════════

@app.before_request
def _update_remote_heartbeat():
    global last_remote_request, _watchdog_landed
    try:
        if request.path.startswith("/api/"):
            last_remote_request = time.time()
            _watchdog_landed = False
    except Exception:
        pass

@app.before_request
def _log_incoming_api_command():
    try:
        if request.method != "POST" or not request.path.startswith("/api/") or request.path.startswith("/api/logging/"):
            return
        payload = request.get_json(silent=True) or {}
        append_command_log(request.path, payload)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Flask endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    with conn_lock:
        connected = conn_state["connected"]
    return jsonify(ok=True, service="unified_pi_api_server", drone_type=drone_type, connected=connected)


@app.get("/api/debug")
def api_debug():
    """Debug endpoint — curl http://PI:8080/api/debug to inspect state."""
    b = backend
    with conn_lock:
        cs = dict(conn_state)
    with telemetry_lock:
        tel = dict(telemetry)
    return jsonify(
        code_version="2026-03-26-v2",
        drone_type=drone_type,
        drone_ip=drone_ip,
        conn_state=cs,
        telemetry=tel,
        flying=flying,
        last_state_seen=last_state_seen,
        state_age=round(time.time() - last_state_seen, 3) if last_state_seen else None,
        last_remote_request=last_remote_request,
        remote_age=round(time.time() - last_remote_request, 3) if last_remote_request else None,
        backend_type=type(b).__name__ if b else None,
        backend_drone_exists=b.drone is not None if b and hasattr(b, "drone") else None,
        has_olympe=HAS_OLYMPE_SDK,
        has_tello=HAS_TELLO_SDK,
        has_cv2=HAS_CV2,
    )


@app.get("/api/heartbeat")
def api_heartbeat():
    with conn_lock:
        connected = conn_state["connected"]
    return jsonify(ok=True, flying=flying, connected=connected, drone_type=drone_type, t=time.time())


@app.get("/api/magneto")
def api_magneto_get():
    """Return the current magnetometer calibration status for this drone.
    Returns {ok, status} where status is a human-readable string like
    'REQUIRED, axes=x0y0z0' or 'not-required, all-axes-ok'. Returns None
    for status if the drone isn't Anafi or the state hasn't been reported."""
    b = backend
    mag = None
    reader = getattr(b, "_read_magnetometer_state", None) if b else None
    if callable(reader):
        try:
            mag = reader()
        except Exception as e:
            mag = f"err:{e}"
    required = _magneto_needs_calibration(mag)
    return jsonify(ok=True, status=mag, required=required, drone_type=drone_type)


@app.post("/api/magneto/calibrate")
def api_magneto_calibrate():
    """Trigger magnetometer calibration on the drone. After calling this,
    the operator must rotate the drone in a figure-8 motion around each
    axis (≈ 30 seconds total) until the drone confirms completion via
    MagnetoCalibrationStateChanged. Safer to do this via the FreeFlight
    app where the drone's LEDs give visual feedback on each axis."""
    if not HAS_OLYMPE_SDK or not HAS_MAGNETO_CALIB_CMD:
        return jsonify(ok=False, error="magnetometer calibration not supported "
                                        "by this Olympe build"), 400
    b = backend
    if b is None or getattr(b, "drone", None) is None:
        return jsonify(ok=False, error="drone not connected"), 503
    try:
        # '1' = start calibration
        with command_lock:
            b.drone(StartMagnetoCalibration(1)).wait(_timeout=2)
        print("[ANAFI] Magnetometer calibration REQUESTED — operator must "
              "rotate drone in figure-8 around each axis.")
        return jsonify(ok=True, message="calibration started — rotate drone in figure-8")
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


# ─── Wi-Fi control (Anafi is dual-band 2.4/5 GHz) ───────────────────────────

def _anafi_backend():
    """Return the active OlympeBackend or None."""
    b = backend
    if b is None or not isinstance(b, OlympeBackend):
        return None
    if b.drone is None:
        return None
    return b


@app.get("/api/wifi/status")
def api_wifi_status():
    """Report the current AP channel + band the Anafi is broadcasting on."""
    if not HAS_WIFI_CTRL:
        return jsonify(ok=False, error="Wi-Fi control not supported by this "
                                        "Olympe build"), 501
    b = _anafi_backend()
    if b is None:
        return jsonify(ok=False, error="drone not connected"), 503
    try:
        st = b._get_state(WifiApChannelChanged) if WifiApChannelChanged else None
        if st is None:
            return jsonify(ok=True, status="not-reported")
        return jsonify(ok=True, status=dict(st))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/wifi/scan")
def api_wifi_scan():
    """Scan Wi-Fi channels in the requested band and return authorized
    channels with their measured RSSI. Pass {"band": "2_4_GHz"} or
    {"band": "5_GHz"} or {"band": "all"} (default).

    The scan takes ~2-3 seconds. Results come from multiple
    AuthorizedChannel events that Olympe collects in response."""
    if not HAS_WIFI_CTRL:
        return jsonify(ok=False, error="Wi-Fi control not supported"), 501
    b = _anafi_backend()
    if b is None:
        return jsonify(ok=False, error="drone not connected"), 503
    data = request.get_json(silent=True) or {}
    band_str = str(data.get("band", "all")).lower()
    try:
        band = {
            "2_4_ghz": getattr(WifiBand, "2_4_GHz", None) if "WifiBand" in globals() else "2_4_GHz",
            "2.4":     getattr(WifiBand, "2_4_GHz", None) if "WifiBand" in globals() else "2_4_GHz",
            "5_ghz":   getattr(WifiBand, "5_GHz",   None) if "WifiBand" in globals() else "5_GHz",
            "5":       getattr(WifiBand, "5_GHz",   None) if "WifiBand" in globals() else "5_GHz",
            "all":     getattr(WifiBand, "all",     None) if "WifiBand" in globals() else "all",
        }.get(band_str, band_str)
        with command_lock:
            b.drone(WifiScan(band=band)).wait(_timeout=5)
        # Harvest the authorized-channel events. Olympe's state dict keeps
        # the latest per channel, so we pull them all.
        channels = []
        try:
            # AuthorizedChannel is a multi-valued state. Try state() to get
            # all known channels, but API varies by Olympe version.
            all_ch = b.drone.get_state(WifiAuthorizedChannel)
            if isinstance(all_ch, dict):
                channels = list(all_ch.values())
            elif isinstance(all_ch, list):
                channels = all_ch
        except Exception:
            pass
        return jsonify(ok=True, band=str(band), channels=channels)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/wifi/channel")
def api_wifi_channel():
    """Set the Anafi's AP channel. Two modes:

    - Automatic (recommended): {"auto": true, "band": "5_GHz"} — drone
      scans and picks the cleanest channel in the requested band.
    - Manual: {"auto": false, "band": "5_GHz", "channel": 36} — pin to
      a specific channel. Valid channels: 1–13 for 2.4 GHz, 36/40/44/48/
      149/153/157/161/165 for 5 GHz (region-dependent).

    IMPORTANT: changing the AP channel momentarily drops the Wi-Fi
    connection (drone re-associates on the new channel). The watchdog
    will reconnect within a few seconds. Only issue this command when
    the drone is ON THE GROUND.
    """
    if not HAS_WIFI_CTRL:
        return jsonify(ok=False, error="Wi-Fi control not supported"), 501
    b = _anafi_backend()
    if b is None:
        return jsonify(ok=False, error="drone not connected"), 503
    data = request.get_json(silent=True) or {}
    auto_mode = bool(data.get("auto", True))
    band_str  = str(data.get("band", "5_GHz")).lower()
    channel   = int(data.get("channel", 0))

    try:
        # Map string → enum (defensive against naming drift between Olympe versions)
        band = {
            "2_4_ghz": getattr(WifiBand, "2_4_GHz", None) if "WifiBand" in globals() else "2_4_GHz",
            "2.4":     getattr(WifiBand, "2_4_GHz", None) if "WifiBand" in globals() else "2_4_GHz",
            "5_ghz":   getattr(WifiBand, "5_GHz",   None) if "WifiBand" in globals() else "5_GHz",
            "5":       getattr(WifiBand, "5_GHz",   None) if "WifiBand" in globals() else "5_GHz",
        }.get(band_str, band_str)
        if auto_mode:
            # Auto on this band — channel=0 means "pick the best one"
            sel_type = getattr(WifiSelectionType, "auto_all",
                         getattr(WifiSelectionType, "auto_2_4_ghz", "auto_all")) \
                       if "WifiSelectionType" in globals() else "auto"
            if band_str in ("5_ghz", "5"):
                sel_type = getattr(WifiSelectionType, "auto_5_ghz",
                                   getattr(WifiSelectionType, "auto_all", "auto_5_ghz")) \
                           if "WifiSelectionType" in globals() else "auto"
            elif band_str in ("2_4_ghz", "2.4"):
                sel_type = getattr(WifiSelectionType, "auto_2_4_ghz",
                                   getattr(WifiSelectionType, "auto_all", "auto_2_4_ghz")) \
                           if "WifiSelectionType" in globals() else "auto"
            with command_lock:
                b.drone(WifiSetApChannel(type=sel_type, band=band, channel=0)).wait(_timeout=5)
            print(f"[ANAFI] Wi-Fi: AUTO channel selection in band {band_str}")
            return jsonify(ok=True, mode="auto", band=band_str,
                           message=f"drone picks best {band_str} channel; "
                                   "connection will drop briefly")
        else:
            manual_type = getattr(WifiSelectionType, "manual", "manual") \
                          if "WifiSelectionType" in globals() else "manual"
            with command_lock:
                b.drone(WifiSetApChannel(type=manual_type, band=band,
                                         channel=channel)).wait(_timeout=5)
            print(f"[ANAFI] Wi-Fi: MANUAL → band={band_str}, channel={channel}")
            return jsonify(ok=True, mode="manual", band=band_str, channel=channel,
                           message=f"channel set to {channel}; connection will "
                                   "drop briefly")
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.get("/api/capabilities")
def api_capabilities():
    b = backend
    caps = b.capabilities() if b else {}
    caps["drone_type"] = drone_type
    caps["drone_ip"] = drone_ip
    caps["tello_sdk"] = HAS_TELLO_SDK
    caps["olympe_sdk"] = HAS_OLYMPE_SDK
    return jsonify(**caps)


@app.post("/api/key_down")
def api_key_down():
    data = request.get_json(silent=True) or {}
    add_key(data.get("key", ""))
    return jsonify(ok=True)


@app.post("/api/key_up")
def api_key_up():
    data = request.get_json(silent=True) or {}
    remove_key(data.get("key", ""))
    return jsonify(ok=True)


@app.post("/api/takeoff")
def api_takeoff():
    global flying, takeoff_cooldown_until
    b = backend
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    try:
        if not flying:
            hold_s = SAFE_TAKEOFF_S if safe_takeoff_enabled else 3.0
            start_discrete_window(hold_s)
            b.before_discrete_command()
            ok, msg = b.takeoff()
            b.after_discrete_command()
            if ok:
                flying = True
                takeoff_cooldown_until = time.time() + hold_s
            else:
                print(f"[{drone_type.upper()}] Takeoff returned ok=False msg={msg}")
                return jsonify(ok=False, error=msg), 500
        return jsonify(ok=True, flying=flying, safe_takeoff=safe_takeoff_enabled)
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Tello: attempt recovery on failure
        if drone_type == "tello":
            rok, rmsg = b.recover()
            return jsonify(ok=False, error="takeoff_failed", recovered=rok, message=rmsg), 500
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/land")
def api_land():
    global flying, rc_override, rc_override_until
    b = backend
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    try:
        if not flying:
            return jsonify(ok=True, flying=False)
        start_discrete_window(3.0)
        with pressed_lock:
            pressed_web.clear()
            key_last_seen.clear()
        with rc_lock:
            rc_override = None
            rc_override_until = 0.0
        b.before_discrete_command()
        ok, msg = b.land()
        b.after_discrete_command()
        if ok:
            flying = False
            return jsonify(ok=True, flying=False)
        print(f"[{drone_type.upper()}] Land returned ok=False msg={msg}")
        return jsonify(ok=False, error=msg), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        if drone_type == "tello":
            rok, rmsg = b.recover()
            return jsonify(ok=False, error="land_failed", recovered=rok, message=rmsg), 500
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/flip")
def api_flip():
    b = backend
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    direction = str(data.get("dir", data.get("direction", ""))).lower()
    if direction not in {"l", "r", "f", "b", "left", "right", "front", "back"}:
        return jsonify(ok=False, error="dir must be one of l|r|f|b"), 400
    now = time.time()
    if now < takeoff_cooldown_until:
        return jsonify(ok=False, error="flip_blocked_takeoff_cooldown"), 409
    with telemetry_lock:
        bat = telemetry.get("battery")
        is_flying = bool(telemetry.get("flying"))
    if not is_flying:
        return jsonify(ok=False, error="flip_requires_flying"), 409
    if bat is not None and bat < 50:
        return jsonify(ok=False, error="flip_requires_battery_50_plus", battery=bat), 409
    try:
        start_discrete_window(2.0 if drone_type == "anafi" else 1.2)
        ok, msg = b.flip(direction)
        if ok:
            return jsonify(ok=True, dir=direction)
        return jsonify(ok=False, error=msg), 500
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/emergency")
def api_emergency():
    global flying
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    try:
        ok, msg = b.emergency()
        flying = False
        return jsonify(ok=ok)
    except Exception as e:
        flying = False
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/move")
def api_move():
    b = backend
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    direction = str(data.get("dir", "")).lower()
    if direction not in {"up", "down", "left", "right", "forward", "back"}:
        return jsonify(ok=False, error="dir must be one of up|down|left|right|forward|back"), 400
    try:
        dist_cm = max(20, min(500, int(data.get("cm", 20))))
        start_discrete_window(1.0 if drone_type == "tello" else max(1.0, dist_cm / 50))
        ok, msg = b.move(direction, dist_cm)
        if ok:
            return jsonify(ok=True, dir=direction, cm=dist_cm)
        return jsonify(ok=False, error=msg), 500
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/rotate")
def api_rotate():
    b = backend
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    direction = str(data.get("dir", "")).lower()
    if direction not in {"cw", "ccw"}:
        return jsonify(ok=False, error="dir must be one of cw|ccw"), 400
    try:
        degrees = max(1, min(360, int(data.get("deg", 45))))
        start_discrete_window(1.0 if drone_type == "tello" else max(1.0, degrees / 90))
        ok, msg = b.rotate(direction, degrees)
        if ok:
            return jsonify(ok=True, dir=direction, deg=degrees)
        return jsonify(ok=False, error=msg), 500
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/go")
def api_go():
    b = backend
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    try:
        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        z = int(data.get("z", 0))
        speed = max(10, min(100, int(data.get("speed", 20))))
        start_discrete_window(1.5)
        ok, msg = b.go_xyz(x, y, z, speed)
        if ok:
            return jsonify(ok=True, x=x, y=y, z=z, speed=speed)
        return jsonify(ok=False, error=msg), 500
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/rc")
def api_rc():
    global rc_override, rc_override_until
    data = request.get_json(silent=True) or {}
    def clamp(v):
        try:
            return max(-100, min(100, int(v)))
        except Exception:
            return 0
    lr = clamp(data.get("lr", 0))
    fb = clamp(data.get("fb", 0))
    ud = clamp(data.get("ud", 0))
    yaw = clamp(data.get("yaw", 0))
    dur_ms = max(50, min(2000, int(data.get("duration_ms", 250))))
    with rc_lock:
        rc_override = (lr, fb, ud, yaw)
        rc_override_until = time.time() + (dur_ms / 1000.0)
    return jsonify(ok=True, rc={"lr": lr, "fb": fb, "ud": ud, "yaw": yaw}, duration_ms=dur_ms)


@app.post("/api/recover")
def api_recover():
    global flying
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    with pressed_lock:
        pressed_web.clear()
        key_last_seen.clear()
    try:
        ok, msg = b.recover()
        flying = False
        with conn_lock:
            conn_state["connected"] = ok
            conn_state["last_reconnect"] = time.time()
        return jsonify(ok=ok, message=msg)
    except Exception as e:
        with conn_lock:
            conn_state["connected"] = False
        return jsonify(ok=False, message=str(e)), 500


# --- Tello-specific endpoints ---
@app.post("/api/speed")
def api_speed():
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    speed = max(10, min(100, int(data.get("speed", 30))))
    try:
        ok, msg = b.set_speed(speed)
        if ok:
            return jsonify(ok=True, speed=speed)
        return jsonify(ok=False, error=msg), 501
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/curve")
def api_curve():
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    try:
        x1, y1, z1 = int(data.get("x1", 20)), int(data.get("y1", 0)), int(data.get("z1", 0))
        x2, y2, z2 = int(data.get("x2", 40)), int(data.get("y2", 0)), int(data.get("z2", 0))
        speed = max(10, min(60, int(data.get("speed", 20))))
        start_discrete_window(1.5)
        ok, msg = b.curve(x1, y1, z1, x2, y2, z2, speed)
        if ok:
            return jsonify(ok=True, x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2, speed=speed)
        return jsonify(ok=False, error=msg), 501
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/stream")
def api_stream():
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    action = str(data.get("action", "on")).lower()
    if action not in {"on", "off"}:
        return jsonify(ok=False, error="action must be on|off"), 400
    try:
        ok, msg = b.stream_control(action)
        if ok:
            return jsonify(ok=True, action=action)
        return jsonify(ok=False, error=msg), 501
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/sdk")
def api_sdk_passthrough():
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    command = str(data.get("command", "")).strip()
    if not command:
        return jsonify(ok=False, error="command required"), 400
    try:
        start_discrete_window(0.8)
        ok, resp = b.sdk_passthrough(command)
        if ok:
            return jsonify(ok=True, command=command, response=resp)
        return jsonify(ok=False, error=resp), 501
    except Exception as e:
        return jsonify(ok=False, error=str(e), command=command), 500


# --- Anafi-specific endpoints ---
@app.post("/api/camera/photo")
def api_camera_photo():
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    ok, msg = b.camera_photo()
    return jsonify(ok=ok) if ok else (jsonify(ok=False, error=msg), 501 if msg == "not_supported" else 500)


@app.post("/api/camera/record/start")
def api_camera_record_start():
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    ok, msg = b.camera_record_start()
    return jsonify(ok=ok, recording=True) if ok else (jsonify(ok=False, error=msg), 501 if msg == "not_supported" else 500)


@app.post("/api/camera/record/stop")
def api_camera_record_stop():
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    ok, msg = b.camera_record_stop()
    return jsonify(ok=ok, recording=False) if ok else (jsonify(ok=False, error=msg), 501 if msg == "not_supported" else 500)


@app.post("/api/gimbal")
def api_gimbal():
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    tilt = float(data.get("tilt", 0))
    pan = float(data.get("pan", 0))
    ok, msg = b.gimbal_set(tilt, pan)
    return jsonify(ok=ok, tilt=tilt, pan=pan) if ok else (jsonify(ok=False, error=msg), 501 if msg == "not_supported" else 500)


@app.post("/api/rth")
def api_rth():
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    action = str(data.get("action", "start")).lower()
    ok, msg = b.rth(action)
    return jsonify(ok=ok, action=action) if ok else (jsonify(ok=False, error=msg), 501 if msg == "not_supported" else 500)


@app.post("/api/moveto")
def api_moveto():
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    lat, lon = data.get("lat"), data.get("lon")
    if lat is None or lon is None:
        return jsonify(ok=False, error="lat and lon required"), 400
    alt = float(data.get("alt", 2.0))
    heading = float(data.get("heading", 0))
    start_discrete_window(10.0)
    ok, msg = b.moveto(lat, lon, alt, heading)
    return jsonify(ok=ok, lat=float(lat), lon=float(lon), alt=alt) if ok else (jsonify(ok=False, error=msg), 501 if msg == "not_supported" else 500)


# --- Video endpoints ---
@app.get("/api/video")
def api_video_feed():
    if _video_mode != "mjpeg":
        return jsonify(ok=False, error="MJPEG not active. POST /api/video/start with mode=mjpeg"), 400
    b = backend
    def gen():
        while running and _video_mode == "mjpeg":
            jpg = b.get_video_jpeg() if b else b""
            if jpg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(1.0 / max(1, VIDEO_FPS))
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/video/start")
def api_video_start():
    global _video_mode
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    try:
        data = request.get_json(silent=True) or {}
        mode = data.get("mode", "mjpeg")
        b.video_stop_all()
        if mode == "mjpeg":
            ok, msg = b.video_start_mjpeg()
            if ok:
                _video_mode = "mjpeg"
                return jsonify(ok=True, mode="mjpeg", message=msg, stream_url=f"http://{request.host}/api/video")
            print(f"[{drone_type.upper()}] video_start_mjpeg failed: {msg}")
            return jsonify(ok=False, error=msg), 500
        elif mode == "forward":
            target_host = data.get("target_host")
            target_port = int(data.get("target_port", VIDEO_UDP_FORWARD_PORT))
            if not target_host:
                return jsonify(ok=False, error="target_host required"), 400
            ok, msg = b.video_start_forward(target_host, target_port)
            if ok:
                _video_mode = "forward"
                return jsonify(ok=True, mode="forward", message=msg, target=f"{target_host}:{target_port}",
                               viewer_cmd=f"ffplay -fflags nobuffer -flags low_delay -framedrop -probesize 32 -analyzeduration 0 udp://0.0.0.0:{target_port}")
            print(f"[{drone_type.upper()}] video_start_forward failed: {msg}")
            return jsonify(ok=False, error=msg), 500
        return jsonify(ok=False, error=f"unknown mode: {mode}"), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/video/stop")
def api_video_stop():
    b = backend
    if b:
        b.video_stop_all()
    return jsonify(ok=True, mode="off")


@app.get("/api/video/status")
def api_video_status():
    b = backend
    status = b.video_status() if b else {"mode": "off"}
    if _video_mode == "mjpeg":
        status["stream_url"] = f"http://{request.host}/api/video"
    return jsonify(**status)


# --- Settings endpoints ---
@app.get("/api/settings")
def api_settings_get():
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    return jsonify(ok=True, **b.get_settings())


@app.post("/api/settings")
def api_settings_set():
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    results = b.set_settings(data)
    _save_flight_config(results)
    return jsonify(ok=True, **results)


# --- Telemetry endpoints ---
@app.get("/api/telemetry")
def api_telemetry():
    now = time.time()
    age = (now - last_state_seen) if last_state_seen else 9999.0
    with telemetry_lock:
        payload = dict(telemetry)
    payload["state_age_s"] = round(age, 3)
    payload["state_fresh"] = age <= 2.0
    payload["drone_type"] = drone_type
    # Connection watchdog telemetry — lets the C2 show reconnect activity
    with conn_lock:
        payload["connected"] = bool(conn_state.get("connected", False))
        payload["reconnect_failures"] = int(conn_state.get("consecutive_failures", 0))
        payload["reconnect_last_error"] = conn_state.get("last_error", "") or ""
    # Magnetometer calibration status (Anafi only) so the C2 can show a
    # "CAL NEEDED" indicator. Read defensively because older Olympe versions
    # may not have the calibration messages at all.
    try:
        b = backend
        reader = getattr(b, "_read_magnetometer_state", None) if b else None
        if callable(reader):
            mag = reader()
            payload["magneto_status"] = mag
            payload["magneto_required"] = _magneto_needs_calibration(mag)
    except Exception:
        pass
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.get("/api/telemetry/stream")
def api_telemetry_stream():
    def gen():
        while running:
            with telemetry_lock:
                payload = dict(telemetry)
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.4)
    return Response(gen(), mimetype="text/event-stream")


# --- Safety ---
@app.get("/api/safety/takeoff")
def api_safe_takeoff_get():
    return jsonify(enabled=safe_takeoff_enabled, hold_s=SAFE_TAKEOFF_S)


@app.post("/api/safety/takeoff")
def api_safe_takeoff_set():
    global safe_takeoff_enabled
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled")
    if not isinstance(enabled, bool):
        return jsonify(ok=False, error="enabled must be boolean"), 400
    safe_takeoff_enabled = enabled
    return jsonify(ok=True, enabled=safe_takeoff_enabled, hold_s=SAFE_TAKEOFF_S)


# --- Logging endpoints ---
@app.get("/api/logging/commands")
def api_command_log_status():
    return jsonify(enabled=COMMAND_LOG_ENABLED, path=str(COMMAND_LOG_PATH))


@app.get("/api/logging/commands/download")
def api_command_log_download():
    p = COMMAND_LOG_PATH
    if not p.exists():
        return jsonify(ok=False, error="not found"), 404
    return send_file(p, as_attachment=True, download_name=p.name, mimetype="application/x-ndjson")


@app.post("/api/logging/commands/clear")
def api_command_log_clear():
    try:
        COMMAND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        COMMAND_LOG_PATH.write_text("", encoding="utf-8")
        return jsonify(ok=True, cleared=True)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.get("/api/logging/telemetry")
def api_telemetry_log_status():
    with telemetry_log_lock:
        return jsonify(enabled=telemetry_log_enabled, path=str(telemetry_log_path))


@app.post("/api/logging/telemetry")
def api_telemetry_log_config():
    global telemetry_log_enabled, telemetry_log_path
    data = request.get_json(silent=True) or {}
    with telemetry_log_lock:
        if isinstance(data.get("enabled"), bool):
            telemetry_log_enabled = data["enabled"]
        if isinstance(data.get("path"), str) and data["path"].strip():
            telemetry_log_path = Path(data["path"].strip())
        return jsonify(enabled=telemetry_log_enabled, path=str(telemetry_log_path))


@app.get("/api/logging/telemetry/download")
def api_telemetry_log_download():
    with telemetry_log_lock:
        p = telemetry_log_path
    if not p.exists():
        return jsonify(ok=False, error="not found"), 404
    return send_file(p, as_attachment=True, download_name=p.name, mimetype="application/x-ndjson")


@app.post("/api/logging/telemetry/clear")
def api_telemetry_log_clear():
    with telemetry_log_lock:
        p = telemetry_log_path
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        return jsonify(ok=True, cleared=True)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


# ═══════════════════════════════════════════════════════════════════════════
# Positioning subsystem
# ═══════════════════════════════════════════════════════════════════════════

def _pos_snapshot_to_js(snapshot: dict) -> dict:
    """Convert flat _pos_st snapshot to the nested format the JS updatePosUI expects.

    Velocity uses frame-matched telemetry (cm/s -> m/s), then rotates into
    arena frame using the ArUco heading direction.
    """
    x, y, z = snapshot.get("x"), snapshot.get("y"), snapshot.get("z")
    pos = [x, y, z] if x is not None else None
    dx, dy = snapshot.get("dx"), snapshot.get("dy")
    direction = [dx, dy] if dx is not None else None

    tel_vx = snapshot.get("tel_vgx") or 0.0   # drone-frame forward, cm/s
    tel_vy = snapshot.get("tel_vgy") or 0.0   # drone-frame lateral, cm/s
    tel_vz = snapshot.get("tel_vgz") or 0.0   # vertical, cm/s

    # Rotate drone-frame velocity into arena frame using ArUco heading
    fwd = tel_vx / 100.0   # m/s
    lat = tel_vy / 100.0
    if direction is not None and (dx**2 + dy**2) > 1e-6:
        # heading angle from +Y axis (arena north)
        heading = math.atan2(dx, dy)
        arena_vx = fwd * math.sin(heading) + lat * math.cos(heading)
        arena_vy = fwd * math.cos(heading) - lat * math.sin(heading)
    else:
        arena_vx, arena_vy = fwd, lat

    with _pos_cfg_lock:
        enabled = _pos_cfg.get("enabled", False)
        lat_ms = round(_pos_cfg.get("latency_comp_s", 0.05) * 1000)
    return {
        "pos": pos,
        "vel": [round(arena_vx, 3), round(arena_vy, 3), round(tel_vz / 100.0, 3)],
        "dir": direction,
        "fps": snapshot.get("fps"),
        "ref_markers": snapshot.get("ref_markers", []),
        "seen_markers": snapshot.get("seen_markers", []),
        "seen_count": snapshot.get("seen_count", 0),
        "stale": snapshot.get("stale", True),
        "enabled": enabled,
        "latency_ms": lat_ms,
        "sync_quality": snapshot.get("sync_quality", "none"),
        "sync_age_ms": snapshot.get("sync_age_ms"),
        "frame_w": snapshot.get("frame_w"),
        "frame_h": snapshot.get("frame_h"),
    }


def _broadcast_pos_sse(snapshot: dict):
    msg = f"data: {json.dumps(_pos_snapshot_to_js(snapshot))}\n\n"
    with _pos_sse_lock:
        dead = []
        for q in _pos_sse_queues:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _pos_sse_queues.remove(q)


def _default_camera_matrix_pos(frame_w: int, frame_h: int):
    f = frame_w * 0.9
    return np.array([[f, 0, frame_w / 2],
                     [0, f, frame_h / 2],
                     [0, 0, 1]], dtype=np.float64)


def _apply_arena_cfg_to_processor(processor):
    """Override processor marker layout from _arena_cfg."""
    if not HAS_POSITIONING or processor is None:
        return
    try:
        import numpy as _np
        with _arena_cfg_lock:
            cfg = dict(_arena_cfg)
        markers = cfg.get("markers", [])
        mk_size = float(cfg.get("marker_size_m", 0.5))
        if markers:
            new_positions = {}
            new_wall_types = {}
            for m in markers:
                mid = int(m["id"])
                new_positions[mid] = _np.array([float(m.get("x", 0)), float(m.get("y", 0)), float(m.get("z", 0))])
                new_wall_types[mid] = m.get("wall", "north")
            processor.marker_positions = new_positions
            processor.marker_wall_type = new_wall_types
        processor.marker_size = mk_size
        half = mk_size / 2.0
        processor.MARKER_3D_POINTS = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float32)
    except Exception as e:
        print(f"[POS] apply_arena_cfg error: {e}")


def positioning_loop():
    """Background thread: reads frames from _pos_frame_q, runs ArUco positioning."""
    global _pos_annotated_jpeg
    if not HAS_POSITIONING:
        print("[POS] positioning_loop: HAS_POSITIONING=False, exiting")
        return

    processor = None
    calib_w = calib_h = None

    while running:
        # Check if positioning and/or recording is active
        with _pos_cfg_lock:
            cfg = dict(_pos_cfg)
        pos_enabled = cfg.get("enabled", False)
        with _rec_lock:
            rec_active = _rec_enabled
        if not pos_enabled and not rec_active:
            time.sleep(0.2)
            processor = None  # reset so it reinitialises on enable
            continue

        # Get a frame
        try:
            frame, ts = _pos_frame_q.get(timeout=0.5)
        except queue.Empty:
            continue

        h, w = frame.shape[:2]

        # Recording-only fast path: if positioning is disabled but recording is
        # active, write the raw frame straight to the recorder and skip ArUco.
        if not pos_enabled:
            _rec_write_frame(frame)
            continue

        # (Re-)initialise processor if needed
        if processor is None:
            cam_mat = cfg.get("camera_matrix")
            dist = cfg.get("dist_coeffs")
            if cam_mat is None:
                # Try loading from calib file
                try:
                    if POSITION_CALIB_PATH.exists():
                        npz = np.load(str(POSITION_CALIB_PATH))
                        cam_mat = npz["camera_matrix"]
                        dist = npz["dist_coeffs"]
                        print(f"[POS] Loaded calibration from {POSITION_CALIB_PATH}")
                except Exception as ce:
                    print(f"[POS] Calib load error: {ce}")
            if cam_mat is None:
                cam_mat = _default_camera_matrix_pos(w, h)
                dist = np.zeros((5, 1), dtype=np.float64)
                print(f"[POS] Using default camera matrix for {w}x{h}")
            cam_mat = np.array(cam_mat, dtype=np.float64)
            dist = np.array(dist, dtype=np.float64)
            calib_w, calib_h = w, h
            profile = cfg.get("detect_profile", "default")
            try:
                global _pos_processor
                with _arena_cfg_lock:
                    init_marker_size = float(_arena_cfg.get("marker_size_m", 0.5))
                # Runtime overrides from _pos_cfg
                with _pos_cfg_lock:
                    cfg_marker = _pos_cfg.get("marker_size_m")
                    if cfg_marker:
                        init_marker_size = float(cfg_marker)
                    init_kalman = bool(_pos_cfg.get("enable_kalman_filter", True))
                processor = _HeadlessAruCo(cam_mat, dist, detect_profile=profile,
                                           marker_size=init_marker_size, enable_kalman_filter=init_kalman)
                _apply_arena_cfg_to_processor(processor)
                _pos_processor = processor
                print(f"[POS] Processor initialised (profile={profile})")
            except Exception as ie:
                print(f"[POS] Processor init error: {ie}")
                time.sleep(1)
                continue

        # Detect markers for visual annotation (separate fast pass before heavy pose estimation)
        ann = frame.copy()
        raw_corners = raw_ids = None
        try:
            from cv2 import aruco as _aruco
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_corners, raw_ids, _ = processor.detector.detectMarkers(gray)
            if raw_ids is not None and len(raw_ids) > 0:
                _aruco.drawDetectedMarkers(ann, raw_corners, raw_ids)
        except Exception as ae:
            if _video_frame_count < 5:
                print(f"[POS] annotation detect error: {ae}")

        # Match telemetry FIRST so we can feed IMU velocity to the processor
        # before pose estimation. This gives the motion model a fresh prediction
        # to blend against the upcoming vision measurement.
        with _pos_cfg_lock:
            latency_comp_s = max(0.0, float(_pos_cfg.get("latency_comp_s", 0.05)))
            sync_max_gap_s = max(0.01, float(_pos_cfg.get("sync_max_gap_s", 0.20)))
            imu_weight_cfg = float(_pos_cfg.get("imu_weight", 0.3))
        tel_target_ts = ts - latency_comp_s
        tel_match_pre = _telemetry_at(tel_target_ts, max_gap_s=sync_max_gap_s)
        tel_sample_pre = tel_match_pre.get("sample") or {}
        if tel_match_pre.get("quality") not in {"none", "stale"}:
            # Anafi telemetry: vgx/vgy/vgz in body-frame cm/s (Parrot NED: x=fwd, y=right, z=down)
            # Rotate by drone yaw (deg) to get approximate arena-frame m/s.
            # NOTE: assumes arena +y ≈ magnetic north. If the arena is rotated relative
            # to magnetic north, a per-arena yaw offset should be added to tel_yaw here.
            tel_vgx_pre = float(tel_sample_pre.get("vgx") or 0.0) / 100.0   # m/s forward
            tel_vgy_pre = float(tel_sample_pre.get("vgy") or 0.0) / 100.0   # m/s right
            tel_vgz_pre = float(tel_sample_pre.get("vgz") or 0.0) / 100.0   # m/s down
            tel_yaw_pre = float(tel_sample_pre.get("yaw") or 0.0)
            yaw_rad = math.radians(tel_yaw_pre)
            # Body→world yaw rotation: forward axis maps to +y when yaw=0
            v_world_x =  tel_vgx_pre * math.sin(yaw_rad) + tel_vgy_pre * math.cos(yaw_rad)
            v_world_y =  tel_vgx_pre * math.cos(yaw_rad) - tel_vgy_pre * math.sin(yaw_rad)
            v_world_z = -tel_vgz_pre  # NED → Z-up
            processor.set_imu_velocity([v_world_x, v_world_y, v_world_z], ts=ts)
        # Apply live IMU/ArUco blend from UI slider
        processor.set_imu_weight(imu_weight_cfg)

        # Run pose estimation (latency-aware)
        try:
            result = processor.process_frame(
                frame,
                frame_ts=ts,
                latency_s=latency_comp_s,
                now_ts=time.time(),
            )
        except Exception as pe:
            print(f"[POS] process_frame error: {pe}")
            result = None

        # process_frame returns None when no reference markers visible
        if result is None:
            result = {"stale": True, "cam": None, "dir": None,
                      "marker_weights": {}, "ref_markers": []}

        cam = result.get("cam")
        direction = result.get("dir")
        stale = result.get("stale", True)

        # Match telemetry to this frame timestamp (compensate camera pipeline delay)
        tel_target_ts = ts - latency_comp_s
        tel_match = _telemetry_at(tel_target_ts, max_gap_s=sync_max_gap_s)
        tel_sample = tel_match.get("sample") or {}
        tel_quality = tel_match.get("quality", "none")
        tel_age_s = tel_match.get("age_s")
        if tel_quality in {"none", "stale"}:
            tel_vgx = tel_vgy = tel_vgz = 0.0
        else:
            tel_vgx = float(tel_sample.get("vgx") or 0.0)
            tel_vgy = float(tel_sample.get("vgy") or 0.0)
            tel_vgz = float(tel_sample.get("vgz") or 0.0)

        # FPS tracking
        global _pos_fps_counter, _pos_fps_last_reset, _pos_fps_current
        _pos_fps_counter += 1
        _fps_now = time.time()
        if _fps_now - _pos_fps_last_reset >= 1.0:
            _pos_fps_current = round(
                _pos_fps_counter / max(0.001, _fps_now - _pos_fps_last_reset), 1)
            _pos_fps_counter = 0
            _pos_fps_last_reset = _fps_now

        with _pos_st_lock:
            if cam is not None:
                _pos_st["x"] = round(float(cam[0]), 4)
                _pos_st["y"] = round(float(cam[1]), 4)
                _pos_st["z"] = round(float(cam[2]), 4)
            if direction is not None:
                _pos_st["dx"] = round(float(direction[0]), 4)
                _pos_st["dy"] = round(float(direction[1]), 4)
            _pos_st["markers"] = {str(k): round(float(v), 4)
                                   for k, v in (result.get("marker_weights") or {}).items()}
            _pos_st["ref_markers"] = list(result.get("ref_markers") or [])
            _pos_st["seen_markers"] = list(result.get("seen_markers") or [])
            _pos_st["seen_count"] = int(result.get("seen_count") or 0)
            _pos_st["stale"] = stale
            _pos_st["ts"] = ts
            _pos_st["tel_vgx"] = round(tel_vgx, 3)
            _pos_st["tel_vgy"] = round(tel_vgy, 3)
            _pos_st["tel_vgz"] = round(tel_vgz, 3)
            _pos_st["sync_quality"] = tel_quality
            _pos_st["sync_age_ms"] = round(float(tel_age_s) * 1000.0, 1) if tel_age_s is not None else None
            _pos_st["fps"] = _pos_fps_current
            _pos_st["frame_w"] = w
            _pos_st["frame_h"] = h
            snapshot = dict(_pos_st)

        _broadcast_pos_sse(snapshot)

        # Overlay position text and encode annotated JPEG
        try:
            n_detected = len(raw_ids) if raw_ids is not None else 0
            if cam is not None and not stale:
                txt = f"x={cam[0]:.2f} y={cam[1]:.2f} z={cam[2]:.2f}"
                cv2.putText(ann, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
            elif n_detected > 0:
                cv2.putText(ann, f"{n_detected} marker(s) - no ref pose",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            else:
                cv2.putText(ann, "no markers", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (100, 100, 100), 2)
            ok2, jpg2 = cv2.imencode(".jpg", ann, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok2:
                with _pos_annotated_lock:
                    _pos_annotated_jpeg = jpg2.tobytes()
            # Write frame to recording if active (raw or annotated)
            _rec_write_frame(frame if _rec_raw else ann)
        except Exception:
            pass


# ─── Positioning API routes ──────────────────────────────────────────────────

@app.get("/api/position")
def api_position_get():
    with _pos_st_lock:
        snap = dict(_pos_st)
    return jsonify(ok=True, **_pos_snapshot_to_js(snap))


@app.get("/api/position/events")
def api_position_events():
    q: queue.Queue = queue.Queue(maxsize=30)
    with _pos_sse_lock:
        _pos_sse_queues.append(q)

    def gen():
        try:
            while running:
                try:
                    msg = q.get(timeout=5.0)
                    yield msg
                except queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            with _pos_sse_lock:
                try:
                    _pos_sse_queues.remove(q)
                except ValueError:
                    pass

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/position/video")
def api_position_video():
    """MJPEG stream with ArUco annotations — served at the rate frames are produced."""
    def gen():
        last_sent = None
        while running:
            with _pos_annotated_lock:
                jpg = _pos_annotated_jpeg
            if not jpg:
                with _video_jpeg_lock:
                    jpg = _video_last_jpeg
            if jpg and jpg is not last_sent:
                last_sent = jpg
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            else:
                time.sleep(0.02)   # 50 Hz poll — tight but not a busy-spin
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache"})


def _rec_write_frame(bgr_frame):
    """Write a frame to the active recording, lazy-creating VideoWriter from
    the actual incoming frame dimensions. Logs errors instead of swallowing
    them silently."""
    global _rec_writer, _rec_writer_size, _rec_frame_count
    if bgr_frame is None:
        return
    with _rec_lock:
        if not _rec_enabled:
            return
        h, w = bgr_frame.shape[:2]
        # Lazy-create writer on first frame, or recreate on dimension change
        if _rec_writer is None or _rec_writer_size != (w, h):
            # Close any previous writer (dimension change mid-recording)
            if _rec_writer is not None:
                try: _rec_writer.release()
                except Exception: pass
                _rec_writer = None
            try:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(_rec_path, fourcc, _rec_fps_target, (w, h))
                if not writer.isOpened():
                    print(f"[REC] ERROR: VideoWriter failed to open for {_rec_path} @ {w}x{h}")
                    return
                _rec_writer = writer
                _rec_writer_size = (w, h)
                print(f"[REC] Writer opened: {_rec_path} @ {w}x{h} {_rec_fps_target}fps "
                      f"({'raw' if _rec_raw else 'annotated'})")
            except Exception as e:
                print(f"[REC] ERROR opening writer: {e}")
                return
        try:
            _rec_writer.write(bgr_frame)
            _rec_frame_count += 1
        except Exception as e:
            print(f"[REC] write failed (frame {_rec_frame_count}): {e}")


@app.get("/api/video/record/status")
def api_rec_status():
    with _rec_lock:
        return jsonify(ok=True, recording=_rec_enabled, raw=_rec_raw, path=_rec_path, frames=_rec_frame_count)


@app.post("/api/video/record/start")
def api_rec_start():
    global _rec_enabled, _rec_raw, _rec_writer, _rec_path, _rec_frame_count
    global _rec_fname, _rec_fps_target, _rec_writer_size
    data = request.get_json(silent=True) or {}
    with _rec_lock:
        if _rec_enabled:
            return jsonify(ok=False, error="already recording", path=_rec_path)
        raw_mode = bool(data.get("raw", False))
        suffix = "_raw" if raw_mode else "_ann"
        fname = data.get("filename") or f"rec{suffix}_{int(time.time())}.mp4"
        # Ensure recordings dir exists
        rec_dir = Path(__file__).parent / "recordings"
        rec_dir.mkdir(exist_ok=True)
        full_path = str(rec_dir / fname)
        # DEFER VideoWriter creation to first frame — that way we get the
        # real frame dimensions (avoids silent write-fails when the writer
        # was opened at 960x720 but frames arrive at 1280x720, which
        # produced the old 257-byte empty-MP4 bug).
        _rec_writer = None
        _rec_writer_size = (0, 0)
        _rec_path = full_path
        _rec_fname = fname
        _rec_frame_count = 0
        _rec_raw = raw_mode
        _rec_fps_target = float(_pos_fps_current or 5.0)
        _rec_enabled = True
        print(f"[REC] Recording armed ({'raw' if raw_mode else 'annotated'}): {full_path} — waiting for first frame")
        return jsonify(ok=True, path=full_path)


@app.post("/api/video/record/stop")
def api_rec_stop():
    global _rec_enabled, _rec_writer, _rec_path, _rec_frame_count
    with _rec_lock:
        if not _rec_enabled:
            return jsonify(ok=False, error="not recording")
        _rec_enabled = False
        frames = _rec_frame_count
        path = _rec_path
        try:
            if _rec_writer:
                _rec_writer.release()
                _rec_writer = None
        except Exception:
            pass
        print(f"[REC] Recording stopped: {path} ({frames} frames)")
    return jsonify(ok=True, path=path, frames=frames)


@app.get("/api/position/config")
def api_pos_config_get():
    with _pos_cfg_lock:
        cfg = dict(_pos_cfg)
    cfg.pop("camera_matrix", None)
    cfg.pop("dist_coeffs", None)
    has_calib = POSITION_CALIB_PATH.exists()
    # Also return stored config
    try:
        stored = json.loads(POSITION_CONFIG_PATH.read_text()) if POSITION_CONFIG_PATH.exists() else {}
    except Exception:
        stored = {}
    return jsonify(ok=True, config=cfg, stored=stored, has_calibration=has_calib)


@app.post("/api/position/config")
def api_pos_config_set():
    data = request.get_json(silent=True) or {}
    apply_marker_size = False
    with _pos_cfg_lock:
        if "enabled" in data:
            _pos_cfg["enabled"] = bool(data["enabled"])
        if "fps" in data:
            _pos_cfg["fps"] = max(1, min(30, int(data["fps"])))
        if "detect_profile" in data:
            _pos_cfg["detect_profile"] = str(data["detect_profile"])
        if "latency_comp_s" in data:
            _pos_cfg["latency_comp_s"] = max(0.0, min(1.0, float(data["latency_comp_s"])))
        if "sync_max_gap_s" in data:
            _pos_cfg["sync_max_gap_s"] = max(0.01, min(2.0, float(data["sync_max_gap_s"])))
        if "telemetry_buffer_s" in data:
            _pos_cfg["telemetry_buffer_s"] = max(1.0, min(30.0, float(data["telemetry_buffer_s"])))
        # Live filter params — applied to _pos_processor below
        if "enable_kalman_filter" in data:
            _pos_cfg["enable_kalman_filter"] = bool(data["enable_kalman_filter"])
        if "marker_size_m" in data:
            _pos_cfg["marker_size_m"] = max(0.05, min(2.0, float(data["marker_size_m"])))
            apply_marker_size = True
        if "top_k_markers" in data:
            _pos_cfg["top_k_markers"] = max(0, min(10, int(data["top_k_markers"])))
        if "outlier_reject_m" in data:
            _pos_cfg["outlier_reject_m"] = max(0.1, min(20.0, float(data["outlier_reject_m"])))
        if "imu_weight" in data:
            _pos_cfg["imu_weight"] = max(0.0, min(1.0, float(data["imu_weight"])))
        cfg_snap = dict(_pos_cfg)

    # Apply live filter changes to running processor
    try:
        if _pos_processor is not None:
            if "enable_kalman_filter" in data:
                _pos_processor.enable_kalman_filter = bool(data["enable_kalman_filter"])
                # Reset Kalman state so toggling doesn't carry stale history
                try:
                    for kf in _pos_processor.kf_pos:
                        kf.reset()
                except Exception:
                    pass
                print(f"[POS] Kalman filter {'ENABLED' if _pos_processor.enable_kalman_filter else 'DISABLED'} (live)")
            if apply_marker_size:
                # Also reflect in arena_cfg so _apply_arena_cfg_to_processor can be used
                with _arena_cfg_lock:
                    _arena_cfg["marker_size_m"] = cfg_snap["marker_size_m"]
                _apply_arena_cfg_to_processor(_pos_processor)
                print(f"[POS] marker_size_m set to {cfg_snap['marker_size_m']}m (live)")
            if "top_k_markers" in data:
                tk = int(cfg_snap["top_k_markers"])
                _pos_processor.top_k_markers = tk if tk > 0 else 4
                print(f"[POS] top_k_markers set to {tk if tk > 0 else '4 (auto)'} (live)")
            if "outlier_reject_m" in data:
                _pos_processor.outlier_reject_m = float(cfg_snap["outlier_reject_m"])
                print(f"[POS] outlier_reject_m set to {cfg_snap['outlier_reject_m']}m (live)")
    except Exception as e:
        print(f"[POS] live config apply error: {e}")

    cfg_snap.pop("camera_matrix", None)
    cfg_snap.pop("dist_coeffs", None)
    try:
        POSITION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        POSITION_CONFIG_PATH.write_text(json.dumps(cfg_snap, indent=2))
    except Exception as e:
        print(f"[POS] config save error: {e}")
    return jsonify(ok=True, config=cfg_snap)


@app.post("/api/position/calibration")
def api_pos_calib_upload():
    if not HAS_CV2:
        return jsonify(ok=False, error="cv2 not available"), 500
    f = request.files.get("file")
    if f is None:
        return jsonify(ok=False, error="no file"), 400
    try:
        import io, tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        f.save(tmp.name)
        npz = np.load(tmp.name)
        cam_mat = npz["camera_matrix"]
        dist = npz["dist_coeffs"]
        with _pos_cfg_lock:
            _pos_cfg["camera_matrix"] = cam_mat.tolist()
            _pos_cfg["dist_coeffs"] = dist.tolist()
        POSITION_CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
        import shutil as _sh
        _sh.copy(tmp.name, str(POSITION_CALIB_PATH))
        os.unlink(tmp.name)
        return jsonify(ok=True, shape_cam=list(cam_mat.shape), shape_dist=list(dist.shape))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.get("/api/position/calibration")
def api_pos_calib_download():
    if not POSITION_CALIB_PATH.exists():
        return jsonify(ok=False, error="no calibration file"), 404
    return send_file(str(POSITION_CALIB_PATH), as_attachment=True,
                     download_name="position_calib.npz", mimetype="application/octet-stream")


# ─── Arena config routes ─────────────────────────────────────────────────────

@app.get("/api/arena/config")
def api_arena_config_get():
    with _arena_cfg_lock:
        cfg = dict(_arena_cfg)
    return jsonify(ok=True, **cfg)


@app.post("/api/arena/config")
def api_arena_config_set():
    data = request.get_json(silent=True) or {}
    with _arena_cfg_lock:
        if "arena_width_m" in data:
            _arena_cfg["arena_width_m"] = float(data["arena_width_m"])
        if "arena_height_m" in data:
            _arena_cfg["arena_height_m"] = float(data["arena_height_m"])
        if "marker_size_m" in data:
            _arena_cfg["marker_size_m"] = float(data["marker_size_m"])
        if "markers" in data and isinstance(data["markers"], list):
            _arena_cfg["markers"] = data["markers"]
        cfg = dict(_arena_cfg)
    _save_arena_config(cfg)
    _apply_arena_cfg_to_processor(_pos_processor)
    return jsonify(ok=True, **cfg)


@app.post("/api/arena/config/reset")
def api_arena_config_reset():
    global _arena_cfg
    with _arena_cfg_lock:
        _arena_cfg = dict(_ARENA_CONFIG_DEFAULT)
        cfg = dict(_arena_cfg)
    _save_arena_config(cfg)
    _apply_arena_cfg_to_processor(_pos_processor)
    return jsonify(ok=True, **cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global backend, drone_type, drone_ip

    drone_type, drone_ip = detect_drone_type()

    if drone_type == "tello":
        if not HAS_TELLO_SDK:
            print("ERROR: djitellopy not installed. pip install djitellopy")
            return
        logging.getLogger("djitellopy").setLevel(logging.CRITICAL)
        backend = TelloBackend(drone_ip)
    elif drone_type == "anafi":
        if not HAS_OLYMPE_SDK:
            print("ERROR: olympe not installed.")
            return
        logging.getLogger("olympe").setLevel(logging.WARNING)
        backend = OlympeBackend(drone_ip)
    else:
        print(f"ERROR: Unknown drone type: {drone_type}")
        return

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    with conn_lock:
        conn_state["connected"] = False
        conn_state["last_reconnect"] = 0.0
        conn_state["last_verify"] = 0.0

    atexit.register(backend.shutdown)

    # Start background threads
    if drone_type == "tello" and isinstance(backend, TelloBackend):
        threading.Thread(target=backend.wifi_connect_loop, daemon=True).start()

    threading.Thread(target=telemetry_loop, daemon=True).start()
    threading.Thread(target=reconnect_loop, daemon=True).start()
    threading.Thread(target=rc_loop, daemon=True).start()
    threading.Thread(target=watchdog_loop, daemon=True).start()

    # Load persisted flight limits
    _load_flight_config()

    # Start positioning loop (Anafi only, requires HAS_POSITIONING)
    if drone_type == "anafi" and HAS_POSITIONING:
        # Load persisted position config
        try:
            if POSITION_CONFIG_PATH.exists():
                stored = json.loads(POSITION_CONFIG_PATH.read_text())
                with _pos_cfg_lock:
                    _pos_cfg.update({k: stored[k] for k in stored if k in _pos_cfg})
        except Exception as pce:
            print(f"[POS] position_config load error: {pce}")
        threading.Thread(target=positioning_loop, daemon=True, name="positioning").start()
        print(f"[ANAFI] Positioning thread started (enabled={_pos_cfg.get('enabled', False)})")

    tag = drone_type.upper()
    print(f"[{tag}] Unified API server: http://{HTTP_HOST}:{HTTP_PORT}")
    print(f"[{tag}] Drone: {drone_type} @ {drone_ip} (auto-reconnect; watchdog={REMOTE_TIMEOUT_S}s)")
    print(f"[{tag}] SDKs available: tello={HAS_TELLO_SDK}, olympe={HAS_OLYMPE_SDK}")
    print(f"[{tag}] Code version: 2026-03-26-v2")
    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
