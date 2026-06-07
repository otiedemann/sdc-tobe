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
try:
    from flask.json.provider import DefaultJSONProvider
    _HAS_JSON_PROVIDER = True
except ImportError:
    # Flask < 2.2 exposes app.json_encoder instead
    DefaultJSONProvider = None
    _HAS_JSON_PROVIDER = False
import json as _json

# Drone-control core: route handlers above /api/takeoff, /api/land,
# /api/rc, /api/emergency and /api/telemetry delegate to this module's
# do_*/get_* functions, so an in-process consumer (marker_mission) can
# bypass the HTTP layer entirely. drone_core lazy-imports this module
# back to reach the shared globals — no circular-import cost because
# the back-references are resolved at first call, not at module load.
# Sibling import (no relative ``.``) because this file is launched as
# a top-level script (`python unified_api_server.py`) with cwd set to
# the controller_unified/ directory by the launcher.
import drone_core

# ---------------------------------------------------------------------------
# Architecture detection. The SDC26 flight controller is an x86 Linux
# box running Olympe + Anafi (NOT a Raspberry Pi — comments and module
# names may still say "Pi" for historical reasons, but the host is x86).
# The ARM / Tello-only fallback below is legacy for older deployments;
# safe to leave in place, simply not exercised in the current fleet.
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
# Camera-settings probes -- gated per-feature so a partial Olympe build
# still gets whichever knobs it does expose. See the probed import block
# inside ``if HAS_OLYMPE_SDK:`` for the message names.
HAS_CAMERA_EXPOSURE = False
HAS_CAMERA_WB = False
HAS_CAMERA_HDR = False
HAS_CAMERA_EV = False
HAS_CAMERA_ANTIFLICKER = False
HAS_CAMERA_RECORDING_MODE = False
HAS_CAMERA_ZOOM = False
HAS_VIDEO_STABILIZATION = False
HAS_STREAM_MODE = False
HAS_AF_LEGACY = False

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
        try:
            # Confirmation state — drone ACKs the MaxAltitude command by
            # emitting this message. Reading it back lets us VERIFY the
            # firmware accepted the value (vs silently clamping to its
            # own range). Some Olympe versions don't expose this — we
            # just skip the verify if so.
            from olympe.messages.ardrone3.PilotingSettingsState import (
                MaxAltitudeChanged,
            )
            _HAS_MAXALT_STATE = True
        except Exception:
            MaxAltitudeChanged = None
            _HAS_MAXALT_STATE = False
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
    # Drone serial number is reported as two halves -- "high" and "low" --
    # via SettingsState messages that fire once during the initial
    # state-ack flood after connect. We cache the concatenation in
    # OlympeBackend.serial_number and surface it through /api/telemetry.
    try:
        from olympe.messages.common.SettingsState import (
            ProductSerialHighChanged,
            ProductSerialLowChanged,
        )
        HAS_PRODUCT_SERIAL = True
    except (ImportError, KeyError):
        HAS_PRODUCT_SERIAL = False


# ─── Optional Wi-Fi control messages ────────────────────────────────────────
# The Anafi is dual-band 2.4/5 GHz 802.11n. Olympe exposes AP-channel
# config, but the exact Python symbol names drift between Olympe versions
# (set_ap_channel vs SetApChannel vs Command.SetApChannel, etc.).
# We probe the module and log every import error so the operator can
# see what's actually available. Everything downstream is guarded by
# HAS_WIFI_CTRL.
HAS_WIFI_CTRL = False
WifiSetApChannel = WifiScan = None
WifiApChannelChanged = WifiAuthorizedChannel = WifiRssiChanged = None
WifiBand = WifiSelectionType = None
_WIFI_IMPORT_ERRORS: list[str] = []   # populated for /api/wifi/debug

if HAS_OLYMPE_SDK:
    # Candidate message paths. Try each; first one that binds wins.
    _WIFI_CANDIDATES = [
        # (label, module_path, set_attr, scan_attr, chan_evt, auth_evt, rssi_evt)
        ("wifi.<snake>", "olympe.messages.wifi",
            "set_ap_channel", "scan",
            "ap_channel_changed", "authorized_channel", "rssi_changed"),
        ("wifi.<Camel>", "olympe.messages.wifi",
            "SetApChannel", "ScanChannels",
            "ApChannelChanged", "AuthorizedChannel", "Rssi_changed"),
        ("wifi.Command", "olympe.messages.wifi.Command",
            "SetApChannel", "ScanChannels",
            None, None, None),   # events not under Command
    ]
    import importlib as _importlib
    for label, mod_path, set_name, scan_name, chan_name, auth_name, rssi_name in _WIFI_CANDIDATES:
        try:
            mod = _importlib.import_module(mod_path)
            s = getattr(mod, set_name, None)
            sc = getattr(mod, scan_name, None)
            if not (s and sc):
                _WIFI_IMPORT_ERRORS.append(
                    f"{label}: {mod_path}.{set_name}={bool(s)}, "
                    f".{scan_name}={bool(sc)}")
                continue
            WifiSetApChannel = s
            WifiScan = sc
            if chan_name:
                WifiApChannelChanged = getattr(mod, chan_name, None) or \
                    getattr(_importlib.import_module("olympe.messages.wifi"), chan_name, None)
            if auth_name:
                WifiAuthorizedChannel = getattr(mod, auth_name, None) or \
                    getattr(_importlib.import_module("olympe.messages.wifi"), auth_name, None)
            if rssi_name:
                WifiRssiChanged = getattr(mod, rssi_name, None) or \
                    getattr(_importlib.import_module("olympe.messages.wifi"), rssi_name, None)
            print(f"[WIFI] Using Olympe Wi-Fi bindings from {label} ({mod_path})")
            HAS_WIFI_CTRL = True
            break
        except Exception as _e:
            _WIFI_IMPORT_ERRORS.append(f"{label}: {type(_e).__name__}: {_e}")

    # Enums. Olympe 7.x uses lowercase names (band, selection_type),
    # older 1.x used CamelCase (Band, SelectionType). Accept either.
    for ep in ("olympe.enums.wifi", "olympe.messages.wifi"):
        try:
            em = _importlib.import_module(ep)
            WifiBand = (getattr(em, "band", None)
                        or getattr(em, "Band", None)
                        or WifiBand)
            WifiSelectionType = (getattr(em, "selection_type", None)
                                 or getattr(em, "SelectionType", None)
                                 or getattr(em, "Type", None)
                                 or WifiSelectionType)
            if WifiBand and WifiSelectionType:
                print(f"[WIFI] Band/SelectionType enums loaded from {ep} "
                      f"(band={WifiBand.__name__ if hasattr(WifiBand,'__name__') else WifiBand}, "
                      f"sel={WifiSelectionType.__name__ if hasattr(WifiSelectionType,'__name__') else WifiSelectionType})")
                break
        except Exception as _e:
            _WIFI_IMPORT_ERRORS.append(f"enums from {ep}: {type(_e).__name__}: {_e}")

    if HAS_WIFI_CTRL:
        print(f"[WIFI] Control ready — band={bool(WifiBand)}, "
              f"sel_type={bool(WifiSelectionType)}")
    else:
        print("[WIFI] Control NOT available. Errors:")
        for err in _WIFI_IMPORT_ERRORS:
            print(f"[WIFI]   {err}")

    # Camera / gimbal / GPS / RTH / move-to / geofence Olympe messages.
    # These were previously buried below an unconditional ``return False``
    # inside ``_magneto_needs_calibration`` (dead code) which silently
    # left HAS_CAMERA / HAS_GIMBAL / HAS_RTH / HAS_MOVE_TO /
    # HAS_MAX_HORIZ_SPEED / HAS_GEOFENCE at False and the corresponding
    # endpoints broken with NameError. Now imported at module scope so
    # the existing /api/camera/{photo,record/*,zoom}, /api/gimbal,
    # /api/rth, /api/moveto endpoints actually work.
    try:
        from olympe.messages.camera import start_recording, stop_recording, take_photo
        HAS_CAMERA = True
    except (ImportError, KeyError):
        pass
    try:
        from olympe.messages.camera import set_zoom_target as _CameraSetZoom
        HAS_CAMERA_ZOOM = True
    except (ImportError, KeyError):
        HAS_CAMERA_ZOOM = False
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

    # Camera-settings (modern camera.* namespace). Each axis is gated
    # independently so a partial Olympe build still works.
    try:
        from olympe.messages.camera import (
            set_exposure_settings, exposure_settings,
            set_white_balance, white_balance,
            set_hdr_setting, hdr_setting,
            set_ev_compensation, ev_compensation,
            set_antiflicker_mode, antiflicker_mode,
            set_recording_mode, recording_mode,
            zoom_level, zoom_info,
        )
        HAS_CAMERA_EXPOSURE = True
        HAS_CAMERA_WB = True
        HAS_CAMERA_HDR = True
        HAS_CAMERA_EV = True
        HAS_CAMERA_ANTIFLICKER = True
        HAS_CAMERA_RECORDING_MODE = True
    except (ImportError, KeyError):
        pass
    # ardrone3.* fallbacks (image stabilisation lives only here; the
    # legacy anti-flicker is the fallback when ``camera.set_antiflicker_mode``
    # isn't in this Olympe build).
    try:
        from olympe.messages.ardrone3.PictureSettings import VideoStabilizationMode
        from olympe.messages.ardrone3.PictureSettingsState import VideoStabilizationModeChanged
        HAS_VIDEO_STABILIZATION = True
    except (ImportError, KeyError):
        pass
    try:
        from olympe.messages.ardrone3.MediaStreaming import VideoStreamMode
        from olympe.messages.ardrone3.MediaStreamingState import VideoStreamModeChanged
        HAS_STREAM_MODE = True
    except (ImportError, KeyError):
        pass
    try:
        from olympe.messages.ardrone3.Antiflickering import setMode as AntiflickerLegacy
        from olympe.messages.ardrone3.AntiflickeringState import modeChanged as AntiflickerLegacyChanged
        HAS_AF_LEGACY = True
    except (ImportError, KeyError):
        pass


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

# ── Version / git revision reporting ───────────────────────────────
# Surfaced via /api/version so the C2 can detect when it's running a
# newer/older build than the FC (the single most confusing bug source:
# "I updated the code but it still behaves like the old version" —
# because only one of the two was actually restarted).
CODE_VERSION = "2026-04-24-cr (FC version endpoint + C2/FC mismatch check)"


def _read_fc_git_revision() -> dict:
    """Read the FC's git HEAD info — sha, short_sha, branch, subject,
    dirty-flag. Runs once at startup; result is static until restart."""
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def _git(*args) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", repo] + list(args),
                stderr=subprocess.DEVNULL, text=True, timeout=3,
            ).strip()
        except Exception:
            return ""
    sha = _git("rev-parse", "HEAD")
    if not sha:
        return {"sha": "", "short_sha": "", "branch": "",
                "subject": "", "dirty": False}
    short = sha[:7]
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    subject = _git("log", "-1", "--pretty=%s")
    dirty_out = _git("status", "--porcelain")
    return {
        "sha": sha, "short_sha": short, "branch": branch,
        "subject": subject, "dirty": bool(dirty_out),
    }


_FC_GIT_REVISION = _read_fc_git_revision()
RC_HZ = 20
STICK = 60
YAW_STICK_ANAFI = 90
MAX_YAW_SPEED = 150
MAX_ALTITUDE_M = float(os.getenv("MAX_ALTITUDE_M", "5.0"))
# Hard ceiling guard — runtime state surfaced via /api/config/ceiling and
# read in the RC tick loop. `_ceiling_engaged` flashes True whenever the
# guard clamps or overrides an RC command; `_ceiling_last_reason` is a
# human-readable explanation surfaced in /api/telemetry for UI banners.
_ceiling_engaged: bool = False
_ceiling_last_reason: str = ""

# ── Arena XY boundary guard (Pi-side, applies to ALL RC) ─────────────
# Previously only the autonomous observer enforced a safety margin from
# arena walls. If the operator flew manually (WASD) or took off near a
# wall, nothing clamped the RC — which is how a recent flight ended up
# inside the net at y=10.95 (arena back wall y=10.8). This guard runs
# in the Pi's own rc_loop so it applies to BOTH transport paths (WS +
# HTTP) and BOTH control modes (manual + autonomous). Values persisted
# to flight_config.json.
_ARENA_BOUNDS_DEFAULT = {
    "x_min": -10.0, "x_max": 10.0,
    "y_min":   0.0, "y_max": 10.8,
}
_arena_bounds: dict = dict(_ARENA_BOUNDS_DEFAULT)
_arena_margin_m: float = float(os.getenv("ARENA_SAFETY_MARGIN_M", "1.5"))
# Toggle: applies to BOTH manual and autonomous flight. Default ON.
# When OFF the Pi's RC tick makes no XY boundary decisions — operator
# takes full responsibility. The altitude ceiling remains on regardless.
# Arena XY boundary guard. DISABLED by default — the bounds the
# guard uses come from a stale non-centered arena (``y∈[0, 10.8]``)
# while the unified positioning subsystem now publishes positions in
# the centered marker_mission frame (``y∈[-10, +10]``). The two
# clocks disagree, so the guard either clamps too aggressively
# (legitimate PD corrections refused, drone stuck in front of a
# marker it can't centre) or not at all. Until the bounds are
# regenerated from the active arena config, default off.
#
# Re-enable per-host via env: ``ARENA_GUARD_ENABLED=1`` in a systemd
# drop-in, or live via ``POST /api/config/arena_safety {"enabled": true}``.
_arena_guard_enabled: bool = os.getenv("ARENA_GUARD_ENABLED", "0") not in {"0", "false", "False"}
_arena_engaged: bool = False
_arena_last_reason: str = ""
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

# Drone fleet CSV — rows: id, name (= WiFi SSID), password
DRONES_CSV_PATH = Path(__file__).with_name("drones.csv")


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def _host_reachable(host: str) -> bool:
    r = subprocess.run(["ping", "-c", "1", "-W", "1", host], capture_output=True, text=True)
    return r.returncode == 0


def _ping_rtt_ms(host: str, timeout_s: float = 0.8) -> Optional[float]:
    """Return the round-trip time in milliseconds to `host`, or None on
    failure. Uses a single ICMP echo. Parses `time=XX.X ms` from the
    `ping` output — works on both macOS and Linux.
    """
    try:
        # -c 1: one packet; -W ms or s depending on OS. BSD ping uses
        # -W ms (integer), Linux ping uses -W seconds (integer). Use
        # large enough timeout for both: -W 1 second is portable.
        r = subprocess.run(
            ["ping", "-c", "1", "-W", "1", host],
            capture_output=True, text=True, timeout=timeout_s)
        if r.returncode != 0:
            return None
        import re as _re
        m = _re.search(r"time[=<]([\d.]+)\s*ms", r.stdout)
        return float(m.group(1)) if m else None
    except Exception:
        return None


# Background ping sampler — keeps a rolling RTT to drone_ip so /api/drone_ping
# can return the current value instantly (ping takes ~1s so we don't want the
# web request to block on it).
_drone_ping_state = {"rtt_ms": None, "last_update": 0.0, "host": ""}
_drone_ping_lock = threading.Lock()


def _drone_ping_loop():
    """Runs in the background. Pings the drone once per second and
    caches the latest RTT so /api/drone_ping returns immediately."""
    while running:
        host = drone_ip
        if host:
            rtt = _ping_rtt_ms(host, timeout_s=1.2)
            with _drone_ping_lock:
                _drone_ping_state["rtt_ms"] = rtt
                _drone_ping_state["last_update"] = time.time()
                _drone_ping_state["host"] = host
        time.sleep(1.0)


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

    # On ARM hosts only Tello is supported (no Olympe SDK). The
    # current SDC26 flight controller is x86 Linux, so this branch
    # is only hit on legacy setups.
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

# ── WebSocket channel (C2 ↔ this FC) ──────────────────────────────────
# Three endpoints that exist alongside the HTTP API and carry the same
# payloads without the TCP+HTTP framing overhead per call. Falls back to
# "import flask_sock" so an older Pi without the package still runs;
# only the HTTP endpoints will be available there.
try:
    from flask_sock import Sock as _Sock
    sock = _Sock(app)
    HAS_WS = True
except Exception as _e:
    sock = None
    HAS_WS = False
    print(f"[WS] flask-sock not available ({_e}) — WS endpoints disabled")

# Grab simple-websocket's ConnectionClosed so we can silence the (very
# chatty) "Connection closed: 1000" message on clean client disconnects.
# 1000 is the RFC 6455 "normal closure" code — not an error condition.
try:
    from simple_websocket import ConnectionClosed as _WSConnectionClosed
except Exception:
    _WSConnectionClosed = None


def _ws_is_clean_close(exc: Exception) -> bool:
    """True if the exception represents a clean peer disconnect. We
    treat these as info, not errors — otherwise every reconnect logs
    a scary-looking stack trace."""
    if _WSConnectionClosed is not None and isinstance(exc, _WSConnectionClosed):
        return True
    msg = str(exc)
    return "Connection closed" in msg or "1000" in msg


# Enum-aware JSON encoder. Olympe returns enum instances (class `band`,
# `selection_type`, etc.) in its state dicts; Flask's stock encoder raises
# "Object of type band is not JSON serializable" for these. Install a
# global provider so EVERY jsonify() in this app handles them.
def _enum_to_jsonable(obj):
    name = getattr(obj, "name", None)
    if isinstance(name, str):
        return name
    # ArsdkEnum sometimes exposes .value as the serializable form
    val = getattr(obj, "value", None)
    if isinstance(val, (str, int, float, bool)):
        return val
    return str(obj)


try:
    if _HAS_JSON_PROVIDER and hasattr(app, "json"):
        # Flask 2.2+: app.json is a JSONProvider INSTANCE with a callable
        # `.default` attribute (a staticmethod on DefaultJSONProvider).
        # Wrap the existing default so we fall back to enum-naming.
        _orig_default = getattr(app.json, "default", None)
        def _enum_aware_default(obj, _orig=_orig_default):
            if _orig is not None:
                try:
                    return _orig(obj)
                except TypeError:
                    pass
            return _enum_to_jsonable(obj)
        app.json.default = _enum_aware_default
        print("[JSON] Enum-aware default installed on app.json")
    else:
        # Flask < 2.2
        class _EnumAwareJSONEncoder(_json.JSONEncoder):
            def default(self, obj):
                try:
                    return super().default(obj)
                except TypeError:
                    return _enum_to_jsonable(obj)
        app.json_encoder = _EnumAwareJSONEncoder
        print("[JSON] Enum-aware encoder installed on app.json_encoder")
except Exception as _e:
    # If the Flask version has an even different API, don't crash the
    # whole server. Fall back to per-endpoint _jsonable wrapping.
    print(f"[JSON] Could not install enum-aware encoder: {type(_e).__name__}: {_e}")


# Build marker — GET /api/version returns this so the operator can verify
# which commit the flight controller is actually running.
BUILD_TAG = "wifi-scan-v3-with-scanned-item"
BUILD_AT = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


@app.get("/api/version")
def api_version():
    """Return the FC's code version, git revision, and legacy BUILD_TAG
    so the C2 can detect when it's running a newer build than the FC
    (the single most confusing bug source — a code update that only
    lands on one side of the link)."""
    return jsonify(
        ok=True,
        code_version=CODE_VERSION,
        git_revision=_FC_GIT_REVISION,
        build=BUILD_TAG,
        build_at=BUILD_AT,
        file=__file__,
    )
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
# Dedicated lock for discrete_until reads/writes. MUST NOT be the same
# as command_lock — Olympe blocking calls (takeoff/land/flip) hold
# command_lock for up to 10 s per attempt × 3 retries. If the RC tick
# loop also waited on command_lock for the brief discrete_until read,
# every keystroke during that window would stall, then all queued
# keys would slam the drone simultaneously once Olympe released. That
# was the 10-18 s WASD delay after the second takeoff.
discrete_lock = threading.Lock()
takeoff_cooldown_until = 0.0
safe_takeoff_enabled = SAFE_TAKEOFF_DEFAULT

last_remote_request = 0.0
_watchdog_landed = False

# Video state
_video_mode = "off"
_video_last_jpeg = b""
_video_jpeg_lock = threading.Lock()
_video_streaming = False

# Raw BGR frame for in-proc consumers (InProcMjpegReader).
# Updated by _video_frame_cb on every Anafi frame; readers wait on the
# condition so they wake immediately on a new frame instead of polling.
_video_bgr_latest: Optional["np.ndarray"] = None
_video_bgr_ts: float = 0.0
_video_bgr_cond = threading.Condition(threading.Lock())

# ── H.264 fan-out state (sim only) ────────────────────────
# The sim video producer (_sim_video_loop below) writes raw BGR frames
# here in addition to its JPEG output, so the /api/video/h264 endpoint
# can feed an ffmpeg subprocess that NVENC-encodes H.264 over MPEG-TS.
# Producer is reused; we don't run a second pysphinx reader.
#
# The publish path is LAZY: when no /api/video/h264 client is connected,
# _sim_h264_active_clients == 0 and the producer skips the bgr.copy()
# + condition.notify_all() — saving ~40 MB/s of memcpy on the hot path.
# Counter is incremented when a client enters api_video_h264_stream and
# decremented when its generator exits.
_sim_bgr_latest = None              # np.ndarray (h, w, 3) BGR, or None
_sim_bgr_resolution = None          # (width, height) once first frame seen
_sim_bgr_frame_id = 0               # monotonic counter — new frame iff this ticks
_sim_bgr_lock = threading.Lock()
_sim_bgr_condition = threading.Condition(_sim_bgr_lock)
_sim_h264_active_clients = 0        # only publish BGR if > 0

# ── SIM VIDEO PATCH ─────────────────────────────────
# Read frames from a SIMULATED Parrot Anafi (Sphinx) via pysphinx
# instead of Olympe's pdraw/RTSP path — the sim drone does NOT host
# an RTSP server. We reuse the same _video_last_jpeg cache and
# _pos_frame_q so all downstream consumers (web /api/video, position
# tracker, recorder) see the sim feed identically to a real Anafi.
import os as _os_sim_video
_SIM_IP_PREFIXES = ("10.202.0.", "172.21.0.")


def _is_sim_anafi_ip(ip):
    return any(str(ip).startswith(p) for p in _SIM_IP_PREFIXES)


def _sim_video_loop(machine_name="anafi", camera_name="horizontal_camera"):
    """Read Sphinx camera frames and feed the same pipeline as Olympe.

    Uses the high-level pysphinx API:
        s = Sphinx()                       # connects to localhost:8383
        cam = s.get_camera(machine, name)  # returns SphinxShmCamera
        ok, fr = cam.read()                # ok=bool, fr=SphinxShmFrame
        bgra = fr.buffer                   # ndarray (h, w, 4) uint8 B8G8R8A8
        fr.release()                       # return frame to pool
    """
    global _video_last_jpeg, _video_frame_count, _video_streaming
    sphx_lib = "/opt/parrot-sphinx/usr/lib/python/site-packages"
    if sphx_lib not in sys.path:
        sys.path.insert(0, sphx_lib)
    sphx_so = "/opt/parrot-sphinx/usr/lib"
    cur_ld = _os_sim_video.environ.get("LD_LIBRARY_PATH", "")
    if sphx_so not in cur_ld:
        _os_sim_video.environ["LD_LIBRARY_PATH"] = (sphx_so + ":" + cur_ld) if cur_ld else sphx_so
    try:
        from pysphinx.sphinx import Sphinx  # noqa
    except Exception as e:
        print(f"[SIM-VIDEO] pysphinx import failed: {e}")
        _video_streaming = False
        return

    sphx = None
    cam = None
    try:
        sphx = Sphinx()
        info = sphx.get_info()
        if info is None:
            print("[SIM-VIDEO] sphinx daemon not reachable on :8383")
            _video_streaming = False
            return
        names = sphx.get_camera_names(machine_name)
        if camera_name not in names:
            print(f"[SIM-VIDEO] camera '{camera_name}' not in {names}; falling back")
            if names:
                camera_name = names[0]
            else:
                print("[SIM-VIDEO] no cameras at all — aborting")
                _video_streaming = False
                return
        cam = sphx.get_camera(machine_name, camera_name)
        print(f"[SIM-VIDEO] machine={machine_name} cam={camera_name} type={type(cam).__name__}")
    except Exception as e:
        print(f"[SIM-VIDEO] camera setup failed: {e}")
        _video_streaming = False
        return

    print("[SIM-VIDEO] reader thread started")
    last_warn = 0.0
    # Rate-cap the heavy per-frame work (JPEG encode + BGR copy + queue
    # pushes). Sphinx happily delivers 28+ fps on a fast host; sustaining
    # that rate through cv2.imencode + BGR memcpy eats ~4 CPU cores in
    # this Python process, which starves Olympe's command path and the
    # operator sees "video reacts to my key presses 20+ s late". Capping
    # the *processed* rate to SIM_PRODUCER_FPS keeps the FC light and
    # the displayed video at a uniform cadence. Frames that arrive
    # faster than the cap are read-and-released (sphinx shm doesn't
    # back up) but skipped before encode.
    SIM_PRODUCER_FPS = int(os.getenv("SIM_PRODUCER_FPS", "15"))
    _min_dt = 1.0 / max(1, SIM_PRODUCER_FPS)
    _last_emit = 0.0
    try:
        while _video_streaming:
            try:
                ok, fr = cam.read(timeout=1.0)
            except Exception as e:
                t = time.monotonic()
                if t - last_warn > 5.0:
                    print(f"[SIM-VIDEO] cam.read err: {e}")
                    last_warn = t
                time.sleep(0.05)
                continue
            if not ok or fr is None:
                time.sleep(0.005)
                continue
            # Skip frames that arrive faster than the producer target
            # rate. We must still consume + release the shm frame so
            # sphinx's queue doesn't back up, hence the read-then-skip
            # rather than a sleep before read.
            _now = time.monotonic()
            if _now - _last_emit < _min_dt:
                try:
                    fr.release()
                except Exception:
                    pass
                continue
            _last_emit = _now
            try:
                buf = fr.buffer  # ndarray (h, w, ch)
                # Sphinx delivers B8G8R8A8 → drop alpha, already BGR for cv2
                if buf.ndim == 3 and buf.shape[2] >= 3:
                    bgr = buf[:, :, :3]
                else:
                    bgr = buf
                # Publish raw BGR for H.264 fan-out only when a client
                # is actively consuming /api/video/h264. The .copy() is
                # required when we do publish: ``fr.buffer`` is backed
                # by sphinx shm and sphinx will overwrite it on
                # ``fr.release()`` below. When no client is connected
                # we skip the copy entirely — ~40 MB/s of avoided
                # memcpy on the steady-state hot path.
                if _sim_h264_active_clients > 0:
                    global _sim_bgr_latest, _sim_bgr_resolution, _sim_bgr_frame_id
                    with _sim_bgr_condition:
                        _sim_bgr_latest = bgr.copy()
                        _sim_bgr_resolution = (bgr.shape[1], bgr.shape[0])
                        _sim_bgr_frame_id += 1
                        _sim_bgr_condition.notify_all()
                ok2, jpg = cv2.imencode(".jpg", bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), VIDEO_JPEG_QUALITY])
                if ok2:
                    with _video_jpeg_lock:
                        _video_last_jpeg = jpg.tobytes()
                    _video_frame_count += 1
                    if _video_frame_count == 1:
                        h, w = bgr.shape[:2]
                        print(f"[SIM-VIDEO] First frame: {w}x{h}")
                with _pos_cfg_lock:
                    _pos_enabled = _pos_cfg.get("enabled", False)
                with _rec_lock:
                    _rec_active = _rec_enabled
                if _pos_enabled or _rec_active:
                    try:
                        _pos_frame_q.put_nowait((bgr.copy(), time.monotonic()))
                    except queue.Full:
                        pass
            except Exception as ex:
                if _video_frame_count == 0:
                    print(f"[SIM-VIDEO] frame process error: {ex}")
            finally:
                try:
                    fr.release()
                except Exception:
                    pass
    finally:
        if cam is not None:
            try:
                cam.release()
            except Exception:
                pass
        print(f"[SIM-VIDEO] reader thread exiting (frames={_video_frame_count})")


# ── END SIM VIDEO PATCH ─────────────────────────────────
_video_forward_proc = None
_video_forward_target = ""
_video_frame_count = 0

# ---------------------------------------------------------------------------
# Positioning state
# ---------------------------------------------------------------------------
# DEPRECATED — unified-server positioning subsystem.
#
# In the combined marker_mission.app world the actual aruco-based
# world-position estimator lives in marker_mission/aruco_detector.py +
# marker_mission/arena.py (the one wired to the /arena tab the
# operator edits). This subsystem here is a separate, second
# solvePnP path that reads controller_unified/arena_config.json
# instead. It used to be the primary positioning path and is kept
# alive only for legacy HTTP-only deployments and for the few
# clients that still consume its SSE / MJPEG outputs. Do NOT add
# new code that depends on it — use marker_mission's detector +
# arena_holder instead.
#
# To minimise CPU cost in the combined deployment, the loop below
# is gated on subscribers (recording / SSE / MJPEG-video viewers):
# it no longer runs solvePnP every tick just because the cfg has
# enabled=True. See positioning_loop() for the gate.
# ---------------------------------------------------------------------------
_pos_frame_q: "queue.Queue" = queue.Queue(maxsize=3)   # BGR frames tapped from video cb
_pos_processor = None   # live HeadlessAruCoPositioning instance (set by positioning_loop)
_pos_sse_queues: list = []
_pos_sse_lock = threading.Lock()
# Live count of clients holding the /api/position/video MJPEG generator
# open. Incremented inside the generator's try; decremented in its
# finally. Read by positioning_loop's subscriber gate.
_pos_video_clients: int = 0
_pos_video_clients_lock = threading.Lock()
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
    # Multiplicative correction on the camera↔marker distance estimated by
    # solvePnP. 1.0 = no correction. Use to compensate for systematic
    # scale error caused by marker_size_m mismatch or uncalibrated focal
    # length. Field example: UI reads 7 m, real distance is 9 m →
    # distance_scale = 9/7 ≈ 1.286.
    "distance_scale":       1.0,
    # ── Precision tuning (mirrors ctrl_position.py module constants) ──
    # Exposed over the UI so operators can tighten / loosen the fusion
    # without a server restart. Values here get pushed into the running
    # positioner via live-patch in /api/position/config POST.
    "pose_hold_sec":        0.8,    # keep publishing pose for N s after last valid fix
    "min_ref_count":        1,      # min visible markers for a fused pose
    "min_ref_weight":       0.0,    # min weight of best reference marker
    "meas_blend_min":       0.35,   # EMA α lower bound (high-quality fixes)
    "meas_blend_max":       0.85,   # EMA α upper bound (low-quality fixes)
    "vel_blend":            0.25,   # IMU vel (0) ↔ Kalman-state vel (1) blend
    "max_state_dt":         1.0,    # reset state if more than N s between updates
    "kalman_process_var":   1e-3,   # Q — process-noise variance per axis
    "kalman_meas_var":      1e-1,   # R — measurement-noise variance per axis
    # ── IMU low-pass filter (first-order IIR) ───────────────────────
    # Applied to vgx/vgy/vgz (body-frame velocity, cm/s) at telemetry
    # ingestion time. Downstream: the position fusion + the sync buffer
    # both see the filtered values. Set 0 (or negative) to disable.
    # Default 5 Hz cut-off smooths Anafi's noisy per-sample velocity
    # without dulling reaction to actual rapid moves.
    "imu_lowpass_hz":       5.0,
    # Seen-marker hold time (seconds). A marker that's been detected
    # within this window still appears as "seen" even if the current
    # frame missed it. Stops the arena-view halos flickering on
    # single-frame detection dropouts — the per-frame detection rate
    # is still visible on seen_markers_raw for diagnostics.
    "seen_hold_s":          0.6,
    # Pose-jump gate (metres). Reject a fresh fix if it disagrees with
    # the Kalman-predicted state by more than this. Kills the
    # catastrophic >10 m single-marker glitches. 0 = disabled.
    "max_pose_jump_m":      0.0,
    # Target-box marker size. SDC26 target boxes use 19 cm ArUco
    # stickers; arena walls use 50 cm. Markers with ID ≥ 30 are
    # solvePnP-ed against this smaller corner set so target positions
    # land at their true location instead of being over-reported by
    # 50/19 ≈ 2.6 ×.
    "target_marker_size_m": 0.19,
    # Zero-Velocity Update — reverted per operator request (the dual-
    # marker-size path that shipped alongside ZUPT blew the tracker's
    # pose up outside the arena). Defaults are 0 so the feature is
    # inactive even if an older ctrl_position.py exposes the hooks.
    "zupt_speed_m_s":       0.0,   # 0 = disabled
    "zupt_hold_frames":     0,     # 0 = disabled
    # ── Auto Positioning ──────────────────────────────────────────
    # When True, the FC ignores every user-tuned filter/precision knob
    # in this dict (imu_weight, marker_size_m, top_k, outlier, all the
    # Kalman vars, the blends, distance_scale, max_pose_jump_m, …) and
    # uses a known-good built-in profile (CLAUDE_AUTO_CONFIG below)
    # that was derived from field-log analysis. The operator's saved
    # values are preserved in _pos_cfg but simply not applied to the
    # running processor. Toggling this off re-applies the stored
    # operator values. Default True for "just works" out of the box.
    "auto_positioning":     True,
}


# ─────────────────────────────────────────────────────────────────
# Built-in positioning profile — applied when auto_positioning=True.
#
# Derived from flight-log analysis of 19:11:40 (commit 91b87ce), which
# tracked well with 96 % fresh fixes and 0.06 m median step. The two
# additions over that baseline:
#   - min_ref_count = 1 but min_ref_weight = 0.15 → still accept a
#     single high-quality fix without letting junk detections in
#   - max_pose_jump_m = 3.0 → gate blocks the 16 m outliers we saw
#     in that same log (1.4 % rate of catastrophic single-frame
#     glitches, which would now be silently discarded)
# Anything NOT in this dict (e.g. `enabled`, `fps`, `latency_comp_s`,
# `sync_max_gap_s`, `telemetry_buffer_s`) is left under operator
# control — those are about pipeline cadence, not fusion behaviour.
# ─────────────────────────────────────────────────────────────────
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
    "target_marker_size_m": 0.19,   # SDC26 target stickers (dual-size re-enabled)
    "zupt_speed_m_s":       0.0,    # ZUPT stays off (was the real revert culprit)
    "zupt_hold_frames":     0,
}


def _effective_pos_cfg() -> dict:
    """Return the position config that should actually be applied to the
    running processor. If auto_positioning is on, CLAUDE_AUTO_CONFIG wins;
    otherwise the operator's saved _pos_cfg wins."""
    with _pos_cfg_lock:
        cfg = dict(_pos_cfg)
    if cfg.get("auto_positioning", True):
        for k, v in CLAUDE_AUTO_CONFIG.items():
            cfg[k] = v
    return cfg


# ── IMU low-pass filter state ────────────────────────────────────────
# First-order IIR with configurable cut-off frequency. Applied per-axis
# to the body-frame velocity components (vgx, vgy, vgz). The filter
# uses a time-based alpha (alpha = dt / (tau + dt)) so it handles
# the uneven telemetry cadence correctly.
_imu_lpf_lock = threading.Lock()
_imu_lpf_state: dict = {
    "vgx": None, "vgy": None, "vgz": None,
    "last_ts": 0.0,
}


def _apply_imu_lpf(vgx, vgy, vgz, ts_mono: float) -> tuple:
    """Filter one telemetry sample of body-frame velocity.

    Returns (vgx_f, vgy_f, vgz_f) — smoothed values, or the originals
    unchanged if the filter is disabled (cut-off <= 0) or any input is
    None (stale sample). Thread-safe; used from the Anafi telemetry
    callback.
    """
    with _pos_cfg_lock:
        fc = float(_pos_cfg.get("imu_lowpass_hz", 0.0) or 0.0)
    if fc <= 0.0:
        # Filter disabled — reset state so re-enable starts cleanly.
        with _imu_lpf_lock:
            _imu_lpf_state["vgx"] = None
            _imu_lpf_state["vgy"] = None
            _imu_lpf_state["vgz"] = None
        return vgx, vgy, vgz
    # Any None axis → bypass this sample (nothing to filter), but
    # don't drop the other axes. Keeps the pipeline robust to partial
    # telemetry gaps.
    if vgx is None and vgy is None and vgz is None:
        return vgx, vgy, vgz

    with _imu_lpf_lock:
        last = _imu_lpf_state["last_ts"]
        dt = ts_mono - last if last > 0 else 1.0 / max(1e-3, fc)
        # Clamp dt to a sane range — first sample or long-gap recovery
        # should not slam the filter into a corner.
        dt = max(1e-3, min(1.0, dt))
        tau = 1.0 / (2.0 * math.pi * fc)
        alpha = dt / (tau + dt)

        def _step(prev, x):
            if x is None:
                return prev   # keep last value if sample missing
            if prev is None:
                return float(x)
            return prev + alpha * (float(x) - prev)

        _imu_lpf_state["vgx"] = _step(_imu_lpf_state["vgx"], vgx)
        _imu_lpf_state["vgy"] = _step(_imu_lpf_state["vgy"], vgy)
        _imu_lpf_state["vgz"] = _step(_imu_lpf_state["vgz"], vgz)
        _imu_lpf_state["last_ts"] = ts_mono
        return (_imu_lpf_state["vgx"],
                _imu_lpf_state["vgy"],
                _imu_lpf_state["vgz"])

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
    # ── Target boxes ─────────────────────────────────────────────
    # SDC26 target stickers are 19 cm (vs 50 cm arena markers). Only two
    # teams: Blue and Red. Marker IDs encode team + box number:
    #   - IDs 31-36 → Blue box 1-6 (first digit 3 = blue)
    #   - IDs 41-46 → Red  box 1-6 (first digit 4 = red)
    # The UI derives the box number from `id % 10` and the team name from
    # the range mapping below.
    "target_marker_size_m": 0.19,
    "target_teams": [
        {"id_range": [31, 36], "team": "Blue", "color": "#3b82f6"},
        {"id_range": [41, 46], "team": "Red",  "color": "#ef4444"},
    ],
    # Optional per-ID overrides: {"id": 34, "team": "Blue 4", "color": "#..."}
    # Leave empty to use the range mapping above.
    "target_overrides": [],
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
    """Apply persisted flight limits to globals on startup.

    The ceiling (MAX_ALTITUDE_M) is the most important one here: it's
    enforced by THIS process's RC tick loop (20 Hz, see rc_loop()) and
    is independent of any remote/C2 connection. If the C2 is down, the
    Pi still reads MAX_ALTITUDE_M every tick and clamps any upward RC
    that would take the drone past it. That's why persisting the value
    to disk matters — after a Pi restart the guard must come back up
    with the operator's last-set value, not just the compile-time
    default.
    """
    global MAX_ALTITUDE_M, MAX_VERTICAL_SPEED, MAX_TILT, MAX_YAW_SPEED
    global _arena_margin_m, _arena_bounds
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
            if "arena_safety_margin_m" in data:
                _arena_margin_m = float(data["arena_safety_margin_m"])
            if "arena_bounds" in data and isinstance(data["arena_bounds"], dict):
                for k, v in data["arena_bounds"].items():
                    if k in _arena_bounds:
                        _arena_bounds[k] = float(v)
            print(f"[UNIFIED] Loaded flight config: alt={MAX_ALTITUDE_M} vs={MAX_VERTICAL_SPEED} tilt={MAX_TILT} yaw={MAX_YAW_SPEED}")
            print(f"[CEILING] Pi-side safety ceiling: {MAX_ALTITUDE_M:.2f} m (loaded from {FLIGHT_CONFIG_PATH.name})")
            print(f"[CEILING] Enforced every RC tick, independent of C2 connection.")
            print(f"[ARENA]   Pi-side safety margin: {_arena_margin_m:.2f} m "
                  f"bounds=x[{_arena_bounds['x_min']},{_arena_bounds['x_max']}] "
                  f"y[{_arena_bounds['y_min']},{_arena_bounds['y_max']}]")
    except Exception as e:
        print(f"[UNIFIED] flight_config load error: {e}")


def _save_flight_config(results: dict):
    """Persist any successfully-updated flight limit to flight_config.json."""
    keys = {"max_altitude_m", "max_vertical_speed", "max_tilt", "max_yaw_speed",
            "arena_safety_margin_m", "arena_bounds"}
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
    with discrete_lock:
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
    def rotate(self, direction: str, degrees: int,
               speed: Optional[int] = None) -> Tuple[bool, str]: ...
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

    def rotate(self, direction: str, degrees: int,
               speed: Optional[int] = None) -> Tuple[bool, str]:
        # Tello SDK has no per-rotation angular-speed control; the optional
        # ``speed`` override (used on Anafi) is accepted and ignored here so
        # the YAW_IMU 2nd arg degrades gracefully on Tello hardware.
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
        # Filled in by poll_telemetry once the drone has reported its
        # ProductSerialHigh + ProductSerialLow state events. Stays None
        # if the drone never reports them (older Olympe / firmware).
        self.serial_number: Optional[str] = None
        self._serial_high: Optional[str] = None
        self._serial_low: Optional[str] = None

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
            for _attempt in range(3):
                try:
                    d.stop_piloting()
                    return
                except Exception:
                    if _attempt < 2:
                        time.sleep(0.15)
        # If stop_piloting() failed on all retries we do NOT fall back to
        # sending a PCMD(0) — that would re-activate the piloting interface
        # and cause the firmware to ignore the subsequent moveBy command.
        # The RC loop is already paused via start_discrete_window() before
        # this is called, so no PCMD is being sent anyway.

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
        # Wait up to 10 s for FlyingStateChanged to arrive after reconnect.
        # Without this, a takeoff attempt immediately post-reconnect sees
        # state=None and is silently refused by the drone firmware.
        fs = self._get_state(FlyingStateChanged)
        if fs is None:
            print("[ANAFI] /api/takeoff — state not ready, waiting up to 10 s...")
            for _ in range(20):
                time.sleep(0.5)
                fs = self._get_state(FlyingStateChanged)
                if fs is not None:
                    break
        if fs is None:
            # State still missing — Olympe's connection is broken.
            # Force a hard reset + reconnect (destroys the Drone object and
            # creates a fresh one) then wait another 10 s.
            print("[ANAFI] /api/takeoff — state still None, forcing hard reset + reconnect...")
            try:
                self.hard_reset()
                time.sleep(1.0)
                self.connect()
            except Exception as e:
                print(f"[ANAFI] hard reset failed: {e}")
            for _ in range(20):
                time.sleep(0.5)
                fs = self._get_state(FlyingStateChanged)
                if fs is not None:
                    break
            if fs is None:
                print("[ANAFI] /api/takeoff — state still None after reset, aborting")
                return False, "state_not_ready"
            print("[ANAFI] /api/takeoff — state recovered after hard reset")
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
                # Try to grab command_lock briefly so we serialise with
                # any in-flight piloting command. If it's held (e.g.
                # the mission's controller thread is mid-moveBy.wait()
                # during an _IMU step), DO NOT block — fall through and
                # send Landing without the lock. Anafi handles
                # preemption at the firmware level: the Landing command
                # supersedes whatever moveBy was running, regardless of
                # who's holding our Python-side serialisation lock.
                #
                # Earlier bug: killswitch /api/land waited on this lock
                # for up to 30 s while moveBy.wait() held it, so the
                # drone never started landing. Now the killswitch
                # always gets through within ~100 ms.
                acquired = command_lock.acquire(timeout=0.1)
                try:
                    result = d(Landing()).wait(_timeout=10)
                finally:
                    if acquired:
                        command_lock.release()
                if not acquired:
                    print(f"[ANAFI] Landing attempt {attempt}: lock "
                          f"contested — sent without serialisation")
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
        time.sleep(0.4)   # give firmware time to exit PCMD mode before moveBy
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

    def rotate(self, direction: str, degrees: int,
               speed: Optional[int] = None) -> Tuple[bool, str]:
        d = self._d()
        if d is None:
            return False, "not_ready"
        rad = math.radians(degrees)
        d_psi = rad if direction == "cw" else -rad
        self._stop_piloting()
        time.sleep(0.1)
        # Optional per-rotation angular-speed override: temporarily raise
        # (or lower) MaxRotationSpeed for THIS moveBy only, then restore the
        # configured global. The moveBy yaw rate is capped by MaxRotationSpeed,
        # so this is how YAW_IMU's 2nd arg speeds up / slows down the turn.
        spd_applied = None
        if speed is not None:
            spd_applied = max(1, min(200, float(speed)))
            try:
                d(MaxRotationSpeed(spd_applied)).wait(_timeout=2)
                print(f"[ANAFI] rotate speed override -> MaxRotationSpeed={spd_applied}°/s")
            except Exception as e:
                print(f"[ANAFI] rotate speed override failed: {e}")
                spd_applied = None
        try:
            with command_lock:
                result = d(moveBy(0, 0, 0, d_psi)).wait(_timeout=15)
        finally:
            # Always restore the global default so a one-off fast turn
            # doesn't leak into subsequent rotations.
            if spd_applied is not None:
                try:
                    d(MaxRotationSpeed(MAX_YAW_SPEED)).wait(_timeout=2)
                except Exception as e:
                    print(f"[ANAFI] restore MaxRotationSpeed failed: {e}")
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

        # Serial number -- only worth polling until we've cached it.
        if HAS_PRODUCT_SERIAL and self.serial_number is None:
            if self._serial_high is None:
                hi_state = self._get_state(ProductSerialHighChanged)
                if hi_state:
                    raw = hi_state.get("high")
                    if raw:
                        self._serial_high = str(raw).strip()
            if self._serial_low is None:
                lo_state = self._get_state(ProductSerialLowChanged)
                if lo_state:
                    raw = lo_state.get("low")
                    if raw:
                        self._serial_low = str(raw).strip()
            if self._serial_high is not None and self._serial_low is not None:
                self.serial_number = self._serial_high + self._serial_low
                print(f"[ANAFI] serial_number = {self.serial_number}")
        if self.serial_number is not None:
            data["serial_number"] = self.serial_number

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

    def camera_config_get(self) -> dict:
        """Snapshot of the camera's current settings, gated per-feature
        on the corresponding ``HAS_CAMERA_*`` probe. Keys are returned
        only for the axes the firmware actually supports."""
        d = self._d()
        if d is None:
            return {"connected": False}
        out: dict = {"connected": True, "cam_id": 0}
        def _state(evt):
            try:
                s = d.get_state(evt)
                return s if isinstance(s, dict) else {}
            except Exception:
                return {}
        if HAS_CAMERA_EXPOSURE:
            s = _state(exposure_settings)
            out["exposure"] = {
                "mode": s.get("mode"),
                "shutter_speed": s.get("shutter_speed"),
                "iso_sensitivity": s.get("iso_sensitivity"),
                "max_iso_sensitivity": s.get("max_iso_sensitivity"),
                "metering_mode": s.get("metering_mode"),
            }
        if HAS_CAMERA_WB:
            s = _state(white_balance)
            out["white_balance"] = {
                "mode": s.get("mode"),
                "temperature": s.get("temperature"),
            }
        if HAS_CAMERA_HDR:
            out["hdr"] = {"value": _state(hdr_setting).get("value")}
        if HAS_CAMERA_EV:
            out["ev_compensation"] = {
                "value": _state(ev_compensation).get("value"),
            }
        if HAS_CAMERA_ANTIFLICKER:
            out["antiflicker"] = {"mode": _state(antiflicker_mode).get("mode")}
        elif HAS_AF_LEGACY:
            out["antiflicker"] = {"mode": _state(AntiflickerLegacyChanged).get("mode")}
        if HAS_VIDEO_STABILIZATION:
            out["video_stabilization"] = {
                "mode": _state(VideoStabilizationModeChanged).get("mode"),
            }
        if HAS_STREAM_MODE:
            out["stream_mode"] = {
                "mode": _state(VideoStreamModeChanged).get("mode"),
            }
        if HAS_CAMERA_RECORDING_MODE:
            s = _state(recording_mode)
            out["recording"] = {
                "mode":       s.get("mode"),
                "resolution": s.get("resolution"),
                "framerate":  s.get("framerate"),
                "hyperlapse": s.get("hyperlapse"),
                "bitrate":    s.get("bitrate"),  # reported, read-only
            }
        if HAS_CAMERA_ZOOM:
            s = _state(zoom_level)
            out["zoom"] = {"level": s.get("level")}
        return out

    def camera_config_set(self, data: dict) -> dict:
        """Apply a partial dict of camera settings. Each top-level key
        is independent (operator can submit just one axis). Returns
        ``{key: bool}`` per-axis success.
        """
        d = self._d()
        if d is None:
            return {"error": "not_ready"}
        results: dict = {}
        if "exposure" in data and HAS_CAMERA_EXPOSURE:
            e = data["exposure"] or {}
            try:
                with command_lock:
                    r = d(set_exposure_settings(
                        cam_id=0,
                        mode=e.get("mode", "automatic"),
                        shutter_speed=e.get("shutter_speed",
                                            "shutter_1_over_30"),
                        iso_sensitivity=e.get("iso_sensitivity", "iso_100"),
                        max_iso_sensitivity=e.get("max_iso_sensitivity",
                                                   "iso_3200"),
                        metering_mode=e.get("metering_mode", "standard"),
                    )).wait(_timeout=2)
                results["exposure"] = bool(r and r.success())
            except Exception as ex:
                results["exposure"] = False
                results["exposure_error"] = str(ex)
        if "white_balance" in data and HAS_CAMERA_WB:
            w = data["white_balance"] or {}
            try:
                with command_lock:
                    r = d(set_white_balance(
                        cam_id=0,
                        mode=w.get("mode", "automatic"),
                        temperature=w.get("temperature", "t_5000"),
                    )).wait(_timeout=2)
                results["white_balance"] = bool(r and r.success())
            except Exception as ex:
                results["white_balance"] = False
                results["white_balance_error"] = str(ex)
        if "hdr" in data and HAS_CAMERA_HDR:
            try:
                with command_lock:
                    r = d(set_hdr_setting(
                        cam_id=0,
                        value=(data["hdr"] or {}).get("value", "inactive"),
                    )).wait(_timeout=2)
                results["hdr"] = bool(r and r.success())
            except Exception as ex:
                results["hdr"] = False
                results["hdr_error"] = str(ex)
        if "ev_compensation" in data and HAS_CAMERA_EV:
            try:
                with command_lock:
                    r = d(set_ev_compensation(
                        cam_id=0,
                        value=(data["ev_compensation"] or {}).get("value",
                                                                  "ev_0_00"),
                    )).wait(_timeout=2)
                results["ev_compensation"] = bool(r and r.success())
            except Exception as ex:
                results["ev_compensation"] = False
                results["ev_compensation_error"] = str(ex)
        if "antiflicker" in data:
            mode = (data["antiflicker"] or {}).get("mode", "auto")
            try:
                with command_lock:
                    if HAS_CAMERA_ANTIFLICKER:
                        r = d(set_antiflicker_mode(mode=mode)).wait(_timeout=2)
                    elif HAS_AF_LEGACY:
                        r = d(AntiflickerLegacy(mode=mode)).wait(_timeout=2)
                    else:
                        r = None
                results["antiflicker"] = bool(r and r.success())
            except Exception as ex:
                results["antiflicker"] = False
                results["antiflicker_error"] = str(ex)
        if "video_stabilization" in data and HAS_VIDEO_STABILIZATION:
            try:
                with command_lock:
                    r = d(VideoStabilizationMode(
                        mode=(data["video_stabilization"] or {}).get(
                            "mode", "roll_pitch"),
                    )).wait(_timeout=2)
                results["video_stabilization"] = bool(r and r.success())
            except Exception as ex:
                results["video_stabilization"] = False
                results["video_stabilization_error"] = str(ex)
        if "stream_mode" in data and HAS_STREAM_MODE:
            try:
                with command_lock:
                    r = d(VideoStreamMode(
                        mode=(data["stream_mode"] or {}).get(
                            "mode", "low_latency"),
                    )).wait(_timeout=2)
                results["stream_mode"] = bool(r and r.success())
            except Exception as ex:
                results["stream_mode"] = False
                results["stream_mode_error"] = str(ex)
        if "recording" in data and HAS_CAMERA_RECORDING_MODE:
            rec = data["recording"] or {}
            try:
                with command_lock:
                    r = d(set_recording_mode(
                        cam_id=0,
                        mode=rec.get("mode", "standard"),
                        resolution=rec.get("resolution", "res_1080p"),
                        framerate=rec.get("framerate", "fps_30"),
                        hyperlapse=rec.get("hyperlapse", "ratio_15"),
                    )).wait(_timeout=2)
                results["recording"] = bool(r and r.success())
            except Exception as ex:
                results["recording"] = False
                results["recording_error"] = str(ex)
        if "zoom" in data and HAS_CAMERA_ZOOM:
            # Accept both {"zoom": 1.5} (float, from controller code) and
            # {"zoom": {"level": 1.5}} (dict, legacy camera-config format).
            _z = data["zoom"]
            level = _z.get("level", 1.0) if isinstance(_z, dict) else float(_z or 1.0)
            try:
                with command_lock:
                    r = d(_CameraSetZoom(
                        cam_id=0, control_mode="level",
                        target=float(level),
                    )).wait(_timeout=2)
                results["zoom"] = bool(r and r.success())
            except Exception as ex:
                results["zoom"] = False
                results["zoom_error"] = str(ex)
        return results

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
                # Send the SDK command (ardrone3.PilotingSettings.MaxAltitude).
                # This is exactly the "MaxAltitude" setting the Anafi
                # autopilot uses internally — we send it via the Olympe
                # SDK, not a custom layer.
                d(MaxAltitude(v)).wait(_timeout=2)
                MAX_ALTITUDE_M = v
                results["max_altitude_m"] = v

                # Verification — read the drone's CONFIRMED MaxAltitude
                # state. The drone may clamp to its own [min..max] range
                # silently; the state tells us what it actually accepted.
                if _HAS_MAXALT_STATE:
                    try:
                        st = d.get_state(MaxAltitudeChanged)
                        if st and "current" in st:
                            accepted = float(st["current"])
                            results["max_altitude_m_firmware_current"] = accepted
                            if abs(accepted - v) > 0.05:
                                print(f"[ANAFI] MaxAltitude: requested {v}m, "
                                      f"drone accepted {accepted}m (clamped by firmware)")
                            else:
                                print(f"[ANAFI] MaxAltitude: {accepted}m confirmed by drone")
                        if st and "min" in st and "max" in st:
                            results["max_altitude_m_firmware_range"] = [
                                float(st["min"]), float(st["max"])]
                    except Exception as e:
                        results["max_altitude_m_verify_error"] = str(e)[:120]
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
    # Class-level mutex so two concurrent callers can't both see
    # "_video_streaming == False" and each fire their own d.streaming.start().
    # That race was leaking VideoDecoder#N on every near-simultaneous call.
    _video_start_mutex = threading.Lock()

    def video_start_mjpeg(self) -> Tuple[bool, str]:
        """Start (or re-use) the Anafi MJPEG pipeline.

        IDEMPOTENT *AND* MUTUALLY EXCLUSIVE — the class-level
        _video_start_mutex prevents two concurrent callers from both
        entering d.streaming.start(). Prior versions, even with the
        "already streaming?" check, could race: thread A reads flag
        False, thread B reads flag False, both call start(). Olympe
        spawns two decoders. Mutex fixes that.

        Also resets the flag if the drone connection was lost — a
        reconnect must rebuild the stream from scratch, otherwise
        the flag stays stuck True and subsequent calls are silent
        no-ops against a disconnected drone.
        """
        global _video_mode, _video_last_jpeg, _video_frame_count, _video_streaming
        d = self._d()
        if d is None:
            return False, "drone not connected"
        if not HAS_CV2:
            return False, "cv2 not installed — pip install opencv-python-headless"

        # ── sim dispatch ── if connected to a Sphinx-simulated Anafi
        # (10.202.0.x / 172.21.0.x), the drone has no RTSP stream;
        # use pysphinx to read the camera directly.
        if _is_sim_anafi_ip(getattr(self, "ip", "")):
            with self.__class__._video_start_mutex:
                if _video_streaming and _video_mode == "mjpeg":
                    return True, "sim mjpeg already streaming"
                _video_mode = "mjpeg"
                _video_last_jpeg = b""
                _video_frame_count = 0
                _video_streaming = True
                th = threading.Thread(target=_sim_video_loop,
                                      daemon=True, name="sphinx-sim-video")
                th.start()
                print("[ANAFI] sim drone detected — using pysphinx camera reader")
                return True, "sim mjpeg started (pysphinx)"

        with self.__class__._video_start_mutex:
            # Re-check inside the mutex — another thread may have started
            # the stream while we were waiting.
            if _video_streaming and _video_mode == "mjpeg":
                print("[ANAFI] video_start_mjpeg: already streaming, skipping")
                return True, "mjpeg already streaming"

            # If we were in forward mode or half-started, tear down FIRST so
            # Olympe can release its decoder before we request a new one.
            if _video_streaming:
                print("[ANAFI] video_start_mjpeg: stopping previous stream before restart")
                try:
                    self.video_stop_mjpeg()
                except Exception:
                    pass
                time.sleep(0.3)   # let Olympe tear down before we re-start

            # Detect API — retry up to 8 times with 2 s gaps (16 s total).
            # After reconnect or a _media_removed_impl callback, Olympe
            # rebuilds its streaming object internally; set_callbacks
            # briefly disappears during that window. 3×1 s was too short
            # after a full drone reconnect which can take 10-20 s.
            api = "none"
            for _attempt in range(9):
                if hasattr(d, "streaming") and hasattr(d.streaming, "set_callbacks"):
                    api = "modern"
                    break
                elif hasattr(d, "set_streaming_callbacks"):
                    api = "legacy"
                    break
                if _attempt < 8:
                    print(f"[ANAFI] streaming API not ready yet, retrying in 2 s "
                          f"(attempt {_attempt + 1}/8)…")
                    time.sleep(2.0)
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
            # Push raw BGR for in-proc readers (no intermediate JPEG decode).
            global _video_bgr_latest, _video_bgr_ts
            with _video_bgr_cond:
                _video_bgr_latest = cv_frame
                _video_bgr_ts = time.monotonic()
                _video_bgr_cond.notify_all()
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
            with discrete_lock:
                in_discrete = time.time() < discrete_until
            if not in_discrete:
                flying = sdk_flying

        # Compute speed
        vgx = data.get("vgx") or 0
        vgy = data.get("vgy") or 0
        vgz = data.get("vgz") or 0

        # ── IMU low-pass filter ──────────────────────────────────
        # Runs before any downstream consumer sees these values, so
        # both the telemetry dict and the sync buffer used for
        # position fusion see the filtered velocity. The raw values
        # are preserved on a suffixed key for diagnostics.
        if vgx or vgy or vgz:
            vgx_f, vgy_f, vgz_f = _apply_imu_lpf(vgx, vgy, vgz, loop_mono)
            data["vgx_raw"], data["vgy_raw"], data["vgz_raw"] = vgx, vgy, vgz
            data["vgx"] = round(float(vgx_f), 3) if vgx_f is not None else vgx
            data["vgy"] = round(float(vgy_f), 3) if vgy_f is not None else vgy
            data["vgz"] = round(float(vgz_f), 3) if vgz_f is not None else vgz
            vgx, vgy, vgz = data["vgx"], data["vgy"], data["vgz"]
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
                with conn_lock:
                    conn_state["consecutive_failures"] = 0
                    conn_state["last_error"] = ""
            else:
                # Drone dropped — reset the video-streaming flag so the
                # next video_start_mjpeg call actually restarts the
                # pipeline against the NEW connection instead of being
                # silently skipped ("already streaming").
                global _video_streaming
                _video_streaming = False

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

        # ══════════════════════════════════════════════════════════════
        # Hard altitude ceiling — active in BOTH manual and autonomous flight.
        #
        # Our drone crashed into the roof once because:
        #   - The firmware MaxAltitude setting can be relaxed / ignored
        #     in indoor mode (where there's no GPS, only TOF/baro).
        #   - The old fence only zeroed positive ud when already at/above
        #     ceiling — no anticipation, no forced descent.
        #
        # New behaviour:
        #   - approaching zone  (h ≥ ceiling − APPROACH_MARGIN)  → clamp ud
        #                                                         proportionally
        #                                                         to remaining
        #                                                         clearance.
        #   - at ceiling        (h ≥ ceiling)                   → block all
        #                                                         upward RC.
        #   - above ceiling     (h ≥ ceiling + HARD_OVERSHOOT)   → force
        #                                                         active descent
        #                                                         regardless of
        #                                                         any input. The
        #                                                         drone yields
        #                                                         before the
        #                                                         pilot does.
        #
        # Applies unconditionally (no has_altitude_fence() gate) because we
        # don't care whether the firmware thinks it has a fence — height_cm
        # comes from TOF/baro and is what matters.
        global _ceiling_engaged, _ceiling_last_reason
        with telemetry_lock:
            cur_h_cm = telemetry.get("height_cm")
        _ceiling_engaged = False
        _ceiling_last_reason = ""
        if cur_h_cm is not None:
            ceil_cm = max(30, int(MAX_ALTITUDE_M * 100))
            APPROACH_MARGIN_CM = 50   # start easing off 50 cm below ceiling
            HARD_OVERSHOOT_CM = 20    # force descent if >20 cm above ceiling
            DESCENT_RC = 18           # gentle but authoritative push-down
            over_cm = cur_h_cm - ceil_cm
            if over_cm >= HARD_OVERSHOOT_CM:
                # Already above the ceiling by a meaningful margin — stop
                # arguing with the operator, push the drone down ourselves.
                ud = -max(DESCENT_RC, abs(int(ud))) if ud < 0 else -DESCENT_RC
                _ceiling_engaged = True
                _ceiling_last_reason = (
                    f"above ceiling: {cur_h_cm/100:.2f}m > {MAX_ALTITUDE_M:.2f}m, forcing descent"
                )
            elif over_cm >= 0:
                # Right at or just above — block any climb, allow descent.
                if ud > 0:
                    ud = 0
                _ceiling_engaged = True
                _ceiling_last_reason = (
                    f"at ceiling: {cur_h_cm/100:.2f}m ≥ {MAX_ALTITUDE_M:.2f}m, climb blocked"
                )
            elif over_cm >= -APPROACH_MARGIN_CM and ud > 0:
                # Approaching — clamp proportionally. 50cm→full input,
                # 0cm→zero. Prevents the drone from smashing the ceiling
                # under its own momentum when the pilot releases the stick.
                remaining = (-over_cm) / APPROACH_MARGIN_CM   # 0..1
                cap = max(1, int(STICK * remaining))
                if ud > cap:
                    ud = cap
                _ceiling_engaged = True
                _ceiling_last_reason = (
                    f"approaching ceiling: {cur_h_cm/100:.2f}m, climb capped to {cap}% "
                    f"(clearance {-over_cm}cm)"
                )
        # Soft floor — never let a descent command push us into the ground.
        FLOOR_CM = 15    # Anafi TOF minimum, roughly
        if cur_h_cm is not None and cur_h_cm <= FLOOR_CM and ud < 0:
            ud = 0

        # ══════════════════════════════════════════════════════════════
        # Arena XY boundary guard — enforced on EVERY RC command,
        # manual or autonomous. Reads the arena-frame pose published by
        # the position processor (_pos_st) and, if the drone is within
        # _arena_margin_m of any wall AND the current RC vector would
        # push it closer, clamps the offending axis to zero (or reverses
        # it if we've already overshot the margin).
        #
        # Applies to manual WASD → the old path where this was missing
        # — that's how the recent flight ended up at y=10.95 past the
        # y=10.8 back wall.
        #
        # Falls back silently when the positioner has no fresh fix
        # (stale=True or never published). In that case the boundary
        # guard contributes nothing; the operator is responsible.
        global _arena_engaged, _arena_last_reason
        _arena_engaged = False
        _arena_last_reason = ""
        try:
            with _pos_st_lock:
                px = _pos_st.get("x")
                py = _pos_st.get("y")
                pos_stale = _pos_st.get("stale", True)
                dx_dir    = _pos_st.get("dx")
                dy_dir    = _pos_st.get("dy")
        except Exception:
            px = py = None
            pos_stale = True
            dx_dir = dy_dir = None

        if (_arena_guard_enabled and
                px is not None and py is not None and not pos_stale):
            margin = float(_arena_margin_m)
            xmin, xmax = _arena_bounds["x_min"], _arena_bounds["x_max"]
            ymin, ymax = _arena_bounds["y_min"], _arena_bounds["y_max"]

            # Distance to each wall — negative means already past the margin.
            clearance = {
                "x_max": xmax - px - margin,
                "x_min": (px - xmin) - margin,
                "y_max": ymax - py - margin,
                "y_min": (py - ymin) - margin,
            }

            # RC interpretation: convert body-frame RC (lr, fb) to world-
            # frame intent using the drone's current heading (from the
            # position processor's dx/dy direction vector — already
            # normalised). Using position-processor direction avoids
            # chasing compass-yaw drift.
            #
            # Convention (matches _apply_boundary_guard in aruco_seek):
            #   body +fb  → world  dir  (dx, dy)
            #   body +lr  → world  right (cos, -sin) where yaw atan2(dx,dy)
            dx = float(dx_dir) if dx_dir is not None else 0.0
            dy = float(dy_dir) if dy_dir is not None else 1.0
            # Right-vector perpendicular to heading (90° CW in world XY):
            rx = dy
            ry = -dx
            wdx = lr * rx + fb * dx
            wdy = lr * ry + fb * dy

            # ACTIVE BRAKING. Inside the margin, clamping RC to 0 is
            # not enough — Anafi inertia keeps the drone coasting
            # forward another 1-3 m depending on speed. To actually
            # hold the drone OUTSIDE the wall, push REVERSE RC
            # proportional to how deep we are past the margin (and
            # capped at 100). This decelerates the drone aggressively
            # and pushes it back into the safe zone.
            #
            # Gain is empirical: 50 RC per metre of penetration. So
            # 0.5 m inside margin → 25 RC reverse; 2 m inside → 100
            # RC reverse (saturated). At cruise speed 2-4 m/s this
            # gives enough deceleration to come to a stop before
            # the actual wall.
            BRAKE_GAIN_RC_PER_M = 50.0
            actions = []
            # x max (right wall) — clearance neg = past margin
            if clearance["x_max"] <= 0 and wdx > 0:
                wdx = max(-100.0,
                          min(0.0, wdx + clearance["x_max"] * BRAKE_GAIN_RC_PER_M))
                actions.append(f"brake-x_max({wdx:.0f})")
            if clearance["x_min"] <= 0 and wdx < 0:
                wdx = min(100.0,
                          max(0.0, wdx - clearance["x_min"] * BRAKE_GAIN_RC_PER_M))
                actions.append(f"brake-x_min({wdx:.0f})")
            if clearance["y_max"] <= 0 and wdy > 0:
                wdy = max(-100.0,
                          min(0.0, wdy + clearance["y_max"] * BRAKE_GAIN_RC_PER_M))
                actions.append(f"brake-y_max({wdy:.0f})")
            if clearance["y_min"] <= 0 and wdy < 0:
                wdy = min(100.0,
                          max(0.0, wdy - clearance["y_min"] * BRAKE_GAIN_RC_PER_M))
                actions.append(f"brake-y_min({wdy:.0f})")

            if actions:
                # Convert world-frame intent back to body-frame RC.
                # inv rotation: lr = wdx·rx + wdy·ry; fb = wdx·dx + wdy·dy
                # (rx,ry) and (dx,dy) are orthonormal so transpose = inverse.
                lr = int(round(wdx * rx + wdy * ry))
                fb = int(round(wdx * dx + wdy * dy))
                _arena_engaged = True
                _arena_last_reason = (
                    f"near wall: pos=({px:.2f},{py:.2f}) "
                    f"clear=x[{clearance['x_max']:.2f},{clearance['x_min']:.2f}] "
                    f"y[{clearance['y_max']:.2f},{clearance['y_min']:.2f}] "
                    f"actions={','.join(actions)}"
                )

        with discrete_lock:
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


@app.get("/api/drone_ping")
def api_drone_ping():
    """Return the flight-controller → drone ICMP ping RTT in milliseconds.
    The value is sampled once per second by a background thread
    (_drone_ping_loop) so this endpoint returns immediately with the
    most recent measurement. None means the last ping timed out."""
    with _drone_ping_lock:
        st = dict(_drone_ping_state)
    age_s = time.time() - st["last_update"] if st.get("last_update") else None
    return jsonify(ok=True,
                   rtt_ms=st.get("rtt_ms"),
                   host=st.get("host") or drone_ip,
                   sample_age_s=round(age_s, 2) if age_s is not None else None)


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


@app.get("/api/wifi/debug")
def api_wifi_debug():
    """Introspect what's actually available in olympe.messages.wifi on this
    flight controller. Use this when /api/wifi/channel returns 'not
    supported' — the response shows what symbol names this Olympe build
    exposes, so we can fix the imports in HAS_WIFI_CTRL."""
    import importlib
    out = {"has_olympe_sdk": HAS_OLYMPE_SDK, "has_wifi_ctrl": HAS_WIFI_CTRL,
           "import_errors": list(_WIFI_IMPORT_ERRORS)}
    for path in ("olympe.messages.wifi", "olympe.enums.wifi"):
        try:
            m = importlib.import_module(path)
            names = sorted(n for n in dir(m) if not n.startswith("_"))
            out[path] = names
        except Exception as e:
            out[path] = f"import failed: {type(e).__name__}: {e}"
    try:
        cmd_mod = importlib.import_module("olympe.messages.wifi.Command")
        out["olympe.messages.wifi.Command"] = sorted(
            n for n in dir(cmd_mod) if not n.startswith("_"))
    except Exception as e:
        out["olympe.messages.wifi.Command"] = f"not importable: {e}"
    # Resolved bindings
    out["resolved"] = {
        "WifiSetApChannel":       getattr(WifiSetApChannel, "__name__", str(WifiSetApChannel)),
        "WifiScan":               getattr(WifiScan, "__name__", str(WifiScan)),
        "WifiBand":               getattr(WifiBand, "__name__", str(WifiBand)),
        "WifiSelectionType":      getattr(WifiSelectionType, "__name__", str(WifiSelectionType)),
        "WifiApChannelChanged":   getattr(WifiApChannelChanged, "__name__", str(WifiApChannelChanged)),
        "WifiAuthorizedChannel":  getattr(WifiAuthorizedChannel, "__name__", str(WifiAuthorizedChannel)),
        "WifiRssiChanged":        getattr(WifiRssiChanged, "__name__", str(WifiRssiChanged)),
    }
    # If Band has enum members, list them so the user knows what "5_GHz" to pass
    if WifiBand is not None:
        try:
            out["WifiBand_members"] = [m.name for m in WifiBand]
        except Exception:
            try:
                out["WifiBand_members"] = [k for k in WifiBand.__dict__.keys()
                                           if not k.startswith("_")]
            except Exception:
                pass
    if WifiSelectionType is not None:
        try:
            out["WifiSelectionType_members"] = [m.name for m in WifiSelectionType]
        except Exception:
            try:
                out["WifiSelectionType_members"] = [k for k in WifiSelectionType.__dict__.keys()
                                                    if not k.startswith("_")]
            except Exception:
                pass
    return jsonify(out)


def _jsonable(v):
    """Recursively convert Olympe state values into JSON-safe Python types.
    Olympe state dicts often contain ArsdkEnum instances (e.g. Band.5_GHz,
    Type.manual) which aren't serializable by Flask's jsonify. Convert
    enums to 'name' (or str(v) as fallback), and recurse into dict/list."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    # ArsdkEnum / IntEnum / any enum-like object
    name = getattr(v, "name", None)
    if name and isinstance(name, str):
        return name
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    try:
        return str(v)
    except Exception:
        return None


@app.get("/api/environment")
def api_environment_get():
    """Read the drone's current flight environment setting (indoor / outdoor).
    Indoor mode relaxes some GPS-based pre-arm checks and is recommended
    for hangar flying."""
    if not HAS_OLYMPE_SDK:
        return jsonify(ok=False, error="Olympe not available"), 501
    b = _anafi_backend()
    if b is None:
        return jsonify(ok=False, error="drone not connected"), 503
    try:
        import importlib
        wifi_mod = importlib.import_module("olympe.messages.wifi")
        EnvCh = (getattr(wifi_mod, "environment_changed", None)
                 or getattr(wifi_mod, "environement_changed", None)
                 or getattr(wifi_mod, "EnvironmentChanged", None))
        if EnvCh is None:
            return jsonify(ok=False, error="environment state message missing"), 501
        st = b._get_state(EnvCh)
        return jsonify(ok=True, status=_jsonable(dict(st) if st else None))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/environment")
def api_environment_set():
    """Set the drone's flight environment. Body: {"mode": "indoor" | "outdoor"}.
    Indoor relaxes GPS-related pre-arm checks; outdoor is the factory default."""
    if not HAS_OLYMPE_SDK:
        return jsonify(ok=False, error="Olympe not available"), 501
    b = _anafi_backend()
    if b is None:
        return jsonify(ok=False, error="drone not connected"), 503
    data = request.get_json(silent=True) or {}
    mode = str(data.get("mode", "indoor")).lower()
    if mode not in ("indoor", "outdoor"):
        return jsonify(ok=False, error="mode must be 'indoor' or 'outdoor'"), 400
    try:
        import importlib
        wifi_mod = importlib.import_module("olympe.messages.wifi")
        SetEnv = (getattr(wifi_mod, "set_environment", None)
                  or getattr(wifi_mod, "SetEnvironment", None))
        if SetEnv is None:
            return jsonify(ok=False, error="set_environment not exposed by this Olympe build"), 501
        # Resolve the enum member for the requested mode
        env_enum = None
        try:
            enums_mod = importlib.import_module("olympe.enums.wifi")
            env_cls = (getattr(enums_mod, "environment", None)
                       or getattr(enums_mod, "Environment", None))
            if env_cls is not None:
                env_enum = (getattr(env_cls, mode, None)
                            or getattr(env_cls, mode.upper(), None)
                            or getattr(env_cls, mode.capitalize(), None))
        except Exception:
            pass
        arg = env_enum if env_enum is not None else mode
        with command_lock:
            b.drone(SetEnv(environement=arg)).wait(_timeout=3)
        print(f"[ANAFI] Environment set to {mode} (arg={_jsonable(arg)})")
        return jsonify(ok=True, mode=mode, arg=_jsonable(arg))
    except TypeError:
        # Some Olympe versions spell the parameter "environment" (no typo)
        try:
            with command_lock:
                b.drone(SetEnv(environment=arg)).wait(_timeout=3)
            print(f"[ANAFI] Environment set to {mode} (arg={_jsonable(arg)}) [fallback kwarg]")
            return jsonify(ok=True, mode=mode, arg=_jsonable(arg))
        except Exception as e2:
            return jsonify(ok=False, error=str(e2)), 500
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


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
        return jsonify(ok=True, status=_jsonable(dict(st)))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


def _wifi_band_enum(s: str):
    """Map a user-supplied band string → the Olympe enum member.
    Accepts '2.4', '2.4ghz', '2_4_ghz', '5', '5ghz', '5_ghz', 'all'.
    Falls back to the raw string if WifiBand isn't an enum (Olympe 1.x)."""
    key = str(s).lower().replace("ghz", "").replace(".", "_").strip(" _")
    alias = {"2": "2_4_ghz", "2_4": "2_4_ghz", "24": "2_4_ghz",
             "5": "5_ghz",
             "": "all", "all": "all"}.get(key, key)
    if WifiBand is None:
        return alias
    # Try every casing Olympe might use
    for attempt in (alias, alias.lower(), alias.upper(),
                    alias.replace("_ghz", "_GHz"),
                    alias.replace("2_4", "2_4").upper()):
        if hasattr(WifiBand, attempt):
            return getattr(WifiBand, attempt)
    # Final fallback: iterate members, match by .name
    try:
        for m in WifiBand:
            if m.name.lower() == alias.lower():
                return m
    except Exception:
        pass
    return alias


def _wifi_sel_type_enum(which: str):
    """which is 'auto_all' | 'auto_2_4' | 'auto_5' | 'manual'.
    Returns the corresponding Olympe enum member (or the string if the
    enum isn't exposed)."""
    key = str(which).lower()
    # Candidate names across Olympe versions
    candidates = {
        "auto_all":  ["auto_all", "auto"],
        "auto_2_4":  ["auto_2_4_ghz", "auto_2_4Ghz", "auto_2_4GHz", "auto_2.4"],
        "auto_5":    ["auto_5_ghz", "auto_5GHz", "auto_5"],
        "manual":    ["manual", "fixed"],
    }.get(key, [key])
    if WifiSelectionType is None:
        return candidates[0]
    for name in candidates:
        if hasattr(WifiSelectionType, name):
            return getattr(WifiSelectionType, name)
    try:
        for m in WifiSelectionType:
            if m.name.lower() in [c.lower() for c in candidates]:
                return m
    except Exception:
        pass
    return candidates[0]


@app.post("/api/wifi/scan")
def api_wifi_scan():
    """Scan Wi-Fi channels in the requested band and return authorized
    channels with their measured RSSI. Pass {"band": "2_4_GHz"} or
    {"band": "5_GHz"} or {"band": "all"} (default).

    Olympe's scan trigger returns quickly but the interesting data comes
    asynchronously as 'scanned_item' events (one per detected AP) and
    'update_authorized_channels' events (with the final authorised
    channel list including occupancy). We subscribe, trigger the scan,
    wait a few seconds, then harvest everything we saw."""
    if not HAS_WIFI_CTRL:
        return jsonify(ok=False, error="Wi-Fi control not supported"), 501
    b = _anafi_backend()
    if b is None:
        return jsonify(ok=False, error="drone not connected"), 503
    data = request.get_json(silent=True) or {}
    band_str = str(data.get("band", "all")).lower()
    band = _wifi_band_enum(band_str)
    wait_s = float(data.get("wait_s", 4.0))

    # Import the per-item events defensively — they weren't probed in
    # the HAS_WIFI_CTRL block because they're additive.
    import importlib
    try:
        wifi_mod = importlib.import_module("olympe.messages.wifi")
        ScannedItem = getattr(wifi_mod, "scanned_item", None) \
                      or getattr(wifi_mod, "ScannedItem", None)
        UpdateAuthCh = getattr(wifi_mod, "update_authorized_channels", None) \
                      or getattr(wifi_mod, "UpdateAuthorizedChannels", None)
    except Exception:
        ScannedItem = UpdateAuthCh = None

    scanned_items: list[dict] = []
    auth_updates: list[dict] = []

    def _on_scanned_item(evt, *_):
        try:
            scanned_items.append(dict(evt.args))
        except Exception:
            scanned_items.append({"raw": str(evt)})

    def _on_auth_update(evt, *_):
        try:
            auth_updates.append(dict(evt.args))
        except Exception:
            auth_updates.append({"raw": str(evt)})

    subs = []
    try:
        # Subscribe to the event firehose FIRST so we don't miss early
        # emissions from the scan trigger.
        if ScannedItem is not None:
            subs.append(b.drone.subscribe(_on_scanned_item, ScannedItem()))
        if UpdateAuthCh is not None:
            subs.append(b.drone.subscribe(_on_auth_update, UpdateAuthCh()))

        with command_lock:
            b.drone(WifiScan(band=band)).wait(_timeout=2)

        # Let async events arrive
        time.sleep(max(0.5, min(wait_s, 10.0)))

        # As a fallback, also read the per-channel state (some Olympe
        # builds populate a dict keyed by channel number)
        channels_state = []
        try:
            st = b.drone.get_state(WifiAuthorizedChannel)
            if isinstance(st, dict):
                channels_state = list(st.values())
            elif isinstance(st, list):
                channels_state = st
        except Exception:
            pass

        return jsonify(ok=True,
                       band=band_str,
                       band_resolved=_jsonable(band),
                       wait_s=wait_s,
                       scanned_items=_jsonable(scanned_items),
                       scanned_count=len(scanned_items),
                       authorized_updates=_jsonable(auth_updates),
                       channels=_jsonable(channels_state))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500
    finally:
        for h in subs:
            try: h.unsubscribe()
            except Exception: pass


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

    band = _wifi_band_enum(band_str)
    if auto_mode:
        if band_str.startswith("5"):
            sel = _wifi_sel_type_enum("auto_5")
        elif band_str.startswith("2"):
            sel = _wifi_sel_type_enum("auto_2_4")
        else:
            sel = _wifi_sel_type_enum("auto_all")
    else:
        sel = _wifi_sel_type_enum("manual")

    try:
        with command_lock:
            b.drone(WifiSetApChannel(type=sel, band=band,
                                     channel=0 if auto_mode else channel)).wait(_timeout=5)
        mode = "auto" if auto_mode else "manual"
        print(f"[ANAFI] Wi-Fi: {mode.upper()} → band={band_str}, "
              f"channel={'any' if auto_mode else channel} "
              f"(sel={_jsonable(sel)})")
        resp = {"ok": True, "mode": mode, "band": band_str,
                "type_resolved": _jsonable(sel),
                "band_resolved": _jsonable(band),
                "message": f"channel change submitted; connection will drop briefly"}
        if not auto_mode:
            resp["channel"] = channel
        return jsonify(resp)
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


@app.post("/api/key_batch")
def api_key_batch():
    """Refresh a batch of held keys in a single request.

    Replaces the pattern where the C2 used to fire N separate POSTs —
    one per held key — every 100 ms. At 3 held keys × 10 Hz that was
    30 HTTP requests/sec just for key-holding, and it saturated the
    browser's 6-connection pool whenever a WS fallback was in play.
    Body: {"keys": ["w", "a"]}
    """
    data = request.get_json(silent=True) or {}
    keys = data.get("keys", [])
    if not isinstance(keys, list):
        return jsonify(ok=False, error="keys must be a list"), 400
    for k in keys:
        if k:
            add_key(str(k))
    return jsonify(ok=True, n=len(keys))


def _auto_record_after_takeoff(grace_s: float = 2.5):
    """Background one-shot: wait `grace_s` for the C2-side FlightLogger to
    start its own (matched-name) recording; if nothing is recording by then,
    start a fallback recording with a default timestamp filename. Ensures
    every flight leaves a video on disk even without a C2 in the loop.
    """
    try:
        time.sleep(grace_s)
        if not flying:
            return  # already landed in the grace window
        with _rec_lock:
            already = _rec_enabled
        if already:
            return  # C2 FlightLogger (or someone else) won the race — good
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        fname = f"flight_{ts}_auto.mp4"
        ok, payload = _start_recording_internal(fname=fname, raw=False)
        if ok:
            print(f"[REC] Auto-record fallback triggered after takeoff: {fname}")
        else:
            print(f"[REC] Auto-record fallback skipped: {payload}")
    except Exception as e:
        print(f"[REC] Auto-record fallback error: {e}")


@app.post("/api/takeoff")
def api_takeoff():
    payload, status = drone_core.do_takeoff()
    return jsonify(payload), status


@app.post("/api/land")
def api_land():
    payload, status = drone_core.do_land()
    return jsonify(payload), status


# ═══════════════════════════════════════════════════════════════════════════
# Calibration flight — autonomous pattern that scans the arena for Claude
# Automatic Calibration analysis.
#
# Design: stay within ±1.5 m of the arena centre at two altitudes, with
# two full 360° sweeps and a translation cross. Existing boundary guard
# and ceiling enforcement stay active, so even if the operator's preset
# is miscalibrated, the arena keeps the drone safe.
# ═══════════════════════════════════════════════════════════════════════════

CALIBRATION_STEPS: list = [
    # Phase 1 — takeoff + initial fix acquisition
    {"op": "takeoff",                          "name": "Takeoff"},
    {"op": "hover", "s": 4.0,                  "name": "Initial hover — acquire fix"},
    # Phase 2 — full 360° sweep at takeoff altitude, in 90° steps
    {"op": "rotate", "dir": "cw", "deg": 90,   "name": "Rotate CW 90° (1/4)"},
    {"op": "hover", "s": 1.5,                  "name": "Rotation hover"},
    {"op": "rotate", "dir": "cw", "deg": 90,   "name": "Rotate CW 90° (2/4)"},
    {"op": "hover", "s": 1.5,                  "name": "Rotation hover"},
    {"op": "rotate", "dir": "cw", "deg": 90,   "name": "Rotate CW 90° (3/4)"},
    {"op": "hover", "s": 1.5,                  "name": "Rotation hover"},
    {"op": "rotate", "dir": "cw", "deg": 90,   "name": "Rotate CW 90° (4/4) — back to heading 0"},
    {"op": "hover", "s": 2.0,                  "name": "Settle after sweep"},
    # Phase 3 — translation cross (±1.5 m on both axes)
    {"op": "move", "dir": "forward", "cm": 150, "name": "Forward 1.5 m"},
    {"op": "hover", "s": 2.0,                   "name": "Forward hover"},
    {"op": "move", "dir": "back",    "cm": 300, "name": "Back 3.0 m"},
    {"op": "hover", "s": 2.0,                   "name": "Back hover"},
    {"op": "move", "dir": "forward", "cm": 150, "name": "Return to centre (Y)"},
    {"op": "hover", "s": 1.5,                   "name": "Centre hover"},
    {"op": "move", "dir": "right",   "cm": 150, "name": "Right 1.5 m"},
    {"op": "hover", "s": 2.0,                   "name": "Right hover"},
    {"op": "move", "dir": "left",    "cm": 300, "name": "Left 3.0 m"},
    {"op": "hover", "s": 2.0,                   "name": "Left hover"},
    {"op": "move", "dir": "right",   "cm": 150, "name": "Return to centre (X)"},
    {"op": "hover", "s": 2.0,                   "name": "Centre hover"},
    # Phase 4 — altitude change
    {"op": "move", "dir": "up",      "cm": 100, "name": "Up 1.0 m"},
    {"op": "hover", "s": 2.0,                   "name": "High-altitude hover"},
    # Phase 5 — second sweep at high altitude, opposite direction
    {"op": "rotate", "dir": "ccw", "deg": 90,  "name": "Rotate CCW 90° (1/4, high)"},
    {"op": "hover", "s": 1.5,                  "name": "Rotation hover"},
    {"op": "rotate", "dir": "ccw", "deg": 90,  "name": "Rotate CCW 90° (2/4, high)"},
    {"op": "hover", "s": 1.5,                  "name": "Rotation hover"},
    {"op": "rotate", "dir": "ccw", "deg": 90,  "name": "Rotate CCW 90° (3/4, high)"},
    {"op": "hover", "s": 1.5,                  "name": "Rotation hover"},
    {"op": "rotate", "dir": "ccw", "deg": 90,  "name": "Rotate CCW 90° (4/4, high)"},
    {"op": "hover", "s": 2.0,                  "name": "Settle after second sweep"},
    # Phase 6 — descent + land
    {"op": "move", "dir": "down",    "cm": 100, "name": "Down 1.0 m"},
    {"op": "hover", "s": 2.0,                   "name": "Pre-land hover"},
    {"op": "land",                              "name": "Land"},
]


_calib_state: dict = {
    "active":        False,
    "current_step":  0,
    "total_steps":   len(CALIBRATION_STEPS),
    "step_name":     "idle",
    "started_at":    0.0,
    "ended_at":      0.0,
    "elapsed_s":     0.0,
    "aborted":       False,
    "result":        None,   # "ok" | "aborted" | "error" | None (in progress)
    "last_error":    None,
    # Stored after a flight — lets the UI find the matching log + video
    "last_flight_window": None,  # {"start_ts", "end_ts"} — UTC epoch seconds
}
_calib_state_lock = threading.Lock()
_calib_abort = threading.Event()


def _execute_calibration_step(step: dict) -> tuple[bool, str]:
    """Dispatch a single calibration step. Returns (ok, msg)."""
    global flying, takeoff_cooldown_until
    b = backend
    if b is None:
        return False, "backend not ready"
    op = step.get("op")

    if op == "hover":
        # Abort-aware sleep — polls every 100 ms so operator-triggered
        # PAUSE/LAND is honoured promptly.
        dur = float(step.get("s", 1.0))
        start = time.time()
        while time.time() - start < dur:
            if _calib_abort.is_set() or not flying:
                return True, "aborted/landed"
            time.sleep(0.1)
        return True, "ok"

    if op == "takeoff":
        if flying:
            return True, "already flying"
        hold_s = SAFE_TAKEOFF_S if safe_takeoff_enabled else 3.0
        start_discrete_window(hold_s)
        b.before_discrete_command()
        ok, msg = b.takeoff()
        b.after_discrete_command()
        if ok:
            flying = True
            takeoff_cooldown_until = time.time() + hold_s
            # Auto-record kicks in 2.5 s post-takeoff via api_takeoff's
            # fallback thread. But we're bypassing api_takeoff here — so
            # spawn the auto-record fallback manually so the calibration
            # flight gets captured on video.
            try:
                threading.Thread(
                    target=_auto_record_after_takeoff,
                    kwargs={"grace_s": 2.5},
                    daemon=True, name="calib-auto-record",
                ).start()
            except Exception as te:
                print(f"[CALIB] Could not spawn auto-record thread: {te}")
        return (ok, msg)

    if op == "land":
        if not flying:
            return True, "already landed"
        start_discrete_window(3.0)
        b.before_discrete_command()
        ok, msg = b.land()
        b.after_discrete_command()
        if ok:
            flying = False
            try:
                _stop_recording_internal(reason="calib-land")
            except Exception:
                pass
        return (ok, msg)

    if op == "move":
        direction = step["dir"]
        cm = int(step["cm"])
        start_discrete_window(max(2.0, cm / 50))
        return b.move(direction, cm)

    if op == "rotate":
        direction = step["dir"]
        deg = int(step["deg"])
        start_discrete_window(max(1.5, deg / 60))
        return b.rotate(direction, deg)

    return False, f"unknown op: {op}"


def _run_calibration_flight():
    """Background thread: runs through CALIBRATION_STEPS sequentially."""
    global flying
    print(f"[CALIB] Calibration flight starting ({len(CALIBRATION_STEPS)} steps)")
    start_ts = time.time()
    with _calib_state_lock:
        _calib_state.update({
            "active":       True,
            "current_step": 0,
            "total_steps":  len(CALIBRATION_STEPS),
            "step_name":    CALIBRATION_STEPS[0]["name"],
            "started_at":   start_ts,
            "ended_at":     0.0,
            "elapsed_s":    0.0,
            "aborted":      False,
            "result":       None,
            "last_error":   None,
        })
    _calib_abort.clear()
    last_err = None
    try:
        for i, step in enumerate(CALIBRATION_STEPS):
            if _calib_abort.is_set():
                print("[CALIB] Abort requested — stopping sequence")
                break
            with _calib_state_lock:
                _calib_state["current_step"] = i + 1
                _calib_state["step_name"] = step["name"]
                _calib_state["elapsed_s"] = time.time() - start_ts
            print(f"[CALIB] Step {i+1}/{len(CALIBRATION_STEPS)}: {step['name']}")
            ok, msg = _execute_calibration_step(step)
            if not ok:
                last_err = f"Step {i+1} ({step['name']}): {msg}"
                print(f"[CALIB] ABORTED — {last_err}")
                break
    except Exception as e:
        import traceback
        traceback.print_exc()
        last_err = str(e)
    finally:
        end_ts = time.time()
        # Emergency-land if still in the air
        if flying:
            print("[CALIB] Still airborne at end of sequence — emergency land")
            try:
                b = backend
                if b is not None:
                    start_discrete_window(3.0)
                    b.before_discrete_command()
                    b.land()
                    b.after_discrete_command()
                    flying = False
                    try:
                        _stop_recording_internal(reason="calib-final-land")
                    except Exception:
                        pass
            except Exception as e:
                print(f"[CALIB] Final land failed: {e}")
        with _calib_state_lock:
            if _calib_abort.is_set():
                _calib_state["aborted"] = True
                _calib_state["result"] = "aborted"
            elif last_err is not None:
                _calib_state["result"] = "error"
                _calib_state["last_error"] = last_err
            else:
                _calib_state["result"] = "ok"
            _calib_state["ended_at"] = end_ts
            _calib_state["elapsed_s"] = end_ts - start_ts
            _calib_state["active"] = False
            _calib_state["last_flight_window"] = {
                "start_ts": start_ts, "end_ts": end_ts,
            }
        print(f"[CALIB] Done — result={_calib_state['result']} "
              f"elapsed={end_ts - start_ts:.1f}s")


@app.post("/api/calibration/start")
def api_calib_start():
    """Kick off an autonomous calibration flight. Preconditions:
    controller connected, drone on the ground, no calibration already
    in progress. The operator should place the drone in the arena
    centre before calling this."""
    with _calib_state_lock:
        if _calib_state["active"]:
            return jsonify(ok=False,
                           error="calibration already active",
                           state=dict(_calib_state)), 409
    with conn_lock:
        connected = conn_state["connected"]
    if not connected:
        return jsonify(ok=False, error="drone not connected"), 503
    if flying:
        return jsonify(ok=False,
                       error="drone already flying — land first, then start calibration"), 409
    threading.Thread(
        target=_run_calibration_flight,
        daemon=True, name="calibration",
    ).start()
    return jsonify(
        ok=True,
        message="calibration flight started",
        total_steps=len(CALIBRATION_STEPS),
        steps=[{"op": s["op"], "name": s["name"]} for s in CALIBRATION_STEPS],
    )


@app.get("/api/calibration/status")
def api_calib_status():
    with _calib_state_lock:
        snap = dict(_calib_state)
    return jsonify(ok=True, **snap)


@app.post("/api/calibration/abort")
def api_calib_abort():
    """Abort a running calibration flight. Sets the flag that each step
    checks; the drone then lands on the next abort-checkpoint."""
    _calib_abort.set()
    with _calib_state_lock:
        active = _calib_state["active"]
    return jsonify(ok=True, message="abort requested", active=active)


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
    payload, status = drone_core.do_emergency()
    return jsonify(payload), status


@app.post("/api/move")
def api_move():
    data = request.get_json(silent=True) or {}
    # HTTP-layer safety clamp [20, 500] cm — preserved from the old
    # route. In-proc callers (mission's FB_IMU/LR_IMU/UD_IMU) go
    # directly to drone_core.do_move when they need sub-20cm
    # precision.
    try:
        dist_cm = max(20, min(500, int(data.get("cm", 20))))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="cm must be an integer"), 400
    payload, status = drone_core.do_move(
        direction=data.get("dir", ""),
        cm=dist_cm,
    )
    return jsonify(payload), status


@app.post("/api/rotate")
def api_rotate():
    data = request.get_json(silent=True) or {}
    payload, status = drone_core.do_rotate(
        direction=data.get("dir", ""),
        deg=data.get("deg", 45),
        speed=data.get("speed"),
    )
    return jsonify(payload), status


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
    data = request.get_json(silent=True) or {}
    payload, status = drone_core.do_rc(
        lr=data.get("lr", 0),
        fb=data.get("fb", 0),
        ud=data.get("ud", 0),
        yaw=data.get("yaw", 0),
        duration_ms=data.get("duration_ms", 250),
    )
    return jsonify(payload), status


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


@app.post("/api/camera/zoom")
def api_camera_zoom():
    """Set the Anafi's digital camera zoom.
    Body: {"zoom": 1.0..3.0}  (1.0 = no zoom, 3.0 = max 3× digital).
    Implemented via Olympe's camera.set_zoom_target with control_mode=level.
    """
    if not HAS_CAMERA_ZOOM:
        return jsonify(ok=False, error="camera zoom not supported "
                                        "by this Olympe build"), 501
    b = backend
    if b is None or not isinstance(b, OlympeBackend) or b.drone is None:
        return jsonify(ok=False, error="drone not connected"), 503
    data = request.get_json(silent=True) or {}
    try:
        z = float(data.get("zoom", 1.0))
    except Exception:
        return jsonify(ok=False, error="zoom must be numeric"), 400
    # Clamp — Anafi digital zoom range is 1.0..3.0.
    z = max(1.0, min(3.0, z))
    try:
        with command_lock:
            # control_mode="level" = absolute zoom level; "velocity" = rate.
            b.drone(_CameraSetZoom(cam_id=0, control_mode="level",
                                    target=z)).wait(_timeout=2)
        return jsonify(ok=True, zoom=z)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.get("/api/camera/config")
def api_camera_config_get():
    """Read the Anafi's current camera-settings snapshot.
    Returns ``{ok, config}``; ``config`` contains keys per axis the
    firmware exposes (exposure, white_balance, hdr, ev_compensation,
    antiflicker, video_stabilization, stream_mode, recording, zoom).
    Returns connected=False without keys when no drone is bound.
    """
    b = backend
    if b is None or not isinstance(b, OlympeBackend):
        return jsonify(ok=False, error="anafi backend required"), 503
    try:
        return jsonify(ok=True, config=b.camera_config_get())
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/camera/config")
def api_camera_config_set():
    """Apply a partial camera-settings dict.
    Body shape mirrors the GET response (one or more of: exposure,
    white_balance, hdr, ev_compensation, antiflicker,
    video_stabilization, stream_mode, recording, zoom). Each axis is
    applied independently; the response includes ``results`` (per-axis
    bool) and the post-apply ``config`` so the operator sees what the
    firmware actually accepted.
    """
    b = backend
    if b is None or not isinstance(b, OlympeBackend) or b.drone is None:
        return jsonify(ok=False, error="drone not connected"), 503
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify(ok=False, error="body must be a JSON object"), 400
    try:
        results = b.camera_config_set(data)
        return jsonify(ok=True, results=results,
                       config=b.camera_config_get())
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


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
    # The HTTP-layer mjpeg wrapper just frames each jpeg with the
    # multipart boundary. The actual frame-acquisition loop lives in
    # drone_core.iter_video_jpegs so in-proc callers can use the same
    # source.
    def gen():
        for jpg in drone_core.iter_video_jpegs():
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/video/start")
def api_video_start():
    data = request.get_json(silent=True) or {}
    payload, status = drone_core.do_video_start(
        mode=data.get("mode", "mjpeg"),
        target_host=data.get("target_host"),
        target_port=data.get("target_port"),
        # HTTP route fills the absolute stream URL so the response is
        # directly usable by browser/curl clients; in-proc callers
        # leave this as None and consume frames via iter_video_jpegs().
        stream_url=f"http://{request.host}/api/video",
    )
    return jsonify(payload), status


@app.post("/api/video/stop")
def api_video_stop():
    payload, status = drone_core.do_video_stop()
    return jsonify(payload), status


# ── H.264 endpoint (sim only) ───────────────────────────────────────
# Same source as /api/video (the sim_video_loop producer), but encoded
# H.264 over MPEG-TS via NVENC. Browsers play this via the <video>
# element using their hardware H.264 decoder — no MJPEG main-thread
# decode bottleneck, ~5x less bandwidth than MJPEG at the same quality.
#
# Usage:
#   curl -X POST /api/video/h264/start     # ensure producer is running
#   <video src="/api/video/h264">          # browser plays MPEG-TS live
#   ffplay http://host:8080/api/video/h264 # works in any player too
#
# One ffmpeg subprocess per concurrent viewer (cheap on NVENC). The
# subprocess is torn down automatically when the client disconnects.

H264_TARGET_FPS = 15
H264_BITRATE = "1500k"           # ample for 1280x720@15 with deltas
# Keep keyframe interval SHORT for live streams — if a viewer joins
# mid-stream or a packet drops, the player can only resume on the next
# I-frame. With the source occasionally dipping to 5-7 fps, a frame-
# count-based GOP would span >2 s; force keyframes every wall-clock
# second instead.
H264_GOP = 5
H264_FORCE_KEY = "expr:gte(t,n_forced*1)"   # 1 keyframe every 1.0 s


def _h264_encoder_cmd(width: int, height: int) -> list[str]:
    """Build the ffmpeg command line. Prefer GPU (NVENC) on the T4; fall
    back to libx264 if NVENC isn't available."""
    common_in = [
        "ffmpeg", "-loglevel", "error", "-hide_banner",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(H264_TARGET_FPS),
        "-i", "pipe:0",
    ]
    # NVENC low-latency preset (p1 = fastest, tune=ll = low-latency)
    # `-force_key_frames` + `-g` together guarantee the player sees an
    # I-frame within 1 s even if the source rate drops.
    nvenc_out = [
        "-c:v", "h264_nvenc",
        "-preset", "p1", "-tune", "ll", "-zerolatency", "1",
        "-rc", "cbr", "-b:v", H264_BITRATE,
        "-g", str(H264_GOP), "-bf", "0",
        "-force_key_frames", H264_FORCE_KEY,
        "-pix_fmt", "yuv420p",
        "-f", "mpegts", "-muxdelay", "0", "-muxpreload", "0",
        "pipe:1",
    ]
    # libx264 fallback if NVENC is not in this ffmpeg build
    x264_out = [
        "-c:v", "libx264",
        "-preset", "ultrafast", "-tune", "zerolatency",
        "-b:v", H264_BITRATE,
        "-g", str(H264_GOP), "-bf", "0",
        "-force_key_frames", H264_FORCE_KEY,
        "-pix_fmt", "yuv420p",
        "-f", "mpegts", "-muxdelay", "0", "-muxpreload", "0",
        "pipe:1",
    ]
    # Quick probe — done once per process, cached in module attr
    enc = getattr(_h264_encoder_cmd, "_pref_enc", None)
    if enc is None:
        try:
            import subprocess as _sp
            out = _sp.run(["ffmpeg", "-hide_banner", "-encoders"],
                          capture_output=True, text=True, timeout=3).stdout
            enc = "h264_nvenc" if "h264_nvenc" in out else "libx264"
        except Exception:
            enc = "libx264"
        _h264_encoder_cmd._pref_enc = enc
    return common_in + (nvenc_out if enc == "h264_nvenc" else x264_out)


@app.post("/api/video/h264/start")
def api_video_h264_start():
    """Make sure the sim video producer is up so BGR frames are flowing.
    Reuses the existing /api/video/start mjpeg producer — same source,
    different consumer."""
    b = backend
    if b is None:
        return jsonify(ok=False, error="controller not ready"), 503
    if not _is_sim_anafi_ip(getattr(b, "ip", "")):
        return jsonify(ok=False,
                       error="H.264 fan-out is sim-only"), 400
    # If the MJPEG producer isn't already running, kick it off. We don't
    # change _video_mode — that stays tied to MJPEG-vs-forward semantics.
    if not _video_streaming:
        ok, msg = b.video_start_mjpeg()
        if not ok:
            return jsonify(ok=False, error=msg), 500
    return jsonify(ok=True, stream_url=f"http://{request.host}/api/video/h264",
                   mime="video/mp2t",
                   note="play in browser <video> tag, or `ffplay <url>`")


@app.post("/api/video/h264/stop")
def api_video_h264_stop():
    """No-op for the H.264 path — clients tear down their own subprocess
    when they disconnect. Provided for API symmetry."""
    return jsonify(ok=True)


@app.get("/api/video/h264")
def api_video_h264_stream():
    """Live H.264-over-MPEG-TS stream. Spawns one ffmpeg subprocess per
    client; producer is the sim_video_loop's BGR fan-out buffer."""
    # Mark a subscriber so the producer starts publishing BGR frames.
    # The producer skips its bgr.copy() + notify_all() when this is 0,
    # which is the steady-state cost path (no h264 clients).
    global _sim_h264_active_clients
    with _sim_bgr_condition:
        _sim_h264_active_clients += 1
    # Wait up to 5 s for the producer to publish a frame so we know
    # what resolution to encode at.
    deadline = time.monotonic() + 5.0
    with _sim_bgr_condition:
        while _sim_bgr_latest is None and time.monotonic() < deadline:
            _sim_bgr_condition.wait(timeout=0.5)
        if _sim_bgr_latest is None:
            _sim_h264_active_clients -= 1
            return jsonify(error="no frames yet — POST /api/video/h264/start first"), 503
        width, height = _sim_bgr_resolution

    import subprocess as _sp
    cmd = _h264_encoder_cmd(width, height)
    proc = _sp.Popen(cmd, stdin=_sp.PIPE, stdout=_sp.PIPE,
                     stderr=_sp.DEVNULL, bufsize=0)

    # Background feeder: read latest BGR from the shared buffer at the
    # source rate, push to ffmpeg stdin. Stops when client disconnects
    # (BrokenPipeError) or the producer goes idle.
    stop_event = threading.Event()

    def feeder():
        last_id = 0
        frame_interval = 1.0 / H264_TARGET_FPS
        try:
            while not stop_event.is_set():
                with _sim_bgr_condition:
                    waited = _sim_bgr_condition.wait_for(
                        lambda: _sim_bgr_frame_id != last_id
                                or stop_event.is_set(),
                        timeout=2.0,
                    )
                    if stop_event.is_set():
                        return
                    if not waited:
                        # Producer stalled; loop and try again
                        continue
                    frame = _sim_bgr_latest
                    last_id = _sim_bgr_frame_id
                if frame is None:
                    continue
                try:
                    proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, ValueError, OSError):
                    return
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    feeder_t = threading.Thread(target=feeder, daemon=True,
                                name="h264-feeder")
    feeder_t.start()

    def stream_response():
        global _sim_h264_active_clients
        try:
            while True:
                chunk = proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            stop_event.set()
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            # Drop the subscription so the producer reverts to its
            # cheap path. wake any waiters so feeders observing
            # stop_event can exit promptly.
            with _sim_bgr_condition:
                _sim_h264_active_clients = max(0, _sim_h264_active_clients - 1)
                _sim_bgr_condition.notify_all()

    return Response(stream_response(), mimetype="video/mp2t",
                    headers={"Cache-Control": "no-cache, no-store",
                             "X-Encoder": getattr(_h264_encoder_cmd,
                                                  "_pref_enc", "?")})


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


@app.get("/api/config/ceiling")
def api_ceiling_get():
    """Return the active soft-ceiling and live guard state, plus the
    drone's firmware-confirmed MaxAltitude so the operator can verify
    the SDK-level limit matches our software guard.
    """
    firmware_current = None
    firmware_range   = None
    try:
        b = backend
        if b is not None and _HAS_MAXALT_STATE and MaxAltitudeChanged is not None:
            d = getattr(b, "drone", None)
            if d is not None:
                st = d.get_state(MaxAltitudeChanged)
                if st:
                    if "current" in st:
                        firmware_current = round(float(st["current"]), 2)
                    if "min" in st and "max" in st:
                        firmware_range = [round(float(st["min"]), 2),
                                          round(float(st["max"]), 2)]
    except Exception:
        pass
    return jsonify(
        ok=True,
        ceiling_m=round(float(MAX_ALTITUDE_M), 2),
        engaged=bool(_ceiling_engaged),
        reason=_ceiling_last_reason,
        firmware_max_altitude=firmware_current,
        firmware_range=firmware_range,
        persisted_to=str(FLIGHT_CONFIG_PATH),
    )


@app.post("/api/config/ceiling")
def api_ceiling_set():
    data = request.get_json(silent=True) or {}
    payload, status = drone_core.do_set_ceiling(
        data.get("ceiling_m") if data.get("ceiling_m") is not None
        else data.get("max_altitude_m")
    )
    return jsonify(payload), status


@app.get("/api/config/arena_safety")
def api_arena_safety_get():
    """Return Pi-side arena boundary guard state + config.

    The guard runs in rc_loop() for EVERY RC tick (manual + autonomous),
    so it enforces the margin whether the operator is piloting by hand
    or a mission is active. Independent of C2 connection.
    """
    return jsonify(
        ok=True,
        enabled=bool(_arena_guard_enabled),
        margin_m=round(float(_arena_margin_m), 2),
        bounds=dict(_arena_bounds),
        engaged=bool(_arena_engaged),
        reason=_arena_last_reason,
        persisted_to=str(FLIGHT_CONFIG_PATH),
    )


@app.post("/api/config/arena_safety")
def api_arena_safety_set():
    """Update arena safety: any subset of
      {"enabled": bool, "margin_m": float, "bounds": {x_min, x_max, y_min, y_max}}"""
    global _arena_guard_enabled, _arena_margin_m, _arena_bounds
    data = request.get_json(silent=True) or {}
    changed: dict = {}
    if "enabled" in data:
        _arena_guard_enabled = bool(data["enabled"])
        changed["enabled"] = _arena_guard_enabled
    if "margin_m" in data:
        try:
            v = max(0.0, min(5.0, float(data["margin_m"])))
            _arena_margin_m = v
            changed["margin_m"] = v
        except (TypeError, ValueError):
            pass
    if "bounds" in data and isinstance(data["bounds"], dict):
        for k, v in data["bounds"].items():
            if k in _arena_bounds:
                try:
                    _arena_bounds[k] = float(v)
                except (TypeError, ValueError):
                    pass
        changed["bounds"] = dict(_arena_bounds)
    # Persist so a Pi restart keeps operator intent
    try:
        _save_flight_config({
            "arena_safety_margin_m": _arena_margin_m,
            "arena_bounds":           dict(_arena_bounds),
        })
    except Exception:
        pass
    print(f"[ARENA] guard {_arena_guard_enabled and 'ON' or 'OFF'} margin={_arena_margin_m}m "
          f"bounds={_arena_bounds} (changed={changed})")
    return jsonify(ok=True, changed=changed,
                   enabled=_arena_guard_enabled, margin_m=_arena_margin_m,
                   bounds=dict(_arena_bounds))


@app.get("/api/diagnostics")
def api_diagnostics():
    """Pi-side health snapshot — thread count, internal buffer sizes.
    Use from the C2 when chasing "degrades-over-time" symptoms. The
    counters that matter are the ones that should NOT grow
    monotonically across flights: thread count, pressed_web size,
    telemetry buffer length. Video frame-count is expected to grow.
    """
    import threading as _th
    threads = _th.enumerate()
    by_name: dict[str, int] = {}
    for t in threads:
        # Collapse numbered worker threads so the counts stay readable
        name = t.name
        for prefix in ("ThreadPoolExecutor-", "Thread-"):
            if name.startswith(prefix):
                name = prefix + "*"
                break
        by_name[name] = by_name.get(name, 0) + 1

    with telemetry_lock:
        tel_snap = len(telemetry)
    with pressed_lock:
        pressed = sorted(pressed_web)
    with _telemetry_sync_lock:
        tel_buf_len = len(_telemetry_sync_buf)
    with _pos_sse_lock:
        sse_clients = len(_pos_sse_queues)
        ws_pos_clients = len(_pos_ws_queues)

    return jsonify(
        ok=True,
        thread_count=len(threads),
        threads_by_name=by_name,
        telemetry_keys=tel_snap,
        pressed_web=pressed,
        telemetry_sync_buffer_len=tel_buf_len,
        pos_sse_clients=sse_clients,
        pos_ws_clients=ws_pos_clients,
        video_mode=_video_mode,
        video_streaming=_video_streaming,
        video_frame_count=_video_frame_count,
        flying=flying,
        connected=bool(conn_state.get("connected", False)),
        # Safety state — proves the ceiling is Pi-enforced even with
        # no remote/C2 connection. `ceiling_source` tells you where
        # the current value came from (config file or env default).
        ceiling={
            "ceiling_m":        round(float(MAX_ALTITUDE_M), 2),
            "engaged":          bool(_ceiling_engaged),
            "reason":           _ceiling_last_reason or "",
            "config_file":      str(FLIGHT_CONFIG_PATH),
            "config_persists":  FLIGHT_CONFIG_PATH.exists(),
            "enforced_by":      "pi.rc_loop @ 20Hz (independent of C2)",
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# WebSocket endpoints — low-latency C2 ↔ FC channel
# ═══════════════════════════════════════════════════════════════════════
#
# Three long-lived WS connections replace the highest-rate HTTP polls:
#
#   /ws/telemetry  — server-push at TELEMETRY_HZ (no framing per call)
#   /ws/position   — server-push on every pose update (same cadence as
#                    the existing SSE stream, but lower overhead)
#   /ws/rc         — bidirectional. C2 sends {type:"rc",lr,fb,ud,yaw}
#                    or {type:"key",event,key}; server applies them and
#                    acks.
#
# The endpoints only exist if flask-sock was importable (HAS_WS). When
# missing, the HTTP endpoints keep working unchanged.

if HAS_WS:
    # Server-side heartbeat cadence. Short enough that the C2 client's
    # recv timeout (8 s on telemetry/position pull loops) sees a ping
    # before it gives up and reconnects — otherwise every 5–8 seconds
    # of idle positions / telemetry caused a churn cycle.
    _WS_PING_INTERVAL_S = 3.0

    @sock.route("/ws/telemetry")
    def ws_telemetry(ws):
        """Pushes one telemetry JSON per (1/TELEMETRY_HZ) seconds. Stops
        the moment the client disconnects. Wrapped in try/except because
        the receiver can close mid-send and raises ConnectionClosed —
        that's a clean close, not an error condition."""
        _ws_set_nodelay(ws)
        try:
            ws.send(json.dumps({
                "type": "hello", "service": "telemetry",
                "drone_type": drone_type, "hz": TELEMETRY_HZ,
                "server_ts": time.time(),
            }))
            period = 1.0 / max(0.5, float(TELEMETRY_HZ))
            last_ping = time.time()
            while running:
                t0 = time.time()
                payload = _build_telemetry_payload()
                payload["type"] = "telemetry"
                payload["server_ts"] = t0
                try:
                    ws.send(json.dumps(payload, default=str))
                except Exception as e:
                    if _ws_is_clean_close(e):
                        return
                    raise
                if t0 - last_ping > _WS_PING_INTERVAL_S:
                    try:
                        ws.send(json.dumps({"type": "ping", "server_ts": t0}))
                        last_ping = t0
                    except Exception as e:
                        if _ws_is_clean_close(e):
                            return
                        raise
                dt = time.time() - t0
                time.sleep(max(0.01, period - dt))
        except Exception as e:
            if not _ws_is_clean_close(e):
                print(f"[WS] /ws/telemetry error: {e}")

    @sock.route("/ws/position")
    def ws_position(ws):
        """Pushes a payload on every position update. Piggybacks the
        existing broadcast queue (_pos_ws_queues) so there's zero extra
        work in the positioner loop.

        Idle behaviour: the positioner only emits when detections arrive.
        Between frames we send a heartbeat every _WS_PING_INTERVAL_S so
        the client's recv never times out; otherwise a parked drone with
        no detections produces a reconnect loop (see _ws_is_clean_close)."""
        _ws_set_nodelay(ws)
        q: queue.Queue = queue.Queue(maxsize=30)
        with _pos_sse_lock:
            _pos_ws_queues.append(q)
        try:
            ws.send(json.dumps({
                "type": "hello", "service": "position",
                "server_ts": time.time(),
            }))
            last_ping = time.time()
            while running:
                try:
                    # Short queue timeout so we can issue heartbeats even
                    # when the positioner is silent.
                    msg = q.get(timeout=2.0)
                    ws.send(msg)
                except queue.Empty:
                    now = time.time()
                    if now - last_ping > _WS_PING_INTERVAL_S:
                        try:
                            ws.send(json.dumps({"type": "ping", "server_ts": now}))
                            last_ping = now
                        except Exception as e:
                            if _ws_is_clean_close(e):
                                return
                            raise
                except Exception as e:
                    # Clean peer close is expected — don't log as error.
                    if _ws_is_clean_close(e):
                        return
                    raise
        except Exception as e:
            if not _ws_is_clean_close(e):
                print(f"[WS] /ws/position error: {e}")
        finally:
            with _pos_sse_lock:
                try:
                    _pos_ws_queues.remove(q)
                except ValueError:
                    pass

    def _ws_set_nodelay(ws):
        """Set TCP_NODELAY on the underlying socket so small RC frames
        aren't batched by Nagle's algorithm. flask-sock exposes the raw
        socket via `ws.sock` (simple-websocket) — grab it defensively
        in case the API varies."""
        try:
            import socket as _sock
            raw = getattr(ws, "sock", None) or getattr(ws, "_sock", None)
            if raw is not None and hasattr(raw, "setsockopt"):
                raw.setsockopt(_sock.IPPROTO_TCP, _sock.TCP_NODELAY, 1)
                raw.setsockopt(_sock.SOL_SOCKET,  _sock.SO_KEEPALIVE, 1)
        except Exception:
            pass

    @sock.route("/ws/rc")
    def ws_rc(ws):
        """Bidirectional RC channel.

            {"type":"rc",  "lr":0, "fb":0, "ud":0, "yaw":0, "duration_ms":250}
            {"type":"key", "event":"down"|"up", "key":"w"}
            {"type":"ping","client_ts":<unix>}  → replies with pong

        Fire-and-forget by default — no ACKs. The client prefers short
        latency over confirmation; a missed frame is corrected on the
        next one. Applies to the same rc_override / pressed_web globals
        that /api/rc and /api/key_down write, so the tick loop picks it
        up without caring about the transport."""
        global rc_override, rc_override_until
        _ws_set_nodelay(ws)
        try:
            ws.send(json.dumps({
                "type": "hello", "service": "rc", "server_ts": time.time(),
            }))
            while running:
                try:
                    msg = ws.receive(timeout=20.0)
                except Exception:
                    return
                if msg is None:
                    return
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                mtype = data.get("type")
                if mtype == "rc":
                    def clamp(v):
                        try:    return max(-100, min(100, int(v)))
                        except: return 0
                    lr  = clamp(data.get("lr", 0))
                    fb  = clamp(data.get("fb", 0))
                    ud  = clamp(data.get("ud", 0))
                    yaw = clamp(data.get("yaw", 0))
                    dur = max(50, min(2000, int(data.get("duration_ms", 250))))
                    with rc_lock:
                        rc_override = (lr, fb, ud, yaw)
                        rc_override_until = time.time() + (dur / 1000.0)
                elif mtype == "key":
                    ev = (data.get("event") or "").lower()
                    k  = str(data.get("key") or "").lower()
                    if k:
                        if ev == "down": add_key(k)
                        elif ev == "up": remove_key(k)
                elif mtype == "ping":
                    # Respond so the client can measure RTT.
                    try:
                        ws.send(json.dumps({
                            "type": "pong",
                            "server_ts": time.time(),
                            "echo": data.get("client_ts"),
                        }))
                    except Exception:
                        return
        except Exception as e:
            if not _ws_is_clean_close(e):
                print(f"[WS] /ws/rc error: {e}")


# --- Telemetry endpoints ---
def _build_telemetry_payload() -> dict:
    """Shared telemetry builder used by both HTTP (/api/telemetry) and
    WebSocket (/ws/telemetry) paths so every consumer sees identical
    fields with a single implementation to maintain."""
    now = time.time()
    age = (now - last_state_seen) if last_state_seen else 9999.0
    with telemetry_lock:
        payload = dict(telemetry)
    payload["state_age_s"] = round(age, 3)
    payload["state_fresh"] = age <= 2.0
    payload["drone_type"] = drone_type
    with conn_lock:
        payload["connected"] = bool(conn_state.get("connected", False))
        payload["reconnect_failures"] = int(conn_state.get("consecutive_failures", 0))
        payload["reconnect_last_error"] = conn_state.get("last_error", "") or ""
    payload["ceiling_m"] = round(float(MAX_ALTITUDE_M), 2)
    payload["ceiling_engaged"] = bool(_ceiling_engaged)
    payload["ceiling_reason"] = _ceiling_last_reason or ""
    payload["arena_margin_m"] = round(float(_arena_margin_m), 2)
    payload["arena_engaged"] = bool(_arena_engaged)
    payload["arena_reason"] = _arena_last_reason or ""
    try:
        b = backend
        reader = getattr(b, "_read_magnetometer_state", None) if b else None
        if callable(reader):
            mag = reader()
            payload["magneto_status"] = mag
            payload["magneto_required"] = _magneto_needs_calibration(mag)
    except Exception:
        pass
    return payload


@app.get("/api/telemetry")
def api_telemetry():
    payload, status = drone_core.get_telemetry_payload()
    resp = jsonify(payload)
    resp.status_code = status
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
        "targets":     snapshot.get("targets", {}),
        "stale": snapshot.get("stale", True),
        "enabled": enabled,
        "latency_ms": lat_ms,
        "sync_quality": snapshot.get("sync_quality", "none"),
        "sync_age_ms": snapshot.get("sync_age_ms"),
        "frame_w": snapshot.get("frame_w"),
        "frame_h": snapshot.get("frame_h"),
    }


_pos_ws_queues: list = []        # list[queue.Queue] — one per /ws/position client


def _broadcast_pos_sse(snapshot: dict):
    payload = _pos_snapshot_to_js(snapshot)
    msg_sse = f"data: {json.dumps(payload)}\n\n"
    msg_ws  = json.dumps({"type": "position", **payload, "ts": time.time()})
    with _pos_sse_lock:
        dead = []
        for q in _pos_sse_queues:
            try:
                q.put_nowait(msg_sse)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _pos_sse_queues.remove(q)
        # Same snapshot to every WS client subscribed to /ws/position.
        dead = []
        for q in _pos_ws_queues:
            try:
                q.put_nowait(msg_ws)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _pos_ws_queues.remove(q)


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
    """Background thread — DEPRECATED unified-server positioning path.

    Reads frames from _pos_frame_q, runs ArUco positioning, fans the
    result out to /api/position SSE clients + the /api/position/video
    MJPEG and the recorder. Kept alive for legacy HTTP-only clients
    and the visual annotation overlay; new code should use
    marker_mission's detector + arena_holder instead (single
    operator-facing source of truth at /arena, the same one solvePnP
    already runs on for the camera page).

    Subscriber gate: in addition to the cfg ``enabled`` toggle, we
    only do the heavy solvePnP work when SOMEONE is consuming the
    output — at least one of:

      * recording is active (writes annotated frames to disk)
      * SSE subscriber(s) on /api/position/events
      * MJPEG viewer(s) on /api/position/video

    When ``enabled`` is True but no subscribers, we still drain
    _pos_frame_q (so it doesn't grow unbounded) but skip
    detect+solvePnP. This makes the combined marker_mission.app
    deployment effectively free when the operator is on the Camera
    tab (which uses marker_mission's own solvePnP).
    """
    global _pos_annotated_jpeg
    if not HAS_POSITIONING:
        print("[POS] positioning_loop: HAS_POSITIONING=False, exiting")
        return

    print("[POS] DEPRECATED unified-server positioning subsystem started "
          "(subscriber-gated; use marker_mission /arena for the operator "
          "source of truth)")

    processor = None
    calib_w = calib_h = None
    last_idle_drain_warn = 0.0

    while running:
        # Check if positioning and/or recording is active.
        # Use _effective_pos_cfg() so auto_positioning overrides user
        # params transparently — positioning_loop doesn't need to care
        # whether the operator is in auto or manual mode.
        cfg = _effective_pos_cfg()
        pos_enabled = cfg.get("enabled", False)
        with _rec_lock:
            rec_active = _rec_enabled
        if not pos_enabled and not rec_active:
            time.sleep(0.2)
            processor = None  # reset so it reinitialises on enable
            continue

        # Subscriber gate — once enabled, only do the heavy path when
        # someone consumes the output. Cheap: a single int read + a
        # list len. We still drain _pos_frame_q below so the producer
        # callback doesn't block on a full queue.
        with _pos_sse_lock:
            sse_clients = len(_pos_sse_queues)
        with _pos_video_clients_lock:
            video_clients = _pos_video_clients
        has_subscribers = bool(rec_active or sse_clients or video_clients)
        if pos_enabled and not has_subscribers:
            # Drain the frame queue so the producer (video callback)
            # never blocks, then continue. Cheap loop; no solvePnP.
            try:
                while True:
                    _pos_frame_q.get_nowait()
            except queue.Empty:
                pass
            now_drain = time.monotonic()
            if now_drain - last_idle_drain_warn > 30.0:
                print(f"[POS] idle (cfg.enabled=True but no subscribers) "
                      f"— skipping solvePnP, draining frame queue")
                last_idle_drain_warn = now_drain
            time.sleep(0.2)
            processor = None  # reset so it reinitialises on first subscriber
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
                # Runtime overrides. `cfg` is already the _effective_ config
                # (auto_positioning overrides user params transparently).
                cfg_marker = cfg.get("marker_size_m")
                if cfg_marker:
                    init_marker_size = float(cfg_marker)
                init_kalman = bool(cfg.get("enable_kalman_filter", True))
                init_dist_scale = float(cfg.get("distance_scale", 1.0))
                init_max_jump = float(cfg.get("max_pose_jump_m", 0.0))
                processor = _HeadlessAruCo(cam_mat, dist, detect_profile=profile,
                                           marker_size=init_marker_size, enable_kalman_filter=init_kalman)
                _apply_arena_cfg_to_processor(processor)
                processor.distance_scale = init_dist_scale
                processor.max_pose_jump_m = init_max_jump
                # Target-marker-size + ZUPT reverted per operator request — the
                # dual-size solvePnP path caused a massive position blow-up in
                # the field. The attributes below are guarded so this code
                # continues to work if a future ctrl_position.py adds them
                # back cleanly.
                if hasattr(processor, "target_marker_size"):
                    try:
                        processor.target_marker_size = float(cfg.get("target_marker_size_m", 0.19))
                        if hasattr(processor, "_build_marker_point_sets"):
                            processor._build_marker_point_sets()
                    except Exception as se:
                        print(f"[POS] target_marker_size apply skipped: {se}")
                if hasattr(processor, "zupt_speed_m_s"):
                    try:
                        processor.zupt_speed_m_s = float(cfg.get("zupt_speed_m_s", 0.0))
                        processor.zupt_hold_frames = int(cfg.get("zupt_hold_frames", 0))
                    except Exception as se:
                        print(f"[POS] ZUPT apply skipped: {se}")
                _pos_processor = processor
                auto_tag = " [auto]" if cfg.get("auto_positioning") else ""
                print(f"[POS] Processor initialised (profile={profile}, "
                      f"marker_size={init_marker_size:.3f}m, "
                      f"distance_scale={init_dist_scale:.4f}, "
                      f"max_pose_jump_m={init_max_jump:.2f}){auto_tag}")
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
        # Read once from the already-fetched effective cfg so auto_positioning
        # is respected: in auto mode, imu_weight comes from CLAUDE_AUTO_CONFIG.
        latency_comp_s = max(0.0, float(cfg.get("latency_comp_s", 0.05)))
        sync_max_gap_s = max(0.01, float(cfg.get("sync_max_gap_s", 0.20)))
        imu_weight_cfg = float(cfg.get("imu_weight", 0.3))
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

        # ── Seen-marker hold-time (hysteresis) ───────────────────────
        # ArUco detection isn't perfectly reliable per frame — blur,
        # glare, or extreme angle can cause a single-frame miss. The
        # UI halos would then flicker even though the camera is still
        # pointed at the marker. We keep a per-marker last-seen
        # timestamp and treat any marker seen within HOLD_S as still
        # visible, smoothing single-frame drops.
        _SEEN_HOLD_S = float(cfg.get("seen_hold_s", 0.6))
        fresh_ids = [str(m) for m in (result.get("seen_markers") or [])]
        now_mono = time.monotonic()
        if not hasattr(positioning_loop, "_seen_last"):
            positioning_loop._seen_last = {}   # marker_id → last monotonic ts
        seen_last: dict = positioning_loop._seen_last
        for mid in fresh_ids:
            seen_last[mid] = now_mono
        # Prune really stale entries so the dict doesn't grow forever.
        cutoff = now_mono - max(_SEEN_HOLD_S, 2.0)
        for mid in [k for k, ts in seen_last.items() if ts < cutoff]:
            del seen_last[mid]
        # Build the hysteresis-smoothed list: every marker seen within
        # HOLD_S counts. Preserve detection order: most recently-seen first.
        held_ids = sorted(
            (mid for mid, ts in seen_last.items() if (now_mono - ts) <= _SEEN_HOLD_S),
            key=lambda m: seen_last[m], reverse=True,
        )

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
            # Target boxes — marker-ID → arena-frame [x,y,z] dict. These
            # are the SDC26 target markers (ID ≥ 30) projected into world
            # space via each visible reference marker, weighted-fused and
            # EMA-smoothed. The C2 renders them in the "Target Boxes"
            # panel with team/colour derived from arena_config.
            raw_targets = result.get("targets") or {}
            # Carry forward targets seen in earlier frames that haven't
            # been observed in this one — with a TTL so old entries age
            # out. Without this the UI flickers per-frame as the drone
            # glances away from a target. State kept on the loop func.
            if not hasattr(positioning_loop, "_target_last_seen"):
                positioning_loop._target_last_seen = {}   # id → (pos, mono_ts)
            _tgt_ttl_s = 3.0
            _tgt_now = time.monotonic()
            for tid_str, tpos in raw_targets.items():
                try:
                    tid = int(tid_str)
                except Exception:
                    continue
                if isinstance(tpos, (list, tuple)) and len(tpos) == 3:
                    positioning_loop._target_last_seen[tid] = (list(tpos), _tgt_now)
            # Prune expired
            for tid in [k for k, (_, t) in positioning_loop._target_last_seen.items()
                        if _tgt_now - t > _tgt_ttl_s]:
                del positioning_loop._target_last_seen[tid]
            _pos_st["targets"] = {
                str(tid): {
                    "pos":    [round(p[0], 3), round(p[1], 3), round(p[2], 3)],
                    "age_s":  round(_tgt_now - ts_mono, 2),
                    "fresh":  (tid in {int(k) for k in raw_targets.keys()}),
                }
                for tid, (p, ts_mono) in positioning_loop._target_last_seen.items()
            }
            # Publish the SMOOTHED list so the UI halos don't flicker.
            # The raw per-frame detections are still available for
            # diagnostics / flight-log accuracy on "seen_markers_raw".
            _pos_st["seen_markers"]     = held_ids
            _pos_st["seen_markers_raw"] = fresh_ids
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
    """MJPEG stream with ArUco annotations from the DEPRECATED unified-
    server positioning subsystem. Served at the rate frames are
    produced. Use marker_mission's ``/video.mjpg`` (annotated by the
    same /arena the operator edits) for new clients.

    Each open generator increments ``_pos_video_clients`` so
    positioning_loop knows there's a subscriber and runs solvePnP.
    The counter drops back when the client disconnects (try/finally).
    """
    global _pos_video_clients
    def gen():
        global _pos_video_clients
        with _pos_video_clients_lock:
            _pos_video_clients += 1
        last_sent = None
        try:
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
        finally:
            with _pos_video_clients_lock:
                _pos_video_clients = max(0, _pos_video_clients - 1)
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


def _ensure_mjpeg_streaming(reason: str = "recording") -> tuple[bool, str]:
    """Make sure the MJPEG video pipeline is actively streaming so that the
    video callback fires and feeds _pos_frame_q. Without this, `_rec_enabled`
    flips to True but no frames ever arrive → VideoWriter never gets created
    → the mp4 file is never written (the old "no video" bug).

    Idempotent — safe to call whether or not the stream is already running.
    """
    global _video_mode
    b = backend
    if b is None:
        return False, "backend not ready"
    if _video_streaming and _video_mode == "mjpeg":
        return True, "already streaming"
    try:
        ok, msg = b.video_start_mjpeg()
        if ok:
            _video_mode = "mjpeg"
            print(f"[REC] Auto-started MJPEG pipeline for {reason}")
            return True, msg
        print(f"[REC] Auto-start MJPEG failed ({reason}): {msg}")
        return False, msg
    except Exception as e:
        print(f"[REC] Auto-start MJPEG exception ({reason}): {e}")
        return False, str(e)


def _start_recording_internal(fname: str | None, raw: bool) -> tuple[bool, dict]:
    """Internal helper shared by /api/video/record/start and the post-takeoff
    auto-record fallback. Arms recording (deferred writer) and ensures the
    MJPEG pipeline is running so frames actually flow.

    Returns (ok, payload). On conflict (already recording) ok=False with
    error='already recording' and the current path.
    """
    global _rec_enabled, _rec_raw, _rec_writer, _rec_path, _rec_frame_count
    global _rec_fname, _rec_fps_target, _rec_writer_size
    with _rec_lock:
        if _rec_enabled:
            return False, {"error": "already recording", "path": _rec_path}
        raw_mode = bool(raw)
        suffix = "_raw" if raw_mode else "_ann"
        fname = fname or f"rec{suffix}_{int(time.time())}.mp4"
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
    # Kick the MJPEG pipeline AFTER releasing the rec lock to avoid any
    # lock-ordering surprises with the video callback.
    mjpeg_ok, mjpeg_msg = _ensure_mjpeg_streaming(reason="record/start")
    return True, {"path": full_path, "mjpeg_ok": mjpeg_ok, "mjpeg_msg": mjpeg_msg}


@app.post("/api/video/record/start")
def api_rec_start():
    data = request.get_json(silent=True) or {}
    ok, payload = _start_recording_internal(
        fname=data.get("filename"),
        raw=bool(data.get("raw", False)),
    )
    if not ok:
        return jsonify(ok=False, **payload)
    return jsonify(ok=True, **payload)


def _stop_recording_internal(reason: str = "manual") -> tuple[bool, dict]:
    """Internal helper shared by /api/video/record/stop and the auto-stop
    on land. Returns (ok, payload). Idempotent — returns ok=False with
    'not recording' if nothing is active.
    """
    global _rec_enabled, _rec_writer, _rec_path, _rec_frame_count
    with _rec_lock:
        if not _rec_enabled:
            return False, {"error": "not recording"}
        _rec_enabled = False
        frames = _rec_frame_count
        path = _rec_path
        try:
            if _rec_writer:
                _rec_writer.release()
                _rec_writer = None
        except Exception:
            pass
        print(f"[REC] Recording stopped ({reason}): {path} ({frames} frames)")
    return True, {"path": path, "frames": frames}


@app.post("/api/video/record/stop")
def api_rec_stop():
    ok, payload = _stop_recording_internal(reason="api")
    if not ok:
        return jsonify(ok=False, **payload)
    return jsonify(ok=True, **payload)


@app.get("/api/video/recordings")
def api_rec_list():
    """List every video in the recordings folder. Used by the C2 to
    pair per-flight mp4 files with their matching flight-log JSONL."""
    rec_dir = Path(__file__).parent / "recordings"
    if not rec_dir.exists():
        return jsonify(ok=True, files=[])
    out = []
    for p in sorted(rec_dir.glob("*.mp4"), reverse=True):
        try:
            st = p.stat()
            out.append({"name": p.name, "size": st.st_size,
                        "mtime": st.st_mtime})
        except Exception:
            continue
    return jsonify(ok=True, files=out)


@app.get("/api/video/recordings/<path:name>")
def api_rec_download(name):
    """Download a specific recording. Rejects path-traversal attempts."""
    rec_dir = (Path(__file__).parent / "recordings").resolve()
    target = (rec_dir / name).resolve()
    try:
        target.relative_to(rec_dir)
    except ValueError:
        return jsonify(ok=False, error="invalid path"), 403
    if not target.exists():
        return jsonify(ok=False, error="not found"), 404
    return send_file(str(target), mimetype="video/mp4",
                     as_attachment=True, download_name=target.name)


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
        if "distance_scale" in data:
            # Clamped to [0.1, 5.0] — anything outside that range is almost
            # certainly a typo / bad input. 1.0 = no correction.
            _pos_cfg["distance_scale"] = max(0.1, min(5.0, float(data["distance_scale"])))
        # Extended tuning — all affect ctrl_position.py's multi-marker fusion
        if "pose_hold_sec" in data:
            _pos_cfg["pose_hold_sec"] = max(0.0, min(10.0, float(data["pose_hold_sec"])))
        if "min_ref_count" in data:
            _pos_cfg["min_ref_count"] = max(1, min(12, int(data["min_ref_count"])))
        if "min_ref_weight" in data:
            _pos_cfg["min_ref_weight"] = max(0.0, min(1.0, float(data["min_ref_weight"])))
        if "meas_blend_min" in data:
            _pos_cfg["meas_blend_min"] = max(0.0, min(1.0, float(data["meas_blend_min"])))
        if "meas_blend_max" in data:
            _pos_cfg["meas_blend_max"] = max(0.0, min(1.0, float(data["meas_blend_max"])))
        if "vel_blend" in data:
            _pos_cfg["vel_blend"] = max(0.0, min(1.0, float(data["vel_blend"])))
        if "max_state_dt" in data:
            _pos_cfg["max_state_dt"] = max(0.05, min(10.0, float(data["max_state_dt"])))
        if "kalman_process_var" in data:
            _pos_cfg["kalman_process_var"] = max(1e-6, min(10.0, float(data["kalman_process_var"])))
        if "kalman_meas_var" in data:
            _pos_cfg["kalman_meas_var"] = max(1e-6, min(10.0, float(data["kalman_meas_var"])))
        if "imu_lowpass_hz" in data:
            # 0 (or negative) disables the filter. Positive values clamp
            # to [0.1, 100] Hz — below 0.1 Hz it's basically DC-only,
            # above 100 Hz the filter adds nothing useful at TELEMETRY_HZ.
            v = float(data["imu_lowpass_hz"])
            _pos_cfg["imu_lowpass_hz"] = 0.0 if v <= 0 else max(0.1, min(100.0, v))
        if "seen_hold_s" in data:
            # 0 disables hysteresis (halo reflects per-frame detection
            # exactly). Positive values clamped 0..3 s.
            v = float(data["seen_hold_s"])
            _pos_cfg["seen_hold_s"] = max(0.0, min(3.0, v))
        if "max_pose_jump_m" in data:
            # 0 disables the pose-jump gate. Positive values clamped 0..20 m.
            v = float(data["max_pose_jump_m"])
            _pos_cfg["max_pose_jump_m"] = max(0.0, min(20.0, v))
        if "target_marker_size_m" in data:
            # SDC26 defaults: 0.19 m. Clamped to [0.02, 2.0] m.
            v = float(data["target_marker_size_m"])
            _pos_cfg["target_marker_size_m"] = max(0.02, min(2.0, v))
        if "zupt_speed_m_s" in data:
            # 0 disables ZUPT. Positive values clamped 0..1 m/s.
            v = float(data["zupt_speed_m_s"])
            _pos_cfg["zupt_speed_m_s"] = max(0.0, min(1.0, v))
        if "zupt_hold_frames" in data:
            # 0 disables (alongside speed==0). Clamped 0..30 frames.
            v = int(data["zupt_hold_frames"])
            _pos_cfg["zupt_hold_frames"] = max(0, min(30, v))
        if "auto_positioning" in data:
            # Toggling auto flips which set of values actually gets pushed
            # to the processor. The user's manual values are preserved so
            # turning auto off restores them. We re-apply effective cfg to
            # the live processor immediately below.
            _pos_cfg["auto_positioning"] = bool(data["auto_positioning"])
        cfg_snap = dict(_pos_cfg)
    # If auto_positioning is toggled, every downstream filter knob
    # effectively changes, so we apply the full effective-config sweep
    # rather than only the keys present in `data`.
    auto_toggled = "auto_positioning" in data
    eff = _effective_pos_cfg()

    # Apply live filter changes to running processor. When auto_positioning
    # is toggled, we fan out the FULL effective config (since every knob
    # effectively changes); otherwise only the keys present in `data`.
    _touched = (lambda k: auto_toggled or k in data)
    try:
        if _pos_processor is not None:
            if _touched("enable_kalman_filter"):
                _pos_processor.enable_kalman_filter = bool(eff.get("enable_kalman_filter", True))
                # Reset Kalman state so toggling doesn't carry stale history
                try:
                    for kf in _pos_processor.kf_pos:
                        kf.reset()
                except Exception:
                    pass
                print(f"[POS] Kalman filter {'ENABLED' if _pos_processor.enable_kalman_filter else 'DISABLED'} (live)")
            if apply_marker_size or (auto_toggled and "marker_size_m" in eff):
                # Also reflect in arena_cfg so _apply_arena_cfg_to_processor can be used
                with _arena_cfg_lock:
                    _arena_cfg["marker_size_m"] = eff["marker_size_m"]
                _apply_arena_cfg_to_processor(_pos_processor)
                print(f"[POS] marker_size_m set to {eff['marker_size_m']}m (live)")
            if _touched("top_k_markers"):
                tk = int(eff.get("top_k_markers", 4))
                _pos_processor.top_k_markers = tk if tk > 0 else 4
                print(f"[POS] top_k_markers set to {tk if tk > 0 else '4 (auto)'} (live)")
            if _touched("outlier_reject_m"):
                _pos_processor.outlier_reject_m = float(eff.get("outlier_reject_m", 2.5))
                print(f"[POS] outlier_reject_m set to {eff['outlier_reject_m']}m (live)")
            if _touched("distance_scale"):
                _pos_processor.distance_scale = float(eff.get("distance_scale", 1.0))
                print(f"[POS] distance_scale = {eff['distance_scale']:.4f}× (live)")
            if _touched("max_pose_jump_m"):
                _pos_processor.max_pose_jump_m = float(eff.get("max_pose_jump_m", 0.0))
                print(f"[POS] max_pose_jump_m = {eff['max_pose_jump_m']}m (live)")
            # target_marker_size_m + zupt_* reverted in ctrl_position.py.
            # The _pos_cfg fields are still accepted (so saved presets +
            # UI sliders keep working) but we only live-patch when the
            # processor actually exposes the attribute — a no-op on the
            # reverted code path.
            if _touched("target_marker_size_m") and hasattr(_pos_processor, "target_marker_size"):
                try:
                    _pos_processor.target_marker_size = float(eff.get("target_marker_size_m", 0.19))
                    if hasattr(_pos_processor, "_build_marker_point_sets"):
                        _pos_processor._build_marker_point_sets()
                    print(f"[POS] target_marker_size_m = {eff['target_marker_size_m']}m (live)")
                except Exception as se:
                    print(f"[POS] target_marker_size_m live skipped: {se}")
            if _touched("zupt_speed_m_s") and hasattr(_pos_processor, "zupt_speed_m_s"):
                try:
                    _pos_processor.zupt_speed_m_s = float(eff.get("zupt_speed_m_s", 0.0))
                    if hasattr(_pos_processor, "_zupt_slow_count"):
                        _pos_processor._zupt_slow_count = 0
                    print(f"[POS] zupt_speed_m_s = {eff['zupt_speed_m_s']} (live)")
                except Exception as se:
                    print(f"[POS] zupt_speed_m_s live skipped: {se}")
            if _touched("zupt_hold_frames") and hasattr(_pos_processor, "zupt_hold_frames"):
                try:
                    _pos_processor.zupt_hold_frames = int(eff.get("zupt_hold_frames", 0))
                    print(f"[POS] zupt_hold_frames = {eff['zupt_hold_frames']} (live)")
                except Exception as se:
                    print(f"[POS] zupt_hold_frames live skipped: {se}")
            # ── Extended tuning → patched as module globals on ctrl_position
            # because the fusion code reads them as module-level constants. ──
            import ctrl_position as _cp
            if _touched("pose_hold_sec"):
                _cp.POSE_HOLD_SEC = float(eff.get("pose_hold_sec", 0.8))
                print(f"[POS] pose_hold_sec = {_cp.POSE_HOLD_SEC}s (live)")
            if _touched("min_ref_count"):
                _cp.MIN_REF_COUNT = int(eff.get("min_ref_count", 1))
                print(f"[POS] min_ref_count = {_cp.MIN_REF_COUNT} (live)")
            if _touched("min_ref_weight"):
                _cp.MIN_REF_WEIGHT = float(eff.get("min_ref_weight", 0.0))
                print(f"[POS] min_ref_weight = {_cp.MIN_REF_WEIGHT} (live)")
            if _touched("meas_blend_min"):
                _cp.MEAS_BLEND_MIN = float(eff.get("meas_blend_min", 0.35))
                print(f"[POS] meas_blend_min = {_cp.MEAS_BLEND_MIN} (live)")
            if _touched("meas_blend_max"):
                _cp.MEAS_BLEND_MAX = float(eff.get("meas_blend_max", 0.85))
                print(f"[POS] meas_blend_max = {_cp.MEAS_BLEND_MAX} (live)")
            if _touched("vel_blend"):
                _cp.VEL_BLEND = float(eff.get("vel_blend", 0.25))
                print(f"[POS] vel_blend = {_cp.VEL_BLEND} (live)")
            if _touched("max_state_dt"):
                _cp.MAX_STATE_DT = float(eff.get("max_state_dt", 1.0))
                print(f"[POS] max_state_dt = {_cp.MAX_STATE_DT}s (live)")
            if _touched("kalman_process_var"):
                v = float(eff.get("kalman_process_var", 1e-3))
                for kf in _pos_processor.kf_pos:
                    kf.process_variance = v
                print(f"[POS] kalman_process_var = {v} (live, per-axis)")
            if _touched("kalman_meas_var"):
                v = float(eff.get("kalman_meas_var", 0.1))
                for kf in _pos_processor.kf_pos:
                    kf.measurement_variance = v
                print(f"[POS] kalman_meas_var = {v} (live, per-axis)")
            if auto_toggled:
                mode = "AUTO (Claude preset)" if cfg_snap.get("auto_positioning") else "MANUAL"
                print(f"[POS] Positioning mode → {mode}")
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
# Drone fleet CSV — id, name (WiFi SSID), password
# ═══════════════════════════════════════════════════════════════════════════

def _load_drones_csv() -> list:
    """Return list of {id, name, password} dicts from drones.csv."""
    import csv
    if not DRONES_CSV_PATH.exists():
        return []
    try:
        with open(DRONES_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [{"id": r.get("id", ""), "name": r.get("name", ""),
                     "password": r.get("password", "")} for r in reader]
    except Exception as e:
        print(f"[DRONES] CSV load error: {e}")
        return []


def _save_drones_csv(rows: list):
    import csv
    DRONES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DRONES_CSV_PATH.with_suffix(".csv.tmp")
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name", "password"])
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in ["id", "name", "password"]})
        import os as _os
        _os.replace(tmp, DRONES_CSV_PATH)
    except Exception as e:
        print(f"[DRONES] CSV save error: {e}")
        raise


@app.get("/api/drones")
def api_drones_list():
    """List all drones from the fleet CSV (passwords omitted)."""
    try:
        rows = _load_drones_csv()
        return jsonify(ok=True,
                       drones=[{"id": r["id"], "name": r["name"]} for r in rows])
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.get("/api/drones/csv")
def api_drones_csv_download():
    """Download the fleet CSV (includes passwords — handle with care)."""
    if DRONES_CSV_PATH.exists():
        return send_file(str(DRONES_CSV_PATH), mimetype="text/csv",
                         download_name="drones.csv", as_attachment=True)
    template = "id,name,password\n1,ANAFI_XXXXXX,changeme\n"
    return Response(template, mimetype="text/csv",
                    headers={"Content-Disposition":
                             "attachment; filename=\"drones.csv\""})


@app.post("/api/drones/csv")
def api_drones_csv_upload():
    """Upload a new fleet CSV. Body: raw text/csv."""
    import csv, io as _io
    blob = request.get_data(cache=False, as_text=True)
    if not blob.strip():
        return jsonify(ok=False, error="empty body"), 400
    try:
        reader = csv.DictReader(_io.StringIO(blob))
        rows = []
        for row in reader:
            if not all(k in row for k in ("id", "name", "password")):
                return jsonify(ok=False,
                               error="CSV must have id, name, password columns"), 400
            rows.append({"id": row["id"].strip(),
                         "name": row["name"].strip(),
                         "password": row["password"].strip()})
    except Exception as e:
        return jsonify(ok=False, error=f"CSV parse error: {e}"), 400
    try:
        _save_drones_csv(rows)
    except Exception as e:
        return jsonify(ok=False, error=f"write failed: {e}"), 500
    return jsonify(ok=True, count=len(rows))


# Switch state — polled by the UI while the background thread runs
_drone_switch_lock = threading.Lock()
_drone_switch_status: dict = {"status": "idle", "message": "", "drone": None}


@app.get("/api/drones/switch/status")
def api_drones_switch_status():
    return jsonify(**_drone_switch_status)


@app.post("/api/drones/switch")
def api_drones_switch():
    """Switch the FC to a different drone.

    Body: {"id": "<drone id from CSV>"}

    Steps (async, poll /api/drones/switch/status for progress):
      1. Land the current drone if flying
      2. Connect to the new drone's WiFi via nmcli
      3. Hard-reset and reinitialize the Olympe backend
    """
    data = request.get_json(silent=True) or {}
    drone_id = str(data.get("id", "")).strip()
    if not drone_id:
        return jsonify(ok=False, error="'id' field required"), 400

    # Look up drone in CSV
    rows = _load_drones_csv()
    match = next((r for r in rows if r["id"] == drone_id), None)
    if match is None:
        return jsonify(ok=False, error=f"drone id '{drone_id}' not found in CSV"), 404

    if not _drone_switch_lock.acquire(blocking=False):
        return jsonify(ok=False, error="switch already in progress"), 409

    def _do_switch(drone_name: str, drone_password: str, d_id: str):
        global backend, drone_ip
        try:
            # ── Step 1: stop / land current drone ──────────────────────────
            _drone_switch_status.update({
                "status": "stopping",
                "message": "Stopping mission and landing current drone…",
            })
            with conn_lock:
                currently_connected = conn_state["connected"]
            if currently_connected:
                try:
                    import drone_core as _dc
                    _dc.do_land()
                    # Wait up to 15 s for landing
                    for _ in range(30):
                        time.sleep(0.5)
                        if not flying:
                            break
                except Exception as _le:
                    print(f"[SWITCH] land error (ignored): {_le}")

            # ── Step 2: disconnect existing backend ─────────────────────────
            _drone_switch_status.update({
                "status": "disconnecting",
                "message": "Disconnecting from current drone…",
            })
            b = backend
            if b is not None:
                try:
                    b.hard_reset()
                except Exception as _de:
                    print(f"[SWITCH] hard_reset error (ignored): {_de}")
            with conn_lock:
                conn_state["connected"] = False
                conn_state["consecutive_failures"] = 0

            # ── Step 3: connect to new WiFi ────────────────────────────────
            _drone_switch_status.update({
                "status": "wifi",
                "message": f"Connecting to WiFi SSID '{drone_name}'…",
            })
            wifi_cmd = ["nmcli", "device", "wifi", "connect", drone_name]
            if drone_password:
                wifi_cmd += ["password", drone_password]
            r = subprocess.run(wifi_cmd, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                err = (r.stderr.strip() or r.stdout.strip() or
                       f"nmcli exited {r.returncode}")
                _drone_switch_status.update({
                    "status": "error",
                    "message": f"WiFi connect failed: {err}",
                })
                return

            # Give DHCP time to settle before Olympe tries to reach the drone
            _drone_switch_status.update({
                "status": "connecting",
                "message": "WiFi up — waiting for drone link…",
            })
            time.sleep(3.0)

            # ── Step 4: reinitialize Olympe backend ────────────────────────
            _drone_switch_status.update({
                "status": "init",
                "message": "Re-initializing flight controller…",
            })
            if HAS_OLYMPE_SDK:
                new_b = OlympeBackend(ANAFI_DEFAULT_IP)
                backend = new_b
                drone_ip = ANAFI_DEFAULT_IP
            # else: Tello or no SDK — only WiFi switch matters; reconnect_loop picks up

            # Reset serial so it is re-read from the new drone on first connect
            if hasattr(backend, "serial_number"):
                backend.serial_number = None
            if hasattr(backend, "_serial_high"):
                backend._serial_high = None
            if hasattr(backend, "_serial_low"):
                backend._serial_low = None

            # Kick the reconnect loop — it will call backend.connect() on the
            # next cycle and run on_connect() which reloads flight limits and
            # logs magnetometer calibration state.
            with conn_lock:
                conn_state["connected"] = False
                conn_state["last_reconnect"] = 0.0
                conn_state["consecutive_failures"] = 0
                conn_state["last_error"] = ""

            _drone_switch_status.update({
                "status": "done",
                "message": (f"Switched to '{drone_name}' (id={d_id}). "
                            f"FC is reconnecting — check telemetry for serial/battery."),
                "drone": {"id": d_id, "name": drone_name},
            })
            print(f"[SWITCH] Done — drone='{drone_name}' id={d_id}")

        except Exception as exc:
            import traceback as _tb
            _drone_switch_status.update({
                "status": "error",
                "message": str(exc),
            })
            print(f"[SWITCH] Unhandled error: {exc}")
            _tb.print_exc()
        finally:
            _drone_switch_lock.release()

    threading.Thread(
        target=_do_switch,
        args=(match["name"], match["password"], match["id"]),
        daemon=True,
        name="drone-switch",
    ).start()
    return jsonify(ok=True, message="switch started"), 202


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def init_backend_and_threads() -> bool:
    """Bring up the drone backend and start the supporting background
    threads (telemetry, reconnect, RC, watchdog, drone-ping, optional
    positioning). Does NOT start the HTTP server — that's the caller's
    job, so the combined ``marker_mission/app.py`` entry point can
    reuse this setup and then run its own Flask app instance.

    Returns True on success, False if backend selection failed (caller
    should abort). Idempotent: safe to call once per process at boot.
    """
    global backend, drone_type, drone_ip

    drone_type, drone_ip = detect_drone_type()

    if drone_type == "tello":
        if not HAS_TELLO_SDK:
            print("ERROR: djitellopy not installed. pip install djitellopy")
            return False
        logging.getLogger("djitellopy").setLevel(logging.CRITICAL)
        backend = TelloBackend(drone_ip)
    elif drone_type == "anafi":
        if not HAS_OLYMPE_SDK:
            print("ERROR: olympe not installed.")
            return False
        logging.getLogger("olympe").setLevel(logging.WARNING)
        backend = OlympeBackend(drone_ip)
    else:
        print(f"ERROR: Unknown drone type: {drone_type}")
        return False

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
    threading.Thread(target=_drone_ping_loop, daemon=True,
                     name="drone-ping").start()

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
    print(f"[{tag}] Drone: {drone_type} @ {drone_ip} (auto-reconnect; watchdog={REMOTE_TIMEOUT_S}s)")
    print(f"[{tag}] SDKs available: tello={HAS_TELLO_SDK}, olympe={HAS_OLYMPE_SDK}")
    print(f"[{tag}] Code version: {CODE_VERSION}")
    if _FC_GIT_REVISION.get("short_sha"):
        print(f"[{tag}] git revision: {_FC_GIT_REVISION.get('short_sha')} "
              f"({_FC_GIT_REVISION.get('branch','?')}"
              f"{' dirty' if _FC_GIT_REVISION.get('dirty') else ''}) — "
              f"{_FC_GIT_REVISION.get('subject','')[:80]}")
    return True


def main():
    """Standalone HTTP-only entry point. Brings up the backend +
    threads, then runs Flask on the unified port. The combined entry
    point ``marker_mission/app.py`` does NOT call this — it calls
    ``init_backend_and_threads()`` itself and runs the same ``app``
    on the mission's port instead."""
    if not init_backend_and_threads():
        return
    print(f"[{drone_type.upper()}] Unified API server: http://{HTTP_HOST}:{HTTP_PORT}")
    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
