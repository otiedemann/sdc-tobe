import atexit
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import olympe
from flask import Flask, Response, jsonify, request, send_file
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveBy, PCMD
from olympe.messages.ardrone3.PilotingState import (
    FlyingStateChanged,
    AttitudeChanged,
    SpeedChanged,
    AltitudeChanged,
)
from olympe.messages.common.CommonState import BatteryStateChanged
from olympe.messages.ardrone3.Animations import Flip
from olympe.messages.ardrone3.SpeedSettings import MaxRotationSpeed, MaxVerticalSpeed
from olympe.messages.ardrone3.PilotingSettings import MaxAltitude, MaxTilt

# Optional imports – wrapped so the server starts even if SDK version lacks some
try:
    from olympe.messages.ardrone3.Piloting import Emergency as EmergencyCmd
    HAS_EMERGENCY = True
except ImportError:
    HAS_EMERGENCY = False

try:
    from olympe.messages.camera import (
        start_recording,
        stop_recording,
        take_photo,
    )
    HAS_CAMERA = True
except ImportError:
    HAS_CAMERA = False

try:
    from olympe.messages.gimbal import set_target, attitude as gimbal_attitude
    HAS_GIMBAL = True
except ImportError:
    HAS_GIMBAL = False

try:
    from olympe.messages.ardrone3.PilotingState import GpsLocationChanged
    HAS_GPS_STATE = True
except ImportError:
    try:
        from olympe.messages.ardrone3.GPSState import GpsLocationChanged
        HAS_GPS_STATE = True
    except ImportError:
        HAS_GPS_STATE = False

try:
    from olympe.messages.rth import return_to_home, cancel_auto_trigger, state as rth_state
    HAS_RTH = True
except ImportError:
    HAS_RTH = False

try:
    from olympe.messages.move import extended_move_to
    HAS_MOVE_TO = True
except ImportError:
    HAS_MOVE_TO = False

try:
    from olympe.messages.ardrone3.SpeedSettings import MaxHorizontalSpeed
    HAS_MAX_HORIZ_SPEED = True
except ImportError:
    HAS_MAX_HORIZ_SPEED = False

try:
    from olympe.messages.ardrone3.PilotingSettings import MaxDistance, NoFlyOverMaxDistance
    HAS_GEOFENCE = True
except ImportError:
    HAS_GEOFENCE = False

# Optional: OpenCV for Way 1 (on-Pi MJPEG streaming)
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Optional: subprocess for Way 2 (UDP forwarding)
import shutil
import subprocess
import signal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8080
DRONE_IP = os.getenv("ANAFI_IP", "192.168.42.1")
RC_HZ = 20
STICK = 60
YAW_STICK = 90  # Yaw needs higher value for snappy rotation on Anafi
MAX_YAW_SPEED = 150  # deg/s – applied on connect via MaxRotationSpeed
MAX_ALTITUDE_M = float(os.getenv("MAX_ALTITUDE_M", "2.0"))  # hard ceiling in meters
MAX_VERTICAL_SPEED = float(os.getenv("MAX_VERTICAL_SPEED", "0.5"))  # m/s vertical speed limit
MAX_TILT = float(os.getenv("MAX_TILT", "15"))  # degrees – limits horizontal speed
CONNECT_RETRY_S = 3.0
TELEMETRY_HZ = float(os.getenv("TELEMETRY_HZ", "2.0"))
KEY_STALE_S = float(os.getenv("KEY_STALE_S", "1.0"))
SAFE_TAKEOFF_S = float(os.getenv("SAFE_TAKEOFF_S", "3.0"))
SAFE_TAKEOFF_DEFAULT = os.getenv("SAFE_TAKEOFF_DEFAULT", "0") in {"1", "true", "True"}
REMOTE_TIMEOUT_S = float(os.getenv("REMOTE_TIMEOUT_S", "2.0"))  # auto-land if no remote heartbeat for this long

# Video streaming config
VIDEO_JPEG_QUALITY = int(os.getenv("VIDEO_JPEG_QUALITY", "70"))  # MJPEG quality (Way 1)
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "15"))  # target MJPEG frame rate (Way 1)
VIDEO_UDP_FORWARD_PORT = int(os.getenv("VIDEO_UDP_FORWARD_PORT", "55004"))  # outbound port (Way 2)
VIDEO_MODE = os.getenv("VIDEO_MODE", "off")  # "off", "mjpeg", "forward"
TELEMETRY_LOG_DEFAULT = False
TELEMETRY_LOG_PATH_DEFAULT = Path(__file__).with_name("telemetry_log.jsonl")
COMMAND_LOG_ENABLED = os.getenv("API_COMMAND_LOG", "1") in {"1", "true", "True"}
COMMAND_LOG_PATH = Path(
    os.getenv("API_COMMAND_LOG_PATH", str(Path(__file__).with_name("api_command_log.jsonl")))
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
app = Flask(__name__)
running = True
flying = False
drone: Optional[olympe.Drone] = None

pressed_web: Set[str] = set()
key_last_seen: Dict[str, float] = {}
pressed_lock = threading.Lock()

last_state_seen = 0.0

conn_state = {"connected": False, "last_reconnect": 0.0}
conn_lock = threading.Lock()
last_conn_print = None

rc_override: Optional[Tuple[int, int, int, int]] = None
rc_override_until = 0.0
rc_lock = threading.Lock()

telemetry: Dict = {
    "battery": None,
    "temperature": None,
    "height_cm": None,
    "tof_cm": None,
    "barometer_cm": None,
    "flight_time_s": None,
    "pitch": None,
    "roll": None,
    "yaw": None,
    "vgx": None,
    "vgy": None,
    "vgz": None,
    "agx": None,
    "agy": None,
    "agz": None,
    "speed": None,
    "flying": False,
    "connected": False,
    "updated_at": 0.0,
}
telemetry_lock = threading.Lock()
telemetry_log_enabled = TELEMETRY_LOG_DEFAULT
telemetry_log_path = TELEMETRY_LOG_PATH_DEFAULT
telemetry_log_lock = threading.Lock()
command_log_lock = threading.Lock()

command_lock = threading.Lock()
discrete_until = 0.0
takeoff_cooldown_until = 0.0
safe_takeoff_enabled = SAFE_TAKEOFF_DEFAULT

# Remote controller watchdog — auto-land if remote goes silent while flying
last_remote_request = 0.0
_watchdog_landed = False  # prevents repeated land attempts

# Sequence number counter for PCMD
_pcmd_seq = 0
_pcmd_seq_lock = threading.Lock()

# Video streaming state
_video_mode = VIDEO_MODE  # "off", "mjpeg", "forward"
_video_last_jpeg = b""
_video_jpeg_lock = threading.Lock()
_video_streaming = False  # True when pdraw callbacks are active
_video_forward_proc = None  # subprocess for UDP forwarding (Way 2)
_video_forward_target = ""  # "host:port" for UDP forwarding
_video_frame_count = 0


# ---------------------------------------------------------------------------
# Key helpers
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


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def append_telemetry_log(payload: dict):
    global telemetry_log_enabled, telemetry_log_path
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
        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            "payload": payload or {},
        }
        with command_log_lock:
            COMMAND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with COMMAND_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Discrete command window (blocks RC during/after discrete commands)
# ---------------------------------------------------------------------------

def start_discrete_window(seconds: float):
    global discrete_until
    with command_lock:
        discrete_until = max(discrete_until, time.time() + max(0.0, seconds))


# ---------------------------------------------------------------------------
# Olympe piloting interface
# ---------------------------------------------------------------------------
# Some Olympe versions expose start_piloting / piloting_pcmd / stop_piloting
# but they may fail at runtime ("Unable to launch piloting interface").
# We detect at first use, ACTUALLY TEST the call, and fall back to raw
# d(PCMD(...)) if the native API doesn't work.

_has_piloting_api: Optional[bool] = None  # None=untested, True/False=tested


def _detect_piloting_api() -> bool:
    """Check whether the native piloting API actually works (not just exists)."""
    global _has_piloting_api
    if _has_piloting_api is not None:
        return _has_piloting_api
    d = drone
    if d is None:
        return False
    if not callable(getattr(d, "start_piloting", None)):
        _has_piloting_api = False
        print("[ANAFI API] Native piloting API not found – using raw d(PCMD(...))")
        return False
    # Method exists – try calling it to verify it actually works
    try:
        d.start_piloting()
        _has_piloting_api = True
        print("[ANAFI API] Native piloting API works (start_piloting / piloting_pcmd / stop_piloting)")
        return True
    except Exception as e:
        _has_piloting_api = False
        print(f"[ANAFI API] Native piloting API FAILED ({e}) – using raw d(PCMD(...))")
        return False


# ---------------------------------------------------------------------------
# PCMD helper — always uses raw d(PCMD(...)) since native API is unreliable
# ---------------------------------------------------------------------------

def _send_pcmd(roll: int, pitch: int, yaw: int, gaz: int, flag: int = 1):
    """Send a single PCMD frame via raw Olympe command."""
    global _pcmd_seq
    d = drone
    if d is None:
        return
    if _detect_piloting_api():
        try:
            if flag == 0:
                d.piloting_pcmd(0, 0, 0, 0, 0)
            else:
                d.piloting_pcmd(roll, pitch, yaw, gaz, 0)
            return
        except Exception:
            pass
    # Fallback / primary: raw PCMD
    with _pcmd_seq_lock:
        _pcmd_seq = (_pcmd_seq + 1) & 0x7FFFFFFF
        seq = _pcmd_seq
    try:
        d(PCMD(flag, roll, pitch, yaw, gaz, seq))
    except Exception:
        pass


def _start_piloting():
    """Start manual piloting (native API) or no-op for raw PCMD mode."""
    d = drone
    if d is None:
        return
    if _detect_piloting_api():
        try:
            d.start_piloting()
        except Exception:
            pass


def _stop_piloting():
    """Stop manual piloting. For raw PCMD mode, sends flag=0 zeros."""
    d = drone
    if d is None:
        return
    if _detect_piloting_api():
        try:
            d.stop_piloting()
            return
        except Exception:
            pass
    # Raw PCMD fallback: send flag=0 to release manual control
    global _pcmd_seq
    with _pcmd_seq_lock:
        _pcmd_seq = (_pcmd_seq + 1) & 0x7FFFFFFF
        seq = _pcmd_seq
    try:
        d(PCMD(0, 0, 0, 0, 0, seq))
    except Exception:
        pass


def _release_pcmd():
    """Release manual control – stop piloting and send zeros."""
    _stop_piloting()


# ---------------------------------------------------------------------------
# Telemetry collection
# ---------------------------------------------------------------------------

def _get_drone_state(msg_type):
    """Safely get the last state for a message type from Olympe."""
    d = drone
    if d is None:
        return None
    try:
        return d.get_state(msg_type)
    except Exception:
        return None


def _is_flying_state(state_dict) -> bool:
    """Return True if the Olympe FlyingStateChanged state indicates airborne."""
    if not state_dict:
        return False
    try:
        s = state_dict.get("state")
        if s is None:
            return False
        # Olympe enums stringify to e.g. "FlyingStateChanged_State.landed" —
        # we must check ONLY the value after the last dot, otherwise "flying"
        # would match the class name in every state (including "landed").
        name = str(s).lower()
        # Extract just the value part: "flyingstatechanged_state.landed" → "landed"
        if "." in name:
            name = name.rsplit(".", 1)[-1]
        # Also try the .name attribute if it's an enum
        try:
            name = s.name.lower()
        except AttributeError:
            pass
        # state values: landed=0, takingoff=1, hovering=2, flying=3, landing=4, emergency=5
        return name in ("hovering", "flying", "takingoff")
    except Exception:
        return False


def telemetry_loop():
    global flying, last_state_seen
    while running:
        with conn_lock:
            connected_now = conn_state["connected"]

        got_any_state = False

        # --- Battery ---
        bat = None
        bat_state = _get_drone_state(BatteryStateChanged)
        if bat_state:
            got_any_state = True
            try:
                bat = int(bat_state["percent"])
            except Exception:
                pass

        # --- Attitude (pitch/roll/yaw in radians → degrees) ---
        pitch_deg = roll_deg = yaw_deg = None
        att_state = _get_drone_state(AttitudeChanged)
        if att_state:
            got_any_state = True
            try:
                pitch_deg = round(math.degrees(float(att_state["pitch"])), 2)
                roll_deg = round(math.degrees(float(att_state["roll"])), 2)
                yaw_deg = round(math.degrees(float(att_state["yaw"])), 2)
            except Exception:
                pass

        # --- Speed (m/s → cm/s to match Tello convention) ---
        vgx = vgy = vgz = None
        spd_state = _get_drone_state(SpeedChanged)
        if spd_state:
            got_any_state = True
            try:
                # Olympe SpeedChanged: speedX=forward, speedY=right, speedZ=down (NED)
                vgx = round(float(spd_state["speedX"]) * 100, 1)
                vgy = round(float(spd_state["speedY"]) * 100, 1)
                vgz = round(float(spd_state["speedZ"]) * 100, 1)
            except Exception:
                pass

        # --- Altitude (m → cm) ---
        height_cm = None
        alt_state = _get_drone_state(AltitudeChanged)
        if alt_state:
            got_any_state = True
            try:
                height_cm = round(float(alt_state["altitude"]) * 100, 1)
            except Exception:
                pass

        # --- GPS position ---
        gps_lat = gps_lon = gps_alt = None
        if HAS_GPS_STATE:
            gps_state = _get_drone_state(GpsLocationChanged)
            if gps_state:
                got_any_state = True
                try:
                    gps_lat = round(float(gps_state.get("latitude", 500)), 7)
                    gps_lon = round(float(gps_state.get("longitude", 500)), 7)
                    gps_alt = round(float(gps_state.get("altitude", 0)), 2)
                    # 500 = "not available" sentinel in Olympe
                    if gps_lat > 400:
                        gps_lat = None
                    if gps_lon > 400:
                        gps_lon = None
                except Exception:
                    pass

        # --- Gimbal attitude ---
        gimbal_pitch = gimbal_roll = gimbal_yaw = None
        if HAS_GIMBAL:
            gim_state = _get_drone_state(gimbal_attitude)
            if gim_state:
                try:
                    gimbal_pitch = round(float(gim_state.get("pitch_absolute", 0)), 2)
                    gimbal_roll = round(float(gim_state.get("roll_absolute", 0)), 2)
                    gimbal_yaw = round(float(gim_state.get("yaw_absolute", 0)), 2)
                except Exception:
                    pass

        if got_any_state:
            last_state_seen = time.time()
            # If we got state data, the drone IS connected — fix stale conn_state
            with conn_lock:
                if not conn_state["connected"]:
                    conn_state["connected"] = True
                    print("[ANAFI API] Connection detected via telemetry")
                connected_now = True

        # --- Flying state ---
        fly_state = _get_drone_state(FlyingStateChanged)
        sdk_flying = _is_flying_state(fly_state)
        # Update global flying flag from SDK state when not in a discrete command window.
        with command_lock:
            in_discrete = time.time() < discrete_until
        if not in_discrete:
            flying = sdk_flying

        # Compute aggregate speed
        speed = None
        if vgx is not None and vgy is not None and vgz is not None:
            speed = round((vgx ** 2 + vgy ** 2 + vgz ** 2) ** 0.5, 1)

        with telemetry_lock:
            if bat is not None:
                telemetry["battery"] = bat
            if pitch_deg is not None:
                telemetry["pitch"] = pitch_deg
            if roll_deg is not None:
                telemetry["roll"] = roll_deg
            if yaw_deg is not None:
                telemetry["yaw"] = yaw_deg
            if vgx is not None:
                telemetry["vgx"] = vgx
            if vgy is not None:
                telemetry["vgy"] = vgy
            if vgz is not None:
                telemetry["vgz"] = vgz
            if height_cm is not None:
                telemetry["height_cm"] = height_cm
            if speed is not None:
                telemetry["speed"] = speed
            if gps_lat is not None:
                telemetry["gps_lat"] = gps_lat
            if gps_lon is not None:
                telemetry["gps_lon"] = gps_lon
            if gps_alt is not None:
                telemetry["gps_alt"] = gps_alt
            if gimbal_pitch is not None:
                telemetry["gimbal_pitch"] = gimbal_pitch
            if gimbal_roll is not None:
                telemetry["gimbal_roll"] = gimbal_roll
            if gimbal_yaw is not None:
                telemetry["gimbal_yaw"] = gimbal_yaw
            telemetry["flying"] = flying
            telemetry["connected"] = connected_now
            telemetry["updated_at"] = time.time()
            snapshot = dict(telemetry)

        append_telemetry_log(snapshot)
        hz = TELEMETRY_HZ if TELEMETRY_HZ > 0 else 2.0
        time.sleep(max(0.05, 1.0 / hz))


# ---------------------------------------------------------------------------
# Flight safety limits – applied on every (re-)connect
# ---------------------------------------------------------------------------

def _apply_flight_limits(d):
    """Set altitude ceiling, vertical speed cap, max tilt, and yaw speed on the drone."""
    for cmd, label in [
        (MaxAltitude(MAX_ALTITUDE_M), f"MaxAltitude={MAX_ALTITUDE_M}m"),
        (MaxVerticalSpeed(MAX_VERTICAL_SPEED), f"MaxVerticalSpeed={MAX_VERTICAL_SPEED}m/s"),
        (MaxTilt(MAX_TILT), f"MaxTilt={MAX_TILT}°"),
        (MaxRotationSpeed(MAX_YAW_SPEED), f"MaxRotationSpeed={MAX_YAW_SPEED}°/s"),
    ]:
        try:
            d(cmd).wait(_timeout=2)
            print(f"[ANAFI API] Set {label}")
        except Exception as e:
            print(f"[ANAFI API] Failed to set {label}: {e}")


# ---------------------------------------------------------------------------
# Reconnect loop
# ---------------------------------------------------------------------------

def _check_drone_connected(d) -> bool:
    """Verify connection by actually querying a state from the drone."""
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


def reconnect_loop():
    global drone, last_conn_print
    while running:
        with conn_lock:
            last_try = conn_state["last_reconnect"]
            connected_now = conn_state["connected"]

        if connected_now != last_conn_print:
            print("[ANAFI API] Drone connected" if connected_now else "[ANAFI API] Drone disconnected (retrying...)")
            last_conn_print = connected_now

        now = time.time()

        # Periodic health check when we think we're connected
        if connected_now:
            if not _check_drone_connected(drone):
                print("[ANAFI API] Connection lost (health check failed)")
                with conn_lock:
                    conn_state["connected"] = False
            time.sleep(2.0)
            continue

        should_retry = (now - last_try) >= CONNECT_RETRY_S

        if should_retry:
            with conn_lock:
                conn_state["last_reconnect"] = now
            d = drone
            if d is None:
                d = olympe.Drone(DRONE_IP)
                drone = d
            try:
                d.connect()
                # Don't rely on connect() return value — verify with actual state query
                actually_connected = _check_drone_connected(d)
                if actually_connected:
                    _apply_flight_limits(d)
                    _start_piloting()
                    print("[ANAFI API] Connection verified (state query OK)")
                with conn_lock:
                    conn_state["connected"] = actually_connected
            except Exception as e:
                print(f"[ANAFI API] Connect failed: {e}")
                with conn_lock:
                    conn_state["connected"] = False

        time.sleep(1.0)


# ---------------------------------------------------------------------------
# RC/PCMD loop
# ---------------------------------------------------------------------------

def rc_loop():
    global running, flying, rc_override, rc_override_until, takeoff_cooldown_until
    period = 1.0 / RC_HZ
    _was_connected = False
    while running:
        t0 = time.time()
        reap_stale_keys(t0)

        with conn_lock:
            connected = conn_state["connected"]

        if not connected:
            _was_connected = False
            time.sleep(period)
            continue

        # On (re)connect, start Olympe piloting so PCMD values are sent
        if not _was_connected:
            _start_piloting()
            _was_connected = True

        # Takeoff via key
        if has_key("t") and not flying:
            try:
                hold_s = SAFE_TAKEOFF_S if safe_takeoff_enabled else 3.0
                start_discrete_window(hold_s)
                _stop_piloting()
                time.sleep(0.2)
                print("[ANAFI API] Key TakeOff...")
                with command_lock:
                    result = drone(TakeOff()).wait(_timeout=10)
                ok = False
                try:
                    ok = result.success() if result is not None else False
                except Exception:
                    pass
                if ok:
                    flying = True
                    takeoff_cooldown_until = time.time() + hold_s
                else:
                    # Poll for state change
                    for _ in range(4):
                        time.sleep(0.5)
                        fs = _get_drone_state(FlyingStateChanged)
                        if _is_flying_state(fs):
                            flying = True
                            takeoff_cooldown_until = time.time() + hold_s
                            break
                print(f"[ANAFI API] Key TakeOff result: ok={ok}, flying={flying}")
                _start_piloting()
            except Exception as e:
                print(f"[ANAFI API] Takeoff error: {e}")
                _start_piloting()
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
                _stop_piloting()
                time.sleep(0.2)
                print("[ANAFI API] Key Landing...")
                with command_lock:
                    drone(Landing()).wait(_timeout=10)
                flying = False
            except Exception as e:
                print(f"[ANAFI API] Land error: {e}")
            remove_key("l")

        # Build RC axes from held keys
        # Tello mapping: lr=left/right, fb=fwd/back, ud=up/down, yaw=cw/ccw
        # Olympe PCMD: roll=lr, pitch=fb, gaz=ud, yaw=yaw
        lr = axis(has_key("d"), has_key("a")) * STICK
        fb = axis(has_key("w"), has_key("s")) * STICK
        ud = axis(has_key("r"), has_key("f")) * STICK
        yaw = axis(has_key("e"), has_key("q")) * YAW_STICK

        if has_key("space") or has_key("x"):
            lr = fb = ud = yaw = 0

        now = time.time()
        with rc_lock:
            if rc_override is not None and now < rc_override_until:
                lr, fb, ud, yaw = rc_override
            elif rc_override is not None and now >= rc_override_until:
                rc_override = None

        # Software altitude fence: clamp upward gaz when near ceiling
        with telemetry_lock:
            cur_height = telemetry.get("height_cm")
        if cur_height is not None and cur_height >= MAX_ALTITUDE_M * 100 and ud > 0:
            ud = 0

        with command_lock:
            in_discrete = now < discrete_until

        if in_discrete:
            # Don't send ANY PCMD during discrete commands (TakeOff/Land/moveBy).
            # Olympe needs a clear command channel — PCMD flag=1 can block actions.
            pass
        else:
            _send_pcmd(lr, fb, yaw, ud)

        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


# ---------------------------------------------------------------------------
# Remote controller watchdog — auto-land on connection loss
# ---------------------------------------------------------------------------

def watchdog_loop():
    """If the remote controller stops sending requests while airborne, auto-land."""
    global flying, _watchdog_landed
    while running:
        time.sleep(1.0)
        if not flying or _watchdog_landed:
            continue
        if last_remote_request <= 0:
            continue  # never received a request yet
        silence = time.time() - last_remote_request
        if silence >= REMOTE_TIMEOUT_S:
            print(f"[ANAFI API] WATCHDOG: No remote request for {silence:.1f}s while flying — AUTO-LANDING")
            _watchdog_landed = True  # don't spam land attempts
            d = drone
            if d is None:
                continue
            try:
                _stop_piloting()
                time.sleep(0.2)
                with pressed_lock:
                    pressed_web.clear()
                    key_last_seen.clear()
                with command_lock:
                    d(Landing()).wait(_timeout=10)
                flying = False
                print("[ANAFI API] WATCHDOG: Auto-land command sent")
            except Exception as e:
                print(f"[ANAFI API] WATCHDOG: Auto-land failed: {e}")


# ---------------------------------------------------------------------------
# Video streaming
# ---------------------------------------------------------------------------

def _video_frame_callback(yuv_frame):
    """Called by Olympe pdraw for each decoded video frame (Way 1: MJPEG).

    Olympe requires ref()/unref() lifecycle on frames. Without ref() the
    frame buffer can be reclaimed before we finish processing.
    """
    global _video_last_jpeg, _video_frame_count
    if not HAS_CV2:
        return
    try:
        yuv_frame.ref()
    except Exception:
        pass
    try:
        # Detect YUV pixel format — the info() dict structure varies by Olympe version
        cv_cvt = cv2.COLOR_YUV2BGR_I420  # safe default for Anafi (I420/YUV420P)
        try:
            info = yuv_frame.info()
            # Modern Olympe: info["yuv"]["format"]
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
                print(f"[ANAFI API] Frame info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}, using cvt={cv_cvt}")
        except Exception as ie:
            if _video_frame_count == 0:
                print(f"[ANAFI API] Frame info() failed ({ie}), using default I420")

        cv_frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv_cvt)
        ok, jpg = cv2.imencode(
            ".jpg", cv_frame, [int(cv2.IMWRITE_JPEG_QUALITY), VIDEO_JPEG_QUALITY]
        )
        if ok:
            with _video_jpeg_lock:
                _video_last_jpeg = jpg.tobytes()
            _video_frame_count += 1
            if _video_frame_count == 1:
                h, w = cv_frame.shape[:2]
                print(f"[ANAFI API] First video frame decoded: {w}x{h}")
    except Exception as e:
        if _video_frame_count == 0:
            print(f"[ANAFI API] Video frame decode error: {e}")
    finally:
        try:
            yuv_frame.unref()
        except Exception:
            pass


def _video_flush_callback(stream):
    """Required flush callback for Olympe pdraw.

    Must drain all pending buffers and return True to keep the pipeline flowing.
    Without this the pdraw pipeline stalls after a few frames.
    """
    try:
        # Olympe >= 7.x: stream is a queue object
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


def _detect_olympe_streaming_api(d):
    """Detect which Olympe streaming API is available.

    Returns: ("modern", "legacy", or "none"), plus diagnostic info.
    """
    # Modern Olympe >= 7.x: drone.streaming.set_callbacks / .start / .stop
    if hasattr(d, "streaming") and hasattr(d.streaming, "set_callbacks"):
        return "modern"
    # Legacy Olympe: drone.set_streaming_callbacks / drone.start_video_streaming
    if hasattr(d, "set_streaming_callbacks"):
        return "legacy"
    return "none"


def start_video_mjpeg():
    """Way 1: Start pdraw video decoding on the Pi, serve MJPEG over HTTP."""
    global _video_streaming, _video_mode, _video_last_jpeg, _video_frame_count
    d = drone
    if d is None:
        return False, "drone not connected"
    if not HAS_CV2:
        return False, "cv2 (OpenCV) not installed on Pi — pip install opencv-python-headless"

    api = _detect_olympe_streaming_api(d)
    print(f"[ANAFI API] Olympe streaming API detected: {api}")

    _video_mode = "mjpeg"
    _video_last_jpeg = b""
    _video_frame_count = 0

    if api == "modern":
        try:
            d.streaming.set_callbacks(
                raw_cb=_video_frame_callback,
                flush_raw_cb=_video_flush_callback,
            )
            d.streaming.start()
            _video_streaming = True
            print("[ANAFI API] Video MJPEG streaming started (modern API)")
            return True, "mjpeg streaming started"
        except Exception as e:
            print(f"[ANAFI API] Modern streaming API failed: {e}")
            # Fall through to legacy
            api = "legacy" if hasattr(d, "set_streaming_callbacks") else "none"

    if api == "legacy":
        try:
            d.set_streaming_callbacks(raw_cb=_video_frame_callback)
            d.start_video_streaming()
            _video_streaming = True
            print("[ANAFI API] Video MJPEG streaming started (legacy API)")
            return True, "mjpeg streaming started (legacy)"
        except Exception as e:
            print(f"[ANAFI API] Legacy streaming API failed: {e}")
            return False, f"legacy streaming failed: {e}"

    return False, "no streaming API found in this Olympe version"


def stop_video_mjpeg():
    """Stop pdraw MJPEG streaming."""
    global _video_streaming, _video_last_jpeg
    d = drone
    _video_streaming = False
    _video_last_jpeg = b""
    if d is None:
        return
    try:
        d.streaming.stop()
        print("[ANAFI API] Video MJPEG streaming stopped (modern)")
    except Exception:
        try:
            d.stop_video_streaming()
            print("[ANAFI API] Video MJPEG streaming stopped (legacy)")
        except Exception:
            pass


def start_video_forward(target_host: str, target_port: int):
    """Way 2: Forward the Anafi's RTP video stream to C2 server without decoding.

    The Anafi sends RTP/H.264 video to the SkyController/Pi on UDP port 55004.
    We forward those packets to the C2 server using socat (zero-copy, no decode).
    Alternative: GStreamer udpsrc ! udpsink, but socat is lighter and available everywhere.
    """
    global _video_forward_proc, _video_forward_target, _video_mode
    stop_video_forward()  # kill any previous forwarder
    _video_mode = "forward"
    _video_forward_target = f"{target_host}:{target_port}"

    # Try socat first (lightest option, zero-copy UDP relay)
    socat = shutil.which("socat")
    if socat:
        # Listen on local port 55004 (Anafi RTP), forward to target
        cmd = [
            socat, "-u",
            f"UDP-RECV:{VIDEO_UDP_FORWARD_PORT},reuseaddr",
            f"UDP-SENDTO:{target_host}:{target_port}",
        ]
        try:
            _video_forward_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            print(f"[ANAFI API] Video UDP forwarding started via socat → {target_host}:{target_port}")
            return True, f"socat forwarding to {target_host}:{target_port}"
        except Exception as e:
            print(f"[ANAFI API] socat failed: {e}")

    # Fallback: GStreamer
    gst = shutil.which("gst-launch-1.0")
    if gst:
        cmd = [
            gst, "-q",
            "udpsrc", f"port={VIDEO_UDP_FORWARD_PORT}",
            "!", "udpsink", f"host={target_host}", f"port={target_port}",
        ]
        try:
            _video_forward_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            print(f"[ANAFI API] Video UDP forwarding started via GStreamer → {target_host}:{target_port}")
            return True, f"gstreamer forwarding to {target_host}:{target_port}"
        except Exception as e:
            print(f"[ANAFI API] gstreamer failed: {e}")

    # Fallback: pure Python UDP relay (works everywhere, slightly more CPU)
    ok, msg = _start_python_udp_relay(target_host, target_port)
    return ok, msg


def _start_python_udp_relay(target_host: str, target_port: int):
    """Pure-Python UDP relay as last resort (no external deps)."""
    global _video_forward_proc, _video_streaming
    import socket

    _video_streaming = True  # flag to stop the thread

    def relay_thread():
        sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_in.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock_in.bind(("0.0.0.0", VIDEO_UDP_FORWARD_PORT))
        sock_in.settimeout(1.0)
        sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        pkt_count = 0
        print(f"[ANAFI API] Python UDP relay: :{VIDEO_UDP_FORWARD_PORT} → {target_host}:{target_port}")
        while _video_streaming and running:
            try:
                data, _ = sock_in.recvfrom(65536)
                sock_out.sendto(data, (target_host, target_port))
                pkt_count += 1
                if pkt_count == 1:
                    print(f"[ANAFI API] Python UDP relay: first packet forwarded ({len(data)} bytes)")
            except socket.timeout:
                continue
            except Exception as e:
                if _video_streaming:
                    print(f"[ANAFI API] Python UDP relay error: {e}")
                break
        sock_in.close()
        sock_out.close()
        print(f"[ANAFI API] Python UDP relay stopped ({pkt_count} packets forwarded)")

    t = threading.Thread(target=relay_thread, daemon=True)
    t.start()
    print(f"[ANAFI API] Video UDP forwarding started via Python relay → {target_host}:{target_port}")
    return True, f"python relay to {target_host}:{target_port}"


def stop_video_forward():
    """Stop UDP forwarding subprocess."""
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
        print("[ANAFI API] Video UDP forwarding stopped")
    _video_forward_target = ""


def stop_all_video():
    """Stop all video streaming modes."""
    global _video_mode
    stop_video_mjpeg()
    stop_video_forward()
    _video_mode = "off"


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def shutdown():
    global running
    running = False
    stop_all_video()
    d = drone
    if d is None:
        return
    try:
        _stop_piloting()
    except Exception:
        pass
    if flying:
        try:
            d(Landing()).wait(_timeout=5)
        except Exception:
            pass
    try:
        d.disconnect()
        print("[ANAFI API] Drone disconnected.")
    except Exception as e:
        print(f"[ANAFI API] Shutdown error: {e}")


# ---------------------------------------------------------------------------
# Flask middleware
# ---------------------------------------------------------------------------

@app.before_request
def _update_remote_heartbeat():
    """Track last request time from remote controller for watchdog auto-land."""
    global last_remote_request, _watchdog_landed
    try:
        if request.path.startswith("/api/"):
            last_remote_request = time.time()
            _watchdog_landed = False  # reset — remote is back
    except Exception:
        pass


@app.before_request
def _log_incoming_api_command():
    try:
        if request.method != "POST":
            return
        if not request.path.startswith("/api/"):
            return
        if request.path.startswith("/api/logging/"):
            return
        payload = request.get_json(silent=True) or {}
        append_command_log(request.path, payload)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    with conn_lock:
        connected = conn_state["connected"]
    return jsonify(ok=True, service="olympe_pi_api_server", connected=connected)


@app.get("/api/heartbeat")
def api_heartbeat():
    """Ultra-lightweight heartbeat — keeps watchdog alive without locks."""
    with conn_lock:
        connected = conn_state["connected"]
    return jsonify(ok=True, flying=flying, connected=connected, t=time.time())


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
    d = drone
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or d is None:
        return jsonify(ok=False, error="controller not ready"), 503
    try:
        # Log current state so we can diagnose skipped takeoffs
        fs = _get_drone_state(FlyingStateChanged)
        print(f"[ANAFI API] /api/takeoff called — flying={flying}, state={fs}")

        if not flying:
            hold_s = SAFE_TAKEOFF_S if safe_takeoff_enabled else 3.0
            start_discrete_window(hold_s)

            # 1. Stop piloting to clear the PCMD channel for the action command
            _stop_piloting()
            time.sleep(0.2)  # let any in-flight PCMD drain

            # 2. Check pre-takeoff state
            pre_state = _get_drone_state(FlyingStateChanged)
            print(f"[ANAFI API] TakeOff: pre-state={pre_state}")

            # 3. Send TakeOff — use _timeout so we never block forever
            print("[ANAFI API] Sending TakeOff command...")
            with command_lock:
                result = d(TakeOff()).wait(_timeout=10)

            ok = False
            try:
                ok = result.success() if result is not None else False
            except Exception:
                pass
            print(f"[ANAFI API] TakeOff result: ok={ok}, result={result}")

            if ok:
                flying = True
                takeoff_cooldown_until = time.time() + hold_s
            else:
                # Command may still have worked — poll for state change
                for i in range(6):
                    time.sleep(0.5)
                    fs = _get_drone_state(FlyingStateChanged)
                    print(f"[ANAFI API] TakeOff poll {i}: state={fs}")
                    if _is_flying_state(fs):
                        flying = True
                        takeoff_cooldown_until = time.time() + hold_s
                        break
                if not flying:
                    return jsonify(ok=False, error="takeoff_failed"), 500

            # 4. Re-enable piloting for RC control
            _start_piloting()

        return jsonify(ok=True, flying=flying, safe_takeoff=safe_takeoff_enabled)
    except Exception as e:
        print(f"[ANAFI API] TakeOff exception: {e}")
        _start_piloting()  # re-enable piloting even on error
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/land")
def api_land():
    global flying, rc_override, rc_override_until
    d = drone
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or d is None:
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

        # Stop piloting to clear PCMD channel
        _stop_piloting()
        time.sleep(0.2)

        print("[ANAFI API] Sending Landing command...")
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
                print(f"[ANAFI API] Landing attempt {attempt}: ok={ok}")
                if ok:
                    flying = False
                    return jsonify(ok=True, flying=False)
                # Check state even if result wasn't clean
                time.sleep(0.5)
                fs = _get_drone_state(FlyingStateChanged)
                if not _is_flying_state(fs):
                    flying = False
                    return jsonify(ok=True, flying=False)
            except Exception as e:
                last_err = e
                print(f"[ANAFI API] Landing attempt {attempt} error: {e}")
                time.sleep(0.25)
        raise last_err if last_err else RuntimeError("land_failed")
    except Exception as e:
        print(f"[ANAFI API] Land exception: {e}")
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/flip")
def api_flip():
    d = drone
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or d is None:
        return jsonify(ok=False, error="controller not ready"), 503

    data = request.get_json(silent=True) or {}
    direction = str(data.get("dir", data.get("direction", ""))).lower()

    # Accept both l/r/f/b and full names (front/back/left/right)
    dir_map = {
        "f": "front", "front": "front",
        "b": "back",  "back": "back",
        "l": "left",  "left": "left",
        "r": "right", "right": "right",
    }
    if direction not in dir_map:
        return jsonify(ok=False, error="dir must be one of l|r|f|b (or front|back|left|right)"), 400

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

    flip_dir_map = {
        "front": Flip.direction.front,
        "back": Flip.direction.back,
        "left": Flip.direction.left,
        "right": Flip.direction.right,
    }
    olympe_dir = flip_dir_map[dir_map[direction]]

    try:
        start_discrete_window(2.0)
        _stop_piloting()
        time.sleep(0.1)
        with command_lock:
            result = d(Flip(olympe_dir)).wait(_timeout=5)
        _start_piloting()
        if result and result.success():
            return jsonify(ok=True, dir=direction)
        return jsonify(ok=False, error="flip_failed"), 500
    except Exception as e:
        _start_piloting()
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/emergency")
def api_emergency():
    global flying
    d = drone
    if d is None:
        return jsonify(ok=False, error="controller not ready"), 503
    try:
        _stop_piloting()
        if HAS_EMERGENCY:
            with command_lock:
                d(EmergencyCmd()).wait(_timeout=3)
        else:
            # Fallback: land
            with command_lock:
                d(Landing()).wait(_timeout=5)
        flying = False
        return jsonify(ok=True)
    except Exception as e:
        flying = False
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/move")
def api_move():
    """Move in a cardinal direction by a distance in cm (matches Tello API)."""
    d = drone
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or d is None:
        return jsonify(ok=False, error="controller not ready"), 503

    data = request.get_json(silent=True) or {}
    direction = str(data.get("dir", "")).lower()
    if direction not in {"up", "down", "left", "right", "forward", "back"}:
        return jsonify(ok=False, error="dir must be one of up|down|left|right|forward|back"), 400

    try:
        dist_cm = int(data.get("cm", 20))
        dist_cm = max(20, min(500, dist_cm))
        dist_m = dist_cm / 100.0

        # Olympe moveBy: dX=forward(+), dY=right(+), dZ=down(+), dPsi=yaw(rad)
        move_args = {
            "forward": (dist_m, 0, 0, 0),
            "back":    (-dist_m, 0, 0, 0),
            "right":   (0, dist_m, 0, 0),
            "left":    (0, -dist_m, 0, 0),
            "up":      (0, 0, -dist_m, 0),
            "down":    (0, 0, dist_m, 0),
        }[direction]

        start_discrete_window(max(1.0, dist_m * 2))
        _stop_piloting()
        time.sleep(0.1)
        with command_lock:
            result = d(moveBy(*move_args)).wait(_timeout=30)
        _start_piloting()
        if result and result.success():
            return jsonify(ok=True, dir=direction, cm=dist_cm)
        return jsonify(ok=False, error="move_failed"), 500
    except Exception as e:
        _start_piloting()
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/rotate")
def api_rotate():
    """Rotate clockwise or counter-clockwise by degrees (matches Tello API)."""
    d = drone
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or d is None:
        return jsonify(ok=False, error="controller not ready"), 503

    data = request.get_json(silent=True) or {}
    direction = str(data.get("dir", "")).lower()
    if direction not in {"cw", "ccw"}:
        return jsonify(ok=False, error="dir must be one of cw|ccw"), 400

    try:
        degrees = int(data.get("deg", 45))
        degrees = max(1, min(360, degrees))
        rad = math.radians(degrees)
        d_psi = rad if direction == "cw" else -rad

        start_discrete_window(max(1.0, degrees / 90.0))
        _stop_piloting()
        time.sleep(0.1)
        with command_lock:
            result = d(moveBy(0, 0, 0, d_psi)).wait(_timeout=15)
        _start_piloting()
        if result and result.success():
            return jsonify(ok=True, dir=direction, deg=degrees)
        return jsonify(ok=False, error="rotate_failed"), 500
    except Exception as e:
        _start_piloting()
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/go")
def api_go():
    """Move to relative xyz position in cm (matches Tello API; speed param ignored)."""
    d = drone
    with conn_lock:
        connected = conn_state["connected"]
    if not connected or d is None:
        return jsonify(ok=False, error="controller not ready"), 503

    data = request.get_json(silent=True) or {}
    try:
        x = int(data.get("x", 0))  # forward(+)/back(-) in cm
        y = int(data.get("y", 0))  # right(+)/left(-) in cm
        z = int(data.get("z", 0))  # up(+)/down(-) in cm  [Tello convention]

        # Olympe moveBy: dX=fwd, dY=right, dZ=down(+)=up(-)
        dx = x / 100.0
        dy = y / 100.0
        dz = -z / 100.0  # invert: Tello z+ is up, Olympe dZ+ is down

        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        start_discrete_window(max(1.5, dist * 2))
        _stop_piloting()
        time.sleep(0.1)
        with command_lock:
            result = d(moveBy(dx, dy, dz, 0)).wait(_timeout=30)
        _start_piloting()
        if result and result.success():
            return jsonify(ok=True, x=x, y=y, z=z)
        return jsonify(ok=False, error="go_failed"), 500
    except Exception as e:
        _start_piloting()
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/rc")
def api_rc():
    """RC override: set PCMD values for a duration. Matches Tello API (lr/fb/ud/yaw, -100..100)."""
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
    """Attempt to reconnect drone. Resets flying state and clears inputs."""
    global flying, drone, rc_override, rc_override_until
    d = drone
    try:
        if d is not None:
            try:
                d.disconnect()
            except Exception:
                pass
        d = olympe.Drone(DRONE_IP)
        drone = d
        with pressed_lock:
            pressed_web.clear()
            key_last_seen.clear()
        with rc_lock:
            rc_override = None
            rc_override_until = 0.0
        d.connect()
        actually_connected = _check_drone_connected(d)
        if actually_connected:
            _apply_flight_limits(d)
            _start_piloting()
        flying = False
        with conn_lock:
            conn_state["connected"] = actually_connected
            conn_state["last_reconnect"] = time.time()
        return jsonify(ok=actually_connected, message="recovered" if actually_connected else "reconnect_failed")
    except Exception as e:
        with conn_lock:
            conn_state["connected"] = False
        return jsonify(ok=False, message=str(e)), 500


@app.get("/api/telemetry")
def api_telemetry():
    now = time.time()
    age = (now - last_state_seen) if last_state_seen else 9999.0
    with telemetry_lock:
        payload = dict(telemetry)
    payload["state_age_s"] = round(age, 3)
    payload["state_fresh"] = age <= 2.0
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


@app.get("/api/logging/commands")
def api_command_log_status():
    return jsonify(enabled=COMMAND_LOG_ENABLED, path=str(COMMAND_LOG_PATH))


@app.get("/api/logging/commands/download")
def api_command_log_download():
    p = COMMAND_LOG_PATH
    if not p.exists():
        return jsonify(ok=False, error="command log file not found", path=str(p)), 404
    return send_file(p, as_attachment=True, download_name=p.name, mimetype="application/x-ndjson")


@app.post("/api/logging/commands/clear")
def api_command_log_clear():
    p = COMMAND_LOG_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        return jsonify(ok=True, cleared=True, path=str(p))
    except Exception as e:
        return jsonify(ok=False, error=str(e), path=str(p)), 500


@app.get("/api/logging/telemetry")
def api_telemetry_log_status():
    with telemetry_log_lock:
        return jsonify(enabled=telemetry_log_enabled, path=str(telemetry_log_path))


@app.post("/api/logging/telemetry")
def api_telemetry_log_config():
    global telemetry_log_enabled, telemetry_log_path
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled")
    path = data.get("path")
    with telemetry_log_lock:
        if isinstance(enabled, bool):
            telemetry_log_enabled = enabled
        if isinstance(path, str) and path.strip():
            telemetry_log_path = Path(path.strip())
        return jsonify(enabled=telemetry_log_enabled, path=str(telemetry_log_path))


@app.get("/api/logging/telemetry/download")
def api_telemetry_log_download():
    with telemetry_log_lock:
        p = telemetry_log_path
    if not p.exists():
        return jsonify(ok=False, error="telemetry log file not found", path=str(p)), 404
    return send_file(p, as_attachment=True, download_name=p.name, mimetype="application/x-ndjson")


@app.post("/api/logging/telemetry/clear")
def api_telemetry_log_clear():
    with telemetry_log_lock:
        p = telemetry_log_path
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        return jsonify(ok=True, cleared=True, path=str(p))
    except Exception as e:
        return jsonify(ok=False, error=str(e), path=str(p)), 500


# ---------------------------------------------------------------------------
# Camera endpoints
# ---------------------------------------------------------------------------

@app.post("/api/camera/photo")
def api_camera_photo():
    d = drone
    if d is None:
        return jsonify(ok=False, error="controller not ready"), 503
    if not HAS_CAMERA:
        return jsonify(ok=False, error="camera module not available"), 501
    try:
        with command_lock:
            result = d(take_photo(cam_id=0)).wait(_timeout=5)
        if result and result.success():
            return jsonify(ok=True)
        return jsonify(ok=False, error="photo_failed"), 500
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/camera/record/start")
def api_camera_record_start():
    d = drone
    if d is None:
        return jsonify(ok=False, error="controller not ready"), 503
    if not HAS_CAMERA:
        return jsonify(ok=False, error="camera module not available"), 501
    try:
        with command_lock:
            result = d(start_recording(cam_id=0)).wait(_timeout=5)
        if result and result.success():
            return jsonify(ok=True, recording=True)
        return jsonify(ok=False, error="record_start_failed"), 500
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/camera/record/stop")
def api_camera_record_stop():
    d = drone
    if d is None:
        return jsonify(ok=False, error="controller not ready"), 503
    if not HAS_CAMERA:
        return jsonify(ok=False, error="camera module not available"), 501
    try:
        with command_lock:
            result = d(stop_recording(cam_id=0)).wait(_timeout=5)
        if result and result.success():
            return jsonify(ok=True, recording=False)
        return jsonify(ok=False, error="record_stop_failed"), 500
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/gimbal")
def api_gimbal():
    """Set gimbal tilt and pan in degrees. tilt: -90 (down) to +30 (up). pan: ±180."""
    d = drone
    if d is None:
        return jsonify(ok=False, error="controller not ready"), 503
    if not HAS_GIMBAL:
        return jsonify(ok=False, error="gimbal module not available"), 501
    data = request.get_json(silent=True) or {}
    tilt = float(data.get("tilt", 0))
    pan = float(data.get("pan", 0))
    tilt = max(-90, min(30, tilt))
    pan = max(-180, min(180, pan))
    try:
        with command_lock:
            result = d(set_target(
                gimbal_id=0,
                control_mode="position",
                yaw_frame_of_reference="absolute",
                yaw=pan,
                pitch_frame_of_reference="absolute",
                pitch=tilt,
                roll_frame_of_reference="absolute",
                roll=0,
            )).wait(_timeout=5)
        if result and result.success():
            return jsonify(ok=True, tilt=tilt, pan=pan)
        return jsonify(ok=False, error="gimbal_failed"), 500
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


# ---------------------------------------------------------------------------
# Navigation endpoints
# ---------------------------------------------------------------------------

@app.post("/api/rth")
def api_rth():
    """Start or cancel Return-To-Home."""
    d = drone
    if d is None:
        return jsonify(ok=False, error="controller not ready"), 503
    if not HAS_RTH:
        return jsonify(ok=False, error="rth module not available"), 501
    data = request.get_json(silent=True) or {}
    action = str(data.get("action", "start")).lower()
    try:
        if action == "start":
            with command_lock:
                result = d(return_to_home()).wait(_timeout=5)
        elif action == "cancel":
            with command_lock:
                result = d(cancel_auto_trigger()).wait(_timeout=5)
        else:
            return jsonify(ok=False, error="action must be start|cancel"), 400
        if result and result.success():
            return jsonify(ok=True, action=action)
        return jsonify(ok=False, error=f"rth_{action}_failed"), 500
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/moveto")
def api_moveto():
    """Move to absolute GPS coordinates. Requires GPS fix."""
    d = drone
    if d is None:
        return jsonify(ok=False, error="controller not ready"), 503
    if not HAS_MOVE_TO:
        return jsonify(ok=False, error="moveto module not available"), 501
    data = request.get_json(silent=True) or {}
    lat = data.get("lat")
    lon = data.get("lon")
    alt = float(data.get("alt", 2.0))
    heading = float(data.get("heading", 0))
    if lat is None or lon is None:
        return jsonify(ok=False, error="lat and lon required"), 400
    try:
        start_discrete_window(10.0)
        _stop_piloting()
        time.sleep(0.1)
        with command_lock:
            result = d(extended_move_to(
                latitude=float(lat),
                longitude=float(lon),
                altitude=alt,
                orientation_mode="heading_start",
                heading=heading,
                max_horizontal_speed=MAX_TILT / 5.0,
                max_vertical_speed=MAX_VERTICAL_SPEED,
                max_yaw_rotation_speed=MAX_YAW_SPEED,
            )).wait(_timeout=60)
        _start_piloting()
        if result and result.success():
            return jsonify(ok=True, lat=float(lat), lon=float(lon), alt=alt)
        return jsonify(ok=False, error="moveto_failed"), 500
    except Exception as e:
        _start_piloting()
        return jsonify(ok=False, error=str(e)), 500


# ---------------------------------------------------------------------------
# Video streaming endpoints
# ---------------------------------------------------------------------------

@app.get("/api/video")
def api_video_feed():
    """Way 1: MJPEG HTTP stream — decoded on the Pi, viewable in any browser.
    URL: http://<pi-ip>:8080/api/video
    """
    if _video_mode != "mjpeg":
        return jsonify(ok=False, error="MJPEG streaming not active. POST /api/video/start with mode=mjpeg"), 400

    def gen():
        while running and _video_mode == "mjpeg":
            with _video_jpeg_lock:
                jpg = _video_last_jpeg
            if jpg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(1.0 / max(1, VIDEO_FPS))

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/video/start")
def api_video_start():
    """Start video streaming.

    Body JSON:
      mode: "mjpeg" — Way 1: decode on Pi, serve MJPEG at /api/video
      mode: "forward" — Way 2: forward raw UDP/RTP packets to C2 server
        target_host: str — C2 server IP (required for forward mode)
        target_port: int — C2 server port (default: 55004)
    """
    global _video_mode
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "mjpeg")

    # Stop any active streaming first
    stop_all_video()

    if mode == "mjpeg":
        ok, msg = start_video_mjpeg()
        if ok:
            return jsonify(ok=True, mode="mjpeg", message=msg,
                           stream_url=f"http://{request.host}/api/video")
        return jsonify(ok=False, error=msg), 500

    elif mode == "forward":
        target_host = data.get("target_host")
        target_port = int(data.get("target_port", VIDEO_UDP_FORWARD_PORT))
        if not target_host:
            return jsonify(ok=False, error="target_host required for forward mode"), 400
        ok, msg = start_video_forward(target_host, target_port)
        if ok:
            return jsonify(ok=True, mode="forward", message=msg,
                           target=f"{target_host}:{target_port}",
                           viewer_cmd=f"ffplay -fflags nobuffer -flags low_delay -framedrop -probesize 32 -analyzeduration 0 udp://0.0.0.0:{target_port}")
        return jsonify(ok=False, error=msg), 500

    return jsonify(ok=False, error=f"unknown mode: {mode}. Use 'mjpeg' or 'forward'"), 400


@app.post("/api/video/stop")
def api_video_stop():
    """Stop all video streaming."""
    stop_all_video()
    return jsonify(ok=True, mode="off")


@app.get("/api/video/status")
def api_video_status():
    """Get current video streaming status."""
    status = {
        "mode": _video_mode,
        "cv2_available": HAS_CV2,
        "socat_available": shutil.which("socat") is not None,
        "gstreamer_available": shutil.which("gst-launch-1.0") is not None,
    }
    if _video_mode == "mjpeg":
        status["stream_url"] = f"http://{request.host}/api/video"
        status["frames_decoded"] = _video_frame_count
        status["has_frame"] = len(_video_last_jpeg) > 0
    elif _video_mode == "forward":
        status["target"] = _video_forward_target
        status["process_alive"] = _video_forward_proc is not None and _video_forward_proc.poll() is None
    return jsonify(**status)


# ---------------------------------------------------------------------------
# Settings endpoints
# ---------------------------------------------------------------------------

@app.get("/api/settings")
def api_settings_get():
    """Return current flight limit settings."""
    return jsonify(
        max_altitude_m=MAX_ALTITUDE_M,
        max_vertical_speed=MAX_VERTICAL_SPEED,
        max_tilt=MAX_TILT,
        max_yaw_speed=MAX_YAW_SPEED,
        geofence_available=HAS_GEOFENCE,
        camera_available=HAS_CAMERA,
        gimbal_available=HAS_GIMBAL,
        gps_available=HAS_GPS_STATE,
        rth_available=HAS_RTH,
        moveto_available=HAS_MOVE_TO,
        video_mjpeg_available=HAS_CV2,
        video_forward_available=True,
    )


@app.post("/api/settings")
def api_settings_set():
    """Update flight limits on the drone. All fields optional."""
    global MAX_ALTITUDE_M, MAX_VERTICAL_SPEED, MAX_TILT, MAX_YAW_SPEED
    d = drone
    if d is None:
        return jsonify(ok=False, error="controller not ready"), 503
    data = request.get_json(silent=True) or {}
    results = {}
    if "max_altitude_m" in data:
        v = float(data["max_altitude_m"])
        v = max(0.5, min(150, v))
        try:
            d(MaxAltitude(v)).wait(_timeout=2)
            MAX_ALTITUDE_M = v
            results["max_altitude_m"] = v
        except Exception as e:
            results["max_altitude_m_error"] = str(e)
    if "max_vertical_speed" in data:
        v = float(data["max_vertical_speed"])
        v = max(0.1, min(4.0, v))
        try:
            d(MaxVerticalSpeed(v)).wait(_timeout=2)
            MAX_VERTICAL_SPEED = v
            results["max_vertical_speed"] = v
        except Exception as e:
            results["max_vertical_speed_error"] = str(e)
    if "max_tilt" in data:
        v = float(data["max_tilt"])
        v = max(1, min(35, v))
        try:
            d(MaxTilt(v)).wait(_timeout=2)
            MAX_TILT = v
            results["max_tilt"] = v
        except Exception as e:
            results["max_tilt_error"] = str(e)
    if "max_yaw_speed" in data:
        v = float(data["max_yaw_speed"])
        v = max(1, min(200, v))
        try:
            d(MaxRotationSpeed(v)).wait(_timeout=2)
            MAX_YAW_SPEED = v
            results["max_yaw_speed"] = v
        except Exception as e:
            results["max_yaw_speed_error"] = str(e)
    if "geofence_distance" in data and HAS_GEOFENCE:
        v = float(data["geofence_distance"])
        v = max(10, min(4000, v))
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
    return jsonify(ok=True, **results)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def main():
    global drone
    logging.getLogger("olympe").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    drone = olympe.Drone(DRONE_IP)

    atexit.register(shutdown)

    threading.Thread(target=reconnect_loop, daemon=True).start()
    threading.Thread(target=telemetry_loop, daemon=True).start()
    threading.Thread(target=rc_loop, daemon=True).start()
    threading.Thread(target=watchdog_loop, daemon=True).start()

    print(f"[ANAFI API] http://{HTTP_HOST}:{HTTP_PORT} (waiting for Anafi at {DRONE_IP}; auto-reconnect enabled; watchdog={REMOTE_TIMEOUT_S}s)")
    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
