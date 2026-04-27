import json
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# When run as a script (`python controller_modularized/remote_web_controller.py`)
# Python only puts the script's own directory on sys.path. Add the parent
# so `import controller_modularized.…` resolves. Harmless when launched
# as a module (`python -m controller_modularized.remote_web_controller`).
_PKG_PARENT = str(Path(__file__).resolve().parent.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# When invoked as a script, this module loads as ``__main__``. Anything
# that later does ``from controller_modularized.remote_web_controller
# import …`` would otherwise trigger a SECOND load of the same file under
# the qualified name — causing duplicate blueprint registrations and the
# infamous "setup method ... can no longer be called" Flask error. Alias
# the qualified name to this module's slot in sys.modules now, before any
# transitive imports run.
if __name__ == "__main__":
    sys.modules.setdefault("controller_modularized.remote_web_controller",
                            sys.modules[__name__])

import requests
from flask import render_template, Flask, Response, jsonify, request, send_file

# ── WebSocket client — optional dependency ────────────────────────────
# When present, the C2 opens a long-lived WS per drone for telemetry,
# position, and RC. Drops per-call HTTP framing overhead; important
# savings on the RC path that fires per key stroke.
try:
    import websocket as _wsclient  # websocket-client package
    HAS_WSCLIENT = True
except Exception as _e:
    _wsclient = None
    HAS_WSCLIENT = False
    print(f"[WS] websocket-client not available ({_e}) — C2 falls back to HTTP")

# ── ArUco Seek (multi-drone observer / LIVE controller) ────────────────────
# aruco_seek_multi still lives in the original controller/ directory; this
# modularized package depends on it without copying it.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "controller"))
from aruco_seek_multi import (  # noqa: E402
    HoverParams as AsHoverParams,
    MissionManager,
    ObserverFleet,
)

# ── Modularized service layers (Phase 2) ──────────────────────────
from controller_modularized.model.flight_logger import FlightLogger
from controller_modularized.model.drone_ws import DroneWS, drone_ws, start_fleet as _init_drone_ws
from controller_modularized.model.git_rev import read_git_revision as _read_git_revision
from controller_modularized.controller import heartbeat as _heartbeat
from controller_modularized.api import (
    bp_system, bp_drones, bp_flight, bp_safety, bp_camera, bp_telemetry,
    bp_settings, bp_magneto, bp_logs, bp_arena, bp_aruco, bp_missions,
    register_all as _register_blueprints,
)

# ── Connection-pooled HTTP session ───────────────────────────────────────────
# Reuses TCP connections (keep-alive) instead of opening a new one per request.
# Dramatically reduces per-request latency on LAN (~30-50 ms saved per call).
_http_session = requests.Session()
_http_session.headers.update({"Connection": "keep-alive"})
adapter = requests.adapters.HTTPAdapter(
    pool_connections=16,   # one per drone × multiple concurrent types (tel/pos/cmd)
    pool_maxsize=64,       # bumped from 16: the browser fires ~13 polls/sec, each
                           # can fan out to 5 Pis → pool was saturating within a
                           # second and queueing everything else. 64 gives
                           # comfortable headroom for the observed workload.
    max_retries=0,         # fail fast — don't retry on control commands
)
_http_session.mount("http://", adapter)
_http_session.mount("https://", adapter)

# Thread pool for parallel heartbeats / fan-out requests
_heartbeat_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="hb")

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Runs on remote PC. Proxies to Pi API server.
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8090
TIMEOUT_CMD = float(os.getenv("PI_TIMEOUT_CMD", "8.0"))
TIMEOUT_STATUS = float(os.getenv("PI_TIMEOUT_STATUS", "0.5"))
# Fast per-keystroke commands — RC / key_down / key_up. These are
# fire-and-forget on the Pi side; tightly-timeouted so an unreachable
# drone doesn't back up the UI. Anything longer than this on a
# keystroke is a network/server problem, not expected latency.
TIMEOUT_FAST = float(os.getenv("PI_TIMEOUT_FAST", "1.5"))
# Explicitly slow commands — takeoff + land block until the drone
# reports the corresponding state, which on Anafi routinely takes
# 3-5 s. Use this override (via pi_post(..., timeout=TIMEOUT_SLOW))
# so the default TIMEOUT_CMD stays small for responsive UI.
TIMEOUT_SLOW = float(os.getenv("PI_TIMEOUT_SLOW", "15.0"))
VIDEO_UDP_FORWARD_PORT = int(os.getenv("VIDEO_UDP_FORWARD_PORT", "55004"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "30"))
VIDEO_JPEG_QUALITY = int(os.getenv("VIDEO_JPEG_QUALITY", "70"))

# Drone fleet config file — stored alongside this script
DRONES_CONFIG_PATH = Path(__file__).parent / "drones_config.json"

DEFAULT_DRONES = {
    "1": {"name": "Anafi 1", "type": "anafi", "base": "http://flightctrl1:8080"},
    "2": {"name": "Anafi 2", "type": "anafi", "base": "http://flightctrl2:8080"},
    "3": {"name": "Anafi 3", "type": "anafi", "base": "http://flightctrl3:8080"},
    "4": {"name": "Anafi 4", "type": "anafi", "base": "http://flightctrl4:8080"},
}

def load_drones_config() -> dict:
    if DRONES_CONFIG_PATH.exists():
        try:
            with open(DRONES_CONFIG_PATH) as f:
                cfg = json.load(f)
            # Validate structure
            for did, info in cfg.items():
                if not all(k in info for k in ("name", "type", "base")):
                    print(f"[CONFIG] Invalid drone entry {did}, using defaults")
                    return dict(DEFAULT_DRONES)
            print(f"[CONFIG] Loaded {len(cfg)} drones from {DRONES_CONFIG_PATH}")
            return cfg
        except Exception as e:
            print(f"[CONFIG] Error loading {DRONES_CONFIG_PATH}: {e}, using defaults")
    else:
        save_drones_config(DEFAULT_DRONES)
        print(f"[CONFIG] Created default config at {DRONES_CONFIG_PATH}")
    return dict(DEFAULT_DRONES)

def save_drones_config(drones: dict):
    with open(DRONES_CONFIG_PATH, "w") as f:
        json.dump(drones, f, indent=2)

DRONES = load_drones_config()
active_drone_id = "1"
# PI_API_BASE env var overrides the base URL from drones_config.json
_env_base = os.getenv("PI_API_BASE")
if _env_base:
    PI_BASE = _env_base.rstrip("/")
    DRONES[active_drone_id]["base"] = PI_BASE
else:
    PI_BASE = DRONES[active_drone_id]["base"]

app = Flask(__name__,
            template_folder="view/templates",
            static_folder="view/static",
            static_url_path="/static")

# ArUco Seek fleet — one observer per configured drone. LIVE mode is on by
# default (matches tools/aruco_seek_web.py default); disable with REMOTE_NO_LIVE=1.
_aruco_allow_live = os.getenv("REMOTE_NO_LIVE", "0") not in {"1", "true", "True"}
aruco_fleet = ObserverFleet(session=_http_session, allow_live=_aruco_allow_live)
aruco_fleet.configure(DRONES)
mission_manager = MissionManager(aruco_fleet)

# ── Global PAUSE state ─────────────────────────────────────────────────
# When True, every autonomous subsystem (missions, ArUco Seek LIVE) is
# vetoed. Manual WASD / RC / keepalive remain active so the operator
# can nudge a drone manually. Set by /proxy/pause_all, cleared by
# /proxy/resume_all. UI hotkey is '9' (next to '0' = LAND ALL).
_global_paused: bool = False
_global_paused_at: float = 0.0
_global_paused_src: str = ""
_pause_lock = threading.Lock()


def _is_paused() -> bool:
    with _pause_lock:
        return _global_paused


def _pause_guard_response():
    """Return a Flask response to reject an autonomous-control request
    while the fleet is paused, or None if we're allowed to proceed."""
    if _is_paused():
        return jsonify(
            ok=False,
            error="fleet paused — autonomous control is disabled",
            hint="press CONTINUE MISSION (button) or 9 (hotkey) to resume",
            paused=True,
        ), 409
    return None


command_log_enabled = os.getenv("REMOTE_COMMAND_LOG", "0") in {"1", "true", "True"}
command_log_path = Path(os.getenv("REMOTE_COMMAND_LOG_PATH", "remote_command_log.jsonl"))
command_log_last: dict[str, float] = {}

# ── Automatic per-flight logging ───────────────────────────────────────
# Every take-off opens a fresh JSONL file under FLIGHT_LOG_DIR; every
# land (or loss-of-flying-state) closes it. Records are either
# "tick" (5 Hz telemetry + position + visible markers) or "cmd"
# (every command logged through log_command). Files are readable via
# /proxy/flight_logs (list + download).
FLIGHT_LOG_DIR = Path(os.getenv("FLIGHT_LOG_DIR", "flight_logs")).resolve()
FLIGHT_LOG_HZ  = float(os.getenv("FLIGHT_LOG_HZ", "5.0"))


_GIT_REVISION = _read_git_revision()
print(f"[GIT] {_GIT_REVISION.get('branch') or '?'} @ {_GIT_REVISION.get('short_sha') or '?'}"
      f"{' (dirty)' if _GIT_REVISION.get('dirty') else ''} "
      f"— {_GIT_REVISION.get('subject') or ''}")

# ── C2 version string (matches the FC's CODE_VERSION format) ──────
# Used by /proxy/fc_version to flag a mismatch when the FC has been
# started from an older build than the C2.
C2_CODE_VERSION = "2026-04-24-cr (FC version endpoint + C2/FC mismatch check)"


flight_logger = FlightLogger(DRONES, _http_session, FLIGHT_LOG_DIR, FLIGHT_LOG_HZ)
flight_logger.start()

# Spawn the per-drone WebSocket clients now that DRONES is populated.
_init_drone_ws(DRONES)

# Server-side heartbeat loop — keeps the Pi watchdog satisfied.
_heartbeat.start(DRONES, drone_ws, _http_session)


# ── Server-side heartbeat loop ─────────────────────────────────────────
# The Pi's watchdog auto-lands if it sees no remote activity for
# REMOTE_TIMEOUT_S (default 2 s). Historically the browser polled
# /proxy/heartbeat at 2 Hz to keep that alive — a complete waste of
# the browser's HTTP connection pool since the heartbeat has no UI
# purpose. Now fired from a background thread on the C2 itself, once
# per second per drone, skipping drones whose WS is fully down. The
# browser doesn't have to issue ANY heartbeat traffic.


# ── WebSocket client per drone ─────────────────────────────────────────
# Maintains three long-lived WS to each Pi (/ws/telemetry pulls, /ws/position
# pulls, /ws/rc pushes). Caches the latest telemetry + position so HTTP
# proxy calls can answer instantly from RAM instead of going back to the
# Pi. RC/key events take the send path which reuses the already-open
# TCP+WS socket — shaves the ~3-15 ms per-call HTTP framing cost.



def log_command(event: str, payload: dict | None = None):
    # Always feed the per-flight logger — that path runs regardless of the
    # optional debug command log file. When no drones are airborne the
    # call is a cheap no-op.
    try:
        did = None
        if payload is not None:
            did = payload.get("id") or payload.get("drone_id")
        flight_logger.record_command(did, event, payload)
    except Exception:
        pass

    if not command_log_enabled:
        return
    try:
        # High-frequency keepalive events can starve command handling if logged each packet.
        now = time.time()
        key = event
        throttle_s = 0.0
        if event in {"key_down", "key_up"}:
            k = str((payload or {}).get("key", ""))
            key = f"{event}:{k}"
            throttle_s = 0.5
        last = command_log_last.get(key, 0.0)
        if throttle_s > 0 and (now - last) < throttle_s:
            return
        command_log_last[key] = now

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "ts": ts,
            "event": event,
            "payload": payload or {},
        }
        command_log_path.parent.mkdir(parents=True, exist_ok=True)
        with command_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[REMOTE CMD] {ts} {event} payload={payload or {}}")
    except Exception:
        # Logging must never break control flow.
        pass


# ── Diagnostic ring — track recent slow Pi calls so we can see the
# "second takeoff takes 10 s" pattern as real data instead of guessing.
# Any call over 500 ms gets appended. /proxy/diagnostics returns it.
_slow_calls_lock = threading.Lock()
_slow_calls: list = []
_SLOW_CALL_THRESHOLD_S = 0.5


def _record_pi_call(method: str, path: str, dt_s: float, status: int | None,
                     err: str | None = None):
    if dt_s < _SLOW_CALL_THRESHOLD_S and err is None:
        return
    rec = {
        "ts":       time.time(),
        "method":   method,
        "path":     path,
        "dt_ms":    int(dt_s * 1000),
        "status":   status,
        "error":    err,
        "drone_id": active_drone_id,
    }
    with _slow_calls_lock:
        _slow_calls.append(rec)
        # Keep only last 200 entries — more than enough for debugging.
        if len(_slow_calls) > 200:
            del _slow_calls[:len(_slow_calls) - 200]
    # Always print slow calls so the operator sees them live without
    # having to pull /proxy/diagnostics.
    print(f"[PI SLOW] {method} {path} {dt_s:.2f}s "
          f"status={status} err={err or '-'}")


def pi_post(path: str, body: dict | None = None, timeout: float | None = None):
    t0 = time.time()
    status = None
    err = None
    try:
        r = _http_session.post(f"{PI_BASE}{path}", json=body or {},
                               timeout=TIMEOUT_CMD if timeout is None else timeout)
        status = r.status_code
        return r
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        _record_pi_call("POST", path, time.time() - t0, status, err)


def pi_get(path: str, timeout: float | None = None):
    t0 = time.time()
    status = None
    err = None
    try:
        r = _http_session.get(f"{PI_BASE}{path}", timeout=TIMEOUT_CMD if timeout is None else timeout)
        status = r.status_code
        return r
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        _record_pi_call("GET", path, time.time() - t0, status, err)














# ── Per-subsystem transport preference ────────────────────────────────
# Three values accepted:
#   "auto" — prefer WS, fall back to HTTP when WS is down
#   "ws"   — force WS; if WS is down, endpoint returns 503 immediately
#            (no HTTP fallback at all — useful to diagnose whether WS
#            itself is slow)
#   "http" — force HTTP; never use the WS path (useful to confirm WS
#            isn't the source of latency without restarting the server)
#
# Initial values come from env for backward-compat (C2_WS_RC=0 still
# disables WS RC at boot), then overridden at runtime via
# /proxy/config/transport.
_transport_lock = threading.Lock()
_transport = {
    "rc":        "http" if os.getenv("C2_WS_RC", "1") in {"0", "false", "False"} else "auto",
    "telemetry": "auto",
    "position":  "auto",
}


def _transport_mode(subsystem: str) -> str:
    with _transport_lock:
        return _transport.get(subsystem, "auto")


def _ws_enabled_for(subsystem: str) -> bool:
    """Should we attempt the WS path for this subsystem?"""
    return _transport_mode(subsystem) in ("auto", "ws")


def _http_fallback_allowed_for(subsystem: str) -> bool:
    """Should we fall back to HTTP if WS fails?"""
    return _transport_mode(subsystem) in ("auto", "http")


# Legacy alias — still read by a couple of old log messages. Keeps
# existing behaviour if anyone ships a patch that checks this directly.
WS_RC_ENABLED = _ws_enabled_for("rc")


def _active_drone_reachable() -> bool:
    """Cheap check — is the active drone's WS up on any channel? When
    everything is down we can skip slow HTTP fallbacks to avoid blocking
    the Flask request for the full TIMEOUT_CMD (2 s) per keypress."""
    ws = drone_ws.get(str(active_drone_id))
    if ws is None:
        return True   # assume reachable if no WS client (tello, HTTP-only)
    return (ws._ws_connected.get("rc")
            or ws._ws_connected.get("telemetry")
            or ws._ws_connected.get("position"))














def _collect_fleet_videos() -> dict:
    """Gather the set of available recording filenames across all FCs.
    Key = filename (basename), value = {drone_id, size, mtime}. We only
    care about the flight-named ones (prefixed "flight_"). Other
    recordings from the manual Record button are ignored.

    Note: we deliberately do NOT pre-filter by WS status — the WS channel
    can be down while HTTP still works (fallback mode), and we want the
    operator to be able to download videos in that case. A 0.6 s HTTP
    timeout keeps the cost of a fully-offline drone low enough.
    """
    out: dict = {}
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            continue
        try:
            r = _http_session.get(f"{base.rstrip('/')}/api/video/recordings",
                                   timeout=0.6)
            if not r.ok:
                continue
            for f in (r.json().get("files") or []):
                name = f.get("name", "")
                if name.startswith("flight_") and name.endswith(".mp4"):
                    out[name] = {"drone_id": str(did), "size": f.get("size"),
                                  "mtime": f.get("mtime")}
        except Exception:
            continue
    return out












































# -- Anafi / Olympe proxy routes --













































# ---------------------------------------------------------------------------
# UDP → MJPEG bridge for forward mode
# Receives raw H264 UDP from the Pi, decodes with cv2, serves as MJPEG
# ---------------------------------------------------------------------------
_udp_receiver_running = False
_udp_receiver_thread = None
_udp_last_frame_lock = threading.Lock()
_udp_last_jpeg: bytes = b""
_udp_has_frame = False


_ffmpeg_proc = None


def _start_udp_receiver():
    """Start ffmpeg to decode H264 UDP and produce raw frames, then encode to JPEG."""
    global _udp_receiver_running, _udp_receiver_thread, _udp_has_frame, _udp_last_jpeg
    if _udp_receiver_running:
        return
    _udp_receiver_running = True
    _udp_has_frame = False
    _udp_last_jpeg = b""
    _udp_receiver_thread = threading.Thread(target=_udp_receiver_loop, daemon=True, name="udp-video-recv")
    _udp_receiver_thread.start()


def _stop_udp_receiver():
    global _udp_receiver_running, _udp_has_frame, _udp_last_jpeg, _ffmpeg_proc
    _udp_receiver_running = False
    if _ffmpeg_proc is not None:
        try:
            _ffmpeg_proc.terminate()
            _ffmpeg_proc.wait(timeout=3)
        except Exception:
            try:
                _ffmpeg_proc.kill()
            except Exception:
                pass
        _ffmpeg_proc = None
    with _udp_last_frame_lock:
        _udp_last_jpeg = b""
        _udp_has_frame = False


def _udp_receiver_loop():
    """Use ffmpeg to decode H264 UDP stream → raw RGB frames → JPEG."""
    global _udp_last_jpeg, _udp_has_frame, _ffmpeg_proc, _udp_receiver_running
    import subprocess as sp
    import shutil

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("[C2-VIDEO] ffmpeg not found — install ffmpeg to decode UDP forward stream")
        _udp_receiver_running = False
        return

    width, height = 960, 720  # Tello default resolution
    frame_size = width * height * 3  # RGB24

    cmd = [
        ffmpeg_bin,
        "-y",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-framedrop",
        "-probesize", "5000000",
        "-analyzeduration", "5000000",
        "-i", f"udp://0.0.0.0:{VIDEO_UDP_FORWARD_PORT}?overrun_nonfatal=1&fifo_size=50000000",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-an",
        "-sn",
        "-vf", f"scale={width}:{height}",
        "pipe:1",
    ]
    print(f"[C2-VIDEO] Starting ffmpeg on port {VIDEO_UDP_FORWARD_PORT}...")
    try:
        _ffmpeg_proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=frame_size * 2)
    except Exception as e:
        print(f"[C2-VIDEO] ffmpeg failed to start: {e}")
        _udp_receiver_running = False
        return

    # Log stderr in a separate thread so we can see ffmpeg errors
    def _log_stderr():
        for line in _ffmpeg_proc.stderr:
            txt = line.decode(errors="replace").rstrip()
            if txt:
                print(f"[C2-FFMPEG] {txt}")
    threading.Thread(target=_log_stderr, daemon=True, name="ffmpeg-stderr").start()

    frame_count = 0
    buf = b""
    while _udp_receiver_running:
        try:
            to_read = frame_size - len(buf)
            chunk = _ffmpeg_proc.stdout.read(to_read)
            if not chunk:
                # ffmpeg exited — check if it's still running
                rc = _ffmpeg_proc.poll()
                if rc is not None:
                    print(f"[C2-VIDEO] ffmpeg exited with code {rc}")
                    break
                time.sleep(0.01)
                continue
            buf += chunk
            if len(buf) < frame_size:
                continue
            # We have a full frame (BGR24)
            frame_data = buf[:frame_size]
            buf = buf[frame_size:]
            if HAS_CV2:
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
                ok, jpg_buf = cv2.imencode(".jpg", frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, VIDEO_JPEG_QUALITY])
                if ok:
                    with _udp_last_frame_lock:
                        _udp_last_jpeg = jpg_buf.tobytes()
                        _udp_has_frame = True
                    frame_count += 1
                    if frame_count == 1:
                        print("[C2-VIDEO] First frame decoded successfully")
        except Exception as e:
            if _udp_receiver_running:
                print(f"[C2-VIDEO] Error: {e}")
            break

    _udp_receiver_running = False
    if _ffmpeg_proc:
        try:
            _ffmpeg_proc.terminate()
        except Exception:
            pass
    print(f"[C2-VIDEO] Receiver stopped ({frame_count} frames decoded)")








# ─── Latency ping (C2 → flight controller + flight controller → drone) ──────
















# ── Positioning subsystem proxy ───────────────────────────────────────────────













# ── Calibration flight proxies ────────────────────────────────────────
# Operator workflow:
#   1. Place drone in arena centre
#   2. Click "Start Calibration Flight"  →  POST /proxy/calibration/start
#   3. Poll /proxy/calibration/status every 500 ms to drive progress bar
#   4. On completion, the matched flight-log + video show up in the
#      regular Flight Logs panel — user downloads them and uploads to
#      Claude for analysis.
# Any drone in the fleet may be the calibration subject; target is
# passed via `drone_id` query param. Default: the active drone.

def _calib_target_base(drone_id: str | None):
    """Return the base URL for the target drone's FC, or None."""
    did = (drone_id or "").strip() or active_drone_id
    info = DRONES.get(str(did)) if did else None
    return did, (info or {}).get("base")










# ── Arena configuration proxy ─────────────────────────────────────────────────

def _pi_arena_to_js(d: dict) -> dict:
    """Convert Pi API flat arena format → JS-expected nested format.

    Pi API returns:
      {arena_width_m, arena_height_m, marker_size_m,
       markers: [{id, x, y, z, wall, label?}, ...]}

    JS expects:
      {arena: {width_m, depth_m, height_min_m, height_max_m},
       marker_size_m,
       markers: {"0": {pos:[x,y,z], wall, label?}, ...}}
    """
    out: dict = {"ok": d.get("ok", True)}
    out["arena"] = {
        "width_m":      d.get("arena_width_m", 20.0),
        "depth_m":      d.get("arena_height_m", 10.0),
        "height_min_m": d.get("arena_height_min_m", -1.0),
        "height_max_m": d.get("arena_height_max_m",  1.0),
    }
    out["marker_size_m"] = d.get("marker_size_m", 0.5)
    markers_dict: dict = {}
    for m in (d.get("markers") or []):
        mid = str(m.get("id", "?"))
        entry: dict = {
            "pos": [
                float(m.get("x", 0)),
                float(m.get("y", 0)),
                float(m.get("z", 0)),
            ],
            "wall": m.get("wall", "front"),
        }
        if "label" in m:
            entry["label"] = m["label"]
        markers_dict[mid] = entry
    out["markers"] = markers_dict
    # Pass through target-box metadata so the C2 UI can render team labels
    out["target_marker_size_m"] = d.get("target_marker_size_m", 0.19)
    out["target_teams"] = d.get("target_teams") or []
    out["target_overrides"] = d.get("target_overrides") or []
    return out


def _js_arena_to_pi(data: dict) -> dict:
    """Convert JS POST format → Pi API format.

    JS sends:
      {arena: {width_m, depth_m, height_min_m, height_max_m},
       marker_size_m,
       markers: {"0": {pos:[x,y,z], wall, label?}, ...}}

    Pi API expects:
      {arena_width_m, arena_height_m, marker_size_m,
       markers: [{id, x, y, z, wall}, ...]}
    """
    out: dict = {}
    arena = data.get("arena") or {}
    if arena.get("width_m") is not None:
        out["arena_width_m"] = float(arena["width_m"])
    if arena.get("depth_m") is not None:
        out["arena_height_m"] = float(arena["depth_m"])
    if data.get("marker_size_m") is not None:
        out["marker_size_m"] = float(data["marker_size_m"])
    raw_markers = data.get("markers")
    if isinstance(raw_markers, dict):
        arr = []
        for mid_str, m in raw_markers.items():
            pos = m.get("pos") or [0, 0, 0]
            entry: dict = {
                "id":   int(mid_str),
                "x":    float(pos[0]) if len(pos) > 0 else 0.0,
                "y":    float(pos[1]) if len(pos) > 1 else 0.0,
                "z":    float(pos[2]) if len(pos) > 2 else 0.0,
                "wall": m.get("wall", "front"),
            }
            if "label" in m:
                entry["label"] = m["label"]
            arr.append(entry)
        arr.sort(key=lambda e: e["id"])
        out["markers"] = arr
    return out














# ─── ArUco Seek (multi-drone hover-in-front-of-marker) ─────────────────────


def _aruco_resolve(drone_id: str | None):
    """Return observer for the given id, or the active drone if omitted."""
    did = str(drone_id) if drone_id else active_drone_id
    obs = aruco_fleet.get(did)
    if obs is None:
        return None, did
    return obs, did








def asdict_hover_defaults():
    from dataclasses import asdict
    return asdict(AsHoverParams())




# ── Camera-faces-arena-centre toggle ──────────────────────────────────
# Fleet-wide switch. When enabled, every observer's "face override" is
# set to (0, arena_depth/2) so any mission waypoint with no explicit
# face target automatically aims the drone's camera at the arena
# centre — maximising marker visibility, which means better fused
# arena-frame position.
#
# The toggle lives on the C2 so the UI can flip it, the value gets
# broadcast to all observers, and capture_targets mission etc. pick
# it up on the next set_waypoint(...) call.
_camera_face_center_lock = threading.Lock()
_camera_face_center_enabled: bool = os.getenv("C2_CAMERA_FACE_CENTER", "1") not in {"0", "false", "False"}
_camera_face_center_xy: tuple = (0.0, 5.4)   # default arena centre (20×10.8 m)


def _apply_camera_face_to_fleet():
    """Push the current camera-face setting to every observer."""
    with _camera_face_center_lock:
        xy = _camera_face_center_xy if _camera_face_center_enabled else None
    for d_id, o in aruco_fleet._obs.items():
        try:
            o.set_camera_face_override(xy)
        except Exception as e:
            print(f"[CAMERA_FACE] drone {d_id} set failed: {e}")


# Apply at import time so observers created by configure() already
# have the right override before any mission runs.
_apply_camera_face_to_fleet()


















def _aruco_require_live(obs):
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    if obs.mode != "live":
        return jsonify(ok=False,
                       error=f"refused — observer mode is '{obs.mode}', switch to 'live' first"), 409
    return None












# ─── Special Missions (multi-drone coordinated flight) ────────────────────


def _parse_marker_list(raw) -> list[int]:
    """Parse '1-12' / '1,2,3,5-7' / [1,2,3] into a sorted unique int list."""
    if isinstance(raw, list):
        return sorted({int(x) for x in raw if str(x).strip()})
    if not isinstance(raw, str):
        return []
    out: set[int] = set()
    for tok in raw.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, _, b = tok.partition("-")
            try:
                a_i, b_i = int(a), int(b)
                if a_i > b_i:
                    a_i, b_i = b_i, a_i
                for v in range(a_i, b_i + 1):
                    out.add(v)
            except ValueError:
                continue
        else:
            try:
                out.add(int(tok))
            except ValueError:
                continue
    return sorted(out)




# ── Position-tracker presets ──────────────────────────────────────────
# Mirror of the mission-preset system but for Position Tracker tuning.
# Stored on the C2 (controller/position_presets.json) because:
#   - presets should be shared across the whole fleet (one drone's
#     position setup usually applies to all)
#   - loading a preset fans out to every Pi via /proxy/position/config
#   - C2 is the single authority for operator-facing config
POSITION_PRESETS_PATH = Path(os.getenv(
    "POSITION_PRESETS_PATH",
    str(Path(__file__).with_name("position_presets.json"))))
_position_presets_lock = threading.Lock()

_POSITION_PRESETS_DEFAULTS: dict = {
    # Rock-solid defaults derived from field-log analysis (flight 19:11:40
    # tracked cleanly with 96 % fresh fixes and 0.06 m median step). The
    # two tightenings over that baseline are min_ref_weight=0.15 (reject
    # low-quality single detections) and max_pose_jump_m=3.0 (kills the
    # occasional 10+ m single-marker solvePnP mirror-pose glitches).
    # This preset is also what gets applied when auto_positioning=True.
    "Claude": {
        "detect_profile":     "balanced",
        "fov_deg":             69,
        "imu_weight":          0.30,
        "latency_comp_s":      0.20,
        "enable_kalman_filter": True,
        "marker_size_m":       0.50,
        "top_k_markers":       4,
        "outlier_reject_m":    1.5,
        "distance_scale":      1.0,
        "pose_hold_sec":       0.5,
        "min_ref_count":       1,
        "min_ref_weight":      0.15,
        "meas_blend_min":      0.20,
        "meas_blend_max":      0.70,
        "vel_blend":           0.30,
        "max_state_dt":        0.5,
        "kalman_process_var":  5e-4,
        "kalman_meas_var":     0.15,
        "imu_lowpass_hz":      5.0,
        "seen_hold_s":         0.6,
        "max_pose_jump_m":     3.0,
    },
    "balanced": {
        "detect_profile":     "balanced",
        "fov_deg":             69,
        "imu_weight":          0.30,
        "latency_comp_s":      0.20,
        "enable_kalman_filter": True,
        "marker_size_m":       0.50,
        "top_k_markers":       0,
        "outlier_reject_m":    2.5,
        "distance_scale":      1.0,
        "pose_hold_sec":       0.8,
        "min_ref_count":       1,
        "min_ref_weight":      0.0,
        "meas_blend_min":      0.35,
        "meas_blend_max":      0.85,
        "vel_blend":           0.25,
        "max_state_dt":        1.0,
        "kalman_process_var":  1e-3,
        "kalman_meas_var":     1e-1,
        "imu_lowpass_hz":      5.0,
        "seen_hold_s":         0.6,
    },
    "smooth": {
        "detect_profile":     "balanced",
        "fov_deg":             69,
        "imu_weight":          0.50,
        "enable_kalman_filter": True,
        "kalman_process_var":  5e-4,
        "kalman_meas_var":     2e-1,
        "imu_lowpass_hz":      3.0,
        "pose_hold_sec":       1.2,
        "meas_blend_min":      0.20,
        "meas_blend_max":      0.60,
    },
    "responsive": {
        "detect_profile":     "sensitive",
        "fov_deg":             69,
        "imu_weight":          0.15,
        "enable_kalman_filter": True,
        "kalman_process_var":  5e-3,
        "kalman_meas_var":     5e-2,
        "imu_lowpass_hz":      15.0,
        "pose_hold_sec":       0.4,
        "meas_blend_min":      0.60,
        "meas_blend_max":      0.95,
    },
}


def _load_position_presets() -> dict:
    """Load operator-saved presets + auto-inject any missing built-ins.

    The file on disk is the source of truth for anything the operator
    has changed. We only add built-in presets that don't already exist
    (so user edits are never overwritten). This is how the "Claude"
    preset reaches existing installs without us having to rm the file.
    """
    with _position_presets_lock:
        if not POSITION_PRESETS_PATH.exists():
            try:
                POSITION_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
                POSITION_PRESETS_PATH.write_text(
                    json.dumps(_POSITION_PRESETS_DEFAULTS, indent=2))
            except Exception as e:
                print(f"[PRESETS] position seed write failed: {e}")
            return json.loads(json.dumps(_POSITION_PRESETS_DEFAULTS))
        try:
            data = json.loads(POSITION_PRESETS_PATH.read_text())
        except Exception as e:
            print(f"[PRESETS] position load failed ({e}) — using defaults")
            return json.loads(json.dumps(_POSITION_PRESETS_DEFAULTS))
        # Auto-inject new built-ins without touching existing (possibly
        # edited) presets. Write back to disk so the injection is stable.
        added = []
        for name, params in _POSITION_PRESETS_DEFAULTS.items():
            if name not in data:
                data[name] = json.loads(json.dumps(params))
                added.append(name)
        if added:
            try:
                tmp = POSITION_PRESETS_PATH.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(data, indent=2))
                tmp.replace(POSITION_PRESETS_PATH)
                print(f"[PRESETS] injected missing built-in presets: {added}")
            except Exception as e:
                print(f"[PRESETS] inject-write failed: {e}")
        return data


def _save_position_presets(data: dict):
    with _position_presets_lock:
        tmp = POSITION_PRESETS_PATH.with_suffix(".json.tmp")
        POSITION_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(POSITION_PRESETS_PATH)










# ── Mission presets ───────────────────────────────────────────────────
# Each special-mission type has its own JSON parameter block. Operators
# can edit it as code in the UI and save named presets here. Storage is
# a single JSON file on the C2 (the mission runner lives here, not on
# the FC), loaded lazily and written atomically on save/delete.
MISSION_PRESETS_PATH = Path(os.getenv(
    "MISSION_PRESETS_PATH",
    str(Path(__file__).with_name("mission_presets.json"))))
_mission_presets_lock = threading.Lock()


# Built-in starter presets — so the editor never starts empty. Users
# can override or delete them; they re-appear on missing-file.
_MISSION_PRESETS_DEFAULTS: dict = {
    "scan_all": {
        "default": {
            "drone_ids": ["1"],
            "target_markers": "1-12",
            "hover_seconds": 1.5,
            "approach_tolerance_m": 0.30,
            "approach_skew_tol": 0.12,
            "approach_err_x_tol": 0.15,
            "auto_takeoff": False,
        },
    },
    "capture_targets": {
        "default": {
            "drone_ids": ["1"],
            "target_boxes": [
                {"id": 1, "x": -7.0, "y": 2.5, "home_team": "red"},
                {"id": 2, "x": -5.0, "y": 5.4, "home_team": "red"},
                {"id": 3, "x": -7.0, "y": 8.3, "home_team": "red"},
                {"id": 4, "x":  7.0, "y": 2.5, "home_team": "blue"},
                {"id": 5, "x":  5.0, "y": 5.4, "home_team": "blue"},
                {"id": 6, "x":  7.0, "y": 8.3, "home_team": "blue"},
            ],
            "home_xy":        [0.0, 9.0],
            "arena_face_xy":  [0.0, 5.4],
            "hover_above_m":  1.5,
            "hover_seconds":  4.0,
            "nav_tol_xy_m":   0.3,
            "auto_takeoff":   False,
        },
    },
}


def _load_mission_presets() -> dict:
    """Load presets, seeding from defaults on first access."""
    with _mission_presets_lock:
        if not MISSION_PRESETS_PATH.exists():
            try:
                MISSION_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
                MISSION_PRESETS_PATH.write_text(
                    json.dumps(_MISSION_PRESETS_DEFAULTS, indent=2))
            except Exception as e:
                print(f"[PRESETS] seed write failed: {e}")
            return json.loads(json.dumps(_MISSION_PRESETS_DEFAULTS))   # deep copy
        try:
            return json.loads(MISSION_PRESETS_PATH.read_text())
        except Exception as e:
            print(f"[PRESETS] load failed ({e}) — using in-memory defaults")
            return json.loads(json.dumps(_MISSION_PRESETS_DEFAULTS))


def _save_mission_presets(data: dict):
    """Atomic write — avoids truncated file on crash mid-write."""
    with _mission_presets_lock:
        tmp = MISSION_PRESETS_PATH.with_suffix(".json.tmp")
        MISSION_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(MISSION_PRESETS_PATH)














# ═══════════════════════════════════════════════════════════════════════
# Scan & Capture Targets — dynamic-discovery variant of capture_targets
#
# Flow:
#   1. Ensure drone is airborne (takeoff if needed)
#   2. Rotate the drone through a full 360° in 60° steps, with a ~2 s
#      dwell after each step so the positioner can accumulate a clean
#      target-position estimate for whatever it can see.
#   3. Build a target_boxes list from the accumulated ArUco detections
#      (IDs 31-36 = Blue, 41-46 = Red — SDC26 convention). Order the
#      boxes by nearest-neighbour from the drone's current position so
#      the flight path is short.
#   4. Hand the list off to mission_manager.start_capture_all_targets()
#      — from that point the standard capture mission takes over: it
#      navigates to each target using ArUco-fused position, hovers
#      `hover_seconds` over the box centre, then continues.
#
# Runs in a background thread so the HTTP request returns immediately.
# ═══════════════════════════════════════════════════════════════════════

_scan_cap_state: dict = {
    "active":        False,
    "phase":         "idle",  # "takeoff"|"scanning"|"capture"|"done"|"error"|"aborted"
    "drone_id":      None,
    "step_name":     "idle",
    "n_detected":    0,
    "targets_found": {},   # {tid: [x, y, z]}
    "target_boxes":  [],   # [{id, x, y, home_team}, ...] (ordered visit list)
    "started_at":    0.0,
    "ended_at":      0.0,
    "elapsed_s":     0.0,
    "last_error":    None,
    "result":        None,
}
_scan_cap_lock = threading.Lock()
_scan_cap_abort = threading.Event()


def _scan_cap_team_for(tid: int) -> str | None:
    """Return 'blue' / 'red' for the SDC26 ID ranges, else None."""
    if 31 <= tid <= 36:
        return "blue"
    if 41 <= tid <= 46:
        return "red"
    return None


def _scan_cap_nn_order(start_xy, targets: dict) -> list:
    """Nearest-neighbour ordering over XY. `targets` is {tid:[x,y,z]}.
    Returns [{id, x, y, z, home_team}, ...] in visit order. Z is the
    detected marker height — clamped to [0, 2.5] m so an over-reported
    detection (e.g. from a mis-sized marker template) can't make the
    capture mission fly absurdly high. Pass z=0 when we don't trust it."""
    # Keep full (x, y, z) so we can forward the detected Z per box.
    remaining = {int(tid): [float(p[0]), float(p[1]), float(p[2])]
                 for tid, p in targets.items()}
    cx, cy = float(start_xy[0]), float(start_xy[1])
    order = []
    while remaining:
        best_tid = min(
            remaining,
            key=lambda t: (remaining[t][0] - cx) ** 2 + (remaining[t][1] - cy) ** 2,
        )
        bx, by, bz = remaining.pop(best_tid)
        # Clamp Z to a physically-reasonable range for a box on the floor.
        # Anything outside this range is almost certainly a detection
        # error (e.g. mirror-pose ambiguity) — fall back to z=0 so the
        # hover altitude becomes pure hover_above_m.
        if bz < -0.5 or bz > 2.5:
            bz = 0.0
        else:
            bz = max(0.0, bz)
        order.append({
            "id":        best_tid,
            "x":         round(bx, 3),
            "y":         round(by, 3),
            "z":         round(bz, 3),
            "home_team": _scan_cap_team_for(best_tid),
        })
        cx, cy = bx, by
    return order


def _scan_cap_set(**kwargs):
    with _scan_cap_lock:
        _scan_cap_state.update(kwargs)
        if _scan_cap_state.get("started_at"):
            _scan_cap_state["elapsed_s"] = time.time() - _scan_cap_state["started_at"]


def _scan_and_capture_thread(
    drone_ids: list,
    rotation_deg: int,       # kept for backward compat, unused in pursuit mode
    rotation_steps: int,     # kept for backward compat, unused in pursuit mode
    dwell_s: float,          # kept for backward compat, unused in pursuit mode
    hover_seconds: float,
    hover_above_m: float,
    home_xy: tuple,
    arena_face_xy: tuple,
    nav_tol_xy_m: float,
    altitude_stack_m: float,
    min_separation_m: float,
):
    # Pursuit mode: no pre-scan, no pre-planned waypoints. Takeoff (if
    # needed) for every selected drone, then hand control to the reactive
    # DirectTargetPursuitMission — each drone rotates slowly to search,
    # and as soon as a valid SDC26 target appears in its camera, it
    # pursues that target and hovers over it.
    if not drone_ids:
        _scan_cap_set(active=False, phase="error", result="error",
                      last_error="no drones selected", ended_at=time.time())
        return
    try:
        # ── Phase 1: takeoff every selected drone that's still grounded
        _scan_cap_set(phase="takeoff", step_name="Ensuring all selected drones are airborne")
        for did in drone_ids:
            if _scan_cap_abort.is_set():
                raise RuntimeError("aborted before takeoff")
            info = DRONES.get(did) or {}
            b = (info.get("base") or "").rstrip("/")
            if not b:
                print(f"[SCAN_CAP] drone {did} has no base URL — skipping")
                continue
            tel = {}
            try:
                r = _http_session.get(f"{b}/api/telemetry", timeout=TIMEOUT_FAST)
                if r.ok:
                    tel = r.json() or {}
            except Exception:
                pass
            if tel.get("flying"):
                _scan_cap_set(step_name=f"Drone {did}: already airborne")
                continue
            _scan_cap_set(step_name=f"Drone {did}: taking off")
            try:
                r = _http_session.post(f"{b}/api/takeoff", json={},
                                        timeout=TIMEOUT_SLOW)
                body = r.json() if r.ok and r.content else {}
                if not (r.ok and body.get("ok")):
                    raise RuntimeError(
                        f"takeoff failed: HTTP {r.status_code} {body.get('error','')}")
            except Exception as te:
                raise RuntimeError(f"drone {did} takeoff: {te}")
            # Stagger takeoffs by ~1 s so the drones don't all spin up in lockstep.
            time.sleep(1.0)
        # Brief settle before handing off — positioner needs a couple
        # of seconds to latch onto arena markers post-takeoff.
        _scan_cap_set(step_name="Settle before pursuit")
        for _ in range(40):   # ~4 s
            if _scan_cap_abort.is_set(): break
            time.sleep(0.1)

        # ── Phase 2: hand off to DirectTargetPursuitMission ──────
        # No pre-scan, no pre-planned visit list. Each drone slowly
        # rotates to search; the instant a valid SDC26 target is spotted
        # the drone flies directly toward it and hovers.
        _scan_cap_set(phase="pursuit",
                      step_name="Reactive pursuit active — drones chase targets live")
        ok, msg = mission_manager.start_direct_target_pursuit(
            drone_ids=drone_ids,
            hover_above_m=hover_above_m,
            hover_seconds=hover_seconds,
            nav_tol_xy_m=nav_tol_xy_m,
            altitude_stack_m=altitude_stack_m,
            min_separation_m=min_separation_m,
            arena_face_xy=tuple(arena_face_xy),
        )
        if not ok:
            raise RuntimeError(f"pursuit mission refused to start: {msg}")
        _scan_cap_set(phase="done", result="ok",
                      step_name="Handed off to pursuit mission",
                      ended_at=time.time())
        log_command("scan_and_capture_done", {
            "drone_ids": drone_ids,
            "mode": "direct_target_pursuit",
            "altitude_stack_m": altitude_stack_m,
            "min_separation_m": min_separation_m,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        aborted = _scan_cap_abort.is_set()
        _scan_cap_set(
            phase="aborted" if aborted else "error",
            result="aborted" if aborted else "error",
            last_error=str(e),
            ended_at=time.time(),
        )
    finally:
        _scan_cap_set(active=False)












# ─── Environment & Wi-Fi control — pass-through to active drone ─────────────











# ── Magnetometer (Anafi) ────────────────────────────────────────────────────
# Parrot Anafi requires a figure-8 magnetometer calibration whenever the drone
# is moved between locations or re-powered. unified_api_server.py exposes the
# raw GET /api/magneto and POST /api/magneto/calibrate; here we expose a
# higher-level /proxy/magneto/recalibrate that drives the full cycle.





def _parse_magneto_axes(status: str | None) -> dict:
    """Extract per-axis calibration bits from a status string like
    'REQUIRED, axes=x0y1z0, in-progress'. Returns {x, y, z, failed, done,
    in_progress, required} with sensible defaults when the drone hasn't
    reported yet."""
    s = (status or "").lower()
    out = {"x": None, "y": None, "z": None,
           "failed": False, "done": False,
           "in_progress": False, "required": None}
    compact = s.replace(" ", "")
    import re
    m = re.search(r"axes=x(\d)y(\d)z(\d)", compact)
    if m:
        out["x"] = int(m.group(1))
        out["y"] = int(m.group(2))
        out["z"] = int(m.group(3))
    out["failed"] = "failed" in s
    out["done"] = "all-axes-ok" in s or (
        out["x"] == 1 and out["y"] == 1 and out["z"] == 1
    )
    out["in_progress"] = "in-progress" in s
    if "not-required" in s:
        out["required"] = False
    elif "required" in s:
        out["required"] = True
    return out


def _magneto_cycle(timeout_s: float, poll_s: float):
    """Generator that drives one full magnetometer recalibration cycle.

    Yields dicts of shape:
      {"kind": "step", "step": <name>, "ok": bool, ...extra}
      {"kind": "status", "status": str, "axes": {x,y,z,...}}   # during polling
      {"kind": "final", "ok": bool, "pre": ..., "post": ..., ...}

    The caller is responsible for transport (blocking JSON or SSE)."""
    steps: list[dict] = []

    def step(name: str, ok: bool, **info):
        entry = {"kind": "step", "step": name, "ok": ok, **info}
        steps.append({k: v for k, v in entry.items() if k != "kind"})
        print(f"[MAGNETO] {'OK' if ok else 'FAIL'} {name} {info}")
        return entry

    # 1) Heartbeat
    try:
        hb = pi_get("/api/heartbeat", timeout=TIMEOUT_STATUS).json()
    except Exception as e:
        yield step("heartbeat", False, error=str(e))
        yield {"kind": "final", "ok": False, "fatal": "unreachable",
               "error": f"drone unreachable: {e}", "steps": steps}
        return
    connected = bool(hb.get("connected"))
    flying = bool(hb.get("flying"))
    yield step("heartbeat", connected and not flying,
               connected=connected, flying=flying,
               drone_type=hb.get("drone_type"))
    if not connected:
        yield {"kind": "final", "ok": False, "fatal": "not_connected",
               "error": "drone not connected", "steps": steps}
        return
    if flying:
        yield {"kind": "final", "ok": False, "fatal": "flying",
               "error": "refuse to recalibrate while flying — land first",
               "steps": steps}
        return

    # 2) Pre-calibration snapshot
    try:
        pre = pi_get("/api/magneto", timeout=TIMEOUT_STATUS).json()
    except Exception as e:
        pre = {"ok": False, "error": str(e)}
    yield step("pre_status", bool(pre.get("ok")),
               status=pre.get("status"), required=pre.get("required"))

    # 3) Trigger calibration
    try:
        r = pi_post("/api/magneto/calibrate", timeout=TIMEOUT_CMD)
        start = r.json() if r.headers.get("Content-Type",
                                          "").startswith("application/json") else {}
    except Exception as e:
        yield step("start", False, error=str(e))
        yield {"kind": "final", "ok": False, "fatal": "start_failed",
               "error": f"start failed: {e}", "steps": steps, "pre": pre}
        return
    started = bool(start.get("ok"))
    yield step("start", started, http_status=r.status_code,
               message=start.get("message"), error=start.get("error"))
    if not started:
        yield {"kind": "final", "ok": False, "fatal": "start_refused",
               "error": start.get("error", "calibration not started"),
               "steps": steps, "pre": pre, "start": start}
        return

    # 4) Poll status — emit per-axis progress as it flips.
    deadline = time.time() + timeout_s
    post = {"ok": False, "status": None, "required": None}
    finished = False
    failed = False
    last_axes = None
    while time.time() < deadline:
        try:
            post = pi_get("/api/magneto", timeout=TIMEOUT_STATUS).json()
        except Exception as e:
            post = {"ok": False, "error": str(e)}
        axes = _parse_magneto_axes(post.get("status"))
        if axes != last_axes:
            yield {"kind": "status", "status": post.get("status"),
                   "axes": axes, "required": post.get("required")}
            last_axes = axes
        if axes["failed"]:
            failed = True
            break
        if axes["done"]:
            finished = True
            break
        time.sleep(poll_s)

    timed_out = not (finished or failed)
    yield step("poll", finished and not failed,
               final_status=post.get("status"),
               required=post.get("required"), timed_out=timed_out)

    ok = finished and not failed
    if ok:
        msg = "magnetometer calibrated"
    elif failed:
        msg = "magnetometer calibration FAILED — retry the figure-8 dance"
    else:
        msg = ("magnetometer calibration timed out — perform the figure-8 "
               "dance around each axis and retry")
    yield {"kind": "final", "ok": ok, "pre": pre, "post": post,
           "steps": steps, "failed": failed, "timed_out": timed_out,
           "message": msg}







# ── Wire blueprints (Phase 4) ─────────────────────────────
# Importing each api.<bp> module triggers @bp_X.<method>(...)
# decorators inside it, which attach the routes to the blueprint
# object. _register_blueprints(app) then mounts every blueprint
# on the Flask app.
from controller_modularized.api import (  # noqa: E402
    system    as _bp_mod_system,
    drones    as _bp_mod_drones,
    flight    as _bp_mod_flight,
    safety    as _bp_mod_safety,
    camera    as _bp_mod_camera,
    telemetry as _bp_mod_telemetry,
    settings  as _bp_mod_settings,
    magneto   as _bp_mod_magneto,
    logs      as _bp_mod_logs,
    arena     as _bp_mod_arena,
    aruco     as _bp_mod_aruco,
    missions  as _bp_mod_missions,
)
_register_blueprints(app)

def main():
    print("[REMOTE UI] Starting server...")
    print(f"[REMOTE UI] URL: http://{HTTP_HOST}:{HTTP_PORT}")
    print(f"[REMOTE UI] PI_API_BASE={PI_BASE}")
    print(f"[REMOTE UI] timeouts: cmd={TIMEOUT_CMD}s status={TIMEOUT_STATUS}s")
    print(f"[REMOTE UI] HTTP session pool: connections=8 maxsize=16 keep-alive=on")
    print(f"[REMOTE UI] Telemetry: SSE push (fallback poll at 2s)")
    print(f"[REMOTE UI] Heartbeat: parallel fan-out to {len(DRONES)} drones")
    print("[REMOTE UI] Ready (waiting for browser requests)")
    app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
