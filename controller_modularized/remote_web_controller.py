import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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


def _read_git_revision() -> dict:
    """Return a small dict describing the repo state at C2 startup.
    Safe under: no .git dir, detached HEAD, git not on PATH.
    Used to stamp every flight-log header for post-flight traceability."""
    import subprocess
    info = {}
    repo = Path(__file__).resolve().parent.parent
    def _run(args):
        try:
            r = subprocess.run(args, cwd=repo, capture_output=True,
                                text=True, timeout=2)
            if r.returncode == 0:
                return r.stdout.strip()
        except Exception:
            pass
        return ""
    sha     = _run(["git", "rev-parse", "HEAD"])
    short   = _run(["git", "rev-parse", "--short", "HEAD"])
    branch  = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty   = _run(["git", "status", "--porcelain"])
    subject = _run(["git", "log", "-1", "--pretty=%s"])
    info["sha"] = sha
    info["short_sha"] = short
    info["branch"] = branch
    info["dirty"] = bool(dirty)
    info["subject"] = subject
    return info


_GIT_REVISION = _read_git_revision()
print(f"[GIT] {_GIT_REVISION.get('branch') or '?'} @ {_GIT_REVISION.get('short_sha') or '?'}"
      f"{' (dirty)' if _GIT_REVISION.get('dirty') else ''} "
      f"— {_GIT_REVISION.get('subject') or ''}")

# ── C2 version string (matches the FC's CODE_VERSION format) ──────
# Used by /proxy/fc_version to flag a mismatch when the FC has been
# started from an older build than the C2.
C2_CODE_VERSION = "2026-04-24-cr (FC version endpoint + C2/FC mismatch check)"


class FlightLogger:
    """Per-drone flight logger.

    Runs one background thread that polls /api/telemetry + /api/position +
    /proxy/aruco/state for every configured drone at FLIGHT_LOG_HZ. On the
    rising edge of ``flying`` (or airborne detected by height_cm > 30) a
    new JSONL file is opened; on the falling edge it's closed. Commands
    logged via ``log_command()`` are funnelled into the active files so
    every per-flight file is a complete, timestamped audit trail of:
      - telemetry (battery, attitude, velocity, ceiling state, ...)
      - fused arena position (x, y, z, dir, vel, stale)
      - visible ArUco markers (seen + reference lists)
      - every command sent (takeoff, land, rc, mission-start, pause, ...)

    Files land in ``<FLIGHT_LOG_DIR>/flight_<timestamp>_drone-<id>.jsonl``.
    """

    def __init__(self, drones: dict, session, log_dir: Path, hz: float = 5.0):
        self.drones  = drones
        self.session = session
        self.log_dir = log_dir
        self.period  = max(0.1, 1.0 / float(hz or 5.0))
        self._flights: dict[str, dict] = {}
        self._lock    = threading.Lock()
        self._running = False
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True, name="flight-logger")
        t.start()
        print(f"[FLIGHT_LOG] started, dir={self.log_dir}, period={self.period:.2f}s")

    def stop(self):
        self._running = False
        with self._lock:
            for did in list(self._flights.keys()):
                self._close_unlocked(did, reason="shutdown")

    # Public — called from log_command
    def record_command(self, drone_id, event: str, payload: dict | None):
        """Append a command record to the active flight log(s). If
        drone_id is None we broadcast to every active flight (fleet-wide
        commands like PAUSE_ALL / LAND_ALL apply to all airborne drones)."""
        if not self._running:
            return
        did = str(drone_id) if drone_id else None
        with self._lock:
            targets = [did] if did and did in self._flights else list(self._flights.keys())
            for d in targets:
                flt = self._flights.get(d)
                if not flt:
                    continue
                self._write_unlocked(flt, {
                    "type":    "cmd",
                    "ts":      time.time(),
                    "drone_id": d,
                    "event":   event,
                    "payload": payload or {},
                })

    def list_files(self) -> list[dict]:
        """For /proxy/flight_logs — list all flight files with basic meta."""
        out = []
        try:
            for p in sorted(self.log_dir.glob("flight_*.jsonl"), reverse=True):
                try:
                    st = p.stat()
                    out.append({
                        "name": p.name,
                        "size": st.st_size,
                        "mtime": st.st_mtime,
                    })
                except Exception:
                    continue
        except Exception:
            pass
        return out

    def file_path(self, name: str) -> Path | None:
        """Resolve a filename to a path inside log_dir. Rejects path
        traversal attempts."""
        p = (self.log_dir / name).resolve()
        try:
            p.relative_to(self.log_dir)
        except ValueError:
            return None
        return p if p.exists() else None

    # ── internals ──
    def _loop(self):
        while self._running:
            t0 = time.time()
            try:
                self._tick()
            except Exception as e:
                print(f"[FLIGHT_LOG] tick error: {e}")
            dt = time.time() - t0
            time.sleep(max(0.02, self.period - dt))

    def _tick(self):
        for did, info in list(self.drones.items()):
            did  = str(did)
            base = (info or {}).get("base")
            if not base:
                continue
            # Skip HTTP polls entirely for drones whose WS is fully down —
            # otherwise each unreachable Pi burns 1.5 s per tick and the
            # whole fleet-logger thread falls behind at 5 Hz.
            cli = drone_ws.get(did) if 'drone_ws' in globals() else None
            if cli is not None:
                all_down = (not cli._ws_connected.get("telemetry") and
                            not cli._ws_connected.get("position") and
                            not cli._ws_connected.get("rc"))
                if all_down:
                    # Close any open flight for this drone (we can't log
                    # what we can't see) and move on.
                    with self._lock:
                        if did in self._flights:
                            self._close_unlocked(did, reason="unreachable")
                    continue
            try:
                tr  = self.session.get(f"{base.rstrip('/')}/api/telemetry", timeout=0.6)
                tel = tr.json() if tr.ok else {}
            except Exception:
                tel = {}
            try:
                pr  = self.session.get(f"{base.rstrip('/')}/api/position", timeout=0.3)
                pos = pr.json() if pr.ok else {}
            except Exception:
                pos = {}

            # Airborne detection: trust "flying" but fall back to height_cm
            flying   = bool(tel.get("flying"))
            height_cm = (tel.get("height_cm") or 0)
            airborne = flying or (height_cm and height_cm > 30)

            with self._lock:
                flt = self._flights.get(did)
                if airborne and flt is None:
                    self._open_unlocked(did, tel)
                    flt = self._flights.get(did)
                elif not airborne and flt is not None:
                    self._write_unlocked(flt, {
                        "type": "land", "ts": time.time(), "drone_id": did,
                        "telemetry": tel,
                    })
                    self._close_unlocked(did, reason="landed")
                    flt = None
                if flt is None:
                    continue
                # Active flight — emit a tick record.
                #
                # Visible markers: the ArUco observer exposes a `visible_ids`
                # list, but only when the observer thread is started (which
                # happens when ArUco Seek is armed or a mission is running).
                # During plain manual flight the observer is idle and that
                # list is empty. The per-drone position service on the Pi
                # runs its own detection pipeline and publishes `seen_markers`
                # in /api/position — use it as a fallback so the flight log
                # always reflects what the camera is actually seeing.
                vis_markers = []
                try:
                    obs = aruco_fleet.get(did)
                    if obs is not None:
                        st = obs.get_state()
                        vis_markers = st.get("visible_ids") or []
                except Exception:
                    pass
                if not vis_markers:
                    vis_markers = list(pos.get("seen_markers") or [])
                rec = {
                    "type":    "tick",
                    "ts":      time.time(),
                    "drone_id": did,
                    "telemetry": {k: tel.get(k) for k in (
                        "battery", "height_cm", "altitude_m", "flying",
                        "connected", "yaw", "pitch", "roll",
                        "vgx", "vgy", "vgz", "agx", "agy", "agz",
                        "ceiling_m", "ceiling_engaged", "ceiling_reason",
                        "state_age_s", "state_fresh",
                    ) if k in tel},
                    "position": pos.get("pos"),
                    "direction": pos.get("dir"),
                    "pos_vel": pos.get("vel"),
                    "pos_stale": pos.get("stale"),
                    "visible_markers": vis_markers,
                    "ref_markers":  pos.get("ref_markers") or [],
                    "seen_markers": pos.get("seen_markers") or [],
                }
                self._write_unlocked(flt, rec)

    def _open_unlocked(self, did: str, tel: dict):
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        name = (self.drones.get(did, {}).get("name") or did).replace(" ", "_")
        stem = f"flight_{ts}_drone-{did}_{name}"
        path = self.log_dir / f"{stem}.jsonl"
        fh = path.open("w", encoding="utf-8", buffering=1)  # line-buffered
        # Kick off video recording on the FC with a matching basename so the
        # .mp4 and the .jsonl travel together. Annotated mode so the recording
        # has the detected-marker overlay for post-flight review.
        #
        # The FC's /api/video/record/start self-starts MJPEG if it's off, so
        # we intentionally do NOT ping /api/video/start separately here —
        # that endpoint always calls video_stop_all() + video_start_mjpeg()
        # and would restart the decoder on every takeoff (dropping frames
        # and freezing the position tracker for several seconds). No-op
        # when the FC is unreachable — video is nice-to-have, log is the
        # essential record.
        video_name = f"{stem}.mp4"
        video_started = False
        video_err = None
        base = (self.drones.get(did, {}) or {}).get("base")
        if base:
            try:
                r = self.session.post(
                    f"{base.rstrip('/')}/api/video/record/start",
                    json={"filename": video_name, "raw": False},
                    timeout=2.0,
                )
                if r.ok:
                    body = r.json() if r.content else {}
                    if body.get("ok") is True:
                        video_started = True
                    else:
                        video_err = body.get("error") or "record/start returned ok=false"
                else:
                    video_err = f"HTTP {r.status_code}"
            except Exception as e:
                video_err = str(e)
                print(f"[FLIGHT_LOG] video record start failed: {e}")

        # Fetch the FC's version info so the flight log header captures
        # BOTH the C2 git sha (self) and the FC code_version/git_sha.
        # Short 1 s timeout — if the FC is down we'd rather have a
        # no-fc-version log than no log at all.
        fc_version = {}
        if base:
            try:
                vr = self.session.get(f"{base.rstrip('/')}/api/version",
                                       timeout=1.0)
                if vr.ok:
                    vdata = vr.json() or {}
                    fc_version = {
                        "code_version": vdata.get("code_version"),
                        "git_revision": vdata.get("git_revision") or {},
                    }
            except Exception as ve:
                print(f"[FLIGHT_LOG] FC version fetch failed: {ve}")
        header = {
            "type": "takeoff", "ts": time.time(), "drone_id": did,
            "drone_name": self.drones.get(did, {}).get("name"),
            "git_revision": _GIT_REVISION,         # C2 side
            "fc_version":   fc_version,            # FC side
            "drone_base": base,
            "video_filename": video_name if video_started else None,
            "telemetry": tel,
        }
        fh.write(json.dumps(header, default=str) + "\n")
        self._flights[did] = {
            "fh": fh, "path": path, "opened_at": time.time(), "records": 1,
            "stem": stem, "video_name": video_name if video_started else None,
        }
        if video_started:
            print(f"[FLIGHT_LOG] takeoff → {path}  + video → {video_name}")
        else:
            print(f"[FLIGHT_LOG] takeoff → {path}  (no video: {video_err or 'FC unreachable'})")

    def _close_unlocked(self, did: str, reason: str = "landed"):
        flt = self._flights.pop(did, None)
        if flt is None:
            return
        # Stop the matching video recording on the Pi (no-op if not started).
        base = (self.drones.get(did, {}) or {}).get("base")
        video_frames = None
        if base and flt.get("video_name"):
            try:
                r = self.session.post(
                    f"{base.rstrip('/')}/api/video/record/stop",
                    json={}, timeout=1.5,
                )
                if r.ok:
                    j = r.json()
                    video_frames = j.get("frames")
            except Exception as e:
                print(f"[FLIGHT_LOG] video record stop failed: {e}")
        try:
            dur = time.time() - flt["opened_at"]
            flt["fh"].write(json.dumps({
                "type": "close", "ts": time.time(), "reason": reason,
                "duration_s": round(dur, 2), "records": flt["records"],
                "video_filename": flt.get("video_name"),
                "video_frames": video_frames,
            }) + "\n")
            flt["fh"].close()
        except Exception:
            pass
        print(f"[FLIGHT_LOG] closed {flt['path']} "
              f"(reason={reason}, records={flt['records']}, "
              f"duration={time.time() - flt['opened_at']:.1f}s"
              + (f", video={video_frames} frames" if video_frames is not None else "")
              + ")")

    def _write_unlocked(self, flt: dict, rec: dict):
        try:
            flt["fh"].write(json.dumps(rec, default=str) + "\n")
            flt["records"] += 1
        except Exception as e:
            print(f"[FLIGHT_LOG] write error: {e}")


flight_logger = FlightLogger(DRONES, _http_session, FLIGHT_LOG_DIR, FLIGHT_LOG_HZ)
flight_logger.start()


# ── Server-side heartbeat loop ─────────────────────────────────────────
# The Pi's watchdog auto-lands if it sees no remote activity for
# REMOTE_TIMEOUT_S (default 2 s). Historically the browser polled
# /proxy/heartbeat at 2 Hz to keep that alive — a complete waste of
# the browser's HTTP connection pool since the heartbeat has no UI
# purpose. Now fired from a background thread on the C2 itself, once
# per second per drone, skipping drones whose WS is fully down. The
# browser doesn't have to issue ANY heartbeat traffic.
_HEARTBEAT_INTERVAL_S = 1.0


def _heartbeat_loop():
    """Ping every reachable Pi's /api/heartbeat at HEARTBEAT_INTERVAL_S.
    Runs as a daemon thread so it exits with the process. Per-drone
    failures are swallowed silently — the Pi's watchdog only needs
    SOME successful heartbeat per REMOTE_TIMEOUT_S seconds."""
    while True:
        try:
            for did, info in DRONES.items():
                base = (info or {}).get("base")
                if not base:
                    continue
                cli = drone_ws.get(str(did))
                if cli is not None:
                    all_down = (not cli._ws_connected.get("telemetry") and
                                not cli._ws_connected.get("position") and
                                not cli._ws_connected.get("rc"))
                    if all_down:
                        continue
                try:
                    _http_session.get(f"{base.rstrip('/')}/api/heartbeat",
                                      timeout=0.4)
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(_HEARTBEAT_INTERVAL_S)


threading.Thread(target=_heartbeat_loop, daemon=True, name="heartbeat-loop").start()
print(f"[HEARTBEAT] background loop started ({_HEARTBEAT_INTERVAL_S:.1f}s interval)")


# ── WebSocket client per drone ─────────────────────────────────────────
# Maintains three long-lived WS to each Pi (/ws/telemetry pulls, /ws/position
# pulls, /ws/rc pushes). Caches the latest telemetry + position so HTTP
# proxy calls can answer instantly from RAM instead of going back to the
# Pi. RC/key events take the send path which reuses the already-open
# TCP+WS socket — shaves the ~3-15 ms per-call HTTP framing cost.
class DroneWS:
    """One WS channel (really three sockets) to a single Pi. All
    connections auto-reconnect with 1s backoff up to 5s. When the
    websocket-client package isn't present (HAS_WSCLIENT=False) all
    sends become no-ops and callers fall back to HTTP."""

    def __init__(self, drone_id: str, base_http_url: str):
        self.drone_id = str(drone_id)
        self.base_http = base_http_url.rstrip("/")
        # Convert http(s):// → ws(s):// for the WS URL
        if self.base_http.startswith("https://"):
            self.ws_base = "wss://" + self.base_http[len("https://"):]
        elif self.base_http.startswith("http://"):
            self.ws_base = "ws://"  + self.base_http[len("http://"):]
        else:
            self.ws_base = "ws://"  + self.base_http

        self._lock = threading.Lock()
        self._latest_tel: dict | None = None
        self._latest_tel_ts: float = 0.0
        self._latest_pos: dict | None = None
        self._latest_pos_ts: float = 0.0
        self._rc_ws = None                    # websocket.WebSocket
        self._rc_ws_lock = threading.Lock()
        self._rc_seq = 0
        self._last_rc_send_ts: float = 0.0
        self._last_rc_send_ms: float = 0.0      # wall-clock of the most recent ws.send()
        self._rc_rtt_ms: float = 0.0
        self._ws_connected = {"telemetry": False, "position": False, "rc": False}
        # Log-suppression state: only print on state transitions, not every
        # reconnect attempt. Offline hosts would otherwise produce 3 lines
        # every 5 s per drone = a flood that obscures every real log.
        self._was_connected = {"telemetry": False, "position": False, "rc": False}
        self._consec_failures = {"telemetry": 0, "position": 0, "rc": 0}
        self._running = False

    @staticmethod
    def _sockopt_low_latency():
        """TCP_NODELAY kills Nagle's 40 ms batching delay on small
        frames — RC/key events are tiny (50-80 bytes) so without this
        the OS would buffer them. SO_KEEPALIVE on the socket helps us
        notice dead links faster than a timeout."""
        import socket as _sock
        return [
            (_sock.IPPROTO_TCP, _sock.TCP_NODELAY, 1),
            (_sock.SOL_SOCKET,  _sock.SO_KEEPALIVE, 1),
        ]

    # --- Public API ---
    def start(self):
        if not HAS_WSCLIENT or self._running:
            return
        self._running = True
        for name, target in (
            ("telemetry", self._rx_telemetry_loop),
            ("position",  self._rx_position_loop),
            ("rc",        self._rc_connect_loop),
        ):
            t = threading.Thread(target=target, daemon=True,
                                  name=f"ws-{name}-{self.drone_id}")
            t.start()

    def stop(self):
        self._running = False
        with self._rc_ws_lock:
            if self._rc_ws is not None:
                try: self._rc_ws.close()
                except Exception: pass
                self._rc_ws = None

    def latest_telemetry(self) -> tuple[dict | None, float]:
        """Return (cached_telemetry, age_seconds) or (None, +inf) when
        no WS frame has arrived."""
        with self._lock:
            if self._latest_tel is None:
                return None, float("inf")
            return dict(self._latest_tel), time.time() - self._latest_tel_ts

    def latest_position(self) -> tuple[dict | None, float]:
        with self._lock:
            if self._latest_pos is None:
                return None, float("inf")
            return dict(self._latest_pos), time.time() - self._latest_pos_ts

    def send_rc(self, lr: int, fb: int, ud: int, yaw: int,
                duration_ms: int = 250) -> bool:
        """Send an RC frame over WS. Returns True on success, False if
        the socket is not currently connected — caller should fall back
        to HTTP."""
        return self._send_rc_message({
            "type": "rc", "lr": int(lr), "fb": int(fb),
            "ud": int(ud), "yaw": int(yaw),
            "duration_ms": int(duration_ms),
        })

    def send_key(self, key: str, event: str) -> bool:
        """Send a key_down / key_up over WS. event must be 'down' or 'up'."""
        if event not in ("down", "up"):
            return False
        return self._send_rc_message({
            "type": "key", "key": str(key).lower(), "event": event,
        })

    def status(self) -> dict:
        """Connection snapshot for /proxy/ws/status + UI badge."""
        _, tel_age = self.latest_telemetry()
        _, pos_age = self.latest_position()
        with self._lock:
            rc_rtt     = self._rc_rtt_ms
            rc_send_ms = self._last_rc_send_ms
        return {
            "drone_id": self.drone_id,
            "rc":        self._ws_connected["rc"],
            "telemetry": self._ws_connected["telemetry"],
            "position":  self._ws_connected["position"],
            "telemetry_age_ms": int(tel_age * 1000) if tel_age < 1e6 else None,
            "position_age_ms":  int(pos_age * 1000) if pos_age < 1e6 else None,
            "rc_rtt_ms":        rc_rtt     if rc_rtt > 0 else None,
            "rc_send_ms":       rc_send_ms if rc_send_ms > 0 else None,
        }

    # --- Internals ---
    def _send_rc_message(self, msg: dict) -> bool:
        """Send fire-and-forget with a tight timeout budget. If the send
        takes noticeably longer than a round-trip (~50 ms), we treat the
        socket as stalled, abandon it, and let the reconnect loop spin
        up a fresh one — otherwise TCP back-pressure from a wedged
        server can queue RC frames for seconds of perceived lag.

        No sequence numbers / no ACKs — RC is idempotent and bandwidth
        is tiny; any "lost" frame is corrected on the next 100 ms tick.
        """
        if not HAS_WSCLIENT:
            return False
        ws = self._rc_ws        # snapshot ref — no lock needed
        if ws is None:
            return False
        t0 = time.time()
        try:
            ws.send(json.dumps(msg))
            dt_ms = (time.time() - t0) * 1000.0
            with self._lock:
                self._last_rc_send_ts = time.time()
                self._last_rc_send_ms = dt_ms
            # Any RC send over 250 ms is anomalous — the socket is
            # almost certainly wedged by TCP back-pressure. Kill it so
            # the reconnect loop replaces it, otherwise every
            # subsequent send waits behind the same clogged buffer.
            if dt_ms > 250.0:
                print(f"[WS] {self.drone_id} rc send SLOW {dt_ms:.0f}ms — "
                      f"dropping socket")
                with self._rc_ws_lock:
                    if self._rc_ws is ws:
                        try: ws.close()
                        except Exception: pass
                        self._rc_ws = None
                        self._ws_connected["rc"] = False
                return False
            return True
        except Exception as e:
            print(f"[WS] {self.drone_id} rc send failed: {e}")
            with self._rc_ws_lock:
                if self._rc_ws is ws:
                    try: ws.close()
                    except Exception: pass
                    self._rc_ws = None
                    self._ws_connected["rc"] = False
            return False

    def _rx_telemetry_loop(self):
        self._rx_pull_loop("telemetry", f"{self.ws_base}/ws/telemetry",
                            self._on_telemetry_msg)

    def _rx_position_loop(self):
        self._rx_pull_loop("position", f"{self.ws_base}/ws/position",
                            self._on_position_msg)

    def _rx_pull_loop(self, name: str, url: str, handler):
        """Generic pull loop — opens a WS, reads until it closes, marks
        disconnected, retries with backoff.

        recv timeout must be comfortably longer than the server's
        _WS_PING_INTERVAL_S (currently 3 s) so idle channels don't
        false-positive as dead links. 30 s gives ~10× headroom and
        still catches real link losses in under a minute.

        Logging discipline: an offline host would otherwise log every
        reconnect attempt forever. We log only on state transitions
        (first failure after connected, or first success after
        failing). Backoff also ramps to 60 s so the log stays quiet
        and bandwidth is minimal when a drone is simply offline."""
        backoff = 1.0
        while self._running:
            try:
                ws = _wsclient.create_connection(
                    url, timeout=4, sockopt=self._sockopt_low_latency())
                ws.settimeout(30.0)
                self._ws_connected[name] = True
                # Transition FAIL → OK — log only once, reset counters.
                if not self._was_connected[name]:
                    if self._consec_failures[name] > 0:
                        print(f"[WS] {self.drone_id} {name} connected "
                              f"(after {self._consec_failures[name]} failure(s))")
                    self._was_connected[name] = True
                    self._consec_failures[name] = 0
                backoff = 1.0          # reset on successful connect
                while self._running:
                    try:
                        msg = ws.recv()
                    except Exception:
                        break
                    if not msg:
                        break
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue
                    try:
                        handler(data)
                    except Exception as he:
                        print(f"[WS] {self.drone_id} {name} handler error: {he}")
                try: ws.close()
                except Exception: pass
            except Exception as e:
                if self._running:
                    self._consec_failures[name] += 1
                    # Log once on: (a) transition from connected → failed,
                    # or (b) the very first failure at boot so the operator
                    # knows a drone is unreachable. Subsequent reconnect
                    # attempts stay silent so the log doesn't flood while
                    # the drone sits offline.
                    first_failure = (self._consec_failures[name] == 1
                                      and not self._was_connected[name])
                    if self._was_connected[name] or first_failure:
                        self._was_connected[name] = False
                        msg = str(e)
                        if ("Connection refused" not in msg
                                and "Connection closed" not in msg
                                and "1000" not in msg):
                            print(f"[WS] {self.drone_id} {name} disconnect: {e}")
                        else:
                            print(f"[WS] {self.drone_id} {name} offline "
                                  f"(retries will stay silent until recovery)")
            finally:
                self._ws_connected[name] = False
            if self._running:
                time.sleep(backoff)
                # Fast retries while we're likely just between frames (1-5 s),
                # slow retries while the host is clearly offline (>10 failures
                # → 60 s cap). Keeps the reconnect alive without spamming.
                if self._consec_failures[name] <= 3:
                    backoff = min(5.0, backoff * 1.6)
                else:
                    backoff = min(60.0, backoff * 2.0)

    def _rc_connect_loop(self):
        """Keeps the RC send socket alive.

        Crucial latency details:
          - TCP_NODELAY via sockopt — without it, Nagle's 40 ms delay
            batches small RC frames and key events feel sluggish.
          - settimeout(0.3) — short. The timeout applies to BOTH recv
            and send on the socket, so a big value (old: 2 s) meant a
            single stalled send blocked the caller for 2 s. 0.3 s is
            long enough for LAN round-trips but short enough that a
            dropped link fails fast and falls back to HTTP.
          - Ping every ~2 s and measure the round-trip — exposed as
            rc_rtt_ms so operators can see the actual latency."""
        url = f"{self.ws_base}/ws/rc"
        backoff = 1.0
        while self._running:
            try:
                ws = _wsclient.create_connection(
                    url, timeout=4, sockopt=self._sockopt_low_latency())
                ws.settimeout(0.3)
                with self._rc_ws_lock:
                    self._rc_ws = ws
                self._ws_connected["rc"] = True
                # Transition FAIL → OK — log only once.
                if not self._was_connected["rc"]:
                    if self._consec_failures["rc"] > 0:
                        print(f"[WS] {self.drone_id} rc connected "
                              f"(after {self._consec_failures['rc']} failure(s))")
                    self._was_connected["rc"] = True
                    self._consec_failures["rc"] = 0
                backoff = 1.0
                last_ping = 0.0
                ping_pending: dict[int, float] = {}  # client_ts → send_monotonic
                while self._running:
                    # Opportunistic ping for RTT measurement
                    now = time.time()
                    if now - last_ping > 2.0:
                        last_ping = now
                        mono = time.monotonic()
                        try:
                            ws.send(json.dumps({
                                "type": "ping",
                                "client_ts": now,
                            }))
                            ping_pending[int(now * 1000)] = mono
                        except Exception:
                            break
                    try:
                        msg = ws.recv()
                    except _wsclient._exceptions.WebSocketTimeoutException:
                        continue
                    except Exception:
                        break
                    if not msg:
                        break
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue
                    if data.get("type") == "pong":
                        echo = data.get("echo")
                        if echo is not None:
                            sent_mono = ping_pending.pop(int(float(echo) * 1000), None)
                            if sent_mono is not None:
                                with self._lock:
                                    self._rc_rtt_ms = round((time.monotonic() - sent_mono) * 1000.0, 1)
            except Exception as e:
                if self._running:
                    self._consec_failures["rc"] += 1
                    first_failure = (self._consec_failures["rc"] == 1
                                      and not self._was_connected["rc"])
                    if self._was_connected["rc"] or first_failure:
                        self._was_connected["rc"] = False
                        m = str(e)
                        if ("Connection refused" not in m
                                and "Connection closed" not in m
                                and "1000" not in m):
                            print(f"[WS] {self.drone_id} rc disconnect: {e}")
                        else:
                            print(f"[WS] {self.drone_id} rc offline "
                                  f"(retries silent until recovery)")
            finally:
                with self._rc_ws_lock:
                    self._rc_ws = None
                self._ws_connected["rc"] = False
            if self._running:
                time.sleep(backoff)
                if self._consec_failures["rc"] <= 3:
                    backoff = min(5.0, backoff * 1.6)
                else:
                    backoff = min(60.0, backoff * 2.0)

    def _on_telemetry_msg(self, data: dict):
        if data.get("type") not in (None, "telemetry"):
            return
        # Strip framing field; store the rest
        snap = {k: v for k, v in data.items() if k not in ("type",)}
        with self._lock:
            self._latest_tel = snap
            self._latest_tel_ts = time.time()

    def _on_position_msg(self, data: dict):
        if data.get("type") not in (None, "position"):
            return
        snap = {k: v for k, v in data.items() if k not in ("type",)}
        with self._lock:
            self._latest_pos = snap
            self._latest_pos_ts = time.time()


# Create one client per configured drone. They connect on their own
# schedule — failing to reach a drone never blocks the C2 boot.
drone_ws: dict[str, DroneWS] = {}
def _init_drone_ws():
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            continue
        try:
            client = DroneWS(str(did), base)
            client.start()
            drone_ws[str(did)] = client
            print(f"[WS] started client for drone {did} → {client.ws_base}")
        except Exception as e:
            print(f"[WS] failed to start client for drone {did}: {e}")
_init_drone_ws()



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


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/logo.png")
def serve_logo():
    """Serve the team logo for the header of the UI.

    Prefers the alpha-masked variant (team_logo_transparent.png) so the
    logo blends with the page's dark background rather than showing a
    white rectangle. Falls back to the original PNG if the masked file
    isn't present.
    """
    from pathlib import Path as _P
    base = _P(__file__).resolve().parent.parent / "1_Doc"
    for name in ("team_logo_transparent.png", "team_logo.png"):
        p = base / name
        if p.exists():
            return send_file(str(p), mimetype="image/png",
                             max_age=86400)   # cache-1-day
    return jsonify(ok=False, error="logo not found"), 404


@app.get("/proxy/drones")
def proxy_drones():
    return jsonify(drones=DRONES, active=active_drone_id)


@app.post("/proxy/switch")
def proxy_switch():
    global active_drone_id, PI_BASE
    data = request.get_json(silent=True) or {}
    drone_id = str(data.get("id", ""))
    if drone_id not in DRONES:
        return jsonify(ok=False, error="unknown drone id"), 400
    active_drone_id = drone_id
    PI_BASE = DRONES[drone_id]["base"]
    log_command("switch_drone", {"id": drone_id, "name": DRONES[drone_id]["name"]})
    print(f"[REMOTE UI] Switched to {DRONES[drone_id]['name']} @ {PI_BASE}")
    return jsonify(ok=True, active=drone_id, name=DRONES[drone_id]["name"], base=PI_BASE)


@app.get("/proxy/drones/config")
def proxy_drones_config():
    """Return full drone config for editing."""
    return jsonify(drones=DRONES, config_path=str(DRONES_CONFIG_PATH))


@app.post("/proxy/drones/config")
def proxy_drones_config_save():
    """Save updated drone config. Expects {drones: {id: {name, type, base}, ...}}"""
    global DRONES, PI_BASE
    data = request.get_json(silent=True) or {}
    new_drones = data.get("drones")
    if not new_drones or not isinstance(new_drones, dict):
        return jsonify(ok=False, error="drones dict required"), 400
    for did, info in new_drones.items():
        if not all(k in info for k in ("name", "type", "base")):
            return jsonify(ok=False, error=f"drone {did} missing name/type/base"), 400
        if info["type"] not in ("tello", "anafi"):
            return jsonify(ok=False, error=f"drone {did} type must be tello or anafi"), 400
    DRONES.clear()
    DRONES.update(new_drones)
    if active_drone_id in DRONES:
        PI_BASE = DRONES[active_drone_id]["base"]
    save_drones_config(DRONES)
    # Keep ArUco fleet in sync with drone config
    try:
        aruco_fleet.configure(DRONES)
    except Exception as e:
        print(f"[ARUCO] fleet reconfigure failed: {e}")
    print(f"[CONFIG] Saved {len(DRONES)} drones to {DRONES_CONFIG_PATH}")
    return jsonify(ok=True, drones=DRONES)


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


@app.post("/proxy/key_down")
def proxy_key_down():
    data = request.get_json(silent=True) or {}
    log_command("key_down", data)
    mode = _transport_mode("rc")
    if mode in ("auto", "ws"):
        k = str(data.get("key", ""))
        ws = drone_ws.get(str(active_drone_id))
        if ws and ws.send_key(k, "down"):
            return jsonify(ok=True, via="ws")
        if mode == "ws":
            return jsonify(ok=False, error="ws forced but not connected",
                           via="ws"), 503
    # HTTP path — tight TIMEOUT_FAST so an unreachable drone fails fast
    # (was TIMEOUT_CMD=8s, which made every key burn 8s when the Pi
    # didn't answer — the real cause of "HTTP controls don't work").
    # No more _active_drone_reachable() gate: HTTP shouldn't be vetoed
    # by WS status. If HTTP itself is broken the timeout tells us.
    try:
        r = pi_post("/api/key_down", data, timeout=TIMEOUT_FAST)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=f"http: {e}", via="http"), 502


@app.post("/proxy/key_up")
def proxy_key_up():
    data = request.get_json(silent=True) or {}
    log_command("key_up", data)
    mode = _transport_mode("rc")
    if mode in ("auto", "ws"):
        k = str(data.get("key", ""))
        ws = drone_ws.get(str(active_drone_id))
        if ws and ws.send_key(k, "up"):
            return jsonify(ok=True, via="ws")
        if mode == "ws":
            return jsonify(ok=False, error="ws forced but not connected",
                           via="ws"), 503
    try:
        r = pi_post("/api/key_up", data, timeout=TIMEOUT_FAST)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=f"http: {e}", via="http"), 502


@app.post("/proxy/key_batch")
def proxy_key_batch():
    """Batch key_down for all currently held keys in a single request.

    Fires at ~10 Hz from the browser — it's the keep-alive that stops
    the Pi's KEY_STALE_S=1 timer from dropping keys mid-flight. Historic
    path did N serial HTTP POSTs per batch which is the real WASD
    latency culprit (3 held keys × 10 Hz × slow HTTP = backed-up fetch
    queue).

    New path: one WS send per held key on the persistent socket. With
    fire-and-forget semantics, the whole batch clears in microseconds.
    Fallback to HTTP on WS disconnect."""
    data = request.get_json(silent=True) or {}
    keys = data.get("keys", [])
    if not keys:
        return jsonify(ok=True)
    # Dedup so the log and the wire don't carry redundant presses
    uniq = list(dict.fromkeys(keys))

    mode = _transport_mode("rc")
    if mode in ("auto", "ws"):
        ws = drone_ws.get(str(active_drone_id))
        if ws:
            all_sent = True
            for k in uniq:
                if not ws.send_key(str(k), "down"):
                    all_sent = False
                    break
            if all_sent:
                log_command("key_batch", {"keys": uniq, "via": "ws"})
                return jsonify(ok=True, via="ws", n=len(uniq))
        if mode == "ws":
            return jsonify(ok=False, error="ws forced but not connected",
                           via="ws"), 503

    log_command("key_batch", {"keys": uniq, "via": "http"})
    # SINGLE HTTP POST to the Pi's batch endpoint — was N serial
    # per-key calls previously, which saturated the browser's
    # 6-connection pool whenever WS fell back to HTTP.
    try:
        r = pi_post("/api/key_batch", {"keys": uniq}, timeout=TIMEOUT_FAST)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=f"http: {e}", via="http"), 502


@app.post("/proxy/takeoff")
def proxy_takeoff():
    """Relay takeoff to the active drone's Pi.

    Uses TIMEOUT_SLOW (default 15s) not the default TIMEOUT_CMD (8s)
    because Anafi takeoff routinely takes 3-5 seconds for the armament
    sensors + motor spin-up before /api/takeoff returns success. The
    earlier 2s default truncated this and led to "network error
    contacting drone: TypeError: Failed to fetch" on every attempt.
    Also wraps the call in try/except so a slow drone returns clean
    JSON to the UI instead of a Flask stack-trace HTML page.
    """
    log_command("takeoff")
    try:
        r = pi_post("/api/takeoff", timeout=TIMEOUT_SLOW)
    except Exception as e:
        import requests as _rq
        if isinstance(e, _rq.exceptions.ReadTimeout):
            return jsonify(ok=False,
                           error=f"takeoff timed out after {TIMEOUT_SLOW}s — "
                                 f"drone may still be arming; check telemetry"), 504
        return jsonify(ok=False, error=f"network error: {e}"), 502
    return (r.text, r.status_code,
            {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/land")
def proxy_land():
    """Relay land to the active drone's Pi. Same slow-command timeout
    as takeoff — Anafi land waits for ground-contact before returning."""
    log_command("land")
    try:
        r = pi_post("/api/land", timeout=TIMEOUT_SLOW)
    except Exception as e:
        import requests as _rq
        if isinstance(e, _rq.exceptions.ReadTimeout):
            return jsonify(ok=False,
                           error=f"land timed out after {TIMEOUT_SLOW}s — "
                                 f"check drone state before retrying"), 504
        return jsonify(ok=False, error=f"network error: {e}"), 502
    return (r.text, r.status_code,
            {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/land_all")
def proxy_land_all():
    """Emergency panic-button: land every configured drone and halt any
    running mission. Used by the 'q' hotkey in the UI. Tolerates
    individual drones being unreachable — collects per-drone outcomes
    and always returns 200 so the client can render the summary."""
    log_command("land_all")

    # 1) Stop any running mission first so its LIVE-mode state machine
    #    doesn't immediately re-push RC commands that conflict with land.
    mission_stopped = False
    try:
        if mission_manager is not None and mission_manager.current is not None:
            mission_stopped = mission_manager.stop(land=False)
    except Exception as e:
        print(f"[LAND_ALL] mission stop failed: {e}")

    # 2) Send /api/land to every configured drone in parallel-ish
    #    (sequentially but with a short timeout per drone).
    results: dict[str, dict] = {}
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            results[str(did)] = {"ok": False, "error": "no base url"}
            continue
        try:
            # Use the slow-command timeout — Anafi land blocks until
            # ground contact. 3s wasn't enough for gentle descent.
            resp = _http_session.post(f"{base.rstrip('/')}/api/land",
                                      json={}, timeout=TIMEOUT_SLOW)
            try:
                j = resp.json()
            except Exception:
                j = {"raw": resp.text[:120]}
            results[str(did)] = {
                "ok": bool(j.get("ok", resp.status_code == 200)),
                "status": resp.status_code,
                "msg": j.get("msg") or j.get("error") or "",
            }
        except Exception as e:
            results[str(did)] = {"ok": False, "error": str(e)[:120]}

    # 3) Also put every ArUco observer back into OBSERVE mode and clear
    #    any lingering search RC so the drones don't fight the land.
    try:
        for did, obs in aruco_fleet._obs.items():
            try:
                obs.set_search_rc(0, 0, 0, 0)
                obs.set_mode("observe")
            except Exception:
                pass
    except Exception:
        pass

    ok_count = sum(1 for v in results.values() if v.get("ok"))
    print(f"[LAND_ALL] {ok_count}/{len(results)} drones acknowledged land "
          f"(mission_stopped={mission_stopped})")
    return jsonify(
        ok=True,
        landed=ok_count,
        total=len(results),
        mission_stopped=mission_stopped,
        results=results,
    )


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


@app.get("/proxy/flight_logs")
def proxy_flight_logs_list():
    """List archived per-flight log files (newest first), with video
    metadata attached when a matching .mp4 lives on any Pi."""
    files = flight_logger.list_files()
    vids = _collect_fleet_videos()
    for f in files:
        stem = f["name"][:-len(".jsonl")] if f["name"].endswith(".jsonl") else f["name"]
        video_name = stem + ".mp4"
        v = vids.get(video_name)
        if v:
            f["video"] = {
                "name":     video_name,
                "drone_id": v["drone_id"],
                "size":     v["size"],
                "mtime":    v["mtime"],
            }
    return jsonify(ok=True, files=files)


@app.get("/proxy/flight_video/<path:name>")
def proxy_flight_video_download(name):
    """Stream a flight video from whichever Pi has it. Filename is the
    mp4 basename produced by FlightLogger (flight_YYYY-..._drone-N_X.mp4);
    we look up the drone id from the filename and proxy straight through."""
    # Locate the video across the fleet
    vids = _collect_fleet_videos()
    entry = vids.get(name)
    if not entry:
        return jsonify(ok=False, error="video not found on any drone"), 404
    did = entry["drone_id"]
    base = (DRONES.get(did, {}) or {}).get("base")
    if not base:
        return jsonify(ok=False, error="drone has no base url"), 500
    try:
        upstream = _http_session.get(
            f"{base.rstrip('/')}/api/video/recordings/{name}",
            stream=True, timeout=(3, 60))
        if not upstream.ok:
            return jsonify(ok=False, error=f"drone returned {upstream.status_code}"), 502
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502
    def gen():
        for chunk in upstream.iter_content(chunk_size=64 * 1024):
            if chunk:
                yield chunk
    headers = {
        "Content-Type": "video/mp4",
        "Content-Disposition": f'attachment; filename="{name}"',
    }
    size = upstream.headers.get("Content-Length")
    if size:
        headers["Content-Length"] = size
    return Response(gen(), headers=headers)


@app.get("/proxy/flight_logs/<path:name>")
def proxy_flight_logs_download(name):
    p = flight_logger.file_path(name)
    if p is None:
        return jsonify(ok=False, error="file not found"), 404
    # Inline when "as_attachment=false" so the viewer can fetch-and-parse
    # rather than trigger a download.
    as_att = request.args.get("dl", "1") != "0"
    return send_file(str(p), mimetype="application/jsonlines",
                     as_attachment=as_att, download_name=p.name)


@app.get("/flight_log_viewer")
def serve_flight_log_viewer():
    """Interactive replay UI for flight logs.

    Opens ?file=<name> auto-loaded via /proxy/flight_logs/<name>?dl=0.
    Visualises trajectory + events + timeline so an operator can scrub
    through a flight and see exactly where something went wrong.
    """
    from pathlib import Path as _P
    p = _P(__file__).with_name("flight_log_viewer.html")
    if not p.exists():
        return jsonify(ok=False, error="viewer html missing"), 404
    return send_file(str(p), mimetype="text/html", max_age=0)


@app.get("/proxy/config/ceiling")
def proxy_ceiling_get():
    """Aggregate the soft ceiling and engaged state from every drone in
    the fleet. Returns the minimum ceiling across drones (most
    conservative) and engaged=True if ANY drone is currently clamped."""
    results = {}
    any_engaged = False
    min_ceiling = None
    reasons = []
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            results[str(did)] = {"ok": False, "error": "no base url"}
            continue
        try:
            r = _http_session.get(f"{base.rstrip('/')}/api/config/ceiling", timeout=1.0)
            j = r.json()
            results[str(did)] = j
            if j.get("engaged"):
                any_engaged = True
                if j.get("reason"):
                    reasons.append(f"{did}: {j['reason']}")
            if j.get("ceiling_m") is not None:
                c = float(j["ceiling_m"])
                if min_ceiling is None or c < min_ceiling:
                    min_ceiling = c
        except Exception as e:
            results[str(did)] = {"ok": False, "error": str(e)[:120]}
    return jsonify(
        ok=True,
        ceiling_m=min_ceiling,
        engaged=any_engaged,
        reasons=reasons,
        per_drone=results,
    )


@app.get("/proxy/config/arena_safety")
def proxy_arena_safety_get():
    """Aggregate arena-safety state across the fleet — min margin wins,
    any engaged → engaged, any disabled → disabled (most conservative)."""
    results = {}
    any_engaged = False
    all_enabled = True
    min_margin = None
    reasons = []
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            results[str(did)] = {"ok": False, "error": "no base url"}
            continue
        try:
            r = _http_session.get(f"{base.rstrip('/')}/api/config/arena_safety",
                                   timeout=1.0)
            j = r.json() if r.ok else {"ok": False}
            results[str(did)] = j
            if j.get("engaged"):
                any_engaged = True
                if j.get("reason"):
                    reasons.append(f"{did}: {j['reason']}")
            if not j.get("enabled", True):
                all_enabled = False
            m = j.get("margin_m")
            if isinstance(m, (int, float)):
                if min_margin is None or m < min_margin:
                    min_margin = float(m)
        except Exception as e:
            results[str(did)] = {"ok": False, "error": str(e)[:120]}
    return jsonify(
        ok=True,
        enabled=all_enabled,
        margin_m=min_margin,
        engaged=any_engaged,
        reasons=reasons,
        per_drone=results,
    )


@app.post("/proxy/config/arena_safety")
def proxy_arena_safety_set():
    """Fan-out arena safety settings to every drone's Pi."""
    data = request.get_json(silent=True) or {}
    log_command("arena_safety_set", data)
    results = {}
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            continue
        try:
            r = _http_session.post(f"{base.rstrip('/')}/api/config/arena_safety",
                                    json=data, timeout=2.0)
            results[str(did)] = r.json() if r.ok else {"ok": False, "status": r.status_code}
        except Exception as e:
            results[str(did)] = {"ok": False, "error": str(e)[:120]}
    ok_count = sum(1 for v in results.values() if v.get("ok"))
    print(f"[ARENA] fleet-wide: {data} → {ok_count}/{len(results)} drones acknowledged")
    return jsonify(ok=True, applied=ok_count, total=len(results), results=results)


@app.post("/proxy/config/ceiling")
def proxy_ceiling_set():
    """Push a new soft ceiling to every drone in the fleet. Body:
        {"ceiling_m": <float, metres>}
    """
    data = request.get_json(silent=True) or {}
    try:
        v = float(data.get("ceiling_m") or data.get("max_altitude_m"))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="ceiling_m required"), 400
    log_command("ceiling_set", {"ceiling_m": v})
    results = {}
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            results[str(did)] = {"ok": False, "error": "no base url"}
            continue
        try:
            r = _http_session.post(f"{base.rstrip('/')}/api/config/ceiling",
                                   json={"ceiling_m": v}, timeout=2.0)
            results[str(did)] = r.json()
        except Exception as e:
            results[str(did)] = {"ok": False, "error": str(e)[:120]}
    ok_count = sum(1 for v in results.values() if v.get("ok"))
    print(f"[CEILING] fleet-wide set to {v}m: {ok_count}/{len(results)} drones acknowledged")
    return jsonify(ok=True, ceiling_m=v, applied=ok_count, total=len(results), results=results)


@app.post("/proxy/pause_all")
def proxy_pause_all():
    """Fleet-wide PAUSE. Overrides any command — every drone stays
    airborne at its current position (Anafi auto-hovers with zero RC).

    Side-effects, in order:
      1. Raise the global pause flag (blocks mission start + ArUco LIVE).
      2. Abort any running mission (do NOT land — the drones hover).
      3. Yank every ArUco observer out of LIVE back to OBSERVE and clear
         its search RC override, so it stops pushing commands.
      4. Send a hard zero-RC (full brake) to each drone so any residual
         translation command stops immediately.

    The call is idempotent — pressing it again while already paused is
    a no-op beyond re-issuing the zero-RC.
    """
    global _global_paused, _global_paused_at, _global_paused_src
    data = request.get_json(silent=True) or {}
    source = str(data.get("source") or "unknown")
    log_command("pause_all", {"source": source})

    # 1) Raise the flag first so guards start rejecting autonomous starts
    #    even before we're done tearing down the existing activity.
    with _pause_lock:
        _global_paused = True
        _global_paused_at = time.time()
        _global_paused_src = source

    # 2) Stop any running mission (don't land).
    mission_stopped = False
    try:
        if mission_manager is not None and mission_manager.current is not None:
            mission_stopped = mission_manager.stop(land=False)
    except Exception as e:
        print(f"[PAUSE_ALL] mission stop failed: {e}")

    # 3) Observers → OBSERVE, clear search overrides.
    try:
        for did, obs in aruco_fleet._obs.items():
            try:
                obs.set_search_rc(0, 0, 0, 0)
                obs.set_mode("observe")
            except Exception:
                pass
    except Exception:
        pass

    # 4) Full brake — POST rc(0,0,0,0) to every drone in parallel-ish.
    results: dict[str, dict] = {}
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            results[str(did)] = {"ok": False, "error": "no base url"}
            continue
        try:
            # Some APIs want the full RC tuple, others accept a rc_stop
            # shortcut. The unified API exposes /api/rc; send zeros.
            resp = _http_session.post(
                f"{base.rstrip('/')}/api/rc",
                json={"lr": 0, "fb": 0, "ud": 0, "yaw": 0},
                timeout=2.0,
            )
            try:
                j = resp.json()
            except Exception:
                j = {"raw": resp.text[:120]}
            results[str(did)] = {
                "ok": bool(j.get("ok", resp.status_code == 200)),
                "status": resp.status_code,
            }
        except Exception as e:
            results[str(did)] = {"ok": False, "error": str(e)[:120]}

    ok_count = sum(1 for v in results.values() if v.get("ok"))
    print(f"[PAUSE_ALL] fleet paused (source={source}) — "
          f"{ok_count}/{len(results)} drones braked, "
          f"mission_stopped={mission_stopped}")
    return jsonify(
        ok=True, paused=True,
        source=source,
        mission_stopped=mission_stopped,
        braked=ok_count, total=len(results),
        results=results,
    )


@app.post("/proxy/resume_all")
def proxy_resume_all():
    """Clear the global pause. Autonomous endpoints (missions, ArUco
    LIVE) become reachable again, but nothing auto-restarts — the
    operator must re-arm any mission themselves. That's intentional:
    coming out of pause should never surprise the pilot with a drone
    suddenly darting off."""
    global _global_paused, _global_paused_at, _global_paused_src
    data = request.get_json(silent=True) or {}
    source = str(data.get("source") or "unknown")
    log_command("resume_all", {"source": source})
    with _pause_lock:
        was_paused = _global_paused
        _global_paused = False
        _global_paused_at = time.time()
        _global_paused_src = ""
    print(f"[RESUME_ALL] fleet resumed (source={source}, was_paused={was_paused})")
    return jsonify(ok=True, paused=False, source=source, was_paused=was_paused)


@app.get("/proxy/pause_status")
def proxy_pause_status():
    """UI poll target — lets multiple open tabs sync their button state."""
    with _pause_lock:
        return jsonify(
            paused=_global_paused,
            since=_global_paused_at,
            source=_global_paused_src,
        )


@app.post("/proxy/flip")
def proxy_flip():
    data = request.get_json(silent=True) or {}
    log_command("flip", data)
    r = pi_post("/api/flip", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/recover")
def proxy_recover():
    log_command("recover")
    r = pi_post("/api/recover")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/emergency")
def proxy_emergency():
    log_command("emergency")
    r = pi_post("/api/emergency")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/speed")
def proxy_speed():
    data = request.get_json(silent=True) or {}
    log_command("speed", data)
    r = pi_post("/api/speed", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/move")
def proxy_move():
    data = request.get_json(silent=True) or {}
    log_command("move", data)
    r = pi_post("/api/move", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/rotate")
def proxy_rotate():
    data = request.get_json(silent=True) or {}
    log_command("rotate", data)
    r = pi_post("/api/rotate", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/go")
def proxy_go():
    data = request.get_json(silent=True) or {}
    log_command("go", data)
    r = pi_post("/api/go", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/curve")
def proxy_curve():
    data = request.get_json(silent=True) or {}
    log_command("curve", data)
    r = pi_post("/api/curve", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/stream")
def proxy_stream():
    data = request.get_json(silent=True) or {}
    log_command("stream", data)
    r = pi_post("/api/stream", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/sdk")
def proxy_sdk():
    data = request.get_json(silent=True) or {}
    log_command("sdk", data)
    r = pi_post("/api/sdk", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


# -- Anafi / Olympe proxy routes --

@app.post("/proxy/camera/photo")
def proxy_camera_photo():
    log_command("camera_photo")
    r = pi_post("/api/camera/photo")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/camera/record/start")
def proxy_camera_record_start():
    log_command("camera_record_start")
    r = pi_post("/api/camera/record/start")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/camera/record/stop")
def proxy_camera_record_stop():
    log_command("camera_record_stop")
    r = pi_post("/api/camera/record/stop")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/gimbal")
def proxy_gimbal():
    data = request.get_json(silent=True) or {}
    log_command("gimbal", data)
    r = pi_post("/api/gimbal", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/rth")
def proxy_rth():
    data = request.get_json(silent=True) or {}
    log_command("rth", data)
    r = pi_post("/api/rth", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/moveto")
def proxy_moveto():
    data = request.get_json(silent=True) or {}
    log_command("moveto", data)
    r = pi_post("/api/moveto", data, timeout=65)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.get("/proxy/settings")
def proxy_settings_get():
    try:
        r = pi_get("/api/settings", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/settings")
def proxy_settings_set():
    data = request.get_json(silent=True) or {}
    log_command("settings", data)
    r = pi_post("/api/settings", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.get("/proxy/safety/takeoff")
def proxy_safe_takeoff_get():
    try:
        r = pi_get("/api/safety/takeoff", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/safety/takeoff")
def proxy_safe_takeoff_set():
    data = request.get_json(silent=True) or {}
    log_command("safe_takeoff_set", data)
    r = pi_post("/api/safety/takeoff", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.get("/proxy/logging/commands")
def proxy_command_log_status():
    return jsonify(enabled=command_log_enabled, path=str(command_log_path))


@app.post("/proxy/logging/commands")
def proxy_command_log_config():
    global command_log_enabled, command_log_path
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled")
    path = data.get("path")
    if enabled is not None and not isinstance(enabled, bool):
        return jsonify(ok=False, error="enabled must be boolean"), 400
    if path is not None and (not isinstance(path, str) or not path.strip()):
        return jsonify(ok=False, error="path must be non-empty string"), 400
    if isinstance(enabled, bool):
        command_log_enabled = enabled
    if isinstance(path, str) and path.strip():
        command_log_path = Path(path.strip())
    print(f"[REMOTE CMD] command logging {'enabled' if command_log_enabled else 'disabled'}")
    return jsonify(ok=True, enabled=command_log_enabled, path=str(command_log_path))


@app.get("/proxy/logging/commands/download")
def proxy_command_log_download():
    p = command_log_path
    if not p.exists():
        return jsonify(ok=False, error="command log file not found", path=str(p)), 404
    return send_file(p, as_attachment=True, download_name=p.name, mimetype="application/x-ndjson")


@app.post("/proxy/logging/commands/clear")
def proxy_command_log_clear():
    p = command_log_path
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        return jsonify(ok=True, cleared=True, path=str(p))
    except Exception as e:
        return jsonify(ok=False, error=str(e), path=str(p)), 500


@app.get("/proxy/logging/telemetry")
def proxy_log_status():
    try:
        r = pi_get("/api/logging/telemetry", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/logging/telemetry")
def proxy_log_config():
    data = request.get_json(silent=True) or {}
    log_command("telemetry_log_set", data)
    r = pi_post("/api/logging/telemetry", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.get("/proxy/logging/telemetry/download")
def proxy_log_download():
    r = pi_get("/api/logging/telemetry/download")
    headers = {
        "Content-Type": r.headers.get("Content-Type", "application/octet-stream"),
        "Content-Disposition": r.headers.get("Content-Disposition", "attachment; filename=telemetry_log.jsonl"),
    }
    return (r.content, r.status_code, headers)


@app.post("/proxy/logging/telemetry/clear")
def proxy_log_clear():
    log_command("telemetry_log_clear")
    r = pi_post("/api/logging/telemetry/clear")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.get("/proxy/video")
def proxy_video_feed():
    """Proxy the MJPEG video stream from the Pi API server."""
    try:
        r = _http_session.get(f"{PI_BASE}/api/video", stream=True, timeout=30)
        return Response(
            r.iter_content(chunk_size=32768),
            mimetype=r.headers.get("Content-Type", "multipart/x-mixed-replace; boundary=frame"),
        )
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/video/start")
def proxy_video_start():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "mjpeg")
    # For forward mode, auto-detect C2 IP (this machine) so the Pi sends UDP here
    if mode == "forward" and not data.get("target_host"):
        # Use the host part of request.host (what the browser connected to)
        c2_host = request.host.split(":")[0]
        if c2_host in ("127.0.0.1", "localhost"):
            # Try to get our real IP
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                c2_host = s.getsockname()[0]
                s.close()
            except Exception:
                pass
        data["target_host"] = c2_host
        data["target_port"] = data.get("target_port", VIDEO_UDP_FORWARD_PORT)
    log_command("video_start", data)
    # For forward mode, start the UDP receiver BEFORE telling the Pi to forward
    # so ffmpeg is already listening when packets arrive
    if mode == "forward":
        _start_udp_receiver()
        time.sleep(0.5)  # Give ffmpeg time to bind the UDP port
    r = pi_post("/api/video/start", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.post("/proxy/video/stop")
def proxy_video_stop():
    log_command("video_stop")
    _stop_udp_receiver()
    r = pi_post("/api/video/stop")
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


@app.get("/proxy/video/status")
def proxy_video_status():
    try:
        r = pi_get("/api/video/status", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": "application/json"})
    except Exception as e:
        return jsonify(ok=False, error=str(e), mode="off"), 502


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


@app.get("/proxy/video/forward_stream")
def proxy_video_forward_stream():
    """Serve decoded UDP forward frames as MJPEG stream."""
    def gen():
        while _udp_receiver_running:
            with _udp_last_frame_lock:
                jpg = _udp_last_jpeg
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(1.0 / max(1, VIDEO_FPS))
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/proxy/camera/zoom")
def proxy_camera_zoom():
    data = request.get_json(silent=True) or {}
    try:
        r = _http_session.post(f"{PI_BASE}/api/camera/zoom", json=data, timeout=1.5)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.get("/proxy/heartbeat")
def proxy_heartbeat():
    """Send heartbeat to all REACHABLE drones in parallel.

    Skips drones whose WS client reports every channel down — no point
    spending 300 ms × N offline drones every 750 ms. That was the main
    reason the heartbeat poll stalled the HTTP pool when most of the
    fleet was offline.
    """
    def _ping(did_info):
        did, info = did_info
        try:
            r = _http_session.get(f"{info['base']}/api/heartbeat", timeout=0.25)
            return did, r.status_code
        except Exception:
            return did, "timeout"

    # Filter out offline drones using the WS client's health flag.
    live_items = []
    skipped = {}
    for did, info in DRONES.items():
        did_s = str(did)
        cli = drone_ws.get(did_s) if 'drone_ws' in globals() else None
        if cli is not None:
            all_down = (not cli._ws_connected.get("telemetry") and
                        not cli._ws_connected.get("position") and
                        not cli._ws_connected.get("rc"))
            if all_down:
                skipped[did_s] = "offline"
                continue
        live_items.append((did, info))

    results = dict(_heartbeat_pool.map(_ping, live_items)) if live_items else {}
    results.update(skipped)
    return jsonify(ok=True, drones=results)


# ─── Latency ping (C2 → flight controller + flight controller → drone) ──────
@app.get("/proxy/latency")
def proxy_latency():
    """Returns the three-leg latency picture for the ACTIVE drone:
      c2_to_fc_ms     — fresh HTTP round-trip to /api/heartbeat on the Pi
      fc_to_drone_ms  — cached flight-controller-to-drone ICMP ping from /api/drone_ping
      sample_age_s    — age of the fc_to_drone sample
    The client adds these together (plus a user-tuned video-processing
    offset) to get the total command-loop latency."""
    t0 = time.time()
    c2_rtt = None
    fc = None
    try:
        r = _http_session.get(f"{PI_BASE}/api/heartbeat", timeout=1.0)
        c2_rtt = (time.time() - t0) * 1000.0
        fc_ok = (r.status_code == 200)
    except Exception:
        fc_ok = False
    try:
        r2 = _http_session.get(f"{PI_BASE}/api/drone_ping", timeout=1.0)
        if r2.ok:
            fc = r2.json()
    except Exception:
        pass
    resp = {
        "ok": True,
        "c2_to_fc_ms": round(c2_rtt, 2) if c2_rtt is not None else None,
        "fc_reachable": fc_ok,
        "fc_to_drone_ms": (fc.get("rtt_ms") if fc else None),
        "fc_to_drone_host": (fc.get("host") if fc else None),
        "fc_to_drone_sample_age_s": (fc.get("sample_age_s") if fc else None),
        "pi_base": PI_BASE,
    }
    return jsonify(resp)


@app.get("/proxy/telemetry")
def proxy_telemetry():
    mode = _transport_mode("telemetry")
    if mode in ("auto", "ws"):
        ws = drone_ws.get(str(active_drone_id))
        if ws:
            tel, age = ws.latest_telemetry()
            if tel is not None and age < 1.5:
                tel = dict(tel)
                tel["_source"] = "ws"
                tel["_age_ms"] = int(age * 1000)
                headers = {
                    "Content-Type": "application/json",
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                }
                return (json.dumps(tel, default=str), 200, headers)
        if mode == "ws":
            return jsonify(ok=False, error="ws forced but not connected", via="ws"), 503
    try:
        r = pi_get("/api/telemetry", timeout=TIMEOUT_STATUS)
        headers = {
            "Content-Type": r.headers.get("Content-Type", "application/json"),
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        return (r.text, r.status_code, headers)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.get("/proxy/fc_version")
def proxy_fc_version():
    """Fetch the active drone's FC /api/version (short 1 s timeout) and
    compare its code_version string to the C2's. Lets the UI flash a
    banner when the operator deployed a fix to C2 but the FC is still
    running old code — which was silently happening for days."""
    did = str(request.args.get("drone_id") or active_drone_id)
    info = DRONES.get(did) or {}
    base = (info.get("base") or "").rstrip("/")
    fc_code = None
    fc_sha = None
    fc_err = None
    if base:
        try:
            r = _http_session.get(f"{base}/api/version", timeout=1.0)
            if r.ok:
                j = r.json() or {}
                fc_code = j.get("code_version")
                fc_sha = (j.get("git_revision") or {}).get("short_sha")
            else:
                fc_err = f"HTTP {r.status_code}"
        except Exception as e:
            fc_err = str(e)
    # Match = version strings are equal ignoring the trailing parenthetical
    # description (so a small comment change doesn't flag every restart).
    def _strip_tag(v):
        if not v: return ""
        return str(v).split(" ", 1)[0].strip()
    match = (fc_code is not None
             and _strip_tag(fc_code) == _strip_tag(C2_CODE_VERSION))
    return jsonify(
        ok=True,
        drone_id=did,
        c2_code_version=C2_CODE_VERSION,
        c2_git_sha=_GIT_REVISION.get("short_sha"),
        fc_code_version=fc_code,
        fc_git_sha=fc_sha,
        fc_error=fc_err,
        match=match,
    )


@app.get("/proxy/ui_state")
def proxy_ui_state():
    """Single consolidated poll — replaces the many 0.5-5Hz polls the UI
    used to fire for tiny pieces of state (heartbeat, pause, ws_status,
    ceiling, transport, mission status). All of them are cheap reads
    of local C2 state; making them one request means the browser keeps
    its 6 HTTP/1.1 connections free for real traffic (takeoff, key
    events) instead of queueing behind 8 small polls.

    Heartbeat to the Pi is STILL sent here so the Pi watchdog stays
    happy — we reuse the existing parallel-ping logic.
    """
    out: dict = {}

    # --- Pause state ---
    with _pause_lock:
        out["pause"] = {
            "paused":  _global_paused,
            "since":   _global_paused_at,
            "source":  _global_paused_src,
        }

    # --- Ceiling + arena-safety (aggregated from per-drone endpoints) ---
    # Combined fan-out: for every live drone we fetch BOTH endpoints in
    # sequence. That's 2 HTTP calls per drone per uiStatePoll tick (1Hz)
    # instead of two separate per-2s polls issued by the browser — less
    # load on the C2's HTTP pool and on the browser's 6-connection limit.
    try:
        ceilings = []
        any_c_engaged = False
        c_reasons = []
        margins = []
        all_arena_enabled = True
        any_a_engaged = False
        a_reasons = []
        for did, info in DRONES.items():
            base = (info or {}).get("base")
            if not base: continue
            cli = drone_ws.get(str(did)) if 'drone_ws' in globals() else None
            if cli is not None:
                all_down = (not cli._ws_connected.get("telemetry") and
                            not cli._ws_connected.get("position") and
                            not cli._ws_connected.get("rc"))
                if all_down:
                    continue
            # Ceiling
            try:
                r = _http_session.get(f"{base.rstrip('/')}/api/config/ceiling",
                                       timeout=0.25)
                if r.status_code == 200:
                    j = r.json()
                    if j.get("ceiling_m") is not None:
                        ceilings.append(float(j["ceiling_m"]))
                    if j.get("engaged"):
                        any_c_engaged = True
                        if j.get("reason"):
                            c_reasons.append(f"{did}: {j['reason']}")
            except Exception:
                pass
            # Arena safety
            try:
                r = _http_session.get(f"{base.rstrip('/')}/api/config/arena_safety",
                                       timeout=0.25)
                if r.status_code == 200:
                    j = r.json()
                    if j.get("margin_m") is not None:
                        margins.append(float(j["margin_m"]))
                    if not j.get("enabled", True):
                        all_arena_enabled = False
                    if j.get("engaged"):
                        any_a_engaged = True
                        if j.get("reason"):
                            a_reasons.append(f"{did}: {j['reason']}")
            except Exception:
                pass
        out["ceiling"] = {
            "ceiling_m": min(ceilings) if ceilings else None,
            "engaged":   any_c_engaged,
            "reasons":   c_reasons,
        }
        out["arena_safety"] = {
            "enabled":  all_arena_enabled,
            "margin_m": min(margins) if margins else None,
            "engaged":  any_a_engaged,
            "reasons":  a_reasons,
        }
    except Exception:
        out["ceiling"] = {"ceiling_m": None, "engaged": False, "reasons": []}
        out["arena_safety"] = {"enabled": True, "margin_m": None,
                                "engaged": False, "reasons": []}

    # --- WS status (per drone) ---
    try:
        out["ws"] = {
            "available": HAS_WSCLIENT,
            "drones": {did: cli.status() for did, cli in drone_ws.items()},
        }
    except Exception:
        out["ws"] = {"available": HAS_WSCLIENT, "drones": {}}

    # --- Transport selector state ---
    try:
        with _transport_lock:
            out["transport"] = dict(_transport)
    except Exception:
        out["transport"] = {}

    # --- Missions ---
    try:
        out["missions"] = mission_manager.status()
    except Exception:
        out["missions"] = {}

    # Heartbeat removed — now handled by the C2's _heartbeat_loop()
    # background thread directly. No reason for the browser to be
    # involved in keeping the Pi watchdog alive.

    return jsonify(ok=True, **out)


@app.get("/proxy/diagnostics")
def proxy_diagnostics():
    """C2-side + Pi-side diagnostic snapshot.

    Everything that SHOULD NOT grow monotonically across flights is
    here. If a counter keeps climbing flight-after-flight, that's the
    leak. To use:
      1. Open this URL in a browser tab before flight 1.
      2. Note the thread_count + flight_logger.active_files + http_pool counts.
      3. Fly, land, fly, land.
      4. Refresh. Any number growing is a direct clue.
    """
    import threading as _th, gc as _gc
    with _slow_calls_lock:
        slow = list(_slow_calls)

    # --- C2 threads ---
    threads = _th.enumerate()
    by_name: dict[str, int] = {}
    for t in threads:
        n = t.name
        for prefix in ("ThreadPoolExecutor-", "Thread-", "obs-", "ws-"):
            if n.startswith(prefix):
                n = prefix + "*"
                break
        by_name[n] = by_name.get(n, 0) + 1

    # --- HTTP session connection pool ---
    http_pool_stats = {}
    try:
        adapter = _http_session.get_adapter("http://x")
        if hasattr(adapter, "poolmanager"):
            pm = adapter.poolmanager
            http_pool_stats = {
                "pool_connections_limit": getattr(adapter, "_pool_connections", None),
                "pool_maxsize_limit":     getattr(adapter, "_pool_maxsize",     None),
                "pools_cached":           len(pm.pools) if hasattr(pm, "pools") else None,
            }
    except Exception as e:
        http_pool_stats = {"error": str(e)[:120]}

    # --- FlightLogger ---
    try:
        with flight_logger._lock:
            fl_state = {
                "active_files": len(flight_logger._flights),
                "drone_ids":    list(flight_logger._flights.keys()),
                "running":      flight_logger._running,
                "log_dir":      str(flight_logger.log_dir),
            }
    except Exception as e:
        fl_state = {"error": str(e)[:120]}

    # --- DroneWS clients ---
    ws_clients = {}
    for did, cli in drone_ws.items():
        try:
            s = cli.status()
            ws_clients[did] = {
                "rc": s.get("rc"),
                "telemetry": s.get("telemetry"),
                "position":  s.get("position"),
                "rc_rtt_ms": s.get("rc_rtt_ms"),
                "rc_send_ms": s.get("rc_send_ms"),
                "consec_failures": dict(cli._consec_failures),
            }
        except Exception as e:
            ws_clients[did] = {"error": str(e)[:80]}

    # --- Per-drone Pi diagnostics ---
    per_drone = {}
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            continue
        # Skip fan-out for drones whose WS is all-down — same pattern
        # as fleet poll. Keeps this endpoint responsive under partial
        # outage.
        cli = drone_ws.get(str(did))
        if cli is not None:
            all_down = (not cli._ws_connected.get("telemetry") and
                        not cli._ws_connected.get("position") and
                        not cli._ws_connected.get("rc"))
            if all_down:
                per_drone[str(did)] = {"error": "offline (ws all down)"}
                continue
        try:
            r = _http_session.get(f"{base.rstrip('/')}/api/diagnostics", timeout=0.5)
            per_drone[str(did)] = r.json() if r.ok else {"error": f"http {r.status_code}"}
        except Exception as e:
            per_drone[str(did)] = {"error": str(e)[:120]}

    return jsonify(
        ok=True,
        # C2-side counters — these are what usually leak
        c2={
            "thread_count":    len(threads),
            "threads_by_name": by_name,
            "gc_counts":       list(_gc.get_count()),
            "http_pool":       http_pool_stats,
            "flight_logger":   fl_state,
            "ws_clients":      ws_clients,
            "slow_calls":      slow,
        },
        per_drone=per_drone,
        active_drone=active_drone_id,
    )


@app.get("/proxy/ws/status")
def proxy_ws_status():
    """Per-drone WS connection snapshot for the UI badge."""
    out = {did: cli.status() for did, cli in drone_ws.items()}
    return jsonify(
        ok=True,
        available=HAS_WSCLIENT,
        drones=out,
    )


@app.get("/proxy/config/transport")
def proxy_transport_get():
    """Return the current per-subsystem transport preference."""
    with _transport_lock:
        snap = dict(_transport)
    return jsonify(ok=True, transport=snap, ws_available=HAS_WSCLIENT)


@app.post("/proxy/config/transport")
def proxy_transport_set():
    """Update the transport preference for one or more subsystems.
    Body: {"rc":"http", "telemetry":"auto", "position":"ws"}

    Valid values: "auto" (WS if up, HTTP fallback), "ws" (force WS),
    "http" (force HTTP). Unknown subsystems / values are ignored.
    """
    global WS_RC_ENABLED
    data = request.get_json(silent=True) or {}
    allowed = {"auto", "ws", "http"}
    changed = {}
    with _transport_lock:
        for subsys in ("rc", "telemetry", "position"):
            v = data.get(subsys)
            if v is None:
                continue
            v = str(v).lower()
            if v in allowed:
                _transport[subsys] = v
                changed[subsys] = v
        snap = dict(_transport)
    WS_RC_ENABLED = snap["rc"] in ("auto", "ws")
    log_command("transport_set", {"changed": changed, "active": snap})
    print(f"[TRANSPORT] updated: {changed} → now {snap}")
    return jsonify(ok=True, transport=snap, changed=changed)


# ── Positioning subsystem proxy ───────────────────────────────────────────────

@app.get("/proxy/position")
def proxy_position_get():
    try:
        r = pi_get("/api/position", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.get("/proxy/position/events")
def proxy_position_events():
    """SSE proxy — streams ArUco position events from Pi to browser."""
    pi_url = PI_BASE + "/api/position/events"

    def generate():
        try:
            with _http_session.get(pi_url, stream=True, timeout=(3, 300)) as resp:
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
        except Exception as e:
            import json as _json
            yield f"data: {_json.dumps({'error': str(e)})}\n\n".encode()

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/proxy/telemetry/stream")
def proxy_telemetry_stream():
    """SSE proxy — streams telemetry events from Pi to browser (replaces polling)."""
    pi_url = PI_BASE + "/api/telemetry/stream"

    def generate():
        try:
            with _http_session.get(pi_url, stream=True, timeout=(3, 300)) as resp:
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
        except Exception as e:
            import json as _json
            yield f"data: {_json.dumps({'error': str(e)})}\n\n".encode()

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/proxy/position/video")
def proxy_position_video():
    """MJPEG proxy — streams ArUco-annotated frames from Pi."""
    pi_url = PI_BASE + "/api/position/video"

    def generate():
        try:
            with _http_session.get(pi_url, stream=True, timeout=(3, 300)) as resp:
                for chunk in resp.iter_content(chunk_size=16384):
                    if chunk:
                        yield chunk
        except Exception:
            pass

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


@app.get("/proxy/position/config")
def proxy_position_config_get():
    try:
        r = pi_get("/api/position/config", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/position/config")
def proxy_position_config_set():
    data = request.get_json(silent=True) or {}
    r = pi_post("/api/position/config", data)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})


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


@app.post("/proxy/calibration/start")
def proxy_calibration_start():
    did, base = _calib_target_base(request.args.get("drone_id"))
    if not base:
        return jsonify(ok=False, error=f"drone {did} not configured"), 404
    try:
        r = _http_session.post(f"{base.rstrip('/')}/api/calibration/start",
                               json={}, timeout=TIMEOUT_CMD)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.get("/proxy/calibration/status")
def proxy_calibration_status():
    did, base = _calib_target_base(request.args.get("drone_id"))
    if not base:
        return jsonify(ok=False, error=f"drone {did} not configured"), 404
    try:
        r = _http_session.get(f"{base.rstrip('/')}/api/calibration/status",
                              timeout=TIMEOUT_FAST)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/calibration/abort")
def proxy_calibration_abort():
    did, base = _calib_target_base(request.args.get("drone_id"))
    if not base:
        return jsonify(ok=False, error=f"drone {did} not configured"), 404
    try:
        r = _http_session.post(f"{base.rstrip('/')}/api/calibration/abort",
                               json={}, timeout=TIMEOUT_CMD)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/position/calibration")
def proxy_position_calibration():
    """Proxy NPZ calibration file upload to Pi."""
    if "file" not in request.files:
        return jsonify(ok=False, error="no file"), 400
    f = request.files["file"]
    try:
        resp = _http_session.post(
            PI_BASE + "/api/position/calibration",
            files={"file": (f.filename, f.read(), "application/octet-stream")},
            timeout=15,
        )
        return Response(resp.content, status=resp.status_code,
                        content_type=resp.headers.get("Content-Type", "application/json"))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


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


@app.get("/proxy/arena/config")
def proxy_arena_config_get():
    try:
        r = pi_get("/api/arena/config", timeout=TIMEOUT_STATUS)
        d = r.json()
        return jsonify(**_pi_arena_to_js(d))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/arena/config")
def proxy_arena_config_set():
    data = request.get_json(silent=True) or {}
    try:
        pi_payload = _js_arena_to_pi(data)
        r = pi_post("/api/arena/config", pi_payload)
        d = r.json()
        return jsonify(**_pi_arena_to_js(d))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/arena/config/reset")
def proxy_arena_config_reset():
    try:
        r = pi_post("/api/arena/config/reset", {})
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.get("/proxy/video/record/status")
def proxy_rec_status():
    try:
        r = pi_get("/api/video/record/status", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/video/record/start")
def proxy_rec_start():
    try:
        data = request.get_json(silent=True) or {}
        r = pi_post("/api/video/record/start", data)
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/video/record/stop")
def proxy_rec_stop():
    try:
        r = pi_post("/api/video/record/stop", {})
        return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


# ─── ArUco Seek (multi-drone hover-in-front-of-marker) ─────────────────────


def _aruco_resolve(drone_id: str | None):
    """Return observer for the given id, or the active drone if omitted."""
    did = str(drone_id) if drone_id else active_drone_id
    obs = aruco_fleet.get(did)
    if obs is None:
        return None, did
    return obs, did


@app.get("/proxy/aruco/state")
def proxy_aruco_state():
    """Snapshot of ONE observer (query ?id=1, defaults to active drone)."""
    did = request.args.get("id") or active_drone_id
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(running=False, drone_id=did, error="unknown drone"), 404
    return jsonify(obs.get_state())


@app.get("/proxy/aruco/fleet")
def proxy_aruco_fleet():
    """Snapshot of every observer in the fleet.

    Merged data flow:
      1. Start from the observer's own state dict (pos + telemetry) if
         the observer thread is actually running on this drone.
      2. For every drone in DRONES, fan out a lightweight GET to
         <base>/api/position in parallel (300 ms timeout). That endpoint
         returns the live fused pose even when no observer is started,
         which is exactly what manual-flight mode needs — otherwise the
         3D arena view stays empty until someone arms ArUco Seek.

    Merge policy: the observer's pos wins if present (it's been cached
    with IMU dead-reckoning); the /api/position fallback fills any gap.
    """
    observers = dict(aruco_fleet.all_states())
    # Ensure every configured drone has at least an entry to populate
    for did in DRONES.keys():
        did = str(did)
        if did not in observers:
            observers[did] = {"drone_id": did, "running": False}

    # Two data paths, in order of preference:
    #   1. WS cache (zero HTTP on the fleet-poll tick — <1 ms)
    #   2. Parallel fan-out to each Pi's /api/position
    #
    # We SKIP the HTTP fan-out for any drone whose WS is known-disconnected.
    # If the WS can't reach the Pi, HTTP won't either — trying anyway wastes
    # ~200 ms per drone per poll on connection-refused / timeout, which was
    # the real cause of the "500 timeout" and button-press delays (the fleet
    # poll was holding up Flask threads that other endpoints needed).
    need_http: list[tuple[str, str]] = []
    pos_mode = _transport_mode("position")
    for did, info in DRONES.items():
        did = str(did)
        base = (info or {}).get("base")
        if not base:
            continue
        # WS cache first (unless position transport is forced to http)
        cli = drone_ws.get(did) if pos_mode in ("auto", "ws") else None
        if cli is not None:
            pj, age = cli.latest_position()
            if pj is not None and age < 1.5:
                entry = observers.setdefault(did, {"drone_id": did})
                if entry.get("pos") is None and pj.get("pos") is not None:
                    entry["pos"] = pj["pos"]
                if entry.get("dir") is None and pj.get("dir") is not None:
                    entry["dir"] = pj["dir"]
                if entry.get("pos_vel") is None and pj.get("vel") is not None:
                    entry["pos_vel"] = pj["vel"]
                if entry.get("pos_stale") is None and pj.get("stale") is not None:
                    entry["pos_stale"] = pj["stale"]
                if entry.get("ref_markers") is None and pj.get("ref_markers") is not None:
                    entry["ref_markers"] = pj["ref_markers"]
                # Currently-visible markers — MUST be propagated so the 2D
                # halo + 3D highlight light up on manual flight too. When
                # the DroneObserver isn't started, the per-drone position
                # service is the only source for this, and we were dropping
                # it in both the WS-cache and HTTP-fallback paths.
                if entry.get("seen_markers") is None and pj.get("seen_markers") is not None:
                    entry["seen_markers"] = pj["seen_markers"]
                if entry.get("altitude_m") is None and pj.get("pos"):
                    try:
                        entry["altitude_m"] = float(pj["pos"][2])
                    except (IndexError, TypeError, ValueError):
                        pass
                entry["_pos_source"] = "ws"
                continue
            # WS exists but stale — skip HTTP if ALL three sockets are down
            # (i.e. the Pi is unreachable). Keep trying HTTP if only position
            # is down but telemetry/rc still connected (different failure mode).
            all_down = (not cli._ws_connected.get("telemetry") and
                        not cli._ws_connected.get("position") and
                        not cli._ws_connected.get("rc"))
            if all_down:
                observers.setdefault(did, {"drone_id": did, "ws_all_down": True})
                continue
        # Position forced to ws-only: don't attempt HTTP.
        if pos_mode == "ws":
            observers.setdefault(did, {"drone_id": did, "ws_only": True})
            continue
        need_http.append((did, base))

    def _fetch(did: str, base: str):
        try:
            resp = _http_session.get(f"{base.rstrip('/')}/api/position", timeout=0.3)
            if resp.status_code == 200:
                return did, resp.json()
        except Exception:
            pass
        return did, None

    if need_http:
        import concurrent.futures as _cf
        jobs = {}
        pool = _cf.ThreadPoolExecutor(max_workers=max(1, len(need_http)))
        try:
            for did, base in need_http:
                jobs[pool.submit(_fetch, did, base)] = did
            # Loop with a per-future timeout rather than as_completed's
            # global timeout — that one raises TimeoutError instead of
            # returning partial results, which used to bubble up as a
            # 500 whenever any Pi was slow/unreachable.
            deadline = time.time() + 0.6
            for fut in list(jobs):
                remaining = max(0.0, deadline - time.time())
                try:
                    did, pj = fut.result(timeout=remaining)
                except Exception:
                    continue
                if not pj:
                    continue
                entry = observers.setdefault(did, {"drone_id": did})
                if entry.get("pos") is None and pj.get("pos") is not None:
                    entry["pos"] = pj["pos"]
                if entry.get("dir") is None and pj.get("dir") is not None:
                    entry["dir"] = pj["dir"]
                if entry.get("pos_vel") is None and pj.get("vel") is not None:
                    entry["pos_vel"] = pj["vel"]
                if entry.get("pos_stale") is None and pj.get("stale") is not None:
                    entry["pos_stale"] = pj["stale"]
                if entry.get("ref_markers") is None and pj.get("ref_markers") is not None:
                    entry["ref_markers"] = pj["ref_markers"]
                # Currently-visible markers — MUST be propagated so the 2D
                # halo + 3D highlight light up on manual flight too. When
                # the DroneObserver isn't started, the per-drone position
                # service is the only source for this, and we were dropping
                # it in both the WS-cache and HTTP-fallback paths.
                if entry.get("seen_markers") is None and pj.get("seen_markers") is not None:
                    entry["seen_markers"] = pj["seen_markers"]
                if entry.get("altitude_m") is None and pj.get("pos"):
                    try:
                        entry["altitude_m"] = float(pj["pos"][2])
                    except (IndexError, TypeError, ValueError):
                        pass
                entry["_pos_source"] = "http"
        finally:
            # Don't wait for slow workers — let them finish in the
            # background. If we joined here the endpoint could block 4+
            # seconds when a drone is unreachable.
            pool.shutdown(wait=False)

    return jsonify(active=active_drone_id,
                   allow_live=aruco_fleet.allow_live,
                   observers=observers)


@app.get("/proxy/aruco/params")
def proxy_aruco_params_get():
    did = request.args.get("id") or active_drone_id
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(asdict_hover_defaults()), 200
    return jsonify(obs.get_params())


def asdict_hover_defaults():
    from dataclasses import asdict
    return asdict(AsHoverParams())


@app.post("/proxy/aruco/params")
def proxy_aruco_params_set():
    data = request.get_json(silent=True) or {}
    # If no id, broadcast to EVERY observer (fleet-wide knobs like
    # axis_locked should apply to the whole swarm, not just one drone).
    did = data.pop("id", None) or request.args.get("id")
    if did:
        obs, did = _aruco_resolve(str(did))
        if obs is None:
            return jsonify(ok=False, error="unknown drone"), 404
        applied = obs.update_params(data)
        return jsonify(ok=True, applied=applied, drone_id=did)
    # Fleet-wide fan-out
    applied_all = {}
    for d_id, o in aruco_fleet._obs.items():
        try:
            applied_all[d_id] = o.update_params(data)
        except Exception as e:
            applied_all[d_id] = {"error": str(e)[:80]}
    return jsonify(ok=True, applied=applied_all, fleet=True)


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


@app.get("/proxy/config/camera_face_center")
def proxy_camera_face_center_get():
    with _camera_face_center_lock:
        return jsonify(
            ok=True,
            enabled=_camera_face_center_enabled,
            xy=list(_camera_face_center_xy),
        )


@app.post("/proxy/config/camera_face_center")
def proxy_camera_face_center_set():
    """Enable/disable the toggle, optionally override the centre XY.
    Body: {"enabled": true, "xy": [0, 5.4]}"""
    global _camera_face_center_enabled, _camera_face_center_xy
    data = request.get_json(silent=True) or {}
    changed = {}
    with _camera_face_center_lock:
        if "enabled" in data:
            _camera_face_center_enabled = bool(data["enabled"])
            changed["enabled"] = _camera_face_center_enabled
        if "xy" in data and isinstance(data["xy"], (list, tuple)) and len(data["xy"]) >= 2:
            try:
                _camera_face_center_xy = (float(data["xy"][0]), float(data["xy"][1]))
                changed["xy"] = list(_camera_face_center_xy)
            except (TypeError, ValueError):
                pass
    _apply_camera_face_to_fleet()
    log_command("camera_face_center_set", changed)
    print(f"[CAMERA_FACE] enabled={_camera_face_center_enabled} xy={_camera_face_center_xy}")
    return jsonify(ok=True, changed=changed,
                   enabled=_camera_face_center_enabled,
                   xy=list(_camera_face_center_xy))


@app.get("/proxy/aruco/safety_margin")
def proxy_aruco_safety_margin_get():
    """Return the current autonomous-mode arena safety margin (metres).
    Reads from the active drone's observer since the margin is
    per-observer state; fleet-wide consistency is maintained by the
    POST endpoint broadcasting to every observer."""
    obs = aruco_fleet.get(str(active_drone_id))
    if obs is None:
        return jsonify(ok=False, safety_margin_m=None), 200
    with obs._lock:
        return jsonify(
            ok=True,
            safety_margin_m=float(obs._safety_margin_m),
            arena_bounds=dict(obs._arena_bounds),
        )


@app.post("/proxy/aruco/safety_margin")
def proxy_aruco_safety_margin_set():
    """Set the autonomous-mode arena safety margin. Body:
        {"safety_margin_m": 1.5}
    Broadcasts to every drone's observer so the fleet shares one value.
    """
    data = request.get_json(silent=True) or {}
    try:
        v = float(data.get("safety_margin_m"))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="safety_margin_m required"), 400
    v = max(0.1, min(5.0, v))   # clamp to a sane range
    for d_id, o in aruco_fleet._obs.items():
        try:
            o.set_arena_bounds({}, safety_margin_m=v)
        except Exception as e:
            print(f"[SAFETY] drone {d_id} set failed: {e}")
    log_command("safety_margin_set", {"safety_margin_m": v})
    print(f"[SAFETY] arena margin set to {v}m (fleet-wide)")
    return jsonify(ok=True, safety_margin_m=v)


@app.post("/proxy/aruco/start")
def proxy_aruco_start():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    log_command("aruco_start", {"id": did})
    obs.start()
    return jsonify(ok=True, drone_id=did)


@app.post("/proxy/aruco/stop")
def proxy_aruco_stop():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    log_command("aruco_stop", {"id": did})
    obs.stop()
    return jsonify(ok=True, drone_id=did)


@app.post("/proxy/aruco/target")
def proxy_aruco_target():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    mid = data.get("marker")
    if mid is not None:
        try:
            mid = int(mid)
        except (TypeError, ValueError):
            mid = None
    obs.set_target(mid)
    log_command("aruco_target", {"id": did, "marker": mid})
    return jsonify(ok=True, drone_id=did, marker=mid)


@app.post("/proxy/aruco/mode")
def proxy_aruco_mode():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    requested = (data.get("mode") or "").lower()
    # Respect global PAUSE — switching to LIVE while paused is exactly
    # the kind of autonomous-command surprise PAUSE exists to prevent.
    # Going back to OBSERVE is always allowed (it's a safety downgrade).
    if requested == "live":
        guarded = _pause_guard_response()
        if guarded is not None:
            return guarded
    if requested == "live" and not obs.allow_live:
        return jsonify(ok=False, mode=obs.mode,
                       error="LIVE mode disabled on this server (REMOTE_NO_LIVE=1)"), 403
    actual = obs.set_mode(requested)
    log_command("aruco_mode", {"id": did, "mode": actual})
    return jsonify(ok=(actual == requested), drone_id=did, mode=actual)


def _aruco_require_live(obs):
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    if obs.mode != "live":
        return jsonify(ok=False,
                       error=f"refused — observer mode is '{obs.mode}', switch to 'live' first"), 409
    return None


@app.post("/proxy/aruco/takeoff")
def proxy_aruco_takeoff():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    err = _aruco_require_live(obs)
    if err is not None:
        return err
    log_command("aruco_takeoff", {"id": did})
    return jsonify(ok=True, drone_id=did, result=obs.cmd_takeoff())


@app.post("/proxy/aruco/land")
def proxy_aruco_land():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    err = _aruco_require_live(obs)
    if err is not None:
        return err
    log_command("aruco_land", {"id": did})
    return jsonify(ok=True, drone_id=did, result=obs.cmd_land())


@app.post("/proxy/aruco/emergency")
def proxy_aruco_emergency():
    # Allowed at any mode — killswitch must always be reachable
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    log_command("aruco_emergency", {"id": did})
    return jsonify(ok=True, drone_id=did, result=obs.cmd_emergency())


@app.post("/proxy/aruco/rc_stop")
def proxy_aruco_rc_stop():
    data = request.get_json(silent=True) or {}
    did = str(data.get("id") or request.args.get("id") or active_drone_id)
    obs, did = _aruco_resolve(did)
    if obs is None:
        return jsonify(ok=False, error="unknown drone"), 404
    log_command("aruco_rc_stop", {"id": did})
    return jsonify(ok=True, drone_id=did, result=obs.cmd_rc_stop())


@app.get("/proxy/aruco/video.mjpg")
def proxy_aruco_video():
    """Pass-through MJPEG for the observer's drone (separate from /proxy/video
    so the main UI's video stream and the ArUco Seek panel don't collide)."""
    did = request.args.get("id") or active_drone_id
    obs, did = _aruco_resolve(did)
    if obs is None:
        return Response(b"", status=404)
    upstream_url = f"{obs.api_base}/api/position/video"
    try:
        upstream = requests.get(upstream_url, stream=True, timeout=10)
    except Exception as e:
        return Response(f"upstream error: {e}".encode(), status=502, mimetype="text/plain")
    content_type = upstream.headers.get(
        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
    )

    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        except (GeneratorExit, requests.exceptions.RequestException):
            pass
        finally:
            try:
                upstream.close()
            except Exception:
                pass
    return Response(generate(), content_type=content_type)


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


@app.get("/proxy/missions/status")
def proxy_missions_status():
    return jsonify(mission_manager.status())


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


@app.get("/proxy/position/presets")
def proxy_position_presets_list():
    return jsonify(ok=True, presets=_load_position_presets(),
                   path=str(POSITION_PRESETS_PATH))


@app.post("/proxy/position/presets")
def proxy_position_presets_save():
    """Save a named preset. Body: {"name": "X", "params": {...}}"""
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    params = data.get("params")
    if not name or not isinstance(params, dict):
        return jsonify(ok=False, error="name + params required"), 400
    presets = _load_position_presets()
    presets[name] = params
    try:
        _save_position_presets(presets)
    except Exception as e:
        return jsonify(ok=False, error=f"save failed: {e}"), 500
    log_command("position_preset_save", {"name": name})
    return jsonify(ok=True, name=name)


@app.delete("/proxy/position/presets")
def proxy_position_presets_delete():
    name = str(request.args.get("name", "")).strip()
    if not name:
        return jsonify(ok=False, error="name required"), 400
    presets = _load_position_presets()
    if name not in presets:
        return jsonify(ok=False, error="preset not found"), 404
    del presets[name]
    try:
        _save_position_presets(presets)
    except Exception as e:
        return jsonify(ok=False, error=f"save failed: {e}"), 500
    log_command("position_preset_delete", {"name": name})
    return jsonify(ok=True)


@app.post("/proxy/position/presets/apply")
def proxy_position_presets_apply():
    """Load a preset and fan it out to every drone via /proxy/position/config.
    Body: {"name": "..."} — any unknown keys in the preset are accepted
    by the position config endpoint (it silently ignores unknown fields)."""
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    if not name:
        return jsonify(ok=False, error="name required"), 400
    presets = _load_position_presets()
    params = presets.get(name)
    if not isinstance(params, dict):
        return jsonify(ok=False, error="preset not found"), 404
    # Fan out to every drone's /api/position/config
    results = {}
    for did, info in DRONES.items():
        base = (info or {}).get("base")
        if not base:
            results[str(did)] = {"ok": False, "error": "no base url"}
            continue
        try:
            r = _http_session.post(f"{base.rstrip('/')}/api/position/config",
                                   json=params, timeout=2.0)
            results[str(did)] = r.json() if r.ok else {"ok": False, "status": r.status_code}
        except Exception as e:
            results[str(did)] = {"ok": False, "error": str(e)[:120]}
    log_command("position_preset_apply", {"name": name})
    ok_count = sum(1 for v in results.values() if v.get("ok"))
    return jsonify(ok=True, name=name, applied_to=ok_count,
                   total=len(results), results=results)


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


@app.get("/proxy/missions/presets")
def proxy_mission_presets_list():
    """Return all presets grouped by mission type."""
    return jsonify(ok=True, presets=_load_mission_presets(),
                   path=str(MISSION_PRESETS_PATH))


@app.post("/proxy/missions/presets")
def proxy_mission_presets_save():
    """Create or overwrite a named preset.
    Body: {"mission_type": "scan_all", "name": "tight-margin",
           "params": {...}}"""
    data = request.get_json(silent=True) or {}
    mtype = str(data.get("mission_type", "")).strip()
    name  = str(data.get("name", "")).strip()
    params = data.get("params")
    if not mtype or not name or not isinstance(params, dict):
        return jsonify(ok=False, error="mission_type, name, params required"), 400
    if mtype not in ("scan_all", "capture_targets"):
        return jsonify(ok=False, error=f"unknown mission_type: {mtype}"), 400
    # Minimal sanity checks per mission type — we don't want a broken
    # preset to silently save.
    if mtype == "scan_all":
        if "target_markers" not in params:
            return jsonify(ok=False, error="target_markers required for scan_all"), 400
    if mtype == "capture_targets":
        if not isinstance(params.get("target_boxes"), list):
            return jsonify(ok=False, error="target_boxes (list) required for capture_targets"), 400
    presets = _load_mission_presets()
    presets.setdefault(mtype, {})[name] = params
    try:
        _save_mission_presets(presets)
    except Exception as e:
        return jsonify(ok=False, error=f"save failed: {e}"), 500
    log_command("mission_preset_save", {"mission_type": mtype, "name": name})
    return jsonify(ok=True, mission_type=mtype, name=name)


@app.delete("/proxy/missions/presets")
def proxy_mission_presets_delete():
    """Delete a named preset. Query: ?mission_type=X&name=Y"""
    mtype = str(request.args.get("mission_type", "")).strip()
    name  = str(request.args.get("name", "")).strip()
    if not mtype or not name:
        return jsonify(ok=False, error="mission_type, name required"), 400
    presets = _load_mission_presets()
    bucket  = presets.get(mtype, {})
    if name not in bucket:
        return jsonify(ok=False, error="preset not found"), 404
    del bucket[name]
    try:
        _save_mission_presets(presets)
    except Exception as e:
        return jsonify(ok=False, error=f"save failed: {e}"), 500
    log_command("mission_preset_delete", {"mission_type": mtype, "name": name})
    return jsonify(ok=True)


@app.post("/proxy/missions/scan_all/start")
def proxy_missions_scan_all_start():
    guarded = _pause_guard_response()
    if guarded is not None:
        return guarded
    data = request.get_json(silent=True) or {}
    drone_ids = data.get("drone_ids") or []
    if not isinstance(drone_ids, list) or not drone_ids:
        return jsonify(ok=False, error="drone_ids (non-empty list) required"), 400
    target_markers = _parse_marker_list(data.get("target_markers", "1-12"))
    if not target_markers:
        return jsonify(ok=False, error="target_markers must parse to at least one id"), 400
    hover_seconds = float(data.get("hover_seconds", 3.0))
    approach_tolerance_m = float(data.get("approach_tolerance_m", 0.30))
    approach_skew_tol   = float(data.get("approach_skew_tol", 0.12))
    approach_err_x_tol  = float(data.get("approach_err_x_tol", 0.15))
    auto_takeoff = bool(data.get("auto_takeoff", False))
    ok, msg = mission_manager.start_scan_all(
        drone_ids=[str(d) for d in drone_ids],
        target_markers=target_markers,
        hover_seconds=hover_seconds,
        approach_tolerance_m=approach_tolerance_m,
        approach_skew_tol=approach_skew_tol,
        approach_err_x_tol=approach_err_x_tol,
        auto_takeoff=auto_takeoff,
    )
    log_command("mission_scan_all_start", {
        "drone_ids": drone_ids, "target_markers": target_markers,
        "hover_seconds": hover_seconds, "ok": ok, "msg": msg,
    })
    status = 200 if ok else 409
    return jsonify(ok=ok, message=msg, status=mission_manager.status()), status


@app.post("/proxy/missions/capture_targets/start")
def proxy_missions_capture_targets_start():
    """Launch the SDC26 capture-all-targets mission. Body:
    {
      "drone_ids":    ["1", "2", ...],
      "target_boxes": [{"id":1,"x":-5.0,"y":2.0,"z":0.0}, ...],
                                                  # z optional — marker
                                                  # height above floor
                                                  # (e.g. 0.5 for a stand)
      "home_xy":      [x, y],                    # team home coord
      "arena_face_xy":[x, y],                    # where the camera aims
      "hover_above_m": 1.5,                      # HEIGHT above the target
                                                  # marker; waypoint Z =
                                                  # box.z + hover_above_m
      "hover_seconds": 4.0,
      "auto_takeoff":  false
    }
    """
    guarded = _pause_guard_response()
    if guarded is not None:
        return guarded
    data = request.get_json(silent=True) or {}
    drone_ids = data.get("drone_ids") or []
    if not isinstance(drone_ids, list) or not drone_ids:
        return jsonify(ok=False, error="drone_ids (non-empty list) required"), 400
    target_boxes = data.get("target_boxes") or []
    if not isinstance(target_boxes, list) or not target_boxes:
        return jsonify(ok=False, error="target_boxes (non-empty list) required"), 400
    home_xy = data.get("home_xy", [0.0, 1.5])
    arena_face_xy = data.get("arena_face_xy", [0.0, 5.4])
    hover_above_m = float(data.get("hover_above_m", 1.5))
    hover_seconds = float(data.get("hover_seconds", 4.0))
    nav_tol_xy_m  = float(data.get("nav_tol_xy_m", 0.3))
    auto_takeoff  = bool(data.get("auto_takeoff", False))
    ok, msg = mission_manager.start_capture_all_targets(
        drone_ids=[str(d) for d in drone_ids],
        target_boxes=target_boxes,
        home_xy=tuple(home_xy),
        arena_face_xy=tuple(arena_face_xy),
        hover_above_m=hover_above_m,
        hover_seconds=hover_seconds,
        nav_tol_xy_m=nav_tol_xy_m,
        auto_takeoff=auto_takeoff,
    )
    log_command("mission_capture_targets_start", {
        "drone_ids": drone_ids,
        "target_boxes": target_boxes,
        "ok": ok, "msg": msg,
    })
    status = 200 if ok else 409
    return jsonify(ok=ok, message=msg, status=mission_manager.status()), status


@app.post("/proxy/missions/stop")
def proxy_missions_stop():
    data = request.get_json(silent=True) or {}
    land = bool(data.get("land", False))
    ok = mission_manager.stop(land=land)
    log_command("mission_stop", {"land": land, "ok": ok})
    return jsonify(ok=ok, status=mission_manager.status())


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


@app.post("/proxy/missions/scan_and_capture/start")
def proxy_scan_and_capture_start():
    """Scan-and-capture: the first selected drone rotates to discover
    SDC26 target boxes (IDs 31-36 Blue, 41-46 Red) via ArUco. All
    selected drones then visit boxes concurrently at stacked altitudes
    using the standard CaptureAllTargetsMission (target-claim prevents
    two drones from approaching the same box; per-drone altitude offset
    keeps 3-D paths separated). Body:
      {
        "drone_ids":      ["2", "1"],       # multi-drone list; first scans
        "drone_id":       "2",              # (legacy single-drone fallback)
        "rotation_deg":   60,               # degrees per rotate step
        "rotation_steps": 6,                # full 360° by default
        "dwell_s":        2.0,              # observation window per step
        "hover_seconds":  3.0,              # dwell over each target box
        "hover_above_m":  1.5,              # HEIGHT above the target marker
        "home_xy":        [0.0, 1.5],       # landing / return point
        "arena_face_xy":  [0.0, 5.4],       # where camera points during capture
        "nav_tol_xy_m":   0.3,              # arrival tolerance
        "altitude_stack_m": 1.0,            # per-drone Z-offset step
        "min_separation_m": 1.2             # inter-drone 3-D proximity pause
      }
    """
    with _scan_cap_lock:
        if _scan_cap_state["active"]:
            return jsonify(ok=False, error="scan-and-capture already active",
                           state=dict(_scan_cap_state)), 409
    guarded = _pause_guard_response()
    if guarded is not None:
        return guarded
    data = request.get_json(silent=True) or {}
    # Accept either `drone_ids: [list]` (preferred, multi-drone) or the
    # legacy `drone_id: str` (single-drone). Falls back to active drone.
    drone_ids_raw = data.get("drone_ids")
    if isinstance(drone_ids_raw, list) and drone_ids_raw:
        drone_ids = [str(d) for d in drone_ids_raw if str(d) in DRONES]
    else:
        did = str(data.get("drone_id") or active_drone_id)
        drone_ids = [did] if did in DRONES else []
    if not drone_ids:
        return jsonify(ok=False, error="no valid drones selected"), 400
    rotation_deg   = max(10, min(180, int(data.get("rotation_deg", 60))))
    rotation_steps = max(1, min(24, int(data.get("rotation_steps", 6))))
    dwell_s        = max(0.5, min(10.0, float(data.get("dwell_s", 2.0))))
    hover_seconds  = max(1.0, min(20.0, float(data.get("hover_seconds", 3.0))))
    hover_above_m  = max(0.3, min(5.0, float(data.get("hover_above_m", 1.5))))
    home_xy        = data.get("home_xy") or [0.0, 1.5]
    arena_face_xy  = data.get("arena_face_xy") or [0.0, 5.4]
    nav_tol_xy_m   = max(0.1, min(2.0, float(data.get("nav_tol_xy_m", 0.3))))
    altitude_stack_m = max(0.0, min(3.0, float(data.get("altitude_stack_m", 1.0))))
    min_separation_m = max(0.0, min(5.0, float(data.get("min_separation_m", 1.2))))
    with _scan_cap_lock:
        _scan_cap_state.update({
            "active":        True,
            "phase":         "starting",
            "drone_id":      drone_ids[0],    # legacy single-id field — first drone
            "drone_ids":     drone_ids,
            "step_name":     "starting",
            "n_detected":    0,
            "targets_found": {},
            "target_boxes":  [],
            "started_at":    time.time(),
            "ended_at":      0.0,
            "elapsed_s":     0.0,
            "last_error":    None,
            "result":        None,
        })
    _scan_cap_abort.clear()
    threading.Thread(
        target=_scan_and_capture_thread,
        args=(drone_ids, rotation_deg, rotation_steps, dwell_s,
              hover_seconds, hover_above_m,
              tuple(home_xy), tuple(arena_face_xy), nav_tol_xy_m,
              altitude_stack_m, min_separation_m),
        daemon=True, name="scan-and-capture",
    ).start()
    log_command("scan_and_capture_start", {
        "drone_ids": drone_ids, "rotation_steps": rotation_steps,
        "hover_seconds": hover_seconds,
        "altitude_stack_m": altitude_stack_m,
        "min_separation_m": min_separation_m,
    })
    return jsonify(ok=True, message="scan-and-capture started",
                   drone_ids=drone_ids)


@app.get("/proxy/missions/scan_and_capture/status")
def proxy_scan_and_capture_status():
    with _scan_cap_lock:
        snap = dict(_scan_cap_state)
    return jsonify(ok=True, **snap)


@app.post("/proxy/missions/scan_and_capture/abort")
def proxy_scan_and_capture_abort():
    """Abort the scan-and-capture. Sets a flag that the background thread
    checks; the thread exits at the next dwell boundary. If the capture
    mission has already handed off to mission_manager, also stop that."""
    _scan_cap_abort.set()
    # Also try to stop any running mission (best-effort — might not be ours)
    try:
        if _scan_cap_state.get("phase") in {"capture", "done"}:
            mission_manager.stop(land=False)
    except Exception:
        pass
    with _scan_cap_lock:
        active = _scan_cap_state["active"]
    return jsonify(ok=True, message="abort requested", active=active)


@app.get("/proxy/missions/trace")
def proxy_missions_trace():
    """Download the trace log for the most recent mission. The mission
    class writes a JSONL file per run; we return the current one (or the
    most recent if no mission is active)."""
    from pathlib import Path as _P
    import glob as _glob
    path = None
    try:
        cur = mission_manager.current
        if cur is not None and getattr(cur, "trace_path", None):
            path = cur.trace_path
    except Exception:
        pass
    if not path:
        # Fall back to newest file in the logs dir
        try:
            from aruco_seek_multi import MISSION_LOG_DIR
            files = sorted(_glob.glob(str(MISSION_LOG_DIR / "mission_*.jsonl")))
            if files:
                path = files[-1]
        except Exception:
            pass
    if not path or not _P(path).exists():
        return jsonify(ok=False, error="no trace available yet"), 404
    return send_file(path, mimetype="application/x-ndjson",
                     as_attachment=True,
                     download_name=_P(path).name)


@app.get("/proxy/missions/traces")
def proxy_missions_traces():
    """List all mission trace files on disk with size and mtime."""
    import glob as _glob
    from aruco_seek_multi import MISSION_LOG_DIR
    files = []
    try:
        for f in sorted(_glob.glob(str(MISSION_LOG_DIR / "mission_*.jsonl"))):
            p = Path(f)
            st = p.stat()
            files.append({
                "name": p.name,
                "size": st.st_size,
                "mtime": st.st_mtime,
            })
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500
    return jsonify(ok=True, files=files)


# ─── Environment & Wi-Fi control — pass-through to active drone ─────────────

@app.get("/proxy/environment")
def proxy_environment_get():
    try:
        r = _http_session.get(f"{PI_BASE}/api/environment", timeout=3)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/environment")
def proxy_environment_set():
    data = request.get_json(silent=True) or {}
    log_command("environment_set", data)
    try:
        r = _http_session.post(f"{PI_BASE}/api/environment", json=data, timeout=4)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.get("/proxy/wifi/status")
def proxy_wifi_status():
    try:
        r = _http_session.get(f"{PI_BASE}/api/wifi/status", timeout=3)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/wifi/channel")
def proxy_wifi_channel():
    data = request.get_json(silent=True) or {}
    log_command("wifi_channel_set", data)
    try:
        r = _http_session.post(f"{PI_BASE}/api/wifi/channel", json=data, timeout=8)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/wifi/scan")
def proxy_wifi_scan():
    data = request.get_json(silent=True) or {}
    try:
        r = _http_session.post(f"{PI_BASE}/api/wifi/scan", json=data, timeout=10)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


# ── Magnetometer (Anafi) ────────────────────────────────────────────────────
# Parrot Anafi requires a figure-8 magnetometer calibration whenever the drone
# is moved between locations or re-powered. unified_api_server.py exposes the
# raw GET /api/magneto and POST /api/magneto/calibrate; here we expose a
# higher-level /proxy/magneto/recalibrate that drives the full cycle.

@app.get("/proxy/magneto")
def proxy_magneto_status():
    """Current magnetometer calibration status for the active drone."""
    try:
        r = pi_get("/api/magneto", timeout=TIMEOUT_STATUS)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


@app.post("/proxy/magneto/calibrate")
def proxy_magneto_calibrate():
    """One-shot: send the StartMagnetoCalibration command and return.
    The caller is then responsible for the figure-8 dance and for polling
    /proxy/magneto until all axes report ok."""
    log_command("magneto_calibrate")
    try:
        r = pi_post("/api/magneto/calibrate", timeout=TIMEOUT_CMD)
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 502


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


@app.post("/proxy/magneto/recalibrate")
def proxy_magneto_recalibrate():
    """Blocking orchestrator: runs the full recalibration cycle and returns a
    single summary JSON. See /proxy/magneto/recalibrate/stream for live
    per-step progress.

    Body (optional):
      { "timeout_s": 60, "poll_s": 1.0 }"""
    data = request.get_json(silent=True) or {}
    try:
        timeout_s = max(5.0, float(data.get("timeout_s", 60)))
        poll_s = max(0.25, float(data.get("poll_s", 1.0)))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="timeout_s/poll_s must be numeric"), 400

    log_command("magneto_recalibrate", {"timeout_s": timeout_s, "poll_s": poll_s})

    final = {"ok": False, "error": "no final event"}
    for ev in _magneto_cycle(timeout_s, poll_s):
        if ev.get("kind") == "final":
            final = {k: v for k, v in ev.items() if k != "kind"}
    return jsonify(final), 200


@app.get("/proxy/magneto/recalibrate/stream")
def proxy_magneto_recalibrate_stream():
    """SSE stream of the recalibration cycle. Used by the wizard GUI to light
    up each step + per-axis indicator as it happens.

    Query params: timeout_s (default 60), poll_s (default 1.0)."""
    try:
        timeout_s = max(5.0, float(request.args.get("timeout_s", 60)))
        poll_s = max(0.25, float(request.args.get("poll_s", 1.0)))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="timeout_s/poll_s must be numeric"), 400

    log_command("magneto_recalibrate_stream",
                {"timeout_s": timeout_s, "poll_s": poll_s})

    def generate():
        for ev in _magneto_cycle(timeout_s, poll_s):
            yield f"data: {json.dumps(ev)}\n\n".encode()

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


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
