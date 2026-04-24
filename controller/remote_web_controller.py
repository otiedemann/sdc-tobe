import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, send_file

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
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
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

app = Flask(__name__)

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

        header = {
            "type": "takeoff", "ts": time.time(), "drone_id": did,
            "drone_name": self.drones.get(did, {}).get("name"),
            "git_revision": _GIT_REVISION,
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

HTML = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Drone Remote Controller</title>
  <!-- Three.js for optional 3D arena view. Loaded up-front so the 3D
       checkbox handler can init the scene the first time it's ticked.
       Uses ES module imports via importmap, which works in Chrome/Safari
       desktop. Falls back gracefully — if the module fails to load, the
       3D checkbox will log an error and the 2D view still works. -->
  <script type=\"importmap\">
  {
    \"imports\": {
      \"three\": \"https://unpkg.com/three@0.161.0/build/three.module.js\",
      \"three/addons/\": \"https://unpkg.com/three@0.161.0/examples/jsm/\"
    }
  }
  </script>
  <style>
    body { background:#0f172a; color:#e2e8f0; font-family:Arial,sans-serif; margin:0; padding:16px; }
    .row { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }
    .panel { background:#111827; border:1px solid #334155; border-radius:8px; padding:12px; }
    .grid { display:grid; grid-template-columns:repeat(3,70px); gap:8px; }
    button { height:52px; border-radius:8px; border:1px solid #475569; background:#1e293b; color:#e2e8f0; font-weight:700; cursor:pointer; }
    button:active, .active { background:#0ea5e9; color:#001018; }
    .small { color:#94a3b8; font-size:12px; }
    .status-wrap { margin-top:8px; display:flex; gap:16px; flex-wrap:wrap; align-items:center; }
    .meter { min-width:220px; }
    .meter-label { font-size:12px; color:#94a3b8; margin-bottom:4px; }
    .meter-track { height:12px; border-radius:999px; background:#1f2937; border:1px solid #334155; overflow:hidden; }
    .meter-fill { height:100%; width:0%; background:#22c55e; transition:width .2s ease, background .2s ease; }
    .adv { margin-top:8px; border-top:1px solid #334155; padding-top:8px; }
    .adv-grid { display:grid; grid-template-columns:repeat(3,minmax(100px,1fr)); gap:8px; }
    .adv input { height:36px; border-radius:8px; border:1px solid #475569; background:#0f172a; color:#e2e8f0; padding:0 8px; }
    .drone-bar { display:flex; gap:8px; margin-bottom:12px; }
    .drone-btn { height:44px; padding:0 18px; border-radius:8px; border:2px solid #475569; background:#1e293b; color:#e2e8f0; font-weight:700; cursor:pointer; font-size:14px; transition:all .15s; }
    .drone-btn.selected { background:#0ea5e9; color:#001018; border-color:#0ea5e9; }
    .drone-btn:hover:not(.selected) { border-color:#94a3b8; }
    .drone-type { font-size:10px; font-weight:400; opacity:.7; display:block; line-height:1; }
    .video-panel { margin-top:12px; }
    .video-panel img { max-width:100%; border-radius:8px; background:#000; }
    .video-controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
    .video-controls button { height:36px; font-size:12px; padding:0 12px; }
    .video-controls select, .video-controls input { height:36px; border-radius:8px; border:1px solid #475569; background:#0f172a; color:#e2e8f0; padding:0 8px; }
    .video-status { font-size:11px; color:#94a3b8; margin-top:4px; }
    .video-url { font-size:11px; color:#38bdf8; word-break:break-all; }
    .pos-panel { min-width:260px; }
    .pos-coords { font-family:monospace; font-size:14px; letter-spacing:0.05em; margin:8px 0; }
    .pos-x { color:#38bdf8; } .pos-y { color:#4ade80; } .pos-z { color:#fb923c; }
    .pos-stale { color:#f59e0b; font-size:11px; }
    .arena-canvas { display:block; border-radius:6px; background:#0f172a; border:1px solid #334155; }
    .pos-cfg { margin-top:8px; border-top:1px solid #334155; padding-top:8px; }
    .pos-cfg label { font-size:12px; color:#94a3b8; }
    .pos-cfg input, .pos-cfg select { height:32px; border-radius:6px; border:1px solid #475569; background:#0f172a; color:#e2e8f0; padding:0 6px; }
    .pos-cfg button { height:32px; font-size:12px; padding:0 10px; }
    /* ── Info-icon (ⓘ) system for tuning parameters ──────────────── */
    .info-icon {
      display:inline-flex; align-items:center; justify-content:center;
      width:13px; height:13px; margin-left:4px;
      border:1px solid #64748b; border-radius:50%;
      color:#94a3b8; background:transparent;
      font: italic 700 9px/1 Georgia, 'Times New Roman', serif;
      cursor:pointer; user-select:none; vertical-align:middle;
      transition: border-color .12s, color .12s, background .12s;
    }
    .info-icon:hover { border-color:#38bdf8; color:#38bdf8; background:#0b1424; }
    #param_info_modal {
      display:none; position:fixed; inset:0;
      background:rgba(0,0,0,0.6); z-index:2000;
      justify-content:center; align-items:center;
    }
    #param_info_modal .box {
      background:#111827; border:1px solid #38bdf8; border-radius:8px;
      padding:16px 20px; max-width:520px; width:90%;
      color:#e2e8f0; box-shadow:0 12px 44px rgba(0,0,0,0.7);
    }
    #param_info_modal .title   { color:#38bdf8; font-weight:700; font-size:14px; margin-bottom:4px; }
    #param_info_modal .subtitle{ color:#64748b; font-size:11px; font-family:monospace; margin-bottom:10px; }
    #param_info_modal .body    { font-size:13px; line-height:1.55; white-space:pre-wrap; }
    #param_info_modal .range   { font-size:11px; color:#94a3b8; margin-top:10px; font-family:monospace; }
    #param_info_modal .close {
      margin-top:12px; padding:6px 14px; background:#334155;
      border:1px solid #64748b; color:#e2e8f0; border-radius:5px;
      cursor:pointer; font-size:12px;
    }
    #param_info_modal .close:hover { background:#475569; }
    /* ── ArUco Seek panel ─────────────────────────────────────────── */
    #aruco_panel { margin-top:16px; padding:12px; background:#0b1220; border:1px solid #334155; border-radius:8px; }
    #aruco_panel h3 { margin:0 0 10px 0; color:#38bdf8; font-size:15px; }
    #aruco_panel .arc-grid { display:grid; grid-template-columns:minmax(360px,1fr) minmax(420px,1fr); gap:12px; align-items:flex-start; }
    #aruco_panel canvas.arc-topdown { background:#0b1220; border:1px solid #1e293b; border-radius:4px; display:block; width:100%; max-width:420px; height:auto; }
    #aruco_panel img.arc-video { width:100%; max-width:480px; background:#0f172a; border-radius:4px; min-height:180px; }
    #aruco_panel .arc-readout { font-family:'SF Mono','Menlo',monospace; font-size:11px; line-height:1.55; color:#cbd5e1; }
    #aruco_panel .arc-readout .k { color:#94a3b8; display:inline-block; min-width:80px; }
    #aruco_panel .arc-readout .pd { color:#64748b; font-size:10px; }
    #aruco_panel .arc-readout b { color:#38bdf8; font-weight:600; }
    #aruco_panel .arc-params { max-height:360px; overflow-y:auto; padding-right:6px; }
    #aruco_panel .arc-row { display:flex; align-items:center; gap:6px; margin-bottom:3px; font-size:11px; }
    #aruco_panel .arc-row label { width:150px; color:#cbd5e1; flex-shrink:0; font-size:10px; }
    #aruco_panel .arc-row input[type=range] { flex:1; min-width:80px; }
    #aruco_panel .arc-row input[type=number] { width:60px; background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:3px; padding:2px 4px; font-size:10px; }
    #aruco_panel .arc-pgroup { border-top:1px dashed #334155; margin-top:6px; padding-top:4px; }
    #aruco_panel .arc-pgroup-label { font-size:9px; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:3px; }
    #aruco_panel .mode-seg { display:inline-flex; border:1px solid #334155; border-radius:4px; overflow:hidden; vertical-align:middle; }
    #aruco_panel .mode-seg button { background:transparent; border:0; border-radius:0; padding:5px 12px; font-weight:600; color:#94a3b8; font-size:11px; }
    #aruco_panel .mode-seg button.active.observe { background:#065f46; color:#ecfdf5; }
    #aruco_panel .mode-seg button.active.live    { background:#b91c1c; color:#fee2e2; }
    #aruco_panel .mode-seg button:disabled { opacity:0.45; cursor:not-allowed; }
    #aruco_panel .arc-manual { display:none; gap:6px; align-items:center; padding:4px 6px; background:#450a0a; border:1px solid #ef4444; border-radius:4px; }
    #aruco_panel .arc-manual.show { display:inline-flex; }
    #aruco_panel .arc-manual button { font-size:11px; padding:4px 8px; }
    #arc_live_banner { display:none; background:#b91c1c; color:#fee2e2; padding:6px 10px; border-radius:4px; font-weight:700; letter-spacing:0.04em; margin-bottom:8px; box-shadow:0 0 0 2px #fbbf24 inset; text-align:center; font-size:12px; }
    #arc_live_banner.show { display:block; animation:arcpulse 1.6s ease-in-out infinite; }
    @keyframes arcpulse { 0%,100% { box-shadow:0 0 0 2px #fbbf24 inset; } 50% { box-shadow:0 0 0 4px #fbbf24 inset; } }
    /* PAUSED banner pulse (stronger, so it can't be missed) */
    @keyframes pausepulse { 0%,100% { box-shadow:0 0 0 2px #facc15 inset; } 50% { box-shadow:0 0 0 5px #fde68a inset; } }
    /* Dim autonomous-mode controls while paused to make the blocked
       state obvious — operator can still see them, but they're clearly
       disabled. Manual (WASD) controls remain at full opacity. */
    body.paused-mode #mission_panel,
    body.paused-mode #missions_panel,
    body.paused-mode #aruco_panel { opacity:0.45; filter:grayscale(0.6); pointer-events:none; }

    /* Collapsible panels — click the header to toggle. State persists in
       localStorage so each operator keeps their preferred layout. */
    .collapsible-toggle {
      cursor: pointer; user-select: none;
      display: flex; align-items: center; gap: 8px;
      padding: 2px 0;
    }
    .collapsible-toggle:hover { color: #60a5fa; }
    .collapsible-toggle:before {
      content: '▾'; transition: transform 0.15s;
      display: inline-block; font-size: 11px; color: #94a3b8; width: 10px;
    }
    .collapsible.collapsed > .collapsible-toggle:before {
      transform: rotate(-90deg);
    }
    .collapsible.collapsed > .collapsible-body {
      display: none !important;
    }
    /* Video panel special case: keep the <img> alive (= still decoding
       MJPEG) but not visually rendered when the panel is collapsed. */
    #video_panel.collapsible.collapsed > .collapsible-body {
      display: block !important;
      height: 0; overflow: hidden; margin: 0; padding: 0;
    }
    body.arc-live-mode { box-shadow:0 0 0 4px #b91c1c inset; }
    .arc-rc-sent { color:#f87171; font-weight:600; }
    /* ── Special Missions panel ──────────────────────────────────── */
    #missions_panel { margin-top:16px; padding:12px; background:#0b1220; border:1px solid #334155; border-radius:8px; }
    #missions_panel h3 { margin:0 0 10px 0; color:#a78bfa; font-size:15px; }
    #missions_panel .mis-row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; font-size:12px; margin-bottom:6px; }
    #missions_panel select, #missions_panel input { background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:4px; padding:4px 6px; font-size:12px; }
    #missions_panel .mis-status { font-family:monospace; font-size:11px; color:#cbd5e1; background:#0f172a; border:1px solid #1e293b; border-radius:4px; padding:8px; max-height:200px; overflow-y:auto; white-space:pre-wrap; }
    #missions_panel .mis-drone-line { padding:2px 0; border-bottom:1px solid #1e293b; }
    #missions_panel .mis-drone-line b { color:#38bdf8; }
    #missions_panel .mis-badge { display:inline-block; padding:1px 6px; border-radius:3px; font-size:10px; font-weight:600; margin-left:6px; }
    #missions_panel .mis-badge.idle { background:#1e293b; color:#94a3b8; }
    #missions_panel .mis-badge.scan { background:#1e3a5f; color:#60a5fa; }
    #missions_panel .mis-badge.approach { background:#164e63; color:#22d3ee; }
    #missions_panel .mis-badge.hover { background:#065f46; color:#4ade80; }
    #missions_panel .mis-badge.wait { background:#451a03; color:#fbbf24; }
    #missions_panel .mis-badge.done { background:#14532d; color:#86efac; }
    #missions_panel .mis-badge.error { background:#7f1d1d; color:#fca5a5; }
    /* ── Takeoff error banner ───────────────────────────────────── */
    #takeoff_err { display:none; background:#7f1d1d; border:1px solid #ef4444; color:#fee2e2;
                   padding:10px 14px; border-radius:8px; margin-top:10px; font-size:13px; line-height:1.5; }
    #takeoff_err.show { display:block; animation:takeoffpulse 1.2s ease-in-out 2; }
    #takeoff_err .hdr { font-weight:700; letter-spacing:0.02em; margin-bottom:4px; font-size:14px; }
    #takeoff_err .reason { font-family:monospace; font-size:12px; color:#fecaca; margin-bottom:6px; word-break:break-word; }
    #takeoff_err .actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:6px; }
    #takeoff_err .actions button { height:32px; padding:0 12px; font-size:12px; font-weight:600; }
    #takeoff_err .actions .magneto { background:#1e3a5f; border-color:#3b82f6; color:#dbeafe; }
    #takeoff_err .actions .dismiss { background:#374151; border-color:#6b7280; }
    @keyframes takeoffpulse { 0%,100% { box-shadow:0 0 0 2px #fbbf24 inset; } 50% { box-shadow:0 0 0 4px #fbbf24 inset; } }
    /* ── Magnetometer recalibration wizard ──────────────────────── */
    #mag_modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.78); z-index:1100;
                 justify-content:center; align-items:center; }
    #mag_modal.show { display:flex; }
    #mag_modal .card { background:#0b1220; border:1px solid #334155; border-radius:10px;
                       padding:18px; width:min(720px,95vw); max-height:92vh; overflow-y:auto;
                       box-shadow:0 10px 40px rgba(0,0,0,0.5); }
    #mag_modal h3 { margin:0 0 4px 0; color:#60a5fa; font-size:18px; }
    #mag_modal .sub { color:#94a3b8; font-size:12px; margin-bottom:12px; }
    #mag_modal .steps { display:grid; grid-template-columns:1fr; gap:6px; margin-bottom:12px; }
    .mag-step { display:flex; align-items:center; gap:10px; padding:8px 10px; background:#0f172a;
                border:1px solid #1e293b; border-radius:6px; transition:all .2s ease; }
    .mag-step.active { border-color:#3b82f6; background:#0f1e33; box-shadow:0 0 0 1px #3b82f6 inset; }
    .mag-step.ok { border-color:#10b981; background:#052e1b; }
    .mag-step.fail { border-color:#ef4444; background:#2a0a0a; }
    .mag-step .num { flex:0 0 26px; height:26px; border-radius:50%; background:#1e293b;
                     color:#94a3b8; font-weight:700; font-size:13px; display:flex;
                     align-items:center; justify-content:center; border:1px solid #334155; }
    .mag-step.active .num { background:#3b82f6; color:#eff6ff; border-color:#3b82f6; }
    .mag-step.ok .num { background:#10b981; color:#052e1b; border-color:#10b981; }
    .mag-step.fail .num { background:#ef4444; color:#450a0a; border-color:#ef4444; }
    .mag-step .title { flex:1; font-size:13px; color:#e2e8f0; }
    .mag-step .info { font-size:11px; color:#94a3b8; font-family:monospace; }
    #mag_axes { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin:10px 0; }
    .mag-axis { background:#0f172a; border:1px solid #1e293b; border-radius:6px; padding:10px;
                text-align:center; transition:all .2s ease; }
    .mag-axis.active { border-color:#fbbf24; box-shadow:0 0 0 1px #fbbf24 inset;
                       animation:magpulse 1s ease-in-out infinite; }
    .mag-axis.ok { border-color:#10b981; background:#052e1b; }
    .mag-axis .ax-name { font-weight:700; font-size:13px; color:#e2e8f0; letter-spacing:0.04em; }
    .mag-axis .ax-hint { font-size:10px; color:#94a3b8; margin-top:2px; }
    .mag-axis .ax-state { font-size:18px; margin-top:4px; }
    @keyframes magpulse { 0%,100% { background:#0f172a; } 50% { background:#1a2740; } }
    #mag_instructions { background:#0f172a; border:1px solid #334155; border-radius:6px;
                        padding:10px; margin:10px 0; font-size:12px; color:#cbd5e1; line-height:1.55; }
    #mag_instructions b { color:#fbbf24; }
    #mag_log { max-height:140px; overflow-y:auto; background:#0f172a; border:1px solid #1e293b;
               border-radius:6px; padding:8px; font-family:monospace; font-size:11px;
               color:#94a3b8; white-space:pre-wrap; margin-bottom:10px; }
    #mag_result { display:none; padding:10px; border-radius:6px; margin-bottom:10px;
                  font-weight:600; font-size:13px; }
    #mag_result.ok { display:block; background:#052e1b; border:1px solid #10b981; color:#86efac; }
    #mag_result.fail { display:block; background:#2a0a0a; border:1px solid #ef4444; color:#fca5a5; }
    #mag_modal .btnrow { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
    #mag_modal .btnrow button { height:36px; padding:0 14px; font-size:12px; font-weight:600; }
    #mag_start_btn { background:#065f46; border-color:#10b981; }
    #mag_retry_btn { background:#1e3a5f; border-color:#3b82f6; display:none; }
    #mag_close_btn { background:#374151; border-color:#6b7280; }

    /* ── Light theme ───────────────────────────────────────────────
       Activated by setting <html data-theme=\"light\">. Uses !important
       selectively to defeat the many hard-coded inline style=\"\"
       colors scattered through this single-file UI. Inline attribute
       selectors (e.g. [style*=\"background:#0f172a\"]) catch the most
       common dark hex values without rewriting every element. */
    html[data-theme=\"light\"]            { color-scheme: light; }
    html[data-theme=\"light\"] body       { background:#f1f5f9 !important; color:#0f172a !important; }
    html[data-theme=\"light\"] .panel     { background:#ffffff !important; border-color:#cbd5e1 !important; color:#0f172a !important; }
    html[data-theme=\"light\"] button     { background:#e2e8f0 !important; color:#0f172a !important; border-color:#94a3b8 !important; }
    html[data-theme=\"light\"] button:active,
    html[data-theme=\"light\"] .active    { background:#0ea5e9 !important; color:#001018 !important; }
    html[data-theme=\"light\"] input,
    html[data-theme=\"light\"] select,
    html[data-theme=\"light\"] textarea   { background:#ffffff !important; color:#0f172a !important; border-color:#cbd5e1 !important; }
    html[data-theme=\"light\"] h2,
    html[data-theme=\"light\"] h3         { color:#0369a1 !important; }
    html[data-theme=\"light\"] .small     { color:#475569 !important; }
    html[data-theme=\"light\"] #aruco_panel,
    html[data-theme=\"light\"] #tuning_panel,
    html[data-theme=\"light\"] #mission_panel,
    html[data-theme=\"light\"] #anafi_panel,
    html[data-theme=\"light\"] #missions_panel,
    html[data-theme=\"light\"] #video_panel { background:#f8fafc !important; border-color:#cbd5e1 !important; color:#0f172a !important; }
    html[data-theme=\"light\"] .arena-canvas { background:#f8fafc !important; border-color:#cbd5e1 !important; }
    html[data-theme=\"light\"] .info-icon   { border-color:#94a3b8 !important; color:#475569 !important; }
    html[data-theme=\"light\"] .info-icon:hover { border-color:#0369a1 !important; color:#0369a1 !important; background:#e0f2fe !important; }
    html[data-theme=\"light\"] #param_info_modal { background:rgba(15,23,42,0.45) !important; }
    html[data-theme=\"light\"] #param_info_modal .box { background:#ffffff !important; color:#0f172a !important; border-color:#0ea5e9 !important; }
    html[data-theme=\"light\"] #param_info_modal .title { color:#0369a1 !important; }
    html[data-theme=\"light\"] #param_info_modal .subtitle { color:#64748b !important; }
    html[data-theme=\"light\"] #param_info_modal .body { color:#1e293b !important; }
    html[data-theme=\"light\"] #param_info_modal .close { background:#e2e8f0 !important; color:#0f172a !important; border-color:#94a3b8 !important; }
    html[data-theme=\"light\"] .collapsible-toggle { color:#0369a1 !important; }
    html[data-theme=\"light\"] #aruco_panel .arc-readout   { color:#334155 !important; }
    html[data-theme=\"light\"] #aruco_panel .arc-readout .k { color:#64748b !important; }
    html[data-theme=\"light\"] #aruco_panel .arc-readout b  { color:#0284c7 !important; }
    html[data-theme=\"light\"] #aruco_panel .arc-row label  { color:#334155 !important; }
    html[data-theme=\"light\"] #aruco_panel .arc-pgroup-label { color:#64748b !important; }
    html[data-theme=\"light\"] .drone-btn { background:#f1f5f9 !important; color:#1e293b !important; border-color:#94a3b8 !important; }
    html[data-theme=\"light\"] .drone-btn.selected { background:#0ea5e9 !important; color:#001018 !important; border-color:#0ea5e9 !important; }
    html[data-theme=\"light\"] .pos-cfg input,
    html[data-theme=\"light\"] .pos-cfg select { background:#ffffff !important; color:#0f172a !important; border-color:#cbd5e1 !important; }
    html[data-theme=\"light\"] .pos-cfg label { color:#475569 !important; }
    html[data-theme=\"light\"] .meter-track { background:#e2e8f0 !important; border-color:#94a3b8 !important; }
    html[data-theme=\"light\"] .meter-label { color:#475569 !important; }
    html[data-theme=\"light\"] #arc_live_banner { background:#fecaca !important; color:#7f1d1d !important; }
    html[data-theme=\"light\"] #takeoff_err { background:#fef2f2 !important; color:#7f1d1d !important; border-color:#f87171 !important; }
    /* Catch common inline-styled dark backgrounds/colors */
    html[data-theme=\"light\"] [style*=\"background:#0f172a\"],
    html[data-theme=\"light\"] [style*=\"background: #0f172a\"],
    html[data-theme=\"light\"] [style*=\"background:#111827\"],
    html[data-theme=\"light\"] [style*=\"background:#0b1220\"],
    html[data-theme=\"light\"] [style*=\"background:#0b1424\"] { background:#ffffff !important; }
    html[data-theme=\"light\"] [style*=\"background:#1e293b\"],
    html[data-theme=\"light\"] [style*=\"background:#1e2a3a\"] { background:#e2e8f0 !important; }
    html[data-theme=\"light\"] [style*=\"background:#1f2937\"] { background:#e2e8f0 !important; }
    html[data-theme=\"light\"] [style*=\"color:#e2e8f0\"] { color:#0f172a !important; }
    html[data-theme=\"light\"] [style*=\"color:#94a3b8\"],
    html[data-theme=\"light\"] [style*=\"color:#cbd5e1\"],
    html[data-theme=\"light\"] [style*=\"color:#64748b\"] { color:#475569 !important; }
    html[data-theme=\"light\"] [style*=\"border-color:#334155\"],
    html[data-theme=\"light\"] [style*=\"border:1px solid #334155\"],
    html[data-theme=\"light\"] [style*=\"border:1px solid #475569\"],
    html[data-theme=\"light\"] [style*=\"border:1px solid #1e293b\"],
    html[data-theme=\"light\"] [style*=\"border-top:1px solid #1e293b\"] { border-color:#cbd5e1 !important; }
    /* Keep accent-coloured status badges vibrant (keep dark text on the accent) */
    html[data-theme=\"light\"] [style*=\"color:#22c55e\"] { color:#15803d !important; }
    html[data-theme=\"light\"] [style*=\"color:#f59e0b\"] { color:#b45309 !important; }
    html[data-theme=\"light\"] [style*=\"color:#38bdf8\"] { color:#0369a1 !important; }
    html[data-theme=\"light\"] [style*=\"color:#a78bfa\"] { color:#7c3aed !important; }
    html[data-theme=\"light\"] [style*=\"color:#06b6d4\"] { color:#0e7490 !important; }
    html[data-theme=\"light\"] #theme_toggle { color:#0369a1 !important; border-color:#94a3b8 !important; }
  </style>
</head>
<body>
  <!-- Shared parameter-info popup. Any .info-icon click populates and
       shows this modal via window.showParamInfo(key, event). -->
  <div id=\"param_info_modal\" onclick=\"if(event.target===this) this.style.display='none';\">
    <div class=\"box\">
      <div class=\"title\"    id=\"pim_title\">parameter</div>
      <div class=\"subtitle\" id=\"pim_key\">key</div>
      <div class=\"body\"     id=\"pim_body\">explanation</div>
      <div class=\"range\"    id=\"pim_range\"></div>
      <div style=\"text-align:right;\">
        <button class=\"close\" onclick=\"document.getElementById('param_info_modal').style.display='none';\">Close</button>
      </div>
    </div>
  </div>
  <div style=\"display:flex;align-items:center;gap:14px;margin:0 0 6px 0;\">
    <h2 style=\"margin:0;flex:1;\">Drone Remote Controller</h2>
    <button id=\"theme_toggle\" title=\"Toggle light / dark theme\"
            style=\"height:34px;padding:0 12px;font-size:12px;font-weight:600;
                   background:transparent;\">&#127769; Dark</button>
    <img id=\"team_logo\" src=\"/logo.png?v=2\" alt=\"Team logo\"
         title=\"Team To Be Defined — SDC26\"
         style=\"width:110px;height:auto;background:transparent;
                filter:drop-shadow(0 2px 6px rgba(0,0,0,0.45));\"
         onerror=\"this.style.display='none'\" />
  </div>
  <div style=\"display:flex;align-items:center;gap:8px;\">
    <div class=\"drone-bar\" id=\"drone_bar\" style=\"flex:1;\"></div>
    <button id=\"pause_all_btn\" style=\"padding:6px 14px;font-size:13px;font-weight:700;background:#78350f;border-color:#f59e0b;color:#fde68a;letter-spacing:0.4px;\" title=\"Override any command and freeze every drone in place. Autonomous missions abort; drones hover with zero RC. Keyboard shortcut: 9\">&#9208;&#65039; PAUSE ALL (9)</button>
    <button id=\"resume_all_btn\" style=\"padding:6px 14px;font-size:13px;font-weight:700;background:#065f46;border-color:#10b981;color:#d1fae5;letter-spacing:0.4px;display:none;\" title=\"Clear the pause — operators may re-arm missions manually. Keyboard shortcut: 9\">&#9654;&#65039; CONTINUE MISSION</button>
    <button id=\"land_all_btn\" style=\"padding:6px 14px;font-size:13px;font-weight:700;background:#7f1d1d;border-color:#ef4444;color:#fee2e2;letter-spacing:0.4px;\" title=\"Land every drone in the fleet safely. Keyboard shortcut: 0 (zero)\">&#11088; LAND ALL (0)</button>
    <button id=\"edit_drones_btn\" style=\"padding:4px 12px;font-size:12px;background:#1e3a5f;border-color:#3b82f6;\" title=\"Edit drone fleet config\">Config</button>
  </div>
  <!-- Global PAUSE banner — sits just below the drone bar, hidden until
       the fleet is paused. Amber bg + animated pulse to make it impossible
       to miss. Reminds the operator that WASD is the only active control. -->
  <div id=\"global_pause_banner\" style=\"display:none;background:#ca8a04;color:#1c1917;padding:8px 14px;margin-top:6px;border-radius:6px;font-weight:700;letter-spacing:0.04em;text-align:center;box-shadow:0 0 0 2px #facc15 inset;animation:pausepulse 1.4s ease-in-out infinite;\">
    &#9208;&#65039; PAUSED &mdash; autonomous control is disabled. Drones hover at current position. Only WASD / manual RC is live. Press <b>CONTINUE MISSION</b> or <b>9</b> to resume.
  </div>
  <!-- Ceiling guard — the C2 UI sets the value, but the enforcement
       runs ENTIRELY on each drone's flight-controller Pi (its 20 Hz
       rc_loop using its own height_cm telemetry). If the C2 crashes,
       disconnects, or even shuts down, the Pi keeps clamping upward
       RC. The value persists to flight_config.json on the Pi so a
       Pi restart also retains the ceiling. This is the last-line
       safety — independent of C2 connection. -->
  <div style=\"display:flex;align-items:center;gap:10px;margin-top:6px;padding:4px 10px;background:#0b1424;border:1px solid #1e293b;border-radius:5px;font-size:12px;\">
    <span style=\"color:#fca5a5;font-weight:600;letter-spacing:0.04em;\"
          title=\"Safety ceiling — set here, enforced on every Pi independently\">
      &#128737;&#65039; Ceiling
    </span>
    <label style=\"color:#94a3b8;display:flex;align-items:center;gap:4px;cursor:pointer;\"
           title=\"Hard maximum altitude above ground. Stored and enforced LOCALLY on each drone's Pi (no C2 dependency). Proportional clamp within 50cm, hard stop at ceiling, forced descent above. Persists across Pi restarts via flight_config.json.\">
      max
      <input id=\"ceiling_input\" type=\"number\" min=\"0.5\" max=\"20\" step=\"0.1\" value=\"5.0\" style=\"width:64px;height:26px;font-size:12px;\" />
      m
      <span class=\"info-icon\" data-info=\"ceiling_m\">i</span>
    </label>
    <button id=\"ceiling_apply_btn\"
            style=\"height:26px;font-size:11px;padding:0 10px;background:#7f1d1d;border-color:#ef4444;color:#fee2e2;\"
            title=\"POST the new value to every drone's /api/config/ceiling. Each Pi then: (1) stores it in MAX_ALTITUDE_M for its local RC tick, (2) writes it to flight_config.json for persistence, (3) pushes it to the Anafi firmware MaxAltitude as a second-line guard.\">Apply to fleet</button>
    <span id=\"ceiling_status\" class=\"small\" style=\"color:#64748b;\"></span>
    <span id=\"ceiling_engaged_badge\" class=\"small\" style=\"display:none;color:#fde68a;background:#7f1d1d;padding:2px 8px;border-radius:4px;font-weight:700;letter-spacing:0.04em;animation:pausepulse 1.0s ease-in-out infinite;\">
      &#128680; CEILING ENGAGED
    </span>
    <span class=\"small\" style=\"color:#64748b;margin-left:auto;\"
          title=\"The Pi enforces this ceiling on every RC tick regardless of C2 state — no remote connection required.\">
      &#128274; Pi-enforced
    </span>
  </div>
  <!-- Flight safety guards — axis-lock (autonomous only), arena
       boundary (BOTH manual + autonomous when 'Manual guard' is on),
       and safety margin. The arena guard is enforced on each Pi in
       the RC tick loop so it works independently of the C2. -->
  <div style=\"display:flex;align-items:center;gap:14px;margin-top:4px;padding:4px 10px;background:#0b1424;border:1px solid #1e293b;border-radius:5px;font-size:12px;flex-wrap:wrap;\">
    <span style=\"color:#fbbf24;font-weight:600;letter-spacing:0.04em;\">&#129517; Flight guards</span>
    <label style=\"color:#94a3b8;display:flex;align-items:center;gap:4px;cursor:pointer;\"
           title=\"When ON: during missions and autonomous flight the drone only flies parallel to arena walls — yaw snaps to nearest 90°, lateral motion is strafe OR forward/back but never diagonal. Manual WASD is unchanged.\">
      <input type=\"checkbox\" id=\"axis_locked_toggle\" style=\"accent-color:#fbbf24;\" />
      Axis-lock (autonomous)
      <span class=\"info-icon\" data-info=\"axis_locked\">i</span>
    </label>
    <label style=\"color:#94a3b8;display:flex;align-items:center;gap:4px;cursor:pointer;\"
           title=\"When ON: every RC command on the Pi — manual WASD, keystroke, autonomous mission, everything — is clamped so the drone never approaches an arena wall closer than the margin. Enforced by the Pi's own RC tick at 20 Hz, independent of C2 connection. Falls back silently when position is unknown (operator is responsible).\">
      <input type=\"checkbox\" id=\"arena_guard_toggle\" checked style=\"accent-color:#f87171;\" />
      Arena guard (manual+auto)
      <span class=\"info-icon\" data-info=\"arena_guard_enabled\">i</span>
    </label>
    <label style=\"color:#94a3b8;display:flex;align-items:center;gap:4px;cursor:pointer;\"
           title=\"Minimum distance from any arena wall. The Pi-side RC tick clamps any command that would push the drone closer than this.\">
      margin
      <input id=\"safety_margin_input\" type=\"number\" min=\"0.1\" max=\"5.0\" step=\"0.1\" value=\"1.5\" style=\"width:54px;height:24px;font-size:12px;\" />
      m
      <span class=\"info-icon\" data-info=\"safety_margin_m\">i</span>
    </label>
    <button id=\"safety_margin_apply\" style=\"height:24px;font-size:11px;padding:0 8px;background:#78350f;border-color:#f59e0b;color:#fde68a;\"
            title=\"Push margin to every drone's Pi (arena guard) + every observer (autonomous guard)\">Apply</button>
    <span id=\"arena_guard_engaged_badge\" class=\"small\" style=\"display:none;color:#fde68a;background:#7f1d1d;padding:2px 8px;border-radius:4px;font-weight:700;letter-spacing:0.04em;\">
      &#128680; ARENA GUARD ENGAGED
    </span>
    <label style=\"color:#94a3b8;display:flex;align-items:center;gap:4px;cursor:pointer;\"
           title=\"When ON: during ANY autonomous mission, the drone's camera aims at the arena centre (default x=0, y=5.4) regardless of which direction it's flying. This keeps the maximum number of ArUco markers in view at once, which in turn gives the position processor more references and a more accurate fused pose.\">
      <input type=\"checkbox\" id=\"cam_face_center_toggle\" style=\"accent-color:#22d3ee;\" />
      &#128247; Cam → arena centre
      <span class=\"info-icon\" data-info=\"camera_face_center\">i</span>
    </label>
    <span id=\"autonomous_guards_status\" class=\"small\" style=\"color:#64748b;\"></span>
  </div>
  <div id=\"drone_config_modal\" style=\"display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);z-index:1000;justify-content:center;align-items:center;\">
    <div style=\"background:#1e293b;border:1px solid #334155;border-radius:8px;padding:20px;max-width:700px;width:90%;max-height:80vh;overflow-y:auto;\">
      <h3 style=\"margin:0 0 12px 0;color:#e2e8f0;\">Drone Fleet Configuration</h3>
      <div id=\"drone_config_fields\"></div>
      <div style=\"margin-top:12px;display:flex;gap:8px;\">
        <button id=\"drone_config_add\" style=\"background:#065f46;border-color:#10b981;padding:6px 16px;\">Add Drone</button>
        <button id=\"drone_config_save\" style=\"background:#1e3a5f;border-color:#3b82f6;padding:6px 16px;\">Save</button>
        <button id=\"drone_config_cancel\" style=\"background:#374151;border-color:#6b7280;padding:6px 16px;\">Cancel</button>
        <span id=\"drone_config_status\" class=\"small\" style=\"color:#94a3b8;align-self:center;\"></span>
      </div>
    </div>
  </div>
  <div id=\"mag_modal\" role=\"dialog\" aria-labelledby=\"mag_title\">
    <div class=\"card\">
      <h3 id=\"mag_title\">Magnetometer Recalibration</h3>
      <div class=\"sub\">
        Anafi requires a figure-8 dance around each axis whenever it is moved
        between locations or power-cycled. Keep the drone on a flat,
        non-metallic surface; no cables or phones nearby.
      </div>
      <div class=\"steps\" id=\"mag_steps\">
        <div class=\"mag-step\" data-step=\"heartbeat\">
          <div class=\"num\">1</div>
          <div class=\"title\">Pre-check — drone connected &amp; on the ground</div>
          <div class=\"info\" data-role=\"info\"></div>
        </div>
        <div class=\"mag-step\" data-step=\"pre_status\">
          <div class=\"num\">2</div>
          <div class=\"title\">Read current magnetometer status</div>
          <div class=\"info\" data-role=\"info\"></div>
        </div>
        <div class=\"mag-step\" data-step=\"start\">
          <div class=\"num\">3</div>
          <div class=\"title\">Start calibration command</div>
          <div class=\"info\" data-role=\"info\"></div>
        </div>
        <div class=\"mag-step\" data-step=\"poll\">
          <div class=\"num\">4</div>
          <div class=\"title\">Figure-8 around each axis until all axes confirm</div>
          <div class=\"info\" data-role=\"info\"></div>
        </div>
      </div>
      <div id=\"mag_instructions\">
        <b>How to perform the dance:</b> hold the drone firmly and rotate it
        slowly (~2 s per full turn) around each body axis in turn — roll (X),
        pitch (Y), yaw (Z) — tracing a smooth figure-8. Each axis panel below
        lights green once its bit flips.
      </div>
      <div id=\"mag_axes\">
        <div class=\"mag-axis\" data-axis=\"x\">
          <div class=\"ax-name\">X (roll)</div>
          <div class=\"ax-hint\">tilt left/right repeatedly</div>
          <div class=\"ax-state\" data-role=\"state\">&#9675;</div>
        </div>
        <div class=\"mag-axis\" data-axis=\"y\">
          <div class=\"ax-name\">Y (pitch)</div>
          <div class=\"ax-hint\">tilt nose up/down</div>
          <div class=\"ax-state\" data-role=\"state\">&#9675;</div>
        </div>
        <div class=\"mag-axis\" data-axis=\"z\">
          <div class=\"ax-name\">Z (yaw)</div>
          <div class=\"ax-hint\">rotate around vertical</div>
          <div class=\"ax-state\" data-role=\"state\">&#9675;</div>
        </div>
      </div>
      <div id=\"mag_result\"></div>
      <div id=\"mag_log\">ready.</div>
      <div class=\"btnrow\">
        <button id=\"mag_start_btn\">Start Recalibration</button>
        <button id=\"mag_retry_btn\">Retry</button>
        <button id=\"mag_close_btn\">Close</button>
      </div>
    </div>
  </div>
  <div class=\"small\">Active: <span id=\"pi\"></span></div>
  <div class=\"small\">API status: <span id=\"api_status\">checking...</span></div>
  <div class=\"small\">Drone telemetry status: <span id=\"drone_status\">checking...</span></div>
  <!-- Latency indicator (live ms). Click the toggle to auto-push the total
       into the position-tracker's latency_ms slider. Video-decode offset
       slider lets the operator add the C2-side processing time that isn't
       captured by either ping. -->
  <div class=\"small\" id=\"latency_widget\" style=\"display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:4px;padding:4px 8px;background:#0f172a;border:1px solid #1e293b;border-radius:4px;\">
    <b style=\"color:#e2e8f0;\">Latency:</b>
    <span id=\"lat_total\" style=\"font-weight:700;color:#22c55e;min-width:60px;\">—</span>
    <span class=\"small\" style=\"color:#64748b;\">=
      c2→fc <span id=\"lat_c2fc\" style=\"color:#93c5fd;\">—</span> +
      fc→drone <span id=\"lat_fcdr\" style=\"color:#fbbf24;\">—</span> +
      video <span id=\"lat_vid\" style=\"color:#c4b5fd;\">—</span>
    </span>
    <label style=\"display:flex;align-items:center;gap:4px;color:#94a3b8;cursor:pointer;\" title=\"Add this many ms for local video-frame processing/decoding on the C2. Not measured by either ping.\">
      video +<input id=\"lat_video_offset\" type=\"number\" min=\"0\" max=\"500\" step=\"5\" value=\"30\" style=\"width:55px;\" /> ms
    </label>
    <label style=\"display:flex;align-items:center;gap:4px;color:#94a3b8;cursor:pointer;\" title=\"Auto-push the total into the Position Tracker latency slider every poll.\">
      <input type=\"checkbox\" id=\"lat_auto_apply\" checked /> auto-set latency
    </label>
    <span id=\"ws_status_badge\" class=\"small\" title=\"WebSocket channel between C2 and flight controller. Tel/Pos/RC are pushed on a persistent connection — no per-call HTTP framing overhead.\" style=\"padding:2px 8px;border-radius:3px;font-family:monospace;font-size:10.5px;letter-spacing:0.03em;font-weight:700;background:#334155;color:#94a3b8;\">WS —</span>
  </div>
  <!-- ── Transport selector — per-subsystem WS ↔ HTTP switches ──
       auto = prefer WS, fall back to HTTP when WS is down (default)
       ws   = force WS only (503 if WS disconnected)
       http = force HTTP only (never use WS)
       Persisted on the server + mirrored in localStorage so every tab
       reflects the same choice. -->
  <div id=\"transport_widget\" class=\"small\" style=\"display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:4px;padding:4px 8px;background:#0f172a;border:1px solid #1e293b;border-radius:4px;\">
    <b style=\"color:#e2e8f0;\">Transport:</b>
    <span style=\"color:#64748b;\">Controls (RC/WASD)</span>
    <select id=\"transport_rc\" class=\"transport-sel\" data-subsys=\"rc\" style=\"height:22px;font-size:11px;padding:0 4px;\">
      <option value=\"auto\">auto</option>
      <option value=\"ws\">ws</option>
      <option value=\"http\">http</option>
    </select>
    <span style=\"color:#64748b;\">Telemetry</span>
    <select id=\"transport_telemetry\" class=\"transport-sel\" data-subsys=\"telemetry\" style=\"height:22px;font-size:11px;padding:0 4px;\">
      <option value=\"auto\">auto</option>
      <option value=\"ws\">ws</option>
      <option value=\"http\">http</option>
    </select>
    <span style=\"color:#64748b;\">Position</span>
    <select id=\"transport_position\" class=\"transport-sel\" data-subsys=\"position\" style=\"height:22px;font-size:11px;padding:0 4px;\">
      <option value=\"auto\">auto</option>
      <option value=\"ws\">ws</option>
      <option value=\"http\">http</option>
    </select>
    <button id=\"transport_all_http\" style=\"height:22px;font-size:11px;padding:0 8px;background:#1e293b;border-color:#475569;\" title=\"Force every subsystem to HTTP (legacy transport). Useful for diagnosing WS as the source of latency.\">All HTTP</button>
    <button id=\"transport_all_auto\" style=\"height:22px;font-size:11px;padding:0 8px;background:#1e293b;border-color:#475569;\" title=\"Reset: prefer WS with HTTP fallback\">All auto</button>
    <span id=\"transport_status\" class=\"small\" style=\"color:#64748b;\"></span>
  </div>
  <!-- ── Tuning Parameters — one place for all live-tunable knobs ──
       The Observer PD gains (approach speed, skew correction, IMU damping,
       clamps, etc.) AND the Position Tracker config (profile, FOV, latency,
       IMU/ArUco blend, Kalman, marker size, top-K, outlier) all live here
       so operators don't have to hunt for the right slider. Content is
       relocated at runtime from the ArUco Seek and Position Tracker panels
       via JS to keep existing IDs and handlers intact. -->
  <div id=\"tuning_panel\" class=\"panel\" style=\"margin-top:10px;\">
    <h3 style=\"margin:0 0 8px 0;color:#38bdf8;\">Tuning Parameters <span class=\"small\" style=\"color:#64748b;font-weight:400;\">— all live-adjustable knobs in one place</span></h3>
    <div style=\"display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;\">
      <div style=\"flex:1;min-width:360px;\">
        <div class=\"small\" style=\"color:#a78bfa;margin-bottom:4px;font-weight:600;\">
          Observer PD — visual-servo gains (hover in front of markers, used by missions)
        </div>
        <div id=\"tuning_observer_slot\" style=\"min-height:10px;\"></div>
      </div>
      <div style=\"flex:1;min-width:360px;\">
        <div class=\"small\" style=\"color:#22d3ee;margin-bottom:4px;font-weight:600;\">
          Position Tracker — absolute arena-frame pose fusion (all missions + boundary guard)
        </div>
        <div id=\"tuning_position_slot\" style=\"min-height:10px;\"></div>
      </div>
    </div>
  </div>

  <div class=\"row\" style=\"margin-top:10px;\">
    <div class=\"panel\">
      <div class=\"grid\" id=\"grid\">
        <button data-k=\"q\">Q</button><button data-k=\"w\">W</button><button data-k=\"e\">E</button>
        <button data-k=\"a\">A</button><button data-k=\"x\">STOP</button><button data-k=\"d\">D</button>
        <button data-k=\"r\">R</button><button data-k=\"s\">S</button><button data-k=\"f\">F</button>
      </div>
      <div style=\"margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;\">
        <button id=\"takeoff\">Takeoff (T)</button>
        <button id=\"land\">Land (L)</button>
        <button id=\"recover\">Recover</button>
        <button id=\"safe_takeoff\">Safe Takeoff: OFF</button>
      </div>
      <div id=\"takeoff_err\" role=\"alert\">
        <div class=\"hdr\">&#9888; Cannot take off</div>
        <div class=\"reason\" id=\"takeoff_err_reason\">—</div>
        <div class=\"small\" id=\"takeoff_err_hint\" style=\"color:#fecaca;\"></div>
        <!-- Diagnostic state table — populated from the proxy's
             `diagnostic` field (magnetometer status, battery, alert,
             motor, sensors) so the operator sees exactly why the FC
             refused and can act on it. -->
        <div id=\"takeoff_err_diag\" style=\"display:none;margin-top:6px;padding:6px 8px;background:rgba(0,0,0,0.25);border-radius:4px;font-size:12px;color:#fecaca;\"></div>
        <div class=\"actions\">
          <button class=\"magneto\" id=\"takeoff_err_mag\" style=\"display:none;\">Recalibrate Magnetometer</button>
          <button class=\"dismiss\" id=\"takeoff_err_dismiss\">Dismiss</button>
        </div>
      </div>
      <div style=\"margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;\">
        <button id=\"toggle_log\">Enable Telemetry Log</button>
        <button id=\"download_log\">Download Telemetry Log</button>
        <button id=\"clear_log\">Clear Telemetry Log</button>
      </div>
      <div style=\"margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;\">
        <button id=\"toggle_cmd_log\">Command Logging: OFF</button>
        <button id=\"download_cmd_log\">Download Command Log</button>
        <button id=\"clear_cmd_log\">Clear Command Log</button>
      </div>
      <!-- Automatic per-flight logs — a fresh JSONL file per takeoff → land
           pair, containing telemetry + commands + arena position + visible
           ArUco markers. Archived list below; click a row to download. -->
      <div class=\"adv\" style=\"margin-top:10px;\">
        <div class=\"small\" style=\"margin-bottom:6px;\">
          <b>Flight Logs</b> <span style=\"color:#64748b;\">— auto-recorded per takeoff</span>
          <button id=\"flight_logs_refresh\" style=\"height:24px;font-size:11px;padding:0 8px;margin-left:10px;background:#1e293b;\">Refresh</button>
        </div>
        <div id=\"flight_logs_list\" class=\"small\"
             style=\"font-family:monospace;font-size:11px;max-height:180px;overflow-y:auto;border:1px solid #334155;border-radius:4px;padding:6px;background:#0b1220;color:#cbd5e1;\">
          <i style=\"color:#64748b;\">loading…</i>
        </div>
      </div>
      <div class=\"adv\">
        <div class=\"small\" style=\"margin-bottom:6px;\">Advanced SDK controls</div>
        <div class=\"adv-grid\">
          <button id=\"rotate_cw\">Rotate CW 45°</button>
          <button id=\"rotate_ccw\">Rotate CCW 45°</button>
          <button id=\"move_up\">Up 30cm</button>
          <button id=\"move_down\">Down 30cm</button>
          <button id=\"move_fwd\">Forward 30cm</button>
          <button id=\"move_back\">Back 30cm</button>
          <button id=\"move_left\">Left 30cm</button>
          <button id=\"move_right\">Right 30cm</button>
          <button id=\"stream_on\">Stream ON</button>
          <button id=\"stream_off\">Stream OFF</button>
          <button id=\"set_speed\">Set Speed</button>
        </div>
        <div class=\"row\" style=\"margin-top:8px;\">
          <input id=\"speed_val\" type=\"number\" min=\"10\" max=\"100\" value=\"30\" placeholder=\"speed 10..100\" />
          <input id=\"sdk_cmd\" type=\"text\" placeholder=\"raw sdk cmd (e.g. speed? or battery?)\" style=\"min-width:320px;flex:1;\" />
          <button id=\"sdk_send\">Send SDK Command</button>
        </div>
      </div>
      <div class=\"adv\" id=\"anafi_panel\">
        <div class=\"small\" style=\"margin-bottom:6px;\">Anafi / Olympe controls</div>
        <div class=\"row\" style=\"margin-top:8px; align-items:center;\">
          <span class=\"small\" style=\"min-width:80px;\">Gimbal tilt</span>
          <input id=\"gimbal_tilt\" type=\"range\" min=\"-90\" max=\"30\" value=\"0\" style=\"flex:1;\" />
          <span id=\"gimbal_tilt_val\" class=\"small\" style=\"min-width:40px;\">0°</span>
          <button id=\"gimbal_set\">Set</button>
          <button id=\"gimbal_down\">Down (-90)</button>
          <button id=\"gimbal_fwd\">Forward (0)</button>
        </div>
        <div class=\"row\" style=\"margin-top:8px; align-items:center; padding-top:8px; border-top:1px solid #1e293b;\">
          <span class=\"small\" style=\"min-width:80px;\">Magnetometer</span>
          <span id=\"mag_status\" class=\"small\" style=\"color:#94a3b8;font-family:monospace;flex:1;\">—</span>
          <button id=\"mag_open\" style=\"background:#1e3a5f;border-color:#3b82f6;height:32px;padding:0 12px;font-size:12px;\" title=\"Walk through the figure-8 recalibration wizard\">Recalibrate Magnetometer</button>
        </div>
        <div class=\"row\" style=\"margin-top:8px; align-items:center;\">
          <span class=\"small\" style=\"min-width:80px;\">Max altitude (m)</span>
          <input id=\"set_alt\" type=\"number\" min=\"0.5\" max=\"150\" step=\"0.5\" value=\"5\" style=\"width:70px;\" />
          <span class=\"small\" style=\"min-width:80px;\">Max vert spd</span>
          <input id=\"set_vspd\" type=\"number\" min=\"0.1\" max=\"4\" step=\"0.1\" value=\"0.5\" style=\"width:70px;\" />
          <span class=\"small\" style=\"min-width:80px;\">Max tilt (°)</span>
          <input id=\"set_tilt\" type=\"number\" min=\"1\" max=\"35\" step=\"1\" value=\"15\" style=\"width:70px;\" />
          <button id=\"apply_settings\">Apply Settings</button>
        </div>
        <!-- ── Environment (indoor/outdoor) ─────────────────────── -->
        <div class=\"row\" style=\"margin-top:10px; align-items:center; padding-top:8px; border-top:1px solid #1e293b;\">
          <span class=\"small\" style=\"min-width:80px;\">Environment</span>
          <select id=\"env_mode\" style=\"background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:4px 6px;font-size:12px;\">
            <option value=\"indoor\">Indoor (GPS-less, relaxed checks)</option>
            <option value=\"outdoor\">Outdoor (GPS-required, default)</option>
          </select>
          <button id=\"env_apply\" style=\"background:#1e3a5f;border-color:#3b82f6;\">Apply</button>
          <span id=\"env_status\" class=\"small\" style=\"color:#94a3b8;\">—</span>
        </div>
        <!-- ── Wi-Fi band + channel ─────────────────────────────── -->
        <div class=\"row\" style=\"margin-top:8px; align-items:center;\">
          <span class=\"small\" style=\"min-width:80px;\">Wi-Fi band</span>
          <select id=\"wifi_band\" style=\"background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:4px 6px;font-size:12px;\">
            <option value=\"5_GHz\">5 GHz (recommended)</option>
            <option value=\"2_4_GHz\">2.4 GHz</option>
          </select>
          <span class=\"small\" style=\"min-width:56px;\">Channel</span>
          <select id=\"wifi_channel\" style=\"background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:4px 6px;font-size:12px;min-width:130px;\">
            <option value=\"auto\">Auto (drone picks)</option>
            <!-- 5 GHz non-DFS -->
            <option value=\"36\">36 (5 GHz UNII-1)</option>
            <option value=\"40\">40 (5 GHz UNII-1)</option>
            <option value=\"44\">44 (5 GHz UNII-1)</option>
            <option value=\"48\">48 (5 GHz UNII-1)</option>
            <option value=\"149\">149 (5 GHz UNII-3)</option>
            <option value=\"153\">153 (5 GHz UNII-3)</option>
            <option value=\"157\">157 (5 GHz UNII-3)</option>
            <option value=\"161\">161 (5 GHz UNII-3)</option>
            <option value=\"165\">165 (5 GHz UNII-3)</option>
            <!-- 2.4 GHz -->
            <option value=\"1\">1 (2.4 GHz)</option>
            <option value=\"6\">6 (2.4 GHz)</option>
            <option value=\"11\">11 (2.4 GHz)</option>
          </select>
          <button id=\"wifi_apply\" style=\"background:#1e3a5f;border-color:#3b82f6;\" title=\"Apply band+channel. Drone on ground only. Wi-Fi link drops briefly while re-associating.\">Apply Wi-Fi</button>
          <button id=\"wifi_scan\" style=\"background:#374151;border-color:#6b7280;\" title=\"Scan the selected band for in-use channels\">Scan</button>
          <span id=\"wifi_status\" class=\"small\" style=\"color:#94a3b8;margin-left:6px;\">—</span>
        </div>
      </div>
      <div class=\"adv\" id=\"mission_panel\">
        <div class=\"small\" style=\"margin-bottom:6px;\"><b>Mission Planner</b> — enter one command per line</div>
        <textarea id=\"mission_cmds\" rows=\"8\" style=\"width:100%;font-family:monospace;font-size:12px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:6px;resize:vertical;\" placeholder=\"Examples:\n100 forward\n90 cw\n50 back\n45 ccw\n80 up\n60 down\n30 left\n40 right\nwait 2\nhover 3\nland\ntakeoff\"></textarea>
        <div style=\"margin-top:6px;display:flex;gap:8px;align-items:center;\">
          <button id=\"mission_run\" style=\"background:#065f46;border-color:#10b981;\">Run Mission</button>
          <button id=\"mission_stop\" style=\"background:#7f1d1d;border-color:#dc2626;display:none;\">Abort Mission</button>
          <span id=\"mission_status\" class=\"small\" style=\"color:#94a3b8;\">idle</span>
        </div>
        <div id=\"mission_log\" class=\"small\" style=\"margin-top:6px;max-height:120px;overflow-y:auto;white-space:pre-wrap;color:#94a3b8;\"></div>
      </div>
      <div class=\"small\" style=\"margin-top:8px;\">Keyboard in browser: W/A/S/D R/F Q/E, T, L, Space stop</div>
    </div>

    <div style=\"display:flex; flex-direction:column; gap:12px;\">
      <div class=\"panel\" id=\"video_panel\">
        <div><b>Video Stream</b></div>
        <div class=\"video-controls\">
          <select id=\"video_mode\">
            <option value=\"off\">Off</option>
            <option value=\"mjpeg\">Way 1: MJPEG (decoded on Pi)</option>
            <option value=\"forward\">Way 2: UDP Forward (decoded on C2)</option>
          </select>
          <button id=\"video_toggle\">Start Video</button>
        </div>
        <div class=\"video-status\" id=\"video_status\">Mode: off</div>
        <div class=\"video-url\" id=\"video_url\" style=\"display:none;\"></div>
        <!-- Anafi camera zoom. Digital zoom 1.0–3.0×. Slider writes to
             /proxy/camera/zoom, which forwards to Olympe's set_zoom_target
             on the Pi. Value updates live as you drag. -->
        <div class=\"video-controls\" style=\"margin-top:8px;\">
          <span class=\"small\" style=\"min-width:52px;\">Zoom</span>
          <input id=\"video_zoom\" type=\"range\" min=\"1.0\" max=\"3.0\" step=\"0.05\" value=\"1.0\" style=\"flex:1;max-width:200px;\" />
          <span id=\"video_zoom_val\" class=\"small\" style=\"min-width:40px;\">1.00×</span>
          <button id=\"video_zoom_reset\" style=\"padding:2px 8px;font-size:11px;\">Reset</button>
        </div>
        <div id=\"video_container\" style=\"margin-top:8px;display:none;\">
          <img id=\"video_img\" src=\"\" alt=\"video stream\" style=\"width:480px;height:auto;\" />
        </div>
      </div>
      <div class=\"panel\">
        <div><b>Telemetry</b></div>
        <div class=\"status-wrap\">
          <div class=\"meter\">
            <div class=\"meter-label\">Battery SoC: <span id=\"battery_val\">-</span></div>
            <div class=\"meter-track\"><div id=\"battery_bar\" class=\"meter-fill\"></div></div>
          </div>
        </div>
        <div id=\"compass_wrap\" style=\"display:flex;gap:12px;align-items:center;margin-top:10px;padding:8px;background:#0f172a;border:1px solid #1e293b;border-radius:4px;\">
          <canvas id=\"compass_canvas\" width=\"96\" height=\"96\" style=\"flex:0 0 96px;display:block;\"></canvas>
          <div class=\"small\" style=\"flex:1;line-height:1.55;\">
            <div>Heading: <b style=\"color:#e2e8f0;\"><span id=\"compass_abs\">--</span>°</b> <span style=\"color:#64748b;font-size:11px;\">(mag)</span></div>
            <div>Takeoff ref: <span id=\"compass_ref\" style=\"color:#94a3b8;\">—</span></div>
            <div>Relative: <b style=\"color:#3b82f6;\"><span id=\"compass_rel\">--</span>°</b></div>
            <button id=\"compass_reset\" title=\"Re-capture takeoff heading from current yaw\" style=\"margin-top:4px;height:22px;font-size:11px;padding:0 8px;background:#1e3a5f;border-color:#3b82f6;\">Reset ref</button>
          </div>
        </div>
        <div id=\"telemetry\" class=\"small\" style=\"white-space:pre-wrap; margin-top:10px;\">loading...</div>
        <button id=\"graphs_toggle\" onclick=\"window.toggleGraphs && window.toggleGraphs()\" style=\"margin-top:8px;height:32px;font-size:12px;padding:0 14px;background:#1e3a5f;border-color:#3b82f6;\">Show Graphs</button>
      </div>
    </div>
  </div>

  <div id=\"graphs_panel\" style=\"display:none;margin-top:12px;padding:0 16px 16px;\">
    <div style=\"display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:8px;\" id=\"graphs_container\"></div>
  </div>

  <!-- ── ArUco Seek — hover in front of a marker (per active drone) ── -->
  <div id=\"aruco_panel\">
    <h3>ArUco Seek &mdash; hover in front of marker
      <span id=\"arc_status\" class=\"small\" style=\"margin-left:8px;color:#94a3b8;font-weight:400;\">stopped</span>
      <span id=\"arc_drone_label\" class=\"small\" style=\"margin-left:8px;color:#64748b;font-weight:400;\"></span>
    </h3>

    <div id=\"arc_live_banner\">&#9888; LIVE MODE &mdash; DRONE WILL MOVE &mdash; RC commands are being sent</div>

    <div class=\"mis-row\" style=\"margin-bottom:10px;\">
      <button id=\"arc_start\" style=\"background:#065f46;border-color:#10b981;\">&#9654; Start</button>
      <button id=\"arc_stop\" style=\"background:#7f1d1d;border-color:#ef4444;\">&#9632; Stop</button>
      <span style=\"margin-left:8px;color:#94a3b8;font-size:11px;\">Mode:</span>
      <span class=\"mode-seg\">
        <button id=\"arc_mode_observe\" class=\"active observe\" data-mode=\"observe\">OBSERVE</button>
        <button id=\"arc_mode_live\" data-mode=\"live\">LIVE</button>
      </span>
      <span id=\"arc_mode_gate\" style=\"font-size:11px;color:#fbbf24;display:none;\">LIVE disabled (REMOTE_NO_LIVE=1)</span>
      <span id=\"arc_mode_err\" style=\"display:none;margin-left:10px;padding:4px 10px;font-size:11px;background:#7f1d1d;color:#fecaca;border:1px solid #ef4444;border-radius:4px;font-weight:600;\"></span>
      <span id=\"arc_manual\" class=\"arc-manual\">
        <button id=\"arc_takeoff\" style=\"background:#065f46;border-color:#10b981;\">&uarr; Takeoff</button>
        <button id=\"arc_land\">&darr; Land</button>
        <button id=\"arc_rc_stop\">RC Stop</button>
        <button id=\"arc_emergency\" style=\"background:#7f1d1d;border-color:#ef4444;\">&#9940; EMERGENCY</button>
      </span>
      <span style=\"margin-left:12px;color:#94a3b8;font-size:11px;\">Target marker:</span>
      <input id=\"arc_target_input\" type=\"number\" min=\"0\" placeholder=\"auto\" style=\"width:70px;\" />
      <button id=\"arc_target_lock\">Lock</button>
      <button id=\"arc_target_auto\">Auto</button>
    </div>

    <div class=\"arc-grid\">
      <div>
        <div class=\"small\" style=\"color:#94a3b8;margin-bottom:4px;\">Video feed (drone camera)</div>
        <img id=\"arc_video\" class=\"arc-video\" alt=\"(press Start to load video)\" />
        <div class=\"small\" style=\"color:#94a3b8;margin:8px 0 4px 0;\">Top-down &mdash; drone &harr; marker</div>
        <canvas id=\"arc_topdown\" class=\"arc-topdown\" width=\"420\" height=\"380\"></canvas>
        <div id=\"arc_readout\" class=\"arc-readout\" style=\"margin-top:8px;\">&mdash;</div>
      </div>
      <div>
        <div class=\"small\" style=\"color:#94a3b8;margin-bottom:4px;\">Live tuning parameters &mdash; applied immediately</div>
        <div id=\"arc_params\" class=\"arc-params\"></div>
        <div style=\"margin-top:6px;\"><button id=\"arc_reload\" style=\"font-size:11px;\">&#x21bb; Reload params from server</button></div>
      </div>
    </div>
  </div>

  <!-- ── Special Missions — coordinated multi-drone flights ───────── -->
  <div id=\"missions_panel\">
    <h3>Special Missions
      <span id=\"mis_title_status\" class=\"small\" style=\"margin-left:8px;color:#94a3b8;font-weight:400;\">idle</span>
    </h3>

    <div class=\"mis-row\">
      <label style=\"color:#94a3b8;\">Mission:</label>
      <select id=\"mis_type\">
        <option value=\"scan_all\">Scan all ArUco markers (sequential, collision-aware)</option>
        <option value=\"capture_targets\">Capture enemy targets (SDC26 — box-capture, camera on arena centre)</option>
      </select>
    </div>

    <!-- Capture-Targets specific inputs — hidden unless that mission is selected. -->
    <div id=\"mis_capture_rows\" style=\"display:none;\">
      <div class=\"mis-row\">
        <label style=\"color:#94a3b8;\" title=\"One JSON object per target box. Use the Blue team's 3 enemy boxes (the red-home boxes) for standard play.\">Target boxes (JSON):</label>
        <textarea id=\"mis_boxes_json\" rows=\"8\" style=\"width:100%;max-width:520px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:6px;font-family:monospace;font-size:11px;\">[
  {\"id\": 1, \"x\": -7.0, \"y\": 2.5, \"home_team\": \"red\"},
  {\"id\": 2, \"x\": -5.0, \"y\": 5.4, \"home_team\": \"red\"},
  {\"id\": 3, \"x\": -7.0, \"y\": 8.3, \"home_team\": \"red\"},
  {\"id\": 4, \"x\":  7.0, \"y\": 2.5, \"home_team\": \"blue\"},
  {\"id\": 5, \"x\":  5.0, \"y\": 5.4, \"home_team\": \"blue\"},
  {\"id\": 6, \"x\":  7.0, \"y\": 8.3, \"home_team\": \"blue\"}
]</textarea>
      </div>
      <div class=\"mis-row\">
        <label style=\"color:#94a3b8;\" title=\"World-frame XY the drone returns to after all captures. Typically your team's home-zone centre.\">Home XY:</label>
        <input id=\"mis_home_x\" type=\"number\" step=\"0.1\" value=\"0.0\" style=\"width:70px;\" />
        <input id=\"mis_home_y\" type=\"number\" step=\"0.1\" value=\"9.0\" style=\"width:70px;\" />
        <label style=\"color:#94a3b8;margin-left:12px;\" title=\"World-frame XY the drone's camera aims at while moving (typically arena centre so many markers stay in view for triangulation).\">Face XY:</label>
        <input id=\"mis_face_x\" type=\"number\" step=\"0.1\" value=\"0.0\" style=\"width:70px;\" />
        <input id=\"mis_face_y\" type=\"number\" step=\"0.1\" value=\"5.4\" style=\"width:70px;\" />
        <label style=\"color:#94a3b8;margin-left:12px;\" title=\"Altitude above the floor for the capture hover.\">Altitude (m):</label>
        <input id=\"mis_alt\" type=\"number\" step=\"0.1\" min=\"0.5\" value=\"1.5\" style=\"width:70px;\" />
        <label style=\"color:#94a3b8;margin-left:12px;\" title=\"Hover duration above each box. Must be ≥ the 2s capture-hold from SDC26 rules.\">Hover s:</label>
        <input id=\"mis_cap_hover_s\" type=\"number\" step=\"0.5\" min=\"2.0\" value=\"4.0\" style=\"width:70px;\" />
      </div>
    </div>

    <div class=\"mis-row\">
      <label style=\"color:#94a3b8;\">Drones:</label>
      <span id=\"mis_drones\" style=\"display:flex;gap:10px;flex-wrap:wrap;\"></span>
    </div>

    <div class=\"mis-row\">
      <label style=\"color:#94a3b8;\">Target markers:</label>
      <input id=\"mis_markers\" type=\"text\" value=\"1-12\" placeholder=\"e.g. 1-12 or 1,2,3,7\" style=\"width:220px;\" />
      <label style=\"color:#94a3b8;margin-left:12px;\">Hover s:</label>
      <input id=\"mis_hover_s\" type=\"number\" min=\"0.5\" step=\"0.5\" value=\"1.5\" style=\"width:70px;\" title=\"How long the drone hovers in front of each marker before moving on. 1.5s scans cleanly; raise if the score validation needs longer.\" />
      <label style=\"color:#94a3b8;margin-left:12px;\">Approach tol (m):</label>
      <input id=\"mis_tol_m\" type=\"number\" min=\"0.1\" step=\"0.05\" value=\"0.30\" style=\"width:70px;\" title=\"Drone transitions from APPROACH to HOVER once within this many metres of the hover distance AND sufficiently perpendicular (see Skew tol).\" />
      <label style=\"color:#94a3b8;margin-left:12px;\" title=\"Max skew before declaring the drone perpendicular. 0.08 ≈ 6° off the marker normal. Lower values force the drone to align straight-on before hovering.\">Skew tol:</label>
      <input id=\"mis_skew_tol\" type=\"number\" min=\"0.02\" max=\"0.50\" step=\"0.01\" value=\"0.12\" style=\"width:70px;\" title=\"Max perspective skew to accept for HOVER. 0.12 ≈ 9° off the marker normal. Lower = stricter perpendicular (longer to converge).\" />
      <label style=\"color:#94a3b8;margin-left:12px;display:flex;align-items:center;gap:4px;\">
        <input id=\"mis_auto_takeoff\" type=\"checkbox\" />
        auto-takeoff
      </label>
    </div>

    <!-- ── Mission-as-code + presets ──────────────────────────────────
         Every mission type has its own JSON parameter block. The
         editor below is the canonical source of truth; the form
         fields above auto-sync into it when you click "From form".
         Presets are saved server-side in controller/mission_presets.json
         so they survive server restarts and are shared across browser
         tabs. -->
    <div class=\"mis-row\" style=\"margin-top:8px;\">
      <label style=\"color:#94a3b8;\">Preset:</label>
      <select id=\"mis_preset_sel\" style=\"min-width:180px;\"></select>
      <button id=\"mis_preset_load\" style=\"background:#1e293b;border-color:#475569;\" title=\"Load the selected preset's JSON into the editor\">Load</button>
      <input id=\"mis_preset_name\" type=\"text\" placeholder=\"preset name\" style=\"width:140px;\" />
      <button id=\"mis_preset_save\" style=\"background:#1e293b;border-color:#475569;\" title=\"Save the current JSON as a named preset (overwrites if name exists)\">Save</button>
      <button id=\"mis_preset_delete\" style=\"background:#450a0a;border-color:#b91c1c;color:#fecaca;\" title=\"Delete the selected preset\">Delete</button>
      <span id=\"mis_preset_status\" class=\"small\" style=\"color:#64748b;\"></span>
    </div>
    <div class=\"mis-row\">
      <label style=\"color:#94a3b8;align-self:flex-start;\">Code (JSON):</label>
      <textarea id=\"mis_code\" rows=\"10\" spellcheck=\"false\"
                style=\"width:100%;max-width:680px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:6px;font-family:monospace;font-size:11px;line-height:1.4;\"
                placeholder=\"JSON parameters for the selected mission…\"></textarea>
    </div>
    <div class=\"mis-row\">
      <button id=\"mis_code_from_form\" style=\"background:#1e293b;border-color:#475569;\" title=\"Rebuild the JSON from the form fields above\">&#8689; From form</button>
      <button id=\"mis_code_to_form\" style=\"background:#1e293b;border-color:#475569;\" title=\"Apply the JSON values to the form fields (where possible)\">&#8690; To form</button>
      <button id=\"mis_code_run\" style=\"background:#065f46;border-color:#10b981;\" title=\"Parse the JSON and start the mission using those parameters — bypasses the form\">&#9654; Run from code</button>
      <span id=\"mis_code_status\" class=\"small\" style=\"color:#64748b;\"></span>
    </div>

    <div class=\"mis-row\" style=\"margin-top:8px;\">
      <button id=\"mis_start\" style=\"background:#065f46;border-color:#10b981;\">&#9654; Start mission</button>
      <button id=\"mis_stop\" style=\"background:#7f1d1d;border-color:#ef4444;\">&#9632; Stop</button>
      <button id=\"mis_stop_land\" style=\"background:#7f1d1d;border-color:#ef4444;\">&#9632; Stop + Land</button>
      <a id=\"mis_trace_download\" href=\"/proxy/missions/trace\" download style=\"margin-left:10px;color:#93c5fd;text-decoration:underline;font-size:12px;\" title=\"Download the JSONL trace log of the current (or most recent) mission\">⤓ Download trace</a>
      <span id=\"mis_err\" style=\"display:none;margin-left:10px;padding:4px 10px;font-size:11px;background:#7f1d1d;color:#fecaca;border:1px solid #ef4444;border-radius:4px;font-weight:600;\"></span>
      <span id=\"mis_ok\"  style=\"display:none;margin-left:10px;padding:4px 10px;font-size:11px;background:#064e3b;color:#a7f3d0;border:1px solid #22c55e;border-radius:4px;font-weight:600;\"></span>
      <span id=\"mis_progress\" style=\"margin-left:12px;color:#38bdf8;font-weight:600;\">—</span>
    </div>

    <div class=\"mis-row\" style=\"margin-top:6px;\">
      <label style=\"color:#94a3b8;\">Scanned:</label>
      <span id=\"mis_scanned\" style=\"color:#4ade80;font-family:monospace;\">—</span>
      <label style=\"color:#94a3b8;margin-left:12px;\">Remaining:</label>
      <span id=\"mis_remaining\" style=\"color:#fbbf24;font-family:monospace;\">—</span>
    </div>

    <div class=\"small\" style=\"color:#94a3b8;margin:10px 0 4px 0;\">Per-drone status</div>
    <div id=\"mis_status\" class=\"mis-status\">idle — no mission running</div>
  </div>

  <script>
  // ── Live Telemetry Graphs (standalone, runs independently) ─────────
  (function(){
    const WINDOW_S = 10;
    const SAMPLE_HZ = 20;
    const CANVAS_W = 340, CANVAS_H = 130;
    const GROUPS = [
      {title:'Altitude (cm)',     keys:['height_cm','tof_cm','barometer_cm'], colors:['#22d3ee','#f472b6','#a78bfa']},
      {title:'Attitude (deg)',    keys:['pitch','roll','yaw'],                colors:['#22d3ee','#f472b6','#a78bfa']},
      {title:'Velocity (cm/s)',   keys:['vgx','vgy','vgz'],                   colors:['#22d3ee','#f472b6','#a78bfa']},
      {title:'Acceleration',      keys:['agx','agy','agz'],                   colors:['#22d3ee','#f472b6','#a78bfa']},
      {title:'Speed',             keys:['speed'],                              colors:['#22d3ee']},
      {title:'Battery (%)',       keys:['battery'],                            colors:['#34d399']},
      {title:'Temperature (C)',   keys:['temperature'],                        colors:['#fb923c']},
      {title:'Position (m)',      keys:['pos_x','pos_y','pos_z'],              colors:['#22d3ee','#f472b6','#a78bfa']},
    ];
    const graphs = [];
    let visible = false, rafId = null, sampleTimer = null;

    function init() {
      const c = document.getElementById('graphs_container');
      if (!c || graphs.length) return;
      GROUPS.forEach(g => {
        const wrap = document.createElement('div');
        wrap.style.cssText = 'background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:6px;';
        const hdr = document.createElement('div');
        hdr.style.cssText = 'font-size:11px;color:#94a3b8;margin-bottom:4px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;';
        let html = '<b style=\"color:#e2e8f0;\">' + g.title + '</b>';
        g.keys.forEach((k,i) => { html += '<span style=\"color:'+g.colors[i]+';\">'+k+'</span>'; });
        hdr.innerHTML = html;
        wrap.appendChild(hdr);
        const cv = document.createElement('canvas');
        cv.width = CANVAS_W; cv.height = CANVAS_H;
        cv.style.cssText = 'width:100%;height:auto;display:block;background:#020617;border-radius:4px;';
        wrap.appendChild(cv);
        c.appendChild(wrap);
        graphs.push({keys: g.keys, colors: g.colors, samples: [], canvas: cv, ctx: cv.getContext('2d')});
      });
      console.log('[graphs] init done,', graphs.length, 'graphs');
    }

    function sample() {
      const ts = performance.now();
      const t = (typeof window.lastTelemetry === 'object' && window.lastTelemetry) ? Object.assign({}, window.lastTelemetry) : {};
      if (Array.isArray(window._lastPos)) { t.pos_x = window._lastPos[0]; t.pos_y = window._lastPos[1]; t.pos_z = window._lastPos[2]; }
      graphs.forEach(g => {
        const vals = {}; let any = false;
        g.keys.forEach(k => { const v = t[k]; if (v != null && !isNaN(v)) { vals[k] = Number(v); any = true; } else { vals[k] = null; } });
        if (any) g.samples.push({t: ts, vals});
        const cutoff = ts - WINDOW_S * 1000;
        while (g.samples.length > 0 && g.samples[0].t < cutoff) g.samples.shift();
      });
    }

    function draw() {
      if (!visible) { rafId = null; return; }
      const now = performance.now();
      graphs.forEach(g => {
        const ctx = g.ctx, W = g.canvas.width, H = g.canvas.height;
        ctx.fillStyle = '#020617'; ctx.fillRect(0,0,W,H);
        if (g.samples.length < 2) {
          ctx.fillStyle = '#475569'; ctx.font = '11px sans-serif';
          ctx.textAlign = 'center'; ctx.fillText('waiting for data...', W/2, H/2); ctx.textAlign = 'left';
          return;
        }
        const tMin = now - WINDOW_S*1000, tMax = now;
        let yMin = Infinity, yMax = -Infinity;
        g.samples.forEach(s => g.keys.forEach(k => { if (s.vals[k]!=null) { yMin=Math.min(yMin,s.vals[k]); yMax=Math.max(yMax,s.vals[k]); } }));
        if (!isFinite(yMin)) return;
        if (yMin === yMax) { yMin -= 1; yMax += 1; }
        const pad = (yMax-yMin)*0.1 || 1; yMin -= pad; yMax += pad;
        ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 0.5;
        for (let i=0;i<=4;i++) { const y=(i/4)*H; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }
        ctx.fillStyle = '#64748b'; ctx.font = '9px monospace';
        for (let i=0;i<=4;i++) { const v = yMin + ((4-i)/4)*(yMax-yMin); ctx.fillText(v.toFixed(1), 2, (i/4)*H + 9); }
        g.keys.forEach((k, ki) => {
          ctx.strokeStyle = g.colors[ki]; ctx.lineWidth = 1.5; ctx.beginPath();
          let started = false;
          g.samples.forEach(s => {
            if (s.vals[k] == null) { started = false; return; }
            const x = ((s.t - tMin) / (tMax - tMin)) * W;
            const y = H - ((s.vals[k] - yMin) / (yMax - yMin)) * H;
            if (!started) { ctx.moveTo(x,y); started = true; } else { ctx.lineTo(x,y); }
          });
          ctx.stroke();
          const last = g.samples[g.samples.length - 1];
          if (last && last.vals[k] != null) {
            ctx.fillStyle = g.colors[ki]; ctx.font = '10px monospace'; ctx.textAlign = 'right';
            ctx.fillText(last.vals[k].toFixed(1), W - 2, 10 + ki * 11); ctx.textAlign = 'left';
          }
        });
      });
      rafId = requestAnimationFrame(draw);
    }

    window.toggleGraphs = function() {
      visible = !visible;
      const btn = document.getElementById('graphs_toggle');
      const panel = document.getElementById('graphs_panel');
      if (btn) btn.textContent = visible ? 'Hide Graphs' : 'Show Graphs';
      if (panel) panel.style.display = visible ? 'block' : 'none';
      console.log('[graphs] toggle ->', visible);
      if (visible) {
        init();
        if (!sampleTimer) sampleTimer = setInterval(sample, 1000 / SAMPLE_HZ);
        if (!rafId) rafId = requestAnimationFrame(draw);
      } else {
        if (sampleTimer) { clearInterval(sampleTimer); sampleTimer = null; }
        if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
      }
    };
    console.log('[graphs] toggleGraphs registered');
  })();
  </script>

  <!-- ── ArUco Seek client JS ─────────────────────────────────────── -->
  <script>
  (function(){
    const PGROUPS = ['Mission target','Camera filter / deadbands','P gains (camera)','D gains (IMU damping)','Output clamps','Drawing'];
    const SLIDERS = [
      ['hover_distance_m','Hover distance (m)',          0.5, 4.0, 0.05, 0],
      ['fb_max',          'Approach speed (fwd RC %)',   0,100, 1, 0],
      ['fb_back_max',     'Retreat speed (back RC %)',   0,100, 1, 0],
      ['dist_p',          'Approach aggressiveness (P · distance)', 0, 60, 0.5, 0],
      ['ema_alpha',       'EMA α (smoothing)',           0.05,0.95, 0.05, 1],
      ['deadband_x',      'Deadband err_x',              0.00,0.30, 0.01, 1],
      ['deadband_y',      'Deadband err_y',              0.00,0.30, 0.01, 1],
      ['deadband_skew',   'Deadband skew',               0.00,0.30, 0.01, 1],
      ['deadband_dist_m', 'Deadband distance (m)',       0.00,1.00, 0.05, 1],
      ['yaw_p',           'P · yaw     (per err_x)',     0, 50, 1, 2],
      ['skew_p',          'P · lateral (per skew)',      0, 50, 1, 2],
      ['alt_p',           'P · altitude (per err_y)',    0,100, 1, 2],
      ['d_yaw',           'D · yaw     (°/s)',           0,  2, 0.05, 3],
      ['d_lr',            'D · lateral (cm/s vgy)',      0,  2, 0.05, 3],
      ['d_ud',            'D · vertical(cm/s vgz)',      0,  2, 0.05, 3],
      ['d_fb',            'D · fwd/back(cm/s vgx)',      0,  2, 0.05, 3],
      ['yaw_max',         'Clamp · yaw max',             0, 80, 1, 4],
      ['lr_max',          'Clamp · lateral max',         0,100, 1, 4],
      ['ud_max',          'Clamp · vertical max',        0,100, 1, 4],
      ['rc_min',          'RC dead-floor',               0, 10, 1, 4],
      ['cam_hfov_deg',    'Cam HFOV (drawing only)',    30,110, 1, 5],
      ['marker_size_m',   'Marker physical size (m)',    0.05, 2.0, 0.01, 5],
    ];
    let arcParams = {};
    let arcAllowLive = false;
    let arcMode = 'observe';

    // Which drone id is the ArUco panel currently tracking? Mirrors the
    // main UI's active drone, picked up from /proxy/drones on each poll.
    let arcActiveId = null;

    async function arcLoadParams() {
      try {
        const r = await fetch('/proxy/aruco/params');
        arcParams = await r.json();
        arcRenderParams();
      } catch {}
    }
    function arcRenderParams() {
      const cont = document.getElementById('arc_params');
      cont.innerHTML = '';
      let curGroup = -1;
      SLIDERS.forEach(([k,label,mn,mx,st,grp]) => {
        if (grp !== curGroup) {
          curGroup = grp;
          const h = document.createElement('div');
          h.className = 'arc-pgroup';
          h.innerHTML = '<div class=\"arc-pgroup-label\">' + PGROUPS[grp] + '</div>';
          cont.appendChild(h);
        }
        const v = arcParams[k] ?? 0;
        const r = document.createElement('div');
        r.className = 'arc-row';
        r.innerHTML =
          '<label title=\"'+k+'\">'+label+' <span class=\"info-icon\" data-info=\"'+k+'\">i</span></label>' +
          '<input type=\"range\" min=\"'+mn+'\" max=\"'+mx+'\" step=\"'+st+'\" value=\"'+v+'\" data-k=\"'+k+'\" />' +
          '<input type=\"number\" min=\"'+mn+'\" max=\"'+mx+'\" step=\"'+st+'\" value=\"'+v+'\" data-k=\"'+k+'\" />';
        cont.appendChild(r);
      });
      cont.querySelectorAll('input').forEach(el => {
        el.addEventListener('input', () => {
          const k = el.dataset.k;
          const v = parseFloat(el.value);
          if (isNaN(v)) return;
          arcParams[k] = v;
          cont.querySelectorAll('input[data-k=\"'+k+'\"]').forEach(s => { if (s !== el) s.value = v; });
        });
        el.addEventListener('change', () => {
          const k = el.dataset.k;
          const v = parseFloat(el.value);
          if (!isNaN(v)) fetch('/proxy/aruco/params', {method:'POST',
              headers:{'Content-Type':'application/json'},
              body: JSON.stringify({[k]: v})});
        });
      });
    }

    function fmt(x, n) { return (x === undefined || x === null || isNaN(x)) ? '—' : Number(x).toFixed(n); }

    function arcApplyModeUI(mode, allow) {
      arcMode = mode || 'observe';
      if (typeof allow === 'boolean') arcAllowLive = allow;
      const bObs  = document.getElementById('arc_mode_observe');
      const bLive = document.getElementById('arc_mode_live');
      bObs.classList.toggle('active', arcMode === 'observe');
      bLive.classList.toggle('active', arcMode === 'live');
      bLive.classList.toggle('live',   arcMode === 'live');
      // Keep the button clickable — the server is the source of truth for
      // allow_live. We just visually hint with the "gate" badge if the
      // server has REMOTE_NO_LIVE set.
      bLive.disabled = false;
      bLive.style.opacity = arcAllowLive ? '1' : '0.7';
      document.getElementById('arc_mode_gate').style.display = arcAllowLive ? 'none' : 'inline';
      const live = (arcMode === 'live');
      document.getElementById('arc_live_banner').classList.toggle('show', live);
      document.body.classList.toggle('arc-live-mode', live);
      document.getElementById('arc_manual').classList.toggle('show', live);
    }

    function arcRenderReadout(s) {
      if (!s.running) {
        document.getElementById('arc_readout').innerHTML = '<span style=\"color:#64748b;\">stopped — press Start</span>';
        return;
      }
      const html =
        '<b>Marker</b><br>' +
        '<span class=\"k\">ID:</span>'+(s.marker_id ?? '—')+
        '&nbsp;&nbsp;<span class=\"k\">visible:</span>'+((s.visible_ids||[]).join(', ')||'—')+'<br>' +
        '<span class=\"k\">distance:</span>'+fmt(s.distance_m,2)+' m '+
          '<span class=\"pd\">(raw '+fmt(s.raw_distance_m,2)+', target '+fmt(arcParams.hover_distance_m,2)+')</span><br>' +
        '<span class=\"k\">err_x:</span>'+fmt(s.err_x,3)+
          '&nbsp;&nbsp;<span class=\"k\">err_y:</span>'+fmt(s.err_y,3)+
          '&nbsp;&nbsp;<span class=\"k\">skew:</span>'+fmt(s.skew,3)+'<br>' +
        '<br><b>IMU</b>  '+
        '<span class=\"pd\">vgx '+fmt(s.vx_cms,0)+', vgy '+fmt(s.vy_cms,0)+', vgz '+fmt(s.vz_cms,0)+' cm/s; '+
          'yaw '+fmt(s.yaw,1)+'° @ '+fmt(s.yaw_rate_dps,0)+'°/s; alt '+fmt(s.altitude_m,2)+' m</span><br>' +
        (s.mode === 'live'
          ? '<br><b class=\"arc-rc-sent\">RC — SENT to drone</b>'
          : '<br><b>RC — would-send</b> <span class=\"pd\">(observe — not sent)</span>') + '<br>' +
        '<span class=\"k\">lr:</span>'+(s.rc_lr ?? '—')+' <span class=\"pd\">(P='+fmt(s.rc_lr_p,1)+' D='+fmt(s.rc_lr_d,1)+')</span>' +
        '&nbsp; <span class=\"k\">fb:</span>'+(s.rc_fb ?? '—')+' <span class=\"pd\">(P='+fmt(s.rc_fb_p,1)+' D='+fmt(s.rc_fb_d,1)+')</span><br>' +
        '<span class=\"k\">ud:</span>'+(s.rc_ud ?? '—')+' <span class=\"pd\">(P='+fmt(s.rc_ud_p,1)+' D='+fmt(s.rc_ud_d,1)+')</span>' +
        '&nbsp; <span class=\"k\">yaw:</span>'+(s.rc_yaw ?? '—')+' <span class=\"pd\">(P='+fmt(s.rc_yaw_p,1)+' D='+fmt(s.rc_yaw_d,1)+')</span><br>' +
        (s.rc_sent_at ? '<span class=\"k\">last sent:</span><span class=\"arc-rc-sent\">'+((Date.now()/1000 - s.rc_sent_at).toFixed(1))+' s ago</span><br>' : '') +
        (s.rc_send_error ? '<span class=\"k\">send err:</span><span style=\"color:#fca5a5;\">'+s.rc_send_error+'</span><br>' : '') +
        // Arena safety guard banner — red when active, grey when idle.
        (s.guard && s.guard.active
          ? '<br><b style=\"color:#fca5a5;background:#7f1d1d;padding:2px 6px;border-radius:3px;\">⛔ SAFETY GUARD</b> '
              + '<span class=\"pd\">' + (s.guard.actions||[]).join(', ')
              + '  pos=('+ (s.guard.pos||[]).join(',') +')</span>'
          : '');
      document.getElementById('arc_readout').innerHTML = html;
    }

    function arcDrawTopDown(s) {
      const c = document.getElementById('arc_topdown');
      const ctx = c.getContext('2d');
      const W = c.width, H = c.height;
      ctx.fillStyle = '#0b1220'; ctx.fillRect(0,0,W,H);
      const cx = W/2;
      const marker_y = 36;
      const target = arcParams.hover_distance_m || 1.5;
      const maxDist = Math.max(target*2.2, 3.0);
      const ppm = (H-80)/maxDist;
      ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1;
      for (let d=0.5; d<=maxDist; d+=0.5) { ctx.beginPath(); ctx.arc(cx, marker_y, d*ppm, 0, Math.PI, false); ctx.stroke(); }
      ctx.fillStyle = '#334155'; ctx.font = '9px monospace'; ctx.textAlign = 'left';
      for (let d=1; d<=maxDist; d+=1) ctx.fillText(d+'m', cx+4, marker_y + d*ppm - 2);
      ctx.strokeStyle = '#0ea5e9'; ctx.lineWidth = 1.5; ctx.setLineDash([4,4]);
      ctx.beginPath(); ctx.arc(cx, marker_y, target*ppm, 0, Math.PI, false); ctx.stroke();
      ctx.setLineDash([]);
      // marker
      const mw = 60;
      ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 6;
      ctx.beginPath(); ctx.moveTo(cx-mw, marker_y); ctx.lineTo(cx+mw, marker_y); ctx.stroke();
      ctx.fillStyle = '#22c55e'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
      ctx.fillText('marker '+(s.marker_id ?? '?'), cx+mw+6, marker_y+4);
      if (!s.running || s.distance_m == null) {
        ctx.fillStyle = '#475569'; ctx.font = '13px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(s.running ? 'no marker visible' : 'stopped', W/2, H/2);
        return;
      }
      const dist = s.distance_m;
      const lateral_m = -(s.skew || 0) * dist;
      const drone_x = cx + lateral_m*ppm;
      const drone_y = marker_y + dist*ppm;
      const hfov = (arcParams.cam_hfov_deg || 69) * Math.PI / 180;
      const yaw_off_rad = Math.atan((s.err_x || 0) * Math.tan(hfov/2));
      const dxv = cx - drone_x, dyv = marker_y - drone_y;
      const aimAng = Math.atan2(dyv, dxv);
      const droneAng = aimAng - yaw_off_rad;
      // LoS
      ctx.strokeStyle = '#fbbf2470'; ctx.lineWidth = 1.5; ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(drone_x, drone_y); ctx.lineTo(cx, marker_y); ctx.stroke();
      ctx.setLineDash([]);
      // FOV
      ctx.strokeStyle = '#38bdf833'; ctx.fillStyle = '#38bdf815';
      const fovLen = ppm * Math.max(2.5, dist+1.0);
      ctx.save(); ctx.translate(drone_x, drone_y); ctx.rotate(droneAng);
      ctx.beginPath(); ctx.moveTo(0,0);
      ctx.lineTo(fovLen*Math.cos(-hfov/2), fovLen*Math.sin(-hfov/2));
      ctx.lineTo(fovLen*Math.cos( hfov/2), fovLen*Math.sin( hfov/2));
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.restore();
      // drone
      ctx.save(); ctx.translate(drone_x, drone_y); ctx.rotate(droneAng);
      ctx.fillStyle = '#fbbf24'; ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(15,0); ctx.lineTo(-10,-10); ctx.lineTo(-10,10); ctx.closePath();
      ctx.fill(); ctx.stroke();
      const vx = s.vx_cms||0, vy = s.vy_cms||0;
      if (Math.hypot(vx,vy) > 2) {
        ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(vx*0.7, vy*0.7); ctx.stroke();
      }
      ctx.restore();
      ctx.fillStyle = '#fbbf24'; ctx.font = 'bold 13px monospace'; ctx.textAlign = 'center';
      ctx.fillText(dist.toFixed(2)+' m', (drone_x+cx)/2+14, (drone_y+marker_y)/2+4);
      ctx.fillStyle = '#cbd5e1'; ctx.font = '11px monospace'; ctx.textAlign = 'left';
      ctx.fillText('lateral '+lateral_m.toFixed(2)+' m', 8, H-22);
      ctx.fillText('dist err '+(dist-target).toFixed(2)+' m', 8, H-6);
      ctx.textAlign = 'right';
      ctx.fillText('target '+target.toFixed(2)+' m', W-8, H-22);
      ctx.fillText('err_x '+(s.err_x||0).toFixed(3), W-8, H-6);
    }

    async function arcPoll() {
      try {
        const r = await fetch('/proxy/aruco/state');
        const s = await r.json();
        const st = document.getElementById('arc_status');
        if (s.running) { st.textContent = '● ' + (s.status_msg || 'running'); st.style.color = '#22c55e'; }
        else           { st.textContent = '○ stopped';                          st.style.color = '#94a3b8'; }
        const dl = document.getElementById('arc_drone_label');
        if (s.drone_id && arcActiveId !== s.drone_id) {
          arcActiveId = s.drone_id;
        }
        if (arcActiveId) dl.textContent = '[drone '+arcActiveId+']';
        arcApplyModeUI(s.mode || 'observe', s.allow_live);
        arcRenderReadout(s);
        arcDrawTopDown(s);
      } catch(e) {}
    }

    // Buttons
    document.getElementById('arc_start').onclick = async () => {
      await fetch('/proxy/aruco/start', {method:'POST'});
      const av = document.getElementById('arc_video');
      av.src = '/proxy/aruco/video.mjpg?t=' + Date.now();
      av.setAttribute('data-active', '1');
      // Reload params for the now-active drone
      arcLoadParams();
    };
    document.getElementById('arc_stop').onclick = async () => {
      await fetch('/proxy/aruco/stop', {method:'POST'});
      const av = document.getElementById('arc_video');
      av.src = '';
      av.removeAttribute('data-active');
    };
    document.getElementById('arc_target_lock').onclick = async () => {
      const v = document.getElementById('arc_target_input').value;
      await fetch('/proxy/aruco/target', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({marker: v ? parseInt(v) : null})});
    };
    document.getElementById('arc_target_auto').onclick = async () => {
      document.getElementById('arc_target_input').value = '';
      await fetch('/proxy/aruco/target', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({marker: null})});
    };
    document.getElementById('arc_reload').onclick = arcLoadParams;

    function arcShowModeErr(msg, kind) {
      const el = document.getElementById('arc_mode_err');
      if (!el) return;
      clearTimeout(arcShowModeErr._t);
      if (!msg) { el.style.display = 'none'; return; }
      // kind: 'warn' = amber, default = red
      const warn = (kind === 'warn') || msg.startsWith('⚠');
      el.textContent = warn ? msg : ('✗ ' + msg);
      el.style.background = warn ? '#78350f' : '#7f1d1d';
      el.style.color      = warn ? '#fde68a' : '#fecaca';
      el.style.borderColor = warn ? '#f59e0b' : '#ef4444';
      el.style.display = 'inline';
      // Auto-hide warnings in 3s, errors in 8s
      arcShowModeErr._t = setTimeout(() => { el.style.display = 'none'; }, warn ? 3200 : 8000);
    }
    async function arcSetMode(mode) {
      console.log('[arc] set-mode request:', mode);
      // Clear previous error
      document.getElementById('arc_mode_err').style.display = 'none';
      try {
        const r = await fetch('/proxy/aruco/mode', {method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({mode})});
        let j;
        try { j = await r.json(); } catch { j = {}; }
        console.log('[arc] set-mode response:', r.status, j);
        if (!r.ok || !j.ok) {
          const msg = (j.error || j.message || r.statusText || 'unknown') + ' (HTTP ' + r.status + ')';
          arcShowModeErr(msg);
          return;
        }
        arcApplyModeUI(j.mode || mode, arcAllowLive);
      } catch (err) {
        console.error('[arc] set-mode failed:', err);
        arcShowModeErr('network error: ' + err);
      }
    }
    document.getElementById('arc_mode_observe').onclick = () => arcSetMode('observe');
    // LIVE button: always post to server, let it decide. Client-side arcAllowLive
    // is now just a UI hint (disabled state) — if somehow wrong, the server
    // returns 403 with a clear error message.
    // LIVE button: double-click / re-click confirmation (no browser confirm()).
    // First click arms — button pulses + warning shows.
    // Second click within 3 seconds → switches mode.
    // Keeps us off native confirm() dialogs which some browsers auto-dismiss.
    const liveBtn = document.getElementById('arc_mode_live');
    let _arcArmedUntil = 0;
    if (liveBtn) {
      liveBtn.onclick = () => {
        const now = Date.now();
        console.log('[arc] LIVE clicked, arcAllowLive=', arcAllowLive, 'armed=', (now < _arcArmedUntil));
        if (now < _arcArmedUntil) {
          // Armed → actually switch
          _arcArmedUntil = 0;
          liveBtn.style.animation = '';
          liveBtn.textContent = 'LIVE';
          arcShowModeErr(''); // hides
          arcSetMode('live');
          return;
        }
        // First click — arm for 3s
        _arcArmedUntil = now + 3000;
        liveBtn.style.animation = 'arcpulse 0.6s ease-in-out infinite';
        liveBtn.textContent = 'LIVE — click again';
        arcShowModeErr('⚠ Click LIVE again within 3 s to confirm');
        setTimeout(() => {
          if (Date.now() >= _arcArmedUntil) {
            _arcArmedUntil = 0;
            liveBtn.style.animation = '';
            liveBtn.textContent = 'LIVE';
            arcShowModeErr('');
          }
        }, 3100);
      };
      console.log('[arc] LIVE button click handler attached');
    } else {
      console.error('[arc] could not find arc_mode_live button — handler NOT attached');
    }

    async function arcPostCmd(path, confirmMsg) {
      if (confirmMsg && !confirm(confirmMsg)) return;
      const r = await fetch(path, {method:'POST'});
      const j = await r.json();
      if (!j.ok && j.error) alert(path + ' refused: ' + j.error);
    }
    document.getElementById('arc_takeoff').onclick   = () => arcPostCmd('/proxy/aruco/takeoff',   'Send TAKEOFF to the active drone?');
    document.getElementById('arc_land').onclick      = () => arcPostCmd('/proxy/aruco/land',      'Send LAND to the active drone?');
    document.getElementById('arc_rc_stop').onclick   = () => arcPostCmd('/proxy/aruco/rc_stop',   null);
    document.getElementById('arc_emergency').onclick = () => arcPostCmd('/proxy/aruco/emergency', '⛔ EMERGENCY STOP — cut motors immediately. Confirm?');

    arcLoadParams();
    setInterval(arcPoll, 500);    // was 250ms/4Hz → 2Hz; 500ms is plenty for the readout
    arcPoll();
    // Version marker — if this string doesn't appear in the DOM,
    // you're running stale JS (restart the Python server or hard-refresh).
    const BUILD = 'co-takeoff-reasons';
    console.log('[arc] init complete, build=' + BUILD);
    const ver = document.createElement('span');
    ver.id = 'arc_build_tag';
    ver.style.cssText = 'font-size:10px;color:#10b981;margin-left:8px;font-weight:700;';
    ver.textContent = 'build ' + BUILD;
    const hdr = document.querySelector('#aruco_panel h3');
    if (hdr) hdr.appendChild(ver);

    // Immediate visible click counter — proves the button event fires, even
    // if the subsequent fetch hangs or the server is wedged. Counts every
    // LIVE / OBSERVE / Takeoff / Land / RC-stop / Emergency press.
    let _arcClicks = 0;
    const clickTag = document.createElement('span');
    clickTag.id = 'arc_click_counter';
    clickTag.style.cssText = 'font-size:10px;color:#fbbf24;margin-left:8px;';
    clickTag.textContent = 'clicks: 0';
    if (hdr) hdr.appendChild(clickTag);
    function arcBumpClicks(src) {
      _arcClicks += 1;
      clickTag.textContent = 'clicks: ' + _arcClicks + ' (' + src + ')';
      console.log('[arc] click #' + _arcClicks + ' from ' + src);
    }
    // Wire onto the existing buttons defensively. We use addEventListener so
    // we don't overwrite the onclick handlers that actually do the work.
    ['arc_mode_observe','arc_mode_live','arc_takeoff','arc_land','arc_rc_stop','arc_emergency','arc_start','arc_stop'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('click', () => arcBumpClicks(id));
    });
  })();
  </script>

  <!-- ── Special Missions client JS ───────────────────────────────── -->
  <script>
  (function(){
    let misDronesKnown = {};

    async function misLoadDrones() {
      try {
        const r = await fetch('/proxy/drones');
        const d = await r.json();
        misDronesKnown = d.drones || {};
        const cont = document.getElementById('mis_drones');
        const prev = {};
        cont.querySelectorAll('input[type=checkbox]').forEach(cb => { prev[cb.dataset.id] = cb.checked; });
        cont.innerHTML = '';
        const ids = Object.keys(misDronesKnown).sort();
        ids.forEach(id => {
          const info = misDronesKnown[id];
          const wrap = document.createElement('label');
          wrap.style.cssText = 'display:flex;gap:4px;align-items:center;font-size:12px;color:#e2e8f0;cursor:pointer;';
          const checked = (id in prev) ? prev[id] : true;
          wrap.innerHTML = '<input type=\"checkbox\" data-id=\"'+id+'\"'+(checked?' checked':'')+' /> '+info.name+' <span style=\"color:#64748b;\">#'+id+'</span>';
          cont.appendChild(wrap);
        });
      } catch {}
    }

    function misSelectedDroneIds() {
      return Array.from(document.querySelectorAll('#mis_drones input[type=checkbox]'))
        .filter(cb => cb.checked).map(cb => cb.dataset.id);
    }

    function misBadge(phase) {
      const low = (phase || 'idle').toLowerCase();
      let cls = 'idle';
      if (low === 'search') cls = 'scan';
      else if (low === 'approach') cls = 'approach';
      else if (low === 'hover') cls = 'hover';
      else if (low === 'done') cls = 'done';
      else if (low === 'error') cls = 'error';
      return '<span class=\"mis-badge '+cls+'\">'+phase+'</span>';
    }

    function misRenderStatus(st) {
      const title = document.getElementById('mis_title_status');
      const prog = document.getElementById('mis_progress');
      const panel = document.getElementById('mis_status');

      if (!st.has_mission) {
        title.textContent = 'idle';
        title.style.color = '#94a3b8';
        prog.textContent = '—';
        document.getElementById('mis_scanned').textContent = '—';
        document.getElementById('mis_remaining').textContent = '—';
        panel.textContent = 'idle — no mission running';
        return;
      }
      title.textContent = st.active ? '● running' : '○ stopped';
      title.style.color = st.active ? '#4ade80' : '#94a3b8';
      prog.textContent = 'Progress: ' + st.progress;
      document.getElementById('mis_scanned').textContent = (st.scanned || []).join(', ') || '—';
      document.getElementById('mis_remaining').textContent = (st.remaining || []).join(', ') || '—';

      let html = '';
      const drones = st.drones || {};
      Object.keys(drones).sort().forEach(did => {
        const d = drones[did];
        const name = (misDronesKnown[did] && misDronesKnown[did].name) || ('Drone '+did);
        const tgt = d.target != null ? 'target '+d.target : '—';
        html += '<div class=\"mis-drone-line\"><b>'+name+' #'+did+'</b> '+misBadge(d.phase)+
                ' <span style=\"color:#64748b;\">['+tgt+']</span> <span style=\"color:#cbd5e1;\">'+(d.note||'')+'</span></div>';
      });
      if (st.error) html += '<div style=\"color:#fca5a5;margin-top:6px;\">error: '+st.error+'</div>';
      panel.innerHTML = html || 'no drones assigned';
    }

    async function misPoll() {
      try {
        const r = await fetch('/proxy/missions/status');
        const st = await r.json();
        misRenderStatus(st);
        // If a capture-targets mission is running, pull its boxes into the
        // global so the arena views (2D + 3D) can render them.
        if (st && st.has_mission && st.target_boxes &&
            Array.isArray(st.target_boxes) && st.target_boxes.length) {
          window._targetBoxes = st.target_boxes;
        }
        // Capture state → colour the boxes in the arena view
        window._missionClaimedBoxes = (st && st.claimed) || {};
        window._missionCapturedBoxes = (st && st.captured) || [];
      } catch {}
    }

    // Parse the mission's target-boxes JSON textarea live and expose it
    // globally. Both the 2D canvas drawArena() and the Three.js scene
    // read window._targetBoxes to render the boxes on the floor even
    // before the mission starts.
    function _parseTargetBoxesInput() {
      const el = document.getElementById('mis_boxes_json');
      if (!el) return;
      try {
        const v = JSON.parse(el.value);
        if (Array.isArray(v)) {
          window._targetBoxes = v;
          el.style.borderColor = '#334155';
        } else {
          el.style.borderColor = '#ef4444';
        }
      } catch {
        el.style.borderColor = '#ef4444';
      }
    }
    (function(){
      const el = document.getElementById('mis_boxes_json');
      if (el) {
        el.addEventListener('input', _parseTargetBoxesInput);
        _parseTargetBoxesInput();  // initial parse so defaults render
      }
    })();

    // Click-to-arm pattern for Start Mission — no native confirm() dialog
    // (some browsers auto-dismiss rapid-fire dialogs, making the mission
    // never start with no visible feedback).
    let _misArmedUntil = 0;
    document.getElementById('mis_start').onclick = async () => {
      const drone_ids = misSelectedDroneIds();
      const misErr = document.getElementById('mis_err');
      const misOk  = document.getElementById('mis_ok');
      function misShowWarn(msg) {
        if (!misErr) return;
        misErr.textContent = msg;
        misErr.style.background = '#78350f';
        misErr.style.color = '#fde68a';
        misErr.style.borderColor = '#f59e0b';
        misErr.style.display = 'inline-block';
      }
      function misClearMsgs() {
        if (misErr) misErr.style.display = 'none';
        if (misOk)  misOk.style.display  = 'none';
      }
      if (!drone_ids.length) {
        misShowWarn('✗ Select at least one drone first');
        return;
      }
      const missionType = document.getElementById('mis_type').value;
      const auto_takeoff = document.getElementById('mis_auto_takeoff').checked;
      let endpoint, payload;
      if (missionType === 'capture_targets') {
        // Parse target-boxes JSON
        let boxes = [];
        try { boxes = JSON.parse(document.getElementById('mis_boxes_json').value); }
        catch (e) { misShowWarn('✗ target_boxes JSON invalid: ' + e.message); return; }
        if (!Array.isArray(boxes) || boxes.length === 0) {
          misShowWarn('✗ target_boxes must be a non-empty JSON array'); return;
        }
        const home_xy = [
          parseFloat(document.getElementById('mis_home_x').value) || 0,
          parseFloat(document.getElementById('mis_home_y').value) || 0,
        ];
        const face_xy = [
          parseFloat(document.getElementById('mis_face_x').value) || 0,
          parseFloat(document.getElementById('mis_face_y').value) || 0,
        ];
        const alt = parseFloat(document.getElementById('mis_alt').value) || 1.5;
        const hv  = parseFloat(document.getElementById('mis_cap_hover_s').value) || 4.0;
        endpoint = '/proxy/missions/capture_targets/start';
        payload = {
          drone_ids, target_boxes: boxes,
          home_xy, arena_face_xy: face_xy,
          hover_above_m: alt, hover_seconds: hv,
          auto_takeoff,
        };
      } else {
        // scan_all (default)
        const markers = document.getElementById('mis_markers').value;
        const hover_seconds = parseFloat(document.getElementById('mis_hover_s').value) || 3.0;
        const tol = parseFloat(document.getElementById('mis_tol_m').value) || 0.35;
        const skew_tol_el = document.getElementById('mis_skew_tol');
        const skew_tol = skew_tol_el ? (parseFloat(skew_tol_el.value) || 0.08) : 0.08;
        endpoint = '/proxy/missions/scan_all/start';
        payload = {
          drone_ids, target_markers: markers,
          hover_seconds, approach_tolerance_m: tol,
          approach_skew_tol: skew_tol,
          auto_takeoff,
        };
      }
      const btn = document.getElementById('mis_start');
      const now = Date.now();
      if (now >= _misArmedUntil) {
        // First click — arm for 3s, show summary inline
        _misArmedUntil = now + 3000;
        const origLabel = btn.textContent;
        btn._origLabel = origLabel;
        btn.textContent = '⚠ Click again to launch';
        btn.style.animation = 'arcpulse 0.6s ease-in-out infinite';
        const summary = 'Drones: ' + drone_ids.join(', ') +
                        '   •   Markers: ' + markers +
                        '   •   Hover ' + hover_seconds + 's' +
                        (auto_takeoff ? '   •   ⚠ AUTO-TAKEOFF' : '');
        misShowWarn('⚠ Starting mission — ' + summary + '. Click Start again within 3 s.');
        setTimeout(() => {
          if (Date.now() >= _misArmedUntil) {
            _misArmedUntil = 0;
            btn.textContent = btn._origLabel || '▶ Start mission';
            btn.style.animation = '';
            misClearMsgs();
          }
        }, 3100);
        return;
      }
      // Armed — proceed with launch
      _misArmedUntil = 0;
      btn.style.animation = '';
      btn.textContent = btn._origLabel || '▶ Start mission';
      misClearMsgs();
      const origLabel = btn.textContent;
      btn.disabled = true; btn.textContent = '… starting';
      let j = {}; let httpStatus = 0;
      try {
        const r = await fetch(endpoint, {method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify(payload)});
        httpStatus = r.status;
        try { j = await r.json(); } catch { j = {}; }
      } catch (err) {
        j = { ok: false, error: 'network error: ' + err };
      } finally {
        btn.disabled = false; btn.textContent = origLabel;
      }
      console.log('[mission] start response:', httpStatus, j);
      if (!j.ok) {
        const msg = (j.error || j.message || 'unknown error') + (httpStatus ? ' (HTTP ' + httpStatus + ')' : '');
        if (misErr) {
          misErr.textContent = '✗ Mission start refused: ' + msg;
          misErr.style.display = 'inline-block';
        }
      } else {
        const msg = j.message || 'mission started';
        const warn = j.status && j.status.error ? ' — warning: ' + j.status.error : '';
        console.log('[mission] started:', msg, warn);
        if (misOk) {
          misOk.textContent = '✓ ' + msg + warn;
          misOk.style.display = 'inline-block';
        }
      }
      misPoll();
    };

    document.getElementById('mis_stop').onclick = async () => {
      await fetch('/proxy/missions/stop', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({land:false})});
      misPoll();
    };

    document.getElementById('mis_stop_land').onclick = async () => {
      if (!confirm('Stop mission AND land all participating drones?')) return;
      await fetch('/proxy/missions/stop', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({land:true})});
      misPoll();
    };

    misLoadDrones();
    // If the drone config changes, the drones list in the ArUco section
    // repopulates via /proxy/drones poll — mirror that for missions.
    setInterval(misLoadDrones, 5000);
    setInterval(misPoll, 1500);  // was 500ms → 1.5s; mission status barely changes between ticks
    misPoll();

    // Show/hide the capture-targets-specific rows based on mission type
    const misTypeSel = document.getElementById('mis_type');
    const misCaptureRows = document.getElementById('mis_capture_rows');
    const misScanRows = document.querySelector('#missions_panel .mis-row:nth-of-type(3)');
    function syncMissionUI() {
      const t = misTypeSel.value;
      if (misCaptureRows) misCaptureRows.style.display = (t === 'capture_targets') ? '' : 'none';
      if (misScanRows)    misScanRows.style.display    = (t === 'capture_targets') ? 'none' : '';
    }
    if (misTypeSel) { misTypeSel.addEventListener('change', syncMissionUI); syncMissionUI(); }

    // ── Mission-as-code + preset management ─────────────────────────
    // Each mission type has its own JSON block. The editor is the
    // canonical source of truth: "Run from code" parses it and starts
    // directly. "From form" rebuilds the JSON from the form fields so
    // the editor stays in sync with traditional click-around editing.
    // Presets are persisted server-side in mission_presets.json.
    (function wireMissionCode(){
      const code  = document.getElementById('mis_code');
      const sel   = document.getElementById('mis_preset_sel');
      const nameI = document.getElementById('mis_preset_name');
      const psBtn = document.getElementById('mis_preset_save');
      const pdBtn = document.getElementById('mis_preset_delete');
      const plBtn = document.getElementById('mis_preset_load');
      const ffBtn = document.getElementById('mis_code_from_form');
      const tfBtn = document.getElementById('mis_code_to_form');
      const runBtn = document.getElementById('mis_code_run');
      const psStatus = document.getElementById('mis_preset_status');
      const cStatus  = document.getElementById('mis_code_status');
      if (!code) return;
      let _presets = {};

      function flash(el, msg, col) {
        if (!el) return;
        el.textContent = msg;
        el.style.color = col || '#64748b';
        setTimeout(() => { if (el.textContent === msg) el.textContent = ''; }, 3000);
      }
      function currentType() {
        return (misTypeSel && misTypeSel.value) || 'scan_all';
      }
      function renderPresetList() {
        if (!sel) return;
        const t = currentType();
        const names = Object.keys((_presets[t]) || {}).sort();
        sel.innerHTML = '';
        if (!names.length) {
          const opt = document.createElement('option');
          opt.value = ''; opt.textContent = '(no presets)';
          sel.appendChild(opt);
          return;
        }
        names.forEach(n => {
          const opt = document.createElement('option');
          opt.value = n; opt.textContent = n;
          sel.appendChild(opt);
        });
        if (names.includes('default')) sel.value = 'default';
      }
      async function loadPresets() {
        try {
          const j = await (await fetch('/proxy/missions/presets')).json();
          _presets = j.presets || {};
          renderPresetList();
          // Auto-populate the code editor with the default preset on first load
          if (!code.value.trim()) loadIntoEditor('default');
        } catch (e) { console.warn('[mis] preset load failed', e); }
      }
      function loadIntoEditor(name) {
        const t = currentType();
        const params = (_presets[t] || {})[name];
        if (!params) return;
        code.value = JSON.stringify(params, null, 2);
        if (nameI) nameI.value = name;
        flash(psStatus, '\u2713 loaded "' + name + '"', '#22c55e');
      }
      function buildFromForm() {
        // Reuse the same logic as mis_start but return the payload
        // instead of POSTing. Everything below mirrors the existing
        // build in mis_start.
        const drone_ids = misSelectedDroneIds();
        const auto_takeoff = document.getElementById('mis_auto_takeoff').checked;
        if (currentType() === 'capture_targets') {
          let boxes = [];
          try { boxes = JSON.parse(document.getElementById('mis_boxes_json').value); }
          catch {}
          return {
            drone_ids, target_boxes: boxes,
            home_xy: [
              parseFloat(document.getElementById('mis_home_x').value) || 0,
              parseFloat(document.getElementById('mis_home_y').value) || 0,
            ],
            arena_face_xy: [
              parseFloat(document.getElementById('mis_face_x').value) || 0,
              parseFloat(document.getElementById('mis_face_y').value) || 0,
            ],
            hover_above_m: parseFloat(document.getElementById('mis_alt').value) || 1.5,
            hover_seconds: parseFloat(document.getElementById('mis_cap_hover_s').value) || 4.0,
            auto_takeoff,
          };
        }
        return {
          drone_ids,
          target_markers: document.getElementById('mis_markers').value,
          hover_seconds: parseFloat(document.getElementById('mis_hover_s').value) || 3.0,
          approach_tolerance_m: parseFloat(document.getElementById('mis_tol_m').value) || 0.30,
          approach_skew_tol: parseFloat(document.getElementById('mis_skew_tol').value) || 0.12,
          auto_takeoff,
        };
      }
      function applyCodeToForm() {
        try {
          const p = JSON.parse(code.value);
          if (currentType() === 'capture_targets') {
            if (Array.isArray(p.target_boxes))
              document.getElementById('mis_boxes_json').value = JSON.stringify(p.target_boxes, null, 2);
            if (Array.isArray(p.home_xy)) {
              document.getElementById('mis_home_x').value = p.home_xy[0];
              document.getElementById('mis_home_y').value = p.home_xy[1];
            }
            if (Array.isArray(p.arena_face_xy)) {
              document.getElementById('mis_face_x').value = p.arena_face_xy[0];
              document.getElementById('mis_face_y').value = p.arena_face_xy[1];
            }
            if (p.hover_above_m != null) document.getElementById('mis_alt').value = p.hover_above_m;
            if (p.hover_seconds != null) document.getElementById('mis_cap_hover_s').value = p.hover_seconds;
          } else {
            if (p.target_markers != null) document.getElementById('mis_markers').value = p.target_markers;
            if (p.hover_seconds != null) document.getElementById('mis_hover_s').value = p.hover_seconds;
            if (p.approach_tolerance_m != null) document.getElementById('mis_tol_m').value = p.approach_tolerance_m;
            if (p.approach_skew_tol != null) document.getElementById('mis_skew_tol').value = p.approach_skew_tol;
          }
          if (typeof p.auto_takeoff === 'boolean')
            document.getElementById('mis_auto_takeoff').checked = p.auto_takeoff;
          flash(cStatus, '\u2713 form populated from JSON', '#22c55e');
        } catch (e) { flash(cStatus, '\u2717 JSON parse error: ' + e.message, '#ef4444'); }
      }
      // Wire buttons
      if (ffBtn) ffBtn.onclick = () => {
        code.value = JSON.stringify(buildFromForm(), null, 2);
        flash(cStatus, '\u2713 JSON rebuilt from form', '#22c55e');
      };
      if (tfBtn) tfBtn.onclick = applyCodeToForm;
      if (plBtn) plBtn.onclick = () => { if (sel && sel.value) loadIntoEditor(sel.value); };
      if (psBtn) psBtn.onclick = async () => {
        const name = (nameI && nameI.value.trim()) || (sel && sel.value) || '';
        if (!name) { flash(psStatus, '\u2717 enter a preset name', '#ef4444'); return; }
        let params;
        try { params = JSON.parse(code.value); }
        catch (e) { flash(psStatus, '\u2717 invalid JSON: ' + e.message, '#ef4444'); return; }
        try {
          const r = await fetch('/proxy/missions/presets', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({
              mission_type: currentType(), name, params,
            }),
          });
          const j = await r.json();
          if (j.ok) { flash(psStatus, '\u2713 saved "' + name + '"', '#22c55e'); await loadPresets(); }
          else flash(psStatus, '\u2717 ' + (j.error || 'save failed'), '#ef4444');
        } catch (e) { flash(psStatus, '\u2717 ' + e, '#ef4444'); }
      };
      if (pdBtn) pdBtn.onclick = async () => {
        const name = (sel && sel.value) || '';
        if (!name) return;
        if (!confirm('Delete preset "' + name + '" for ' + currentType() + '?')) return;
        try {
          const u = '/proxy/missions/presets?mission_type=' + encodeURIComponent(currentType())
                    + '&name=' + encodeURIComponent(name);
          const r = await fetch(u, {method:'DELETE'});
          const j = await r.json();
          if (j.ok) { flash(psStatus, '\u2713 deleted "' + name + '"', '#22c55e'); await loadPresets(); }
          else flash(psStatus, '\u2717 ' + (j.error || 'delete failed'), '#ef4444');
        } catch (e) { flash(psStatus, '\u2717 ' + e, '#ef4444'); }
      };
      if (runBtn) runBtn.onclick = async () => {
        let payload;
        try { payload = JSON.parse(code.value); }
        catch (e) { flash(cStatus, '\u2717 JSON parse error: ' + e.message, '#ef4444'); return; }
        // Fill drone_ids from selector if JSON didn't provide any
        if (!Array.isArray(payload.drone_ids) || !payload.drone_ids.length) {
          payload.drone_ids = misSelectedDroneIds();
        }
        if (!payload.drone_ids || !payload.drone_ids.length) {
          flash(cStatus, '\u2717 drone_ids missing — select drone(s) or include in JSON', '#ef4444');
          return;
        }
        const endpoint = (currentType() === 'capture_targets')
          ? '/proxy/missions/capture_targets/start'
          : '/proxy/missions/scan_all/start';
        try {
          const r = await fetch(endpoint, {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify(payload),
          });
          const j = await r.json();
          if (r.ok && j.ok) flash(cStatus, '\u2713 mission started', '#22c55e');
          else flash(cStatus, '\u2717 ' + (j.error || j.message || ('HTTP ' + r.status)), '#ef4444');
          misPoll();
        } catch (e) { flash(cStatus, '\u2717 ' + e, '#ef4444'); }
      };
      // Re-render preset list when the user flips mission type
      if (misTypeSel) misTypeSel.addEventListener('change', () => {
        renderPresetList();
        // Also auto-load the default preset of the new type so the
        // editor shows something relevant instead of stale JSON.
        const def = (_presets[currentType()] || {}).default;
        if (def) code.value = JSON.stringify(def, null, 2);
      });
      loadPresets();
    })();

    // Mission panel click counter — same idea as the ArUco one. Proves the
    // Start / Stop buttons receive the click event, independent of what
    // the server does with the request afterwards.
    const misHdr = document.querySelector('#missions_panel h3');
    if (misHdr) {
      const misClickTag = document.createElement('span');
      misClickTag.id = 'mis_click_counter';
      misClickTag.style.cssText = 'font-size:10px;color:#fbbf24;margin-left:8px;font-weight:400;';
      misClickTag.textContent = 'clicks: 0';
      misHdr.appendChild(misClickTag);
      let _misClicks = 0;
      ['mis_start','mis_stop','mis_stop_land'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('click', () => {
          _misClicks += 1;
          misClickTag.textContent = 'clicks: ' + _misClicks + ' (' + id + ')';
          console.log('[mission] click #' + _misClicks + ' from ' + id);
        });
      });
    }
  })();
  </script>

  <div class=\"row\" style=\"margin-top:12px;\">
    <div class=\"panel pos-panel\">
      <div style=\"display:flex;align-items:center;gap:10px;margin-bottom:8px;\">
        <b>Position Tracker</b>
        <label style=\"display:flex;align-items:center;gap:5px;font-size:12px;color:#94a3b8;cursor:pointer;\">
          <input type=\"checkbox\" id=\"pos_enabled\" style=\"accent-color:#0ea5e9;\" />
          Enable
        </label>
        <span id=\"pos_status_badge\" class=\"small\" style=\"color:#64748b;\">disabled</span>
      </div>
      <!-- Auto Positioning: single master toggle that overrides every user
           filter/precision knob with the built-in Claude preset on the FC.
           When ON, sliders below are greyed out and the operator can't
           accidentally dial the pose into garbage. When OFF, full manual
           tuning returns. Default ON for \"just works\" out of the box. -->
      <div id=\"pos_auto_bar\" style=\"display:flex;align-items:center;gap:10px;margin-bottom:8px;padding:7px 10px;background:#052e16;border:1px solid #10b981;border-radius:6px;\">
        <label style=\"display:flex;align-items:center;gap:6px;cursor:pointer;font-weight:600;\">
          <input type=\"checkbox\" id=\"pos_auto_toggle\" checked style=\"accent-color:#10b981;transform:scale(1.3);\" />
          <span style=\"color:#d1fae5;\">Auto Positioning (Claude algorithm)</span>
        </label>
        <span class=\"small\" style=\"color:#86efac;\">recommended · uses a field-log-tuned preset + pose-jump gate</span>
        <span id=\"pos_auto_status\" class=\"small\" style=\"color:#64748b;margin-left:auto;\"></span>
      </div>
      <div style=\"display:flex;gap:8px;align-items:center;margin-bottom:6px;\">
        <label class=\"small\" style=\"color:#94a3b8;display:flex;align-items:center;gap:4px;cursor:pointer;\">
          <input type=\"checkbox\" id=\"arena_show_3d\" checked style=\"accent-color:#0ea5e9;\" />
          3D view (Three.js)
        </label>
        <label class=\"small\" style=\"color:#94a3b8;display:flex;align-items:center;gap:4px;cursor:pointer;\">
          <input type=\"checkbox\" id=\"arena_show_all_drones\" checked style=\"accent-color:#10b981;\" />
          Show all drones
        </label>
        <span class=\"small\" style=\"color:#64748b;margin-left:12px;\">Grid: 1 m</span>
      </div>
      <canvas id=\"arena_canvas\" class=\"arena-canvas\" width=\"960\" height=\"560\" style=\"max-width:100%;\"></canvas>
      <div id=\"arena3d_wrap\" style=\"display:block;margin-top:8px;position:relative;\">
        <div id=\"arena3d_container\" style=\"width:960px;max-width:100%;height:520px;background:#0f172a;border:1px solid #334155;border-radius:6px;\"></div>
        <div class=\"small\" style=\"color:#64748b;margin-top:4px;\">Drag to orbit · scroll to zoom · right-drag to pan</div>
      </div>
      <div class=\"pos-coords\" style=\"margin-top:6px;\">
        <span class=\"pos-x\">X: <span id=\"pos_x\">—</span></span>&nbsp;&nbsp;
        <span class=\"pos-y\">Y: <span id=\"pos_y\">—</span></span>&nbsp;&nbsp;
        <span class=\"pos-z\">Z: <span id=\"pos_z\">—</span></span>
      </div>
      <div class=\"small\" style=\"color:#94a3b8;\">
        Hdg: <span id=\"pos_hdg\">—</span>°&nbsp;&nbsp;
        Vel: <span id=\"pos_vel\">—</span>&nbsp;&nbsp;
        Refs: <span id=\"pos_refs\">—</span>&nbsp;&nbsp;
        FPS: <span id=\"pos_fps\">—</span>
        <span id=\"pos_stale\" class=\"pos-stale\" style=\"display:none;\"> ⚠ STALE</span>
      </div>
      <div class=\"small\" style=\"color:#94a3b8;margin-top:2px;\">
        Vx:&nbsp;<span id=\"tel_vx\">—</span>&nbsp;
        Vy:&nbsp;<span id=\"tel_vy\">—</span>&nbsp;
        Vz:&nbsp;<span id=\"tel_vz\">—</span>&nbsp;cm/s&nbsp;&nbsp;&nbsp;
        Ax:&nbsp;<span id=\"tel_ax\">—</span>&nbsp;
        Ay:&nbsp;<span id=\"tel_ay\">—</span>&nbsp;
        Az:&nbsp;<span id=\"tel_az\">—</span>&nbsp;cm/s²
      </div>
      <!-- Safety distance readout — shows how far the active drone is from
           the nearest safety boundary. Green when inside the safe zone,
           amber when approaching, red when outside the margin. Updated on
           every SSE position event. -->
      <div class=\"small\" style=\"margin-top:2px;font-family:monospace;\">
        <span style=\"color:#94a3b8;\">Safety:</span>
        <span id=\"pos_safety_readout\" style=\"color:#64748b;\">—</span>
      </div>
      <div class=\"pos-cfg\">
        <div style=\"display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px;\">
          <label>Profile <span class=\"info-icon\" data-info=\"detect_profile\">i</span>:
            <select id=\"pos_profile\" class=\"pos-cfg\">
              <option value=\"balanced\">Balanced</option>
              <option value=\"sensitive\">Sensitive</option>
              <option value=\"strict\">Strict</option>
            </select>
          </label>
          <label>FOV° <span class=\"info-icon\" data-info=\"fov_deg\">i</span>:
            <input id=\"pos_fov\" type=\"number\" min=\"40\" max=\"120\" value=\"69\" style=\"width:60px;\" />
          </label>
          <label>Latency ms <span class=\"info-icon\" data-info=\"latency_ms\">i</span>:
            <input id=\"pos_latency\" type=\"range\" min=\"0\" max=\"800\" value=\"200\" style=\"width:90px;vertical-align:middle;\" />
            <span id=\"pos_latency_val\" style=\"font-size:11px;color:#94a3b8;\">200</span>
          </label>
          <label title=\"0 = pure ArUco (vision only), 1 = pure IMU dead-reckoning. Higher = more IMU smoothing, less vision jitter.\" style=\"display:inline-flex;align-items:center;gap:4px;\">
            <span style=\"font-size:11px;color:#94a3b8;\">ArUco</span>
            <input id=\"pos_imu_weight\" type=\"range\" min=\"0\" max=\"100\" value=\"30\" style=\"width:110px;vertical-align:middle;accent-color:#06b6d4;\" />
            <span style=\"font-size:11px;color:#94a3b8;\">IMU</span>
            <span id=\"pos_imu_weight_val\" style=\"font-size:11px;color:#06b6d4;font-weight:bold;min-width:32px;\">30%</span>
            <span class=\"info-icon\" data-info=\"imu_weight\">i</span>
          </label>
        </div>
        <div style=\"display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px;padding:6px;background:#0f172a;border:1px solid #1e293b;border-radius:4px;\">
          <span class=\"small\" style=\"color:#64748b;min-width:70px;\">Filters:</span>
          <label style=\"display:flex;align-items:center;gap:4px;cursor:pointer;\">
            <input type=\"checkbox\" id=\"pos_kalman\" style=\"accent-color:#3b82f6;\" />
            <span class=\"small\" style=\"color:#e2e8f0;\">Kalman filter <span class=\"info-icon\" data-info=\"enable_kalman_filter\">i</span></span>
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\">Marker size (m) <span class=\"info-icon\" data-info=\"marker_size_m\">i</span>:
            <input id=\"pos_marker_size\" type=\"number\" min=\"0.05\" max=\"2.0\" step=\"0.01\" value=\"0.5\" style=\"width:64px;\" />
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\">Top-K <span class=\"info-icon\" data-info=\"top_k_markers\">i</span>:
            <input id=\"pos_top_k\" type=\"number\" min=\"0\" max=\"10\" step=\"1\" value=\"0\" style=\"width:50px;\" title=\"0 = auto (4)\" />
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\">Outlier (m) <span class=\"info-icon\" data-info=\"outlier_reject_m\">i</span>:
            <input id=\"pos_outlier\" type=\"number\" min=\"0.1\" max=\"20\" step=\"0.1\" value=\"2.5\" style=\"width:60px;\" />
          </label>
          <label class=\"small\" style=\"color:#fbbf24;\" title=\"Multiplicative correction on camera↔marker distance from solvePnP. 1.0 = no correction. If the UI shows 7 m but actual is 9 m → set 9/7 ≈ 1.286.\">dist × <span class=\"info-icon\" data-info=\"distance_scale\">i</span>:
            <input id=\"pos_distance_scale\" type=\"number\" min=\"0.1\" max=\"5.0\" step=\"0.001\" value=\"1.000\" style=\"width:70px;\" />
          </label>
          <button id=\"pos_filters_apply\" class=\"pos-cfg\" style=\"height:26px;font-size:11px;padding:0 10px;\">Apply filters</button>
          <button id=\"pos_filters_reset\" class=\"pos-cfg\" style=\"height:26px;font-size:11px;padding:0 10px;background:#1e2a3a;\" title=\"Restore defaults: Kalman on, marker 0.5m, top-K auto, outlier 2.5m, dist scale 1.0\">Defaults</button>
          <span id=\"pos_filters_status\" class=\"small\" style=\"color:#64748b;\"></span>
        </div>
        <!-- ── Precision tuning (advanced) ───────────────────────────
             These parameters control multi-marker fusion, IMU blend,
             the pose-coast window, and per-axis Kalman variances. They
             apply immediately to the running positioner and also to
             Aruco Seek / mission boundary guard since they read the
             same global arena-frame pose.                             -->
        <div style=\"display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px;padding:6px;background:#0f172a;border:1px solid #1e293b;border-radius:4px;\">
          <span class=\"small\" style=\"color:#a78bfa;min-width:70px;font-weight:600;\">Precision:</span>
          <label class=\"small\" style=\"color:#94a3b8;\">pose hold (s) <span class=\"info-icon\" data-info=\"pose_hold_sec\">i</span>:
            <input id=\"pos_pose_hold\" type=\"number\" min=\"0\" max=\"10\" step=\"0.1\" value=\"0.8\" style=\"width:60px;\" />
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\">min refs <span class=\"info-icon\" data-info=\"min_ref_count\">i</span>:
            <input id=\"pos_min_refs\" type=\"number\" min=\"1\" max=\"12\" step=\"1\" value=\"1\" style=\"width:50px;\" />
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\">min ref w <span class=\"info-icon\" data-info=\"min_ref_weight\">i</span>:
            <input id=\"pos_min_ref_w\" type=\"number\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0\" style=\"width:56px;\" />
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\">blend min <span class=\"info-icon\" data-info=\"meas_blend_min\">i</span>:
            <input id=\"pos_blend_min\" type=\"number\" min=\"0\" max=\"1\" step=\"0.05\" value=\"0.35\" style=\"width:56px;\" />
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\">blend max <span class=\"info-icon\" data-info=\"meas_blend_max\">i</span>:
            <input id=\"pos_blend_max\" type=\"number\" min=\"0\" max=\"1\" step=\"0.05\" value=\"0.85\" style=\"width:56px;\" />
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\">vel blend <span class=\"info-icon\" data-info=\"vel_blend\">i</span>:
            <input id=\"pos_vel_blend\" type=\"number\" min=\"0\" max=\"1\" step=\"0.05\" value=\"0.25\" style=\"width:56px;\" />
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\">max Δt (s) <span class=\"info-icon\" data-info=\"max_state_dt\">i</span>:
            <input id=\"pos_max_dt\" type=\"number\" min=\"0.05\" max=\"10\" step=\"0.05\" value=\"1.0\" style=\"width:60px;\" />
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\">Q proc <span class=\"info-icon\" data-info=\"kalman_process_var\">i</span>:
            <input id=\"pos_kf_q\" type=\"number\" min=\"1e-6\" max=\"10\" step=\"1e-5\" value=\"1e-3\" style=\"width:74px;\" />
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\">R meas <span class=\"info-icon\" data-info=\"kalman_meas_var\">i</span>:
            <input id=\"pos_kf_r\" type=\"number\" min=\"1e-6\" max=\"10\" step=\"1e-3\" value=\"0.1\" style=\"width:74px;\" />
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\" title=\"First-order low-pass cut-off frequency applied to IMU velocity (vgx/vgy/vgz) at telemetry ingestion. 0 disables. Lower = more smoothing.\">IMU LPF (Hz) <span class=\"info-icon\" data-info=\"imu_lowpass_hz\">i</span>:
            <input id=\"pos_imu_lpf\" type=\"range\" min=\"0\" max=\"30\" step=\"0.5\" value=\"5\" style=\"width:90px;vertical-align:middle;accent-color:#a78bfa;\" />
            <span id=\"pos_imu_lpf_val\" style=\"font-size:11px;color:#a78bfa;font-weight:bold;min-width:42px;display:inline-block;\">5.0 Hz</span>
          </label>
          <label class=\"small\" style=\"color:#fbbf24;\" title=\"Pose-jump gate. Reject a fresh fix if it disagrees with the Kalman-predicted state by more than this many metres. Kills catastrophic single-marker solvePnP mirror-pose glitches (the 10+ m jumps). 0 disables.\">gate (m) <span class=\"info-icon\" data-info=\"max_pose_jump_m\">i</span>:
            <input id=\"pos_max_jump\" type=\"number\" min=\"0\" max=\"20\" step=\"0.1\" value=\"0\" style=\"width:56px;\" title=\"0 = disabled, 3.0 is a good default\" />
          </label>
          <label class=\"small\" style=\"color:#f97316;\" title=\"SDC26 target-box marker side length in metres. Wall markers use the \\\"Marker size\\\" field above. Target markers (ID ≥ 30) solvePnP against THIS size instead, so target world positions don't get over-reported by 50/19 ≈ 2.6×.\">target size (m) <span class=\"info-icon\" data-info=\"target_marker_size_m\">i</span>:
            <input id=\"pos_target_size\" type=\"number\" min=\"0.02\" max=\"2.0\" step=\"0.01\" value=\"0.19\" style=\"width:60px;\" />
          </label>
          <label class=\"small\" style=\"color:#06b6d4;\" title=\"Zero-Velocity Update. When IMU speed stays below this threshold for &quot;hold&quot; frames, the positioner snaps to the last valid pose instead of integrating noisy ArUco fixes — eliminates the parked-drone drift. 0 disables.\">ZUPT &lt; (m/s) <span class=\"info-icon\" data-info=\"zupt_speed_m_s\">i</span>:
            <input id=\"pos_zupt_speed\" type=\"number\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0.05\" style=\"width:56px;\" />
          </label>
          <label class=\"small\" style=\"color:#06b6d4;\" title=\"Consecutive slow frames needed before ZUPT engages. Higher = more robust (fewer false engagements during slow moves), lower = quicker to freeze when parked. 3 frames ≈ 0.6s at 5Hz is a reasonable default.\">hold <span class=\"info-icon\" data-info=\"zupt_hold_frames\">i</span>:
            <input id=\"pos_zupt_hold\" type=\"number\" min=\"0\" max=\"30\" step=\"1\" value=\"3\" style=\"width:50px;\" />
          </label>
          <button id=\"pos_precision_apply\" class=\"pos-cfg\" style=\"height:26px;font-size:11px;padding:0 10px;\">Apply precision</button>
          <button id=\"pos_precision_reset\" class=\"pos-cfg\" style=\"height:26px;font-size:11px;padding:0 10px;background:#1e2a3a;\" title=\"Restore precision defaults\">Defaults</button>
          <span id=\"pos_precision_status\" class=\"small\" style=\"color:#64748b;\"></span>
        </div>
        <!-- ── Position-tracker presets ─────────────────────────────
             Stored server-side in controller/position_presets.json.
             Loading a preset fans out to every drone via
             /proxy/position/config so the whole fleet gets the new
             tuning in one click. -->
        <div style=\"display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px;padding:6px;background:#0f172a;border:1px solid #1e293b;border-radius:4px;\">
          <span class=\"small\" style=\"color:#4ade80;min-width:70px;font-weight:600;\">Presets:</span>
          <select id=\"pos_preset_sel\" style=\"min-width:140px;height:22px;font-size:11px;\"></select>
          <button id=\"pos_preset_apply\" style=\"height:22px;font-size:11px;padding:0 8px;background:#065f46;border-color:#10b981;color:#d1fae5;\" title=\"Apply the selected preset to every drone in the fleet\">Apply to fleet</button>
          <input id=\"pos_preset_name\" type=\"text\" placeholder=\"preset name\" style=\"width:120px;height:22px;font-size:11px;\" />
          <button id=\"pos_preset_save\" style=\"height:22px;font-size:11px;padding:0 8px;background:#1e293b;border-color:#475569;\" title=\"Save the current values as a named preset (overwrites if name exists)\">Save</button>
          <button id=\"pos_preset_delete\" style=\"height:22px;font-size:11px;padding:0 8px;background:#450a0a;border-color:#b91c1c;color:#fecaca;\" title=\"Delete the selected preset\">Delete</button>
          <span id=\"pos_preset_status\" class=\"small\" style=\"color:#64748b;\"></span>
        </div>
        <div style=\"display:flex;gap:8px;flex-wrap:wrap;align-items:center;\">
          <button id=\"pos_cfg_save\" class=\"pos-cfg\">Apply Config</button>
          <label style=\"cursor:pointer;\">
            <button id=\"pos_calib_btn\" class=\"pos-cfg\" onclick=\"document.getElementById('pos_calib_file').click()\">Upload Calibration (.npz)</button>
            <input type=\"file\" id=\"pos_calib_file\" accept=\".npz\" style=\"display:none;\" />
          </label>
          <span id=\"pos_calib_status\" class=\"small\" style=\"color:#94a3b8;\"></span>
        </div>
        <!-- ── Claude Automatic Calibration ────────────────────────────
             Autonomous arena-scanning flight that generates a log + video
             the operator can upload to Claude for analysis. Claude returns
             a tuned preset JSON which the operator pastes into "Import"
             below; it's saved as a named preset and can be applied to the
             fleet from the Presets dropdown above. -->
        <div style=\"margin-top:8px;padding:8px 10px;background:#0c1a2e;border:1px solid #334155;border-radius:6px;\">
          <div style=\"display:flex;align-items:center;gap:10px;margin-bottom:6px;\">
            <b style=\"color:#fbbf24;\">🎯 Claude Automatic Calibration</b>
            <span class=\"small\" style=\"color:#94a3b8;\">
              Place drone in arena centre → Start → upload log+video → paste returned preset
            </span>
          </div>
          <div style=\"display:flex;gap:8px;align-items:center;flex-wrap:wrap;\">
            <button id=\"calib_start_btn\"
                    style=\"background:#713f12;border:1px solid #fbbf24;color:#fef3c7;padding:4px 12px;border-radius:4px;cursor:pointer;font-weight:600;\"
                    title=\"Drone takes off, flies a ~90 s scan pattern (two 360° sweeps + translation cross + altitude change), then lands. Flight log + video are recorded automatically.\">
              ▶ Start Calibration Flight
            </button>
            <button id=\"calib_abort_btn\"
                    style=\"display:none;background:#450a0a;border:1px solid #b91c1c;color:#fecaca;padding:4px 12px;border-radius:4px;cursor:pointer;font-weight:600;\"
                    title=\"Abort the running calibration — the drone will land at the next safe point.\">
              ✕ Abort
            </button>
            <span id=\"calib_status_txt\" class=\"small\" style=\"color:#64748b;\">idle</span>
          </div>
          <div id=\"calib_progress_wrap\" style=\"display:none;margin-top:6px;\">
            <div style=\"height:8px;background:#1e293b;border-radius:4px;overflow:hidden;\">
              <div id=\"calib_progress_bar\" style=\"height:100%;width:0%;background:linear-gradient(90deg,#fbbf24,#f59e0b);transition:width 0.3s;\"></div>
            </div>
            <div class=\"small\" style=\"color:#94a3b8;margin-top:4px;\">
              Step <span id=\"calib_step_num\">0</span>/<span id=\"calib_step_total\">0</span>:
              <span id=\"calib_step_name\" style=\"color:#fbbf24;\">—</span>
              <span style=\"color:#64748b;margin-left:8px;\">elapsed <span id=\"calib_elapsed\">0</span>s</span>
            </div>
          </div>
          <div id=\"calib_download_hint\" style=\"display:none;margin-top:8px;padding:6px 8px;background:#052e16;border:1px solid #10b981;border-radius:4px;font-size:12px;color:#d1fae5;\">
            ✓ Calibration flight complete. Download the matching log + video from the
            <b>Flight Logs</b> panel below (newest entry), then upload both files to Claude.
            Claude will return a preset JSON that you can paste into the <b>Import Preset</b>
            box below to apply it to the fleet.
          </div>
          <!-- Import preset: paste a JSON blob (e.g. Claude's tuned output)
               and save it as a named preset. -->
          <div style=\"margin-top:8px;display:flex;gap:6px;align-items:flex-start;flex-wrap:wrap;\">
            <input id=\"calib_preset_name\" type=\"text\" placeholder=\"preset name (e.g. Claude calibrated 2026-04-24)\"
                   style=\"flex:1;min-width:260px;height:26px;font-size:12px;\" />
            <button id=\"calib_import_btn\"
                    style=\"background:#1e293b;border:1px solid #475569;color:#e2e8f0;padding:3px 10px;border-radius:4px;cursor:pointer;font-size:12px;\"
                    title=\"Paste a preset JSON below and save it under the name above.\">
              Import from JSON ↓
            </button>
          </div>
          <textarea id=\"calib_preset_json\" rows=\"3\" placeholder='{\"detect_profile\":\"balanced\",\"distance_scale\":1.0,...}'
                    style=\"display:none;width:100%;margin-top:6px;font-family:monospace;font-size:11px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:6px;\"></textarea>
          <div id=\"calib_preset_status\" class=\"small\" style=\"color:#64748b;margin-top:4px;\"></div>
        </div>
        <div style=\"margin-top:6px;display:flex;gap:8px;align-items:center;\">
          <button id=\"pos_video_toggle\" class=\"pos-cfg\">Show ArUco Video</button>
          <button id=\"rec_btn\" class=\"pos-cfg\" style=\"background:#1e3a2e;border-color:#22c55e;color:#22c55e;\">&#9679; Record</button>
          <label style=\"display:flex;align-items:center;gap:4px;font-size:12px;color:#94a3b8;cursor:pointer;\">
            <input type=\"checkbox\" id=\"rec_raw\" style=\"accent-color:#22c55e;\" /> Raw
          </label>
          <span id=\"rec_status\" class=\"small\" style=\"color:#64748b;\"></span>
          <a id=\"pos_tracker_link\" href=\"#\" target=\"_blank\" class=\"small\" style=\"color:#38bdf8;display:none;\">Open full arena tracker ↗</a>
        </div>
        <div id=\"pos_video_container\" style=\"display:none;margin-top:6px;\">
          <img id=\"pos_video_img\" src=\"\" style=\"max-width:100%;border-radius:6px;background:#000;\" />
        </div>
      </div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════════════════
       Target Boxes — identified SDC26 target markers (IDs ≥ 30) with
       team/colour assignment and arena-frame position. Driven by the
       FC's /proxy/ui_state → _pos_st.targets stream. Team & colour
       are resolved from arena_config.target_teams (ID ranges) +
       target_overrides (per-ID override table).
       ══════════════════════════════════════════════════════════════ -->
  <div class=\"row\" style=\"margin-top:12px;\">
    <div class=\"panel\" style=\"min-width:340px;flex:1;\">
      <div style=\"display:flex;align-items:center;gap:10px;margin-bottom:8px;\">
        <b>🎯 Target Boxes</b>
        <span class=\"small\" style=\"color:#94a3b8;\">detected SDC target markers (ID ≥ 30, 19 cm stickers)</span>
        <span id=\"targets_badge\" class=\"small\" style=\"margin-left:auto;color:#64748b;\">0 visible</span>
      </div>
      <div id=\"targets_empty\" class=\"small\" style=\"color:#64748b;font-style:italic;\">
        No target boxes detected yet. Fly the drone so its camera points at a box.
      </div>
      <table id=\"targets_table\" style=\"display:none;width:100%;border-collapse:collapse;font-size:13px;\">
        <thead>
          <tr style=\"text-align:left;color:#94a3b8;border-bottom:1px solid #334155;\">
            <th style=\"padding:4px 6px;\">ID</th>
            <th style=\"padding:4px 6px;\">Team</th>
            <th style=\"padding:4px 6px;\">X (m)</th>
            <th style=\"padding:4px 6px;\">Y (m)</th>
            <th style=\"padding:4px 6px;\">Z (m)</th>
            <th style=\"padding:4px 6px;\">Age</th>
            <th style=\"padding:4px 6px;\">Status</th>
          </tr>
        </thead>
        <tbody id=\"targets_tbody\"></tbody>
      </table>
      <div class=\"small\" style=\"color:#64748b;margin-top:6px;\">
        Team/colour assignments come from <b>Arena Configuration → Target Teams</b>
        (ID ranges) with optional per-ID overrides. SDC26 convention:
        IDs 31-36 = Blue (box 1-6), IDs 41-46 = Red (box 1-6).
      </div>

      <!-- ── Scan & Capture Targets mission ─────────────────────────
           Autonomous: drone takes off (if grounded), rotates 360° to
           discover the boxes via ArUco, then hands off to the standard
           CaptureAllTargetsMission which flies over each box for the
           configured hover time. -->
      <div style=\"margin-top:10px;padding:8px 10px;background:#0c1a2e;border:1px solid #334155;border-radius:6px;\">
        <div style=\"display:flex;align-items:center;gap:10px;margin-bottom:6px;\">
          <b style=\"color:#fbbf24;\">🎯 Scan &amp; Capture Targets</b>
          <span class=\"small\" style=\"color:#94a3b8;\">
            rotate to discover, then fly exactly over each box for N seconds
          </span>
        </div>
        <div style=\"display:flex;gap:10px;align-items:center;flex-wrap:wrap;\">
          <button id=\"scan_cap_start_btn\"
                  style=\"background:#713f12;border:1px solid #fbbf24;color:#fef3c7;padding:4px 12px;border-radius:4px;cursor:pointer;font-weight:600;\"
                  title=\"Drone takes off (if needed), rotates 6×60° scanning for target markers, then visits each box in nearest-neighbour order.\">
            ▶ Start Scan &amp; Capture
          </button>
          <button id=\"scan_cap_abort_btn\"
                  style=\"display:none;background:#450a0a;border:1px solid #b91c1c;color:#fecaca;padding:4px 12px;border-radius:4px;cursor:pointer;font-weight:600;\"
                  title=\"Abort the scan/capture — drone continues its current moveBy command, then stays airborne.\">
            ✕ Abort
          </button>
          <label class=\"small\" style=\"color:#94a3b8;\">
            hover <input id=\"scan_cap_hover\" type=\"number\" min=\"1\" max=\"20\" step=\"0.5\" value=\"3\" style=\"width:50px;height:22px;font-size:11px;\" /> s
          </label>
          <label class=\"small\" style=\"color:#94a3b8;\">
            above <input id=\"scan_cap_above\" type=\"number\" min=\"0.5\" max=\"5\" step=\"0.1\" value=\"1.5\" style=\"width:50px;height:22px;font-size:11px;\" /> m
          </label>
          <span id=\"scan_cap_status_txt\" class=\"small\" style=\"color:#64748b;\">idle</span>
        </div>
        <div id=\"scan_cap_progress\" style=\"display:none;margin-top:6px;\">
          <div style=\"height:8px;background:#1e293b;border-radius:4px;overflow:hidden;\">
            <div id=\"scan_cap_progress_bar\" style=\"height:100%;width:0%;background:linear-gradient(90deg,#fbbf24,#f59e0b);transition:width 0.3s;\"></div>
          </div>
          <div class=\"small\" style=\"color:#94a3b8;margin-top:4px;\">
            <span id=\"scan_cap_phase\" style=\"color:#fbbf24;\">—</span>:
            <span id=\"scan_cap_step\">—</span>
            <span style=\"color:#64748b;margin-left:8px;\">
              <span id=\"scan_cap_ndet\">0</span> target(s) found
              &nbsp;·&nbsp; elapsed <span id=\"scan_cap_elapsed\">0.0</span>s
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class=\"row\" style=\"margin-top:12px;\">
    <div class=\"panel\" style=\"min-width:340px;flex:1;\">
      <div style=\"display:flex;align-items:center;gap:10px;margin-bottom:8px;\">
        <b>Arena Configuration</b>
        <span class=\"small\" style=\"color:#94a3b8;\">marker layout &amp; physical dimensions</span>
        <button id=\"arena_cfg_toggle\" style=\"margin-left:auto;padding:2px 10px;font-size:11px;background:#1e293b;\">Show</button>
      </div>
      <div id=\"arena_cfg_body\" style=\"display:none;\">
        <div style=\"display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px;\">
          <label class=\"small\">Arena width (m):
            <input id=\"ac_width\" type=\"number\" step=\"0.5\" value=\"20\" style=\"width:70px;height:30px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;padding:0 6px;\" />
          </label>
          <label class=\"small\">Arena depth (m):
            <input id=\"ac_depth\" type=\"number\" step=\"0.5\" value=\"10\" style=\"width:70px;height:30px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;padding:0 6px;\" />
          </label>
          <label class=\"small\">Height min (m):
            <input id=\"ac_hmin\" type=\"number\" step=\"0.1\" value=\"-1\" style=\"width:60px;height:30px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;padding:0 6px;\" />
          </label>
          <label class=\"small\">Height max (m):
            <input id=\"ac_hmax\" type=\"number\" step=\"0.1\" value=\"1\" style=\"width:60px;height:30px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;padding:0 6px;\" />
          </label>
          <label class=\"small\">Marker size (m):
            <input id=\"ac_msize\" type=\"number\" step=\"0.05\" value=\"0.5\" style=\"width:65px;height:30px;border-radius:5px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;padding:0 6px;\" />
          </label>
        </div>
        <div class=\"small\" style=\"margin-bottom:4px;color:#94a3b8;\">Marker positions (ID · X · Y · Z · Wall)</div>
        <div id=\"arena_marker_table\" style=\"max-height:280px;overflow-y:auto;font-size:11px;\"></div>
        <div style=\"margin-top:6px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;\">
          <label class=\"small\" style=\"color:#94a3b8;display:flex;align-items:center;gap:4px;\">ID:
            <input id=\"arena_new_marker_id\" type=\"number\" min=\"0\" step=\"1\" placeholder=\"auto\" style=\"width:60px;height:26px;border-radius:4px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;padding:0 6px;\" title=\"Marker ID to add (leave blank for next free)\" />
          </label>
          <button id=\"arena_add_marker\" style=\"padding:4px 10px;font-size:11px;background:#065f46;border-color:#10b981;\">+ Add Marker</button>
          <button id=\"arena_save\" style=\"padding:4px 10px;font-size:11px;background:#1e3a5f;border-color:#3b82f6;\">Save Config</button>
          <button id=\"arena_reset\" style=\"padding:4px 10px;font-size:11px;background:#374151;border-color:#6b7280;\">Reset to Defaults</button>
          <span id=\"arena_cfg_status\" class=\"small\" style=\"color:#94a3b8;\"></span>
        </div>
      </div>
    </div>
  </div>
<script>
// --- Drone fleet selector ---
let drones = {};
let activeDroneId = null;

async function loadDrones() {
  try {
    const r = await fetch('/proxy/drones');
    const d = await r.json();
    drones = d.drones;
    activeDroneId = d.active;
    renderDroneBar();
    updatePiLabel();
  } catch {}
}

function renderDroneBar() {
  const bar = document.getElementById('drone_bar');
  bar.innerHTML = '';
  let slot = 1;
  for (const [id, info] of Object.entries(drones)) {
    const btn = document.createElement('button');
    btn.className = 'drone-btn' + (id === activeDroneId ? ' selected' : '');
    const slotBadge = (slot <= 5)
      ? `<span style="background:#1e3a5f;color:#93c5fd;padding:0 4px;margin-right:4px;border-radius:3px;font-size:10px;font-weight:700;">${slot}</span>`
      : '';
    btn.innerHTML = `${slotBadge}${info.name}<span class="drone-type">${info.type}</span>`;
    btn.title = (slot <= 5) ? `Hotkey: ${slot}` : '';
    btn.onclick = () => switchDrone(id);
    bar.appendChild(btn);
    slot += 1;
  }
  // Show/hide Anafi panel based on drone type
  const anafiPanel = document.getElementById('anafi_panel');
  if (anafiPanel) {
    const droneType = drones[activeDroneId]?.type || '';
    anafiPanel.style.display = droneType === 'anafi' ? '' : 'none';
  }
}

async function switchDrone(id) {
  if (id === activeDroneId) return;
  // Release all keys on the current drone before switching
  releaseAllKeys();
  try {
    await fetch('/proxy/switch', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id:id})});
    activeDroneId = id;
    renderDroneBar();
    updatePiLabel();
    // Restart SSE streams for new drone
    startTelemetrySSE();
    if (document.getElementById('pos_enabled').checked) startPosEvents();
    refreshTelemetry();
    // ── Video feed must also switch to the new drone ───────────────
    // /proxy/video, /proxy/position/video and /proxy/aruco/video.mjpg
    // all proxy to the ACTIVE drone on the server. The browser's
    // existing <img> MJPEG connections are pinned to the OLD drone
    // (long-lived HTTP, established when src was set), so we have to
    // tear them down and re-establish. Be aggressive: always do this,
    // regardless of videoActive/src-empty flags.
    try {
      // 1) Guarantee Way-1 MJPEG is running on the new drone.
      //    Fire-and-forget; if it's already running, the server returns ok.
      fetch('/proxy/video/start', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({mode:'mjpeg'})}).catch(()=>{});

      // 2) Force every <img> to drop its current MJPEG connection.
      //    'about:blank' is more aggressive than an empty string; it
      //    actually tears down the existing HTTP socket.
      const elements = [
        { el: document.getElementById('video_img'),     src: '/proxy/video?' },
        { el: document.getElementById('pos_video_img'), src: '/proxy/position/video?' },
        { el: document.getElementById('arc_video'),     src: '/proxy/aruco/video.mjpg?t=' },
      ];
      elements.forEach(({el}) => { if (el) el.src = 'about:blank'; });

      // 3) After the browser has had time to close the old connections
      //    (browsers process src changes async — 300 ms is a safe
      //    upper bound), reconnect every feed with a fresh cache-buster.
      //    Main feed is always reconnected (even if user hadn't pressed
      //    Start Video on the old drone, the new drone should also
      //    show its feed). Position and ArUco feeds only reconnect
      //    if they had been displayed on the previous drone.
      const posVisible = document.getElementById('pos_video_img') &&
        document.getElementById('pos_video_img').parentElement &&
        document.getElementById('pos_video_img').parentElement.style.display !== 'none';
      const arcVisible = document.getElementById('arc_video') &&
        document.getElementById('arc_video').getAttribute('data-active') === '1';
      setTimeout(() => {
        const ts = Date.now();
        const mainImg = document.getElementById('video_img');
        if (mainImg) { mainImg.src = '/proxy/video?' + ts; videoActive = true; }
        if (posVisible) {
          document.getElementById('pos_video_img').src = '/proxy/position/video?' + ts;
        }
        if (arcVisible) {
          document.getElementById('arc_video').src = '/proxy/aruco/video.mjpg?t=' + ts;
        }
        console.log('[drone-switch] video reconnected → drone', id,
                    ' main=yes pos=', posVisible, ' arc=', arcVisible);
      }, 300);
    } catch (err) {
      console.warn('[drone-switch] video reconnect failed:', err);
    }
    // Refresh the environment + Wi-Fi status for the newly selected drone.
    // Both readouts are per-drone so they need to re-fetch after the switch.
    try {
      if (typeof envRefresh  === 'function') setTimeout(envRefresh,  400);
      if (typeof wifiRefresh === 'function') setTimeout(wifiRefresh, 400);
    } catch (e) {}
  } catch {}
}

function updatePiLabel() {
  const info = drones[activeDroneId];
  document.getElementById('pi').textContent = info ? `${info.name} (${info.type}) @ ${info.base}` : '-';
}

async function post(url, body){
  try {
    await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  } catch {}
}
function keyDown(k){ post('/proxy/key_down',{key:k}); }
function keyUp(k){ post('/proxy/key_up',{key:k}); }

const activeKeys = new Set();
function pressKey(k){
  if (!activeKeys.has(k)) {
    activeKeys.add(k);
    keyDown(k);
  }
}
function releaseKey(k){
  if (activeKeys.has(k)) {
    activeKeys.delete(k);
    keyUp(k);
  }
}
function releaseAllKeys(){
  Array.from(activeKeys).forEach(releaseKey);
}

// Batch all active key states into a single POST (instead of one POST per key)
setInterval(()=>{
  if (activeKeys.size === 0) return;
  const keys = Array.from(activeKeys);
  // Send as single batch request
  fetch('/proxy/key_batch', {method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({keys})}).catch(()=>{});
}, 100);

const holdButtons = document.querySelectorAll('button[data-k]');
holdButtons.forEach(btn=>{
  const k = btn.dataset.k;
  btn.addEventListener('pointerdown', e=>{ e.preventDefault(); btn.classList.add('active'); pressKey(k); });
  btn.addEventListener('pointerup',   e=>{ e.preventDefault(); btn.classList.remove('active'); releaseKey(k); });
  btn.addEventListener('pointerleave',e=>{ btn.classList.remove('active'); releaseKey(k); });
  btn.addEventListener('pointercancel',e=>{ e.preventDefault(); btn.classList.remove('active'); releaseKey(k); });
});

// Takeoff + surface any server-side refusal (magneto / sensors / battery / …).
// The proxy now returns a structured JSON shape:
//   { ok:false, error:"<short title>", reason_code:"<slug>", hint:"<actionable text>",
//     diagnostic:{ magneto_status, magneto_required, battery, flying, ... } }
// The short_title already explains WHAT failed; hint explains WHAT TO DO.
async function tryTakeoff(){
  hideTakeoffError();
  let resp, body;
  try {
    resp = await fetch('/proxy/takeoff', {method:'POST',
                       headers:{'Content-Type':'application/json'}, body:'{}'});
    body = await resp.json().catch(()=>({ok:false, error:'non-JSON response'}));
  } catch (e) {
    showTakeoffError('Network error contacting drone: ' + e, '', '', {});
    return;
  }
  if (!resp.ok || body.ok === false) {
    // Structured proxy shape (new)
    const title = (body && body.error) || ('HTTP ' + resp.status);
    let hint = (body && body.hint) || '';
    const code = (body && body.reason_code) || '';
    const diag = (body && body.diagnostic) || {};
    // Legacy fallback — if the proxy didn't format and we only have a
    // raw string, do keyword-based hinting as before.
    if (!hint) {
      if (/magneto/i.test(title))
        hint = 'Magnetometer needs figure-8 calibration before the drone will arm.';
      else if (/sensor/i.test(title))
        hint = 'A sensor check failed — verify battery, motor state.';
      else if (/battery|low/i.test(title))
        hint = 'Battery too low for takeoff.';
      else if (/motor/i.test(title))
        hint = 'Motor fault — inspect props and power-cycle the drone.';
      else if (/not.?ready|not.?connected/i.test(title))
        hint = 'Controller is not connected. Check Wi-Fi link and API server logs.';
      else if (/angle|tilt|level/i.test(title))
        hint = 'Drone is not level — place on a flat surface and retry.';
    }
    showTakeoffError(title, hint, code, diag);
  }
}

function showTakeoffError(title, hint, code, diagnostic){
  const box = document.getElementById('takeoff_err');
  document.getElementById('takeoff_err_reason').textContent = title;
  document.getElementById('takeoff_err_hint').textContent = hint || '';
  // Show magneto-wizard shortcut when the failure is magnetometer-related
  const isMag = (code === 'magnetometer_calibration_required') ||
                /magneto/i.test(title) ||
                (diagnostic && (diagnostic.magneto_required === true ||
                 (typeof diagnostic.magneto_status === 'string' &&
                  /required/i.test(diagnostic.magneto_status))));
  document.getElementById('takeoff_err_mag').style.display = isMag ? '' : 'none';
  // Render any diagnostic state inline so the operator sees WHY
  const dl = document.getElementById('takeoff_err_diag');
  if (dl && diagnostic && Object.keys(diagnostic).length) {
    const rows = [];
    if (diagnostic.magneto_status)
      rows.push(['magnetometer', String(diagnostic.magneto_status)]);
    if (diagnostic.magneto_required !== undefined)
      rows.push(['magneto calibration', diagnostic.magneto_required ? 'REQUIRED' : 'ok']);
    if (diagnostic.battery !== undefined)
      rows.push(['battery', String(diagnostic.battery) + '%']);
    if (diagnostic.alert) rows.push(['alert', String(diagnostic.alert)]);
    if (diagnostic.motor) rows.push(['motor', String(diagnostic.motor)]);
    if (diagnostic.sensors) rows.push(['sensors', String(diagnostic.sensors)]);
    dl.innerHTML = rows.length
      ? rows.map(([k,v]) =>
          '<div style="display:flex;gap:10px;"><span style="color:#94a3b8;min-width:130px;">' +
          k + '</span><span style="font-family:monospace;">' + v + '</span></div>'
        ).join('')
      : '';
    dl.style.display = rows.length ? '' : 'none';
  } else if (dl) {
    dl.innerHTML = '';
    dl.style.display = 'none';
  }
  box.classList.add('show');
}
function hideTakeoffError(){
  document.getElementById('takeoff_err').classList.remove('show');
}

document.getElementById('takeoff').onclick = tryTakeoff;
document.getElementById('takeoff_err_dismiss').onclick = hideTakeoffError;
document.getElementById('takeoff_err_mag').onclick = ()=>{ hideTakeoffError(); openMagnetoWizard(); };
document.getElementById('land').onclick = ()=>post('/proxy/land',{});
document.getElementById('recover').onclick = ()=>post('/proxy/recover',{});

document.getElementById('safe_takeoff').onclick = async ()=>{
  try {
    const r = await fetch('/proxy/safety/takeoff');
    const s = await r.json();
    await post('/proxy/safety/takeoff', {enabled: !Boolean(s.enabled)});
    refreshSafeTakeoff();
  } catch {}
};

document.getElementById('toggle_log').onclick = async ()=>{
  try {
    const s = await fetch('/proxy/logging/telemetry');
    const cur = await s.json();
    const nextEnabled = !Boolean(cur.enabled);
    await post('/proxy/logging/telemetry',{enabled: nextEnabled});
    document.getElementById('toggle_log').textContent = nextEnabled ? 'Disable Telemetry Log' : 'Enable Telemetry Log';
  } catch {}
};

document.getElementById('download_log').onclick = ()=>{
  window.open('/proxy/logging/telemetry/download', '_blank');
};

document.getElementById('clear_log').onclick = async ()=>{
  try {
    await post('/proxy/logging/telemetry/clear', {});
  } catch {}
};

// (emergency killswitch button removed — was too easy to hit by accident)
document.getElementById('rotate_cw').onclick = ()=>post('/proxy/rotate',{dir:'cw',deg:45});
document.getElementById('rotate_ccw').onclick = ()=>post('/proxy/rotate',{dir:'ccw',deg:45});
document.getElementById('move_up').onclick = ()=>post('/proxy/move',{dir:'up',cm:30});
document.getElementById('move_down').onclick = ()=>post('/proxy/move',{dir:'down',cm:30});
document.getElementById('move_fwd').onclick = ()=>post('/proxy/move',{dir:'forward',cm:30});
document.getElementById('move_back').onclick = ()=>post('/proxy/move',{dir:'back',cm:30});
document.getElementById('move_left').onclick = ()=>post('/proxy/move',{dir:'left',cm:30});
document.getElementById('move_right').onclick = ()=>post('/proxy/move',{dir:'right',cm:30});
document.getElementById('stream_on').onclick = ()=>post('/proxy/stream',{action:'on'});
document.getElementById('stream_off').onclick = ()=>post('/proxy/stream',{action:'off'});
document.getElementById('set_speed').onclick = ()=>{
  const v = Number(document.getElementById('speed_val').value || 30);
  post('/proxy/speed',{speed:v});
};
document.getElementById('sdk_send').onclick = ()=>{
  const cmd = document.getElementById('sdk_cmd').value || '';
  if (cmd.trim()) post('/proxy/sdk',{command:cmd.trim()});
};

// Anafi / Olympe controls
const gimbalSlider = document.getElementById('gimbal_tilt');
const gimbalVal = document.getElementById('gimbal_tilt_val');
gimbalSlider.oninput = ()=>{ gimbalVal.textContent = gimbalSlider.value + '°'; };
document.getElementById('gimbal_set').onclick = ()=>post('/proxy/gimbal',{tilt:Number(gimbalSlider.value),pan:0});
document.getElementById('gimbal_down').onclick = ()=>{ gimbalSlider.value=-90; gimbalVal.textContent='-90°'; post('/proxy/gimbal',{tilt:-90,pan:0}); };
document.getElementById('gimbal_fwd').onclick = ()=>{ gimbalSlider.value=0; gimbalVal.textContent='0°'; post('/proxy/gimbal',{tilt:0,pan:0}); };

document.getElementById('apply_settings').onclick = ()=>{
  const alt = Number(document.getElementById('set_alt').value);
  const vs = Number(document.getElementById('set_vspd').value);
  const tilt = Number(document.getElementById('set_tilt').value);
  post('/proxy/settings', {max_altitude_m:alt, max_vertical_speed:vs, max_tilt:tilt});
};

// ── Environment (indoor / outdoor) ──────────────────────────────────
async function envRefresh() {
  const lbl = document.getElementById('env_status');
  if (!lbl) return;
  try {
    const r = await fetch('/proxy/environment', {cache:'no-store'});
    const d = await r.json();
    if (d.ok && d.status) {
      const cur = (d.status.environement || d.status.environment || '').toString();
      lbl.textContent = 'current: ' + (cur || 'unknown');
      lbl.style.color = cur.toLowerCase().includes('indoor') ? '#22c55e' : '#94a3b8';
      // Sync dropdown
      const sel = document.getElementById('env_mode');
      if (sel && cur) {
        const v = cur.toLowerCase().includes('indoor') ? 'indoor' : 'outdoor';
        if (sel.value !== v) sel.value = v;
      }
    } else {
      lbl.textContent = d.error || 'unavailable';
      lbl.style.color = '#f59e0b';
    }
  } catch (e) { lbl.textContent = 'error'; lbl.style.color = '#ef4444'; }
}
(function(){
  const btn = document.getElementById('env_apply');
  if (!btn) return;
  btn.onclick = async () => {
    const mode = document.getElementById('env_mode').value;
    const lbl = document.getElementById('env_status');
    lbl.textContent = '… applying';
    lbl.style.color = '#fbbf24';
    try {
      const r = await fetch('/proxy/environment', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({mode})});
      const d = await r.json();
      if (d.ok) {
        lbl.textContent = '✓ set to ' + mode;
        lbl.style.color = '#22c55e';
        setTimeout(envRefresh, 800);
      } else {
        lbl.textContent = '✗ ' + (d.error || 'unknown');
        lbl.style.color = '#ef4444';
      }
    } catch(e) { lbl.textContent = '✗ '+e; lbl.style.color = '#ef4444'; }
  };
})();
// Initial read + refresh when the user switches drones
setTimeout(envRefresh, 800);

// ── Wi-Fi band + channel ────────────────────────────────────────────
async function wifiRefresh() {
  const lbl = document.getElementById('wifi_status');
  if (!lbl) return;
  try {
    const r = await fetch('/proxy/wifi/status', {cache:'no-store'});
    const d = await r.json();
    if (d.ok && d.status && typeof d.status === 'object') {
      const band = (d.status.band || '').toString();
      const ch   = d.status.channel;
      const typ  = d.status.type || '';
      lbl.textContent = `band=${band}  ch=${ch}  (${typ})`;
      const is5 = band.toLowerCase().includes('5');
      lbl.style.color = is5 ? '#22c55e' : '#fbbf24';
      // Sync dropdowns
      const bandSel = document.getElementById('wifi_band');
      if (bandSel) bandSel.value = is5 ? '5_GHz' : '2_4_GHz';
      const chSel = document.getElementById('wifi_channel');
      if (chSel && ch != null) {
        // Prefer auto when type=auto_*
        if (typ.toLowerCase().includes('auto')) {
          chSel.value = 'auto';
        } else if (Array.from(chSel.options).some(o => o.value == String(ch))) {
          chSel.value = String(ch);
        }
      }
    } else {
      lbl.textContent = d.error || d.status || 'unavailable';
      lbl.style.color = '#f59e0b';
    }
  } catch(e) { lbl.textContent = 'error'; lbl.style.color = '#ef4444'; }
}
(function(){
  const apply = document.getElementById('wifi_apply');
  if (!apply) return;
  apply.onclick = async () => {
    const band = document.getElementById('wifi_band').value;
    const chStr = document.getElementById('wifi_channel').value;
    const lbl = document.getElementById('wifi_status');
    const auto = (chStr === 'auto');
    const body = auto
      ? {auto: true, band}
      : {auto: false, band, channel: Number(chStr)};
    if (!confirm('⚠ Wi-Fi change will disconnect the drone for ~5-10 s.\\n\\n' +
                 'Drone MUST be on the ground. Continue?')) return;
    lbl.textContent = '… applying  (wait ~10 s for reconnect)';
    lbl.style.color = '#fbbf24';
    try {
      const r = await fetch('/proxy/wifi/channel', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(body)});
      const d = await r.json();
      if (d.ok) {
        lbl.textContent = `✓ ${d.mode}: ${band}${auto ? '' : ' ch ' + body.channel} — reconnecting…`;
        lbl.style.color = '#22c55e';
        // Poll status a few times while the watchdog reconnects
        let n = 0;
        const poll = setInterval(() => {
          wifiRefresh();
          if (++n > 8) clearInterval(poll);
        }, 2000);
      } else {
        lbl.textContent = '✗ ' + (d.error || 'unknown');
        lbl.style.color = '#ef4444';
      }
    } catch(e) { lbl.textContent = '✗ '+e; lbl.style.color = '#ef4444'; }
  };
})();
(function(){
  const scan = document.getElementById('wifi_scan');
  if (!scan) return;
  scan.onclick = async () => {
    const band = document.getElementById('wifi_band').value;
    const lbl = document.getElementById('wifi_status');
    lbl.textContent = '… scanning ' + band;
    lbl.style.color = '#fbbf24';
    try {
      const r = await fetch('/proxy/wifi/scan', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({band, wait_s: 4})});
      const d = await r.json();
      if (!d.ok) {
        lbl.textContent = '✗ ' + (d.error || 'scan failed');
        lbl.style.color = '#ef4444';
        return;
      }
      const items = d.scanned_items || [];
      // Aggregate per channel
      const perCh = {};
      items.forEach(it => {
        const ch = it.channel;
        if (ch == null) return;
        if (!perCh[ch]) perCh[ch] = 0;
        perCh[ch] += 1;
      });
      const summary = Object.keys(perCh)
        .sort((a,b)=>Number(a)-Number(b))
        .map(c => `ch${c}:${perCh[c]}`).join(' ');
      lbl.textContent = summary ? `scan: ${summary}` : 'scan: no APs seen';
      lbl.style.color = '#22c55e';
      console.log('[wifi-scan]', d);
    } catch(e) { lbl.textContent = '✗ '+e; lbl.style.color = '#ef4444'; }
  };
})();
setTimeout(wifiRefresh, 1200);

document.getElementById('toggle_cmd_log').onclick = async ()=>{
  try {
    const r = await fetch('/proxy/logging/commands');
    const s = await r.json();
    await post('/proxy/logging/commands', {enabled: !Boolean(s.enabled)});
    refreshCommandLogStatus();
  } catch {}
};

document.getElementById('download_cmd_log').onclick = ()=>{
  window.open('/proxy/logging/commands/download', '_blank');
};

document.getElementById('clear_cmd_log').onclick = async ()=>{
  try {
    await post('/proxy/logging/commands/clear', {});
  } catch {}
};

const map = new Set(['w','a','s','d','q','e','r','f','t','l','x',' ']);
function _isTyping() {
  const t = document.activeElement?.tagName;
  return t === 'INPUT' || t === 'TEXTAREA' || document.activeElement?.isContentEditable;
}
window.addEventListener('keydown', (e)=>{
  const k = e.key.toLowerCase();
  // ── SAFETY HOTKEYS — always top priority ────────────────────────────
  // 0 (LAND ALL) and 9 (PAUSE / CONTINUE) are the fleet-level emergency
  // controls. They MUST fire even when a preset-name textbox, slider,
  // code editor, or any other input has focus — otherwise "press 0 to
  // land" becomes "sometimes presses 0 to land" which is unacceptable
  // for safety. We bypass the _isTyping() guard for these two keys.
  // Movement keys still respect _isTyping() so typing "w" in a textbox
  // doesn't fly the drone.
  if (k === '0') {
    e.preventDefault();
    // Blur any text field so the keystroke doesn't also land in it.
    if (document.activeElement && document.activeElement !== document.body) {
      try { document.activeElement.blur(); } catch {}
    }
    if (window._landAllInFlight) return;
    window._landAllInFlight = true;
    console.log('[LAND_ALL] 0 pressed — landing every drone (top-priority hotkey)');
    landAllDrones('0 hotkey').finally(() => { window._landAllInFlight = false; });
    return;
  }
  if (k === '9') {
    e.preventDefault();
    if (document.activeElement && document.activeElement !== document.body) {
      try { document.activeElement.blur(); } catch {}
    }
    if (window._pauseInFlight) return;
    window._pauseInFlight = true;
    console.log('[PAUSE] 9 pressed — toggling fleet pause (top-priority hotkey)');
    if (window._globalPaused) {
      console.log('[PAUSE] 9 pressed — resuming fleet');
      resumeAllDrones('9 hotkey').finally(() => { window._pauseInFlight = false; });
    } else {
      console.log('[PAUSE] 9 pressed — pausing fleet');
      pauseAllDrones('9 hotkey').finally(() => { window._pauseInFlight = false; });
    }
    return;
  }
  // Below this point, non-safety keys respect the typing guard.
  if (_isTyping()) return;
  // ── Drone switch hotkey: digits 1-5 select the Nth drone in the bar ──
  // Order follows Object.entries(drones) insertion order, same as the
  // drone-bar buttons top→bottom. '1' = first drone, '2' = second, etc.
  // Accept both the top-row digit key and the numpad digit; e.key is '1'
  // in both cases so simple string comparison works, but also use e.code
  // as a fallback when an exotic layout remaps the character.
  const isDigit15 = /^[1-5]$/.test(k) ||
                    /^(Digit|Numpad)[1-5]$/.test(e.code || '');
  if (isDigit15) {
    e.preventDefault();
    const slot = parseInt(k, 10) || parseInt((e.code || '').slice(-1), 10);
    const idx  = slot - 1;
    const ids  = Object.keys(drones || {});
    console.log('[drone-switch] hotkey', slot, 'fleet=', ids,
                'active=', activeDroneId);
    if (ids.length === 0) {
      console.warn('[drone-switch] drones dict is empty — did loadDrones run?');
      return;
    }
    if (idx < ids.length) {
      const targetId = ids[idx];
      if (targetId !== activeDroneId) {
        switchDrone(targetId);
      } else {
        console.log('[drone-switch] already on', targetId);
      }
    } else {
      console.log('[drone-switch] no drone at slot', slot, '(have', ids.length, ')');
    }
    return;
  }
  if (map.has(k)) {
    e.preventDefault();
    pressKey(k === ' ' ? 'space' : k);
  }
});

// Button wiring for the top-of-page LAND ALL button
(function(){
  const b = document.getElementById('land_all_btn');
  if (!b) return;
  b.addEventListener('click', () => {
    if (window._landAllInFlight) return;
    window._landAllInFlight = true;
    console.log('[LAND_ALL] button clicked');
    landAllDrones('button').finally(() => { window._landAllInFlight = false; });
  });
})();

// ── PAUSE ALL / CONTINUE MISSION wiring ───────────────────────────────
// The PAUSE button overrides any command — every drone freezes at its
// current position (autonomous missions abort, ArUco Seek drops out of
// LIVE, and a zero-RC brake goes out to every drone). Only manual WASD
// / RC remains active. CONTINUE MISSION clears the flag; nothing
// auto-restarts so the operator is never surprised by a darting drone.
window._globalPaused = false;
function applyPauseUI(paused, info) {
  window._globalPaused = !!paused;
  const pbtn = document.getElementById('pause_all_btn');
  const rbtn = document.getElementById('resume_all_btn');
  const ban  = document.getElementById('global_pause_banner');
  if (pbtn) pbtn.style.display = paused ? 'none' : '';
  if (rbtn) rbtn.style.display = paused ? '' : 'none';
  if (ban)  ban.style.display  = paused ? '' : 'none';
  document.body.classList.toggle('paused-mode', !!paused);
  if (paused && info && info.source) {
    if (ban) ban.title = 'paused via ' + info.source;
  }
}
async function pauseAllDrones(source) {
  try {
    const r = await fetch('/proxy/pause_all', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({source: source || 'ui'}),
    });
    const j = await r.json();
    console.log('[PAUSE] result:', j);
    applyPauseUI(true, j);
    showLandAllBanner(
      '\\u23f8 PAUSED — ' + (j.braked || 0) + '/' + (j.total || 0) +
      ' drones braked' + (j.mission_stopped ? ' (mission stopped)' : ''),
      '#78350f', '#fde68a', 4000);
  } catch (err) {
    console.error('[PAUSE] failed:', err);
    showLandAllBanner('\\u2717 PAUSE failed: ' + err, '#7f1d1d', '#fecaca', 5000);
  }
}
async function resumeAllDrones(source) {
  try {
    const r = await fetch('/proxy/resume_all', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({source: source || 'ui'}),
    });
    const j = await r.json();
    console.log('[RESUME] result:', j);
    applyPauseUI(false, j);
    showLandAllBanner('\\u25b6 RESUMED — autonomous control re-enabled',
                      '#064e3b', '#a7f3d0', 3500);
  } catch (err) {
    console.error('[RESUME] failed:', err);
    showLandAllBanner('\\u2717 RESUME failed: ' + err, '#7f1d1d', '#fecaca', 5000);
  }
}
(function wirePauseButtons(){
  const p = document.getElementById('pause_all_btn');
  if (p) p.addEventListener('click', () => {
    if (window._pauseInFlight) return;
    window._pauseInFlight = true;
    pauseAllDrones('button').finally(() => { window._pauseInFlight = false; });
  });
  const r = document.getElementById('resume_all_btn');
  if (r) r.addEventListener('click', () => {
    if (window._pauseInFlight) return;
    window._pauseInFlight = true;
    resumeAllDrones('button').finally(() => { window._pauseInFlight = false; });
  });
})();
// Pause state sync is now handled by the unified uiStatePoll() above.
// Retained applyPauseUI() — it's called from there and from hotkey handlers.

// ── Transport selector (per-subsystem WS ↔ HTTP) ──────────────────────
// Populated from /proxy/config/transport on load + every time the
// server is polled via the WS status check. POSTs on every dropdown
// change so the server state always matches the UI.
(function wireTransport(){
  const selectors = document.querySelectorAll('.transport-sel');
  const status = document.getElementById('transport_status');
  function flash(msg, col) {
    if (!status) return;
    status.textContent = msg;
    status.style.color = col || '#64748b';
    setTimeout(() => { if (status.textContent === msg) status.textContent = ''; }, 2500);
  }
  async function apply(patch, label) {
    try {
      const r = await fetch('/proxy/config/transport', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(patch),
      });
      const j = await r.json();
      if (j.ok) {
        // Reflect server's authoritative state in case some keys were rejected.
        Object.keys(j.transport || {}).forEach(k => {
          const el = document.getElementById('transport_' + k);
          if (el) el.value = j.transport[k];
        });
        flash('\\u2713 ' + (label || 'applied'), '#22c55e');
      } else {
        flash('error: ' + (j.error || 'unknown'), '#ef4444');
      }
    } catch (e) { flash('request failed', '#ef4444'); }
  }
  selectors.forEach(el => {
    el.addEventListener('change', () => {
      apply({[el.dataset.subsys]: el.value}, el.dataset.subsys + '=' + el.value);
    });
  });
  const allHttp = document.getElementById('transport_all_http');
  if (allHttp) allHttp.onclick = () =>
    apply({rc: 'http', telemetry: 'http', position: 'http'}, 'all → http');
  const allAuto = document.getElementById('transport_all_auto');
  if (allAuto) allAuto.onclick = () =>
    apply({rc: 'auto', telemetry: 'auto', position: 'auto'}, 'all → auto');

  // Initial load — fetch current state from the server.
  (async () => {
    try {
      const j = await (await fetch('/proxy/config/transport')).json();
      Object.keys(j.transport || {}).forEach(k => {
        const el = document.getElementById('transport_' + k);
        if (el) el.value = j.transport[k];
      });
      if (!j.ws_available) {
        document.querySelectorAll('.transport-sel').forEach(el => {
          // Disable the WS option when the client library is missing
          el.querySelectorAll('option[value="ws"]').forEach(o => o.disabled = true);
          if (el.value === 'ws') el.value = 'http';
        });
        flash('WS library missing — selectors locked to http/auto', '#fbbf24');
      }
    } catch {}
  })();
})();

// ── WebSocket status badge ────────────────────────────────────────────
// Polls /proxy/ws/status every 2 s. Green when the active drone has
// telemetry + position + rc sockets all up; amber if partial; red if
// the WS client lib is missing or all three are down.
(function wireWsStatus(){
  const el = document.getElementById('ws_status_badge');
  if (!el) return;
  async function tick() {
    try {
      applyWs(await (await fetch('/proxy/ws/status', {cache:'no-store'})).json());
    } catch {
      el.style.background = '#334155';
      el.style.color = '#94a3b8';
      el.textContent = 'WS ?';
    }
  }
  // Exposed on window so the unified uiStatePoll() can drive it without
  // the local timer firing separate /proxy/ws/status requests.
  function applyWs(j) {
    if (!j || !j.available) {
      el.style.background = '#7f1d1d';
      el.style.color = '#fecaca';
      el.textContent = 'WS off';
      el.title = 'websocket-client not installed on C2 — falling back to HTTP';
      return;
    }
    const did = String(window.activeDroneId || activeDroneId);
    const st = (j.drones || {})[did];
    if (!st) {
      el.style.background = '#334155';
      el.style.color = '#94a3b8';
      el.textContent = 'WS —';
      return;
    }
    const up = (st.telemetry ? 1 : 0) + (st.position ? 1 : 0) + (st.rc ? 1 : 0);
    if (up === 3) {
      const slowSend = (st.rc_send_ms != null && st.rc_send_ms > 50);
      el.style.background = slowSend ? '#78350f' : '#064e3b';
      el.style.color      = slowSend ? '#fde68a' : '#86efac';
      const parts = [];
      if (st.rc_send_ms     != null) parts.push('send ' + Math.round(st.rc_send_ms) + 'ms');
      if (st.rc_rtt_ms      != null) parts.push('rtt ' + st.rc_rtt_ms + 'ms');
      if (st.telemetry_age_ms != null) parts.push('tel ' + st.telemetry_age_ms + 'ms');
      if (st.position_age_ms  != null) parts.push('pos ' + st.position_age_ms  + 'ms');
      el.textContent = (slowSend ? 'WS \u26a0 ' : 'WS \u2713 ') + parts.join(' · ');
    } else if (up > 0) {
      el.style.background = '#78350f';
      el.style.color = '#fde68a';
      const flags = (st.telemetry?'T':'·') + (st.position?'P':'·') + (st.rc?'R':'·');
      el.textContent = 'WS ' + flags;
    } else {
      el.style.background = '#7f1d1d';
      el.style.color = '#fecaca';
      el.textContent = 'WS down';
    }
    el.title = 'tel=' + st.telemetry + ' pos=' + st.position + ' rc=' + st.rc;
  }
  window._applyWsStatus = applyWs;
  tick();  // one-time initial fetch; steady state is pushed by uiStatePoll()
})();

// ── Flight Logs list ──────────────────────────────────────────────────
// Lists archived per-flight JSONL files. Polls occasionally so a live
// recording's size updates visibly (the file grows as ticks accumulate).
(function wireFlightLogs(){
  const list = document.getElementById('flight_logs_list');
  const btn  = document.getElementById('flight_logs_refresh');
  if (!list) return;
  function fmtSize(n) {
    if (n < 1024) return n + ' B';
    if (n < 1024*1024) return (n/1024).toFixed(1) + ' KB';
    return (n/1024/1024).toFixed(2) + ' MB';
  }
  function fmtAge(ts) {
    const dt = Math.max(0, Date.now()/1000 - ts);
    if (dt < 60) return Math.round(dt) + 's ago';
    if (dt < 3600) return Math.round(dt/60) + 'm ago';
    if (dt < 86400) return Math.round(dt/3600) + 'h ago';
    return Math.round(dt/86400) + 'd ago';
  }
  async function refresh() {
    try {
      const r = await fetch('/proxy/flight_logs');
      const j = await r.json();
      if (!j.files || !j.files.length) {
        list.innerHTML = '<i style="color:#64748b;">no flights recorded yet</i>';
        return;
      }
      list.innerHTML = j.files.map(f => {
        const vid = f.video;
        const vidHtml = vid
          ? ('<a href="/proxy/flight_video/' + encodeURIComponent(vid.name) + '" ' +
             'style="color:#a78bfa;text-decoration:none;font-weight:600;" ' +
             'title="Download ' + vid.name + ' (' + fmtSize(vid.size || 0) + ')">' +
             '&#127916; Video</a>')
          : '<span style="color:#475569;font-style:italic;" title="No video recorded for this flight">no video</span>';
        return '<div style="display:flex;gap:10px;padding:2px 0;align-items:center;">' +
          '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' +
            '<a href="/proxy/flight_logs/' + encodeURIComponent(f.name) + '" ' +
               'style="color:#38bdf8;text-decoration:none;" title="Download ' + f.name + '">' +
               f.name + '</a>' +
          '</span>' +
          '<a href="/flight_log_viewer?file=' + encodeURIComponent(f.name) + '" ' +
             'target="_blank" ' +
             'style="color:#fbbf24;text-decoration:none;font-weight:600;" ' +
             'title="Open replay viewer in a new tab">&#128065; View</a>' +
          vidHtml +
          '<span style="color:#94a3b8;width:70px;text-align:right;">' + fmtSize(f.size) + '</span>' +
          '<span style="color:#64748b;width:70px;text-align:right;">' + fmtAge(f.mtime) + '</span>' +
        '</div>';
      }).join('');
    } catch (e) {
      list.innerHTML = '<i style="color:#ef4444;">error: ' + e + '</i>';
    }
  }
  if (btn) btn.onclick = refresh;
  refresh();
  setInterval(refresh, 15000);  // 15s — enough to see new flights appear
})();

// ── Flight guards: axis-lock + arena safety (manual + autonomous) ─
(function wireFlightGuards(){
  const axisTog   = document.getElementById('axis_locked_toggle');
  const arenaTog  = document.getElementById('arena_guard_toggle');
  const camFaceTog = document.getElementById('cam_face_center_toggle');
  const inp       = document.getElementById('safety_margin_input');
  const btn       = document.getElementById('safety_margin_apply');
  const status    = document.getElementById('autonomous_guards_status');
  const badge     = document.getElementById('arena_guard_engaged_badge');
  function flash(msg, col) {
    if (!status) return;
    status.textContent = msg;
    status.style.color = col || '#64748b';
    setTimeout(() => { if (status.textContent === msg) status.textContent = ''; }, 2500);
  }

  async function load() {
    try {
      // axis-lock (autonomous-observer param)
      const p = await (await fetch('/proxy/aruco/params')).json();
      if (axisTog && typeof p.axis_locked !== 'undefined') axisTog.checked = !!p.axis_locked;
      // Camera-faces-arena-centre toggle (C2-local fleet-wide setting)
      try {
        const cf = await (await fetch('/proxy/config/camera_face_center')).json();
        if (camFaceTog && typeof cf.enabled !== 'undefined') camFaceTog.checked = !!cf.enabled;
      } catch {}
      // arena guard + margin (Pi-side, enforces on BOTH manual + auto)
      const a = await (await fetch('/proxy/config/arena_safety')).json();
      if (arenaTog && typeof a.enabled !== 'undefined') arenaTog.checked = !!a.enabled;
      if (inp && a.margin_m != null && document.activeElement !== inp)
        inp.value = Number(a.margin_m).toFixed(1);
      if (badge) badge.style.display = a.engaged ? '' : 'none';
      // Cache on window so drawArena + 3D + position readout can render
      // the dashed safety boundary + distance-to-wall readout.
      window._arenaSafety = {
        enabled:  !!a.enabled,
        margin_m: a.margin_m,
        engaged:  !!a.engaged,
        reasons:  a.reasons || [],
      };
      // Redraw the 2D arena so the updated margin rectangle appears
      // even if no fresh position event has arrived (e.g. operator
      // changed the margin value while the drone is stationary).
      if (typeof drawArena === 'function') drawArena();
      // Update the 3D highlight too — safety overlay mesh refresh
      if (window._arena3d && window._arena3d.updateSafetyMargin) {
        window._arena3d.updateSafetyMargin(a.margin_m, !!a.engaged);
      }
    } catch {}
  }
  // Initial load only — subsequent updates ride on the unified 1Hz
  // /proxy/ui_state poll (see uiStatePoll). Avoids adding another
  // per-2s fetch cycle that would compete with the browser's 6-
  // connection HTTP/1.1 pool alongside keydown/batch traffic.
  load();
  // Expose for the unified poller
  window._applyArenaSafety = (a) => {
    if (!a) return;
    if (arenaTog && typeof a.enabled !== 'undefined') arenaTog.checked = !!a.enabled;
    if (inp && a.margin_m != null && document.activeElement !== inp)
      inp.value = Number(a.margin_m).toFixed(1);
    if (badge) badge.style.display = a.engaged ? '' : 'none';
    window._arenaSafety = {
      enabled:  !!a.enabled,
      margin_m: a.margin_m,
      engaged:  !!a.engaged,
      reasons:  a.reasons || [],
    };
    if (typeof drawArena === 'function') drawArena();
    if (window._arena3d && window._arena3d.updateSafetyMargin) {
      window._arena3d.updateSafetyMargin(a.margin_m, !!a.engaged);
    }
  };

  if (axisTog) axisTog.addEventListener('change', async () => {
    try {
      await fetch('/proxy/aruco/params', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({axis_locked: axisTog.checked}),
      });
      flash(axisTog.checked ? '\u2713 axis-lock ON' : '\u2713 axis-lock OFF', '#22c55e');
    } catch (e) { flash('request failed', '#ef4444'); }
  });

  if (camFaceTog) camFaceTog.addEventListener('change', async () => {
    try {
      const r = await fetch('/proxy/config/camera_face_center', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({enabled: camFaceTog.checked}),
      });
      const j = await r.json();
      if (j.ok) flash(camFaceTog.checked
                        ? '\u2713 camera \u2192 arena centre (fleet)'
                        : '\u2713 camera free (mission-driven)',
                       '#22c55e');
      else flash('error: ' + (j.error || 'unknown'), '#ef4444');
    } catch (e) { flash('request failed', '#ef4444'); }
  });

  if (arenaTog) arenaTog.addEventListener('change', async () => {
    try {
      const r = await fetch('/proxy/config/arena_safety', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({enabled: arenaTog.checked}),
      });
      const j = await r.json();
      if (j.ok) flash(arenaTog.checked
                        ? '\u2713 arena guard ON (manual + auto)'
                        : '\u26a0 arena guard OFF — operator owns boundaries',
                       arenaTog.checked ? '#22c55e' : '#fbbf24');
      else flash('error: ' + (j.error || 'unknown'), '#ef4444');
    } catch (e) { flash('request failed', '#ef4444'); }
  });

  if (btn) btn.onclick = async () => {
    const v = parseFloat(inp.value);
    if (!isFinite(v) || v < 0.1 || v > 5.0) {
      flash('enter 0.1 - 5.0 m', '#ef4444');
      return;
    }
    // Push to BOTH the Pi-side guard (manual + auto) AND the observer
    // autonomous guard so they share a single source of truth.
    try {
      const [a, b] = await Promise.all([
        fetch('/proxy/config/arena_safety', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({margin_m: v}),
        }).then(r => r.json()).catch(e => ({ok:false, error:String(e)})),
        fetch('/proxy/aruco/safety_margin', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({safety_margin_m: v}),
        }).then(r => r.json()).catch(e => ({ok:false, error:String(e)})),
      ]);
      if (a.ok && b.ok) flash('\u2713 margin \u2192 ' + v + ' m (Pi + observers)', '#22c55e');
      else flash('partial: pi=' + (a.ok?'ok':'err') + ' obs=' + (b.ok?'ok':'err'), '#fbbf24');
    } catch (e) { flash('request failed', '#ef4444'); }
  };
})();

// ── Ceiling safety wiring ─────────────────────────────────────────────
// The ceiling is enforced on the Pi (can't be bypassed by client code),
// but the UI mirrors its value and flashes an alert banner when the
// guard is actively clamping any drone. Poll every ~1 s so operators
// see engagement immediately on overshoot.
(function wireCeiling(){
  const inp = document.getElementById('ceiling_input');
  const btn = document.getElementById('ceiling_apply_btn');
  const badge = document.getElementById('ceiling_engaged_badge');
  const status = document.getElementById('ceiling_status');
  function apply(j) {
    if (!j) return;
    if (typeof j.ceiling_m === 'number') {
      if (document.activeElement !== inp) inp.value = Number(j.ceiling_m).toFixed(1);
    }
    if (badge) badge.style.display = j.engaged ? '' : 'none';
    if (status) {
      if (j.engaged && j.reasons && j.reasons.length) {
        status.textContent = '\u26a0 ' + j.reasons[0];
        status.style.color = '#fbbf24';
      } else {
        status.textContent = '';
      }
    }
  }
  // Exposed for the unified uiStatePoll — no local setInterval anymore.
  window._applyCeilingStatus = apply;
  async function load() {
    try {
      const j = await (await fetch('/proxy/config/ceiling', {cache:'no-store'})).json();
      apply(j);
    } catch {}
  }
  if (btn) btn.onclick = async () => {
    const v = parseFloat(inp.value);
    if (!isFinite(v) || v < 0.5 || v > 20) {
      status.textContent = 'enter 0.5 - 20 m';
      status.style.color = '#ef4444';
      return;
    }
    status.textContent = 'applying...';
    status.style.color = '#94a3b8';
    try {
      const r = await fetch('/proxy/config/ceiling', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ceiling_m: v}),
      });
      const j = await r.json();
      status.textContent = '\u2713 ' + (j.applied || 0) + '/' + (j.total || 0) + ' drones → ' + v + ' m';
      status.style.color = '#22c55e';
      console.log('[CEILING] apply result:', j);
    } catch (err) {
      status.textContent = 'failed: ' + err;
      status.style.color = '#ef4444';
    }
  };
  load();   // one-time initial fetch; steady state pushed by uiStatePoll()
})();

// Fleet-wide panic land — used by the 'q' hotkey and the big red
// LAND ALL button. Shows a banner with per-drone results.
async function landAllDrones(trigger) {
  // Visible flash so the operator knows the hotkey fired even before the
  // server responds — critical for a panic button.
  showLandAllBanner('⚠ LANDING ALL DRONES (' + trigger + ')…', '#78350f', '#fde68a');
  try {
    const r = await fetch('/proxy/land_all', {method:'POST'});
    const j = await r.json();
    const summary = (j.landed || 0) + '/' + (j.total || 0) +
                    ' acknowledged land' +
                    (j.mission_stopped ? ' (mission stopped)' : '');
    const color = (j.landed === j.total) ? '#064e3b' : '#7f1d1d';
    const txt   = (j.landed === j.total) ? '#a7f3d0' : '#fecaca';
    showLandAllBanner('✓ LAND ALL → ' + summary, color, txt, 6000);
    console.log('[LAND_ALL] result:', j);
  } catch (err) {
    showLandAllBanner('✗ LAND ALL failed: ' + err, '#7f1d1d', '#fecaca', 6000);
    console.error('[LAND_ALL]', err);
  }
}

function showLandAllBanner(msg, bg, fg, autohideMs) {
  let el = document.getElementById('land_all_banner');
  if (!el) {
    el = document.createElement('div');
    el.id = 'land_all_banner';
    el.style.cssText = 'position:fixed;top:8px;left:50%;transform:translateX(-50%);' +
                       'z-index:9999;padding:10px 18px;border-radius:6px;' +
                       'font-weight:700;font-size:14px;box-shadow:0 4px 14px rgba(0,0,0,0.5);' +
                       'border:2px solid rgba(255,255,255,0.15);letter-spacing:0.4px;';
    document.body.appendChild(el);
  }
  el.style.background = bg;
  el.style.color = fg;
  el.textContent = msg;
  el.style.display = 'block';
  clearTimeout(showLandAllBanner._t);
  if (autohideMs) {
    showLandAllBanner._t = setTimeout(() => { el.style.display = 'none'; }, autohideMs);
  }
}
window.addEventListener('keyup', (e)=>{
  if (_isTyping()) { releaseAllKeys(); return; }
  const k = e.key.toLowerCase();
  if (map.has(k)) {
    e.preventDefault();
    releaseKey(k === ' ' ? 'space' : k);
  }
});
window.addEventListener('blur', ()=>releaseAllKeys());
document.addEventListener('visibilitychange', ()=>{
  if (document.hidden) releaseAllKeys();
});

function meterColor(v){
  if (v >= 70) return '#22c55e';
  if (v >= 35) return '#f59e0b';
  return '#ef4444';
}
function setMeter(idBar, idVal, value, suffix=''){
  const bar = document.getElementById(idBar);
  const val = document.getElementById(idVal);
  if (value == null || Number.isNaN(Number(value))) {
    bar.style.width = '0%';
    bar.style.background = '#64748b';
    val.textContent = '-';
    return;
  }
  const v = Math.max(0, Math.min(100, Number(value)));
  bar.style.width = `${v}%`;
  bar.style.background = meterColor(v);
  val.textContent = `${Math.round(v)}${suffix}`;
}

let lastTelemetry = {};

async function refreshTelemetry(){
  const apiEl = document.getElementById('api_status');
  const droneEl = document.getElementById('drone_status');
  try {
    const r = await fetch('/proxy/telemetry', {cache:'no-store'});
    if (!r.ok) throw new Error('api_error');
    const t = await r.json();
    lastTelemetry = t;
    window.lastTelemetry = t;   // expose to standalone scripts (graphs)

    // Update telemetry speed/accel in position tracker section
    const fmt = v => (v != null && !isNaN(v)) ? Number(v).toFixed(1) : '—';
    document.getElementById('tel_vx').textContent = fmt(t.vgx);
    document.getElementById('tel_vy').textContent = fmt(t.vgy);
    document.getElementById('tel_vz').textContent = fmt(t.vgz);
    // Acceleration: only available on Tello; Anafi SDK does not expose it
    const hasAccel = t.agx != null;
    document.getElementById('tel_ax').textContent = hasAccel ? fmt(t.agx) : 'N/A';
    document.getElementById('tel_ay').textContent = hasAccel ? fmt(t.agy) : 'N/A';
    document.getElementById('tel_az').textContent = hasAccel ? fmt(t.agz) : 'N/A';

    apiEl.textContent = 'connected';
    apiEl.style.color = '#22c55e';

    // state_fresh may be absent from older API servers; infer liveness from real data
  const fresh = (t.state_fresh !== undefined) ? Boolean(t.state_fresh) : (t.battery != null || t.yaw != null);
  const live = Boolean(t.connected) && fresh;
    droneEl.textContent = live ? 'live' : 'no live telemetry';
    droneEl.style.color = live ? '#22c55e' : '#f59e0b';

    if (!live) {
      setMeter('battery_bar', 'battery_val', null);
      document.getElementById('telemetry').textContent =
        `no live drone telemetry\n` +
        `api reachable: yes\n` +
        `drone connected: ${t.connected}\n` +
        `state age: ${t.state_age_s ?? '-'} s`;
      return;
    }

    const battery = (typeof t.battery === 'number') ? t.battery : null;
    setMeter('battery_bar', 'battery_val', battery, '%');

    document.getElementById('telemetry').textContent =
      `battery: ${t.battery ?? '-'} %\n` +
      `temperature: ${t.temperature ?? '-'} °C\n` +
      `height: ${t.height_cm ?? '-'} cm\n` +
      `tof: ${t.tof_cm ?? '-'} cm\n` +
      `barometer: ${t.barometer_cm ?? '-'} cm\n` +
      `flight time: ${t.flight_time_s ?? '-'} s\n` +
      `speed: ${t.speed ?? '-'}\n` +
      `wifi snr: ${t.wifi_snr ?? '-'}\n` +
      `attitude p/r/y: ${t.pitch ?? '-'} / ${t.roll ?? '-'} / ${t.yaw ?? '-'}\n` +
      `velocity xyz: ${t.vgx ?? '-'} / ${t.vgy ?? '-'} / ${t.vgz ?? '-'}\n` +
      `accel xyz: ${t.agx ?? '-'} / ${t.agy ?? '-'} / ${t.agz ?? '-'}\n` +
      `sdk version: ${t.sdk_version ?? '-'}\n` +
      `serial number: ${t.serial_number ?? '-'}\n` +
      `mission pad mid/x/y/z/mpry: ${t.mid ?? '-'} / ${t.pad_x ?? '-'} / ${t.pad_y ?? '-'} / ${t.pad_z ?? '-'} / ${t.pad_mpry ?? '-'}\n` +
      `gps: ${t.gps_lat ?? '-'}, ${t.gps_lon ?? '-'} alt=${t.gps_alt ?? '-'}m\n` +
      `gimbal p/r/y: ${t.gimbal_pitch ?? '-'} / ${t.gimbal_roll ?? '-'} / ${t.gimbal_yaw ?? '-'}\n` +
      `state age: ${t.state_age_s ?? '-'} s\n` +
      `flying: ${t.flying}\n` +
      `connected: ${t.connected}`;
  } catch {
    apiEl.textContent = 'disconnected';
    apiEl.style.color = '#ef4444';
    const droneEl = document.getElementById('drone_status');
    droneEl.textContent = 'unknown';
    droneEl.style.color = '#ef4444';
    setMeter('battery_bar', 'battery_val', null);
    document.getElementById('telemetry').textContent = 'telemetry unavailable';
  }
}
async function refreshLogStatus(){
  try {
    const r = await fetch('/proxy/logging/telemetry');
    const s = await r.json();
    document.getElementById('toggle_log').textContent = s.enabled ? 'Disable Telemetry Log' : 'Enable Telemetry Log';
  } catch {}
}
async function refreshSafeTakeoff(){
  try {
    const r = await fetch('/proxy/safety/takeoff');
    const s = await r.json();
    document.getElementById('safe_takeoff').textContent = s.enabled ? 'Safe Takeoff: ON' : 'Safe Takeoff: OFF';
  } catch {}
}
async function refreshCommandLogStatus(){
  try {
    const r = await fetch('/proxy/logging/commands');
    const s = await r.json();
    document.getElementById('toggle_cmd_log').textContent = s.enabled ? 'Command Logging: ON' : 'Command Logging: OFF';
  } catch {}
}
// --- Video stream controls ---
const videoMode = document.getElementById('video_mode');
const videoToggle = document.getElementById('video_toggle');
const videoStatus = document.getElementById('video_status');
const videoUrl = document.getElementById('video_url');
const videoContainer = document.getElementById('video_container');
const videoImg = document.getElementById('video_img');
let videoActive = false;

// Shared video-start logic — used by the manual toggle AND by the
// auto-start path that fires as soon as the page loads. Both paths
// assume Way 1 (MJPEG) unless the user has manually picked forward.
async function startVideoStream(mode) {
  mode = mode || videoMode.value || 'mjpeg';
  if (mode === 'off') return false;
  try {
    const r = await fetch('/proxy/video/start', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({mode})});
    const d = await r.json();
    if (!d.ok) {
      videoStatus.textContent = 'Error: ' + (d.error || 'unknown');
      return false;
    }
    videoActive = true;
    videoToggle.textContent = 'Stop Video';
    videoStatus.textContent = 'Mode: ' + d.mode;
    videoContainer.style.display = '';
    if (d.mode === 'mjpeg') {
      videoImg.src = '/proxy/video?' + Date.now();
      videoUrl.style.display = '';
      videoUrl.innerHTML = 'Direct: <b>' + (d.stream_url || '') + '</b>';
    } else if (d.mode === 'forward') {
      videoImg.src = '/proxy/video/forward_stream?' + Date.now();
      videoUrl.style.display = '';
      videoUrl.innerHTML = 'UDP → C2 decode → MJPEG';
    }
    videoMode.value = d.mode;
    return true;
  } catch (e) {
    videoStatus.textContent = 'Error: ' + e;
    return false;
  }
}

videoToggle.onclick = async () => {
  if (videoActive) {
    await post('/proxy/video/stop', {});
    videoActive = false;
    videoToggle.textContent = 'Start Video';
    videoContainer.style.display = 'none';
    videoUrl.style.display = 'none';
    videoImg.src = '';
    videoStatus.textContent = 'Mode: off';
    return;
  }
  await startVideoStream(videoMode.value);
};

// Auto-start Way 1 (MJPEG) on page load. refreshVideoStatus() below will
// detect if the server reports an already-running stream and skip, so
// reloading the page won't restart the stream unnecessarily.
async function autoStartVideo() {
  try {
    const r = await fetch('/proxy/video/status', {cache:'no-store'});
    const d = await r.json();
    if (d && d.mode && d.mode !== 'off') {
      // Already running — adopt existing state without restarting
      videoActive = true;
      videoMode.value = d.mode;
      videoToggle.textContent = 'Stop Video';
      videoContainer.style.display = '';
      videoStatus.textContent = 'Mode: ' + d.mode + ' (existing)';
      if (d.mode === 'mjpeg') {
        videoImg.src = '/proxy/video?' + Date.now();
      } else if (d.mode === 'forward') {
        videoImg.src = '/proxy/video/forward_stream?' + Date.now();
      }
      console.log('[video] adopted existing stream:', d.mode);
      return;
    }
  } catch {}
  // Nothing running — start Way 1 (MJPEG, decoded on the flight controller).
  videoMode.value = 'mjpeg';
  console.log('[video] auto-starting MJPEG (Way 1)');
  const ok = await startVideoStream('mjpeg');
  if (!ok) console.warn('[video] auto-start failed, click Start Video manually');
}
// Fire a bit after page load so the C2 has booted + WS clients settled.
// Heartbeat is handled server-side now — the browser doesn't wait for it.
setTimeout(autoStartVideo, 300);

// ── Anafi camera zoom slider ─────────────────────────────────────────
(function(){
  const sl  = document.getElementById('video_zoom');
  const lbl = document.getElementById('video_zoom_val');
  const rst = document.getElementById('video_zoom_reset');
  if (!sl || !lbl) return;
  let _zoomTimer = null;
  function sendZoom(v) {
    fetch('/proxy/camera/zoom', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({zoom: v})}).catch(()=>{});
  }
  sl.addEventListener('input', () => {
    const v = Number(sl.value);
    lbl.textContent = v.toFixed(2) + '×';
    // Debounce — don't spam the drone with 100+ posts per drag
    if (_zoomTimer) clearTimeout(_zoomTimer);
    _zoomTimer = setTimeout(() => sendZoom(v), 80);
  });
  if (rst) rst.addEventListener('click', () => {
    sl.value = 1.0;
    lbl.textContent = '1.00×';
    sendZoom(1.0);
  });
})();

// ── Latency measurement ──────────────────────────────────────────────
// Polls /proxy/latency every 2s. Total displayed = c2_to_fc + fc_to_drone
// + video_offset (operator-configurable). When the "auto-set latency"
// checkbox is ticked, the total is pushed into the Position Tracker's
// latency_ms slider each poll so position fusion stays in sync with the
// actual comm-stack delay.
async function latPoll() {
  try {
    const r = await fetch('/proxy/latency', {cache:'no-store'});
    const d = await r.json();
    const fm = (x) => (x == null) ? '—' : Math.round(x) + ' ms';
    const c2fc = d.c2_to_fc_ms;
    const fcdr = d.fc_to_drone_ms;
    const videoEl = document.getElementById('lat_video_offset');
    const videoMs = videoEl ? (parseInt(videoEl.value, 10) || 0) : 0;
    const totalMs = (c2fc || 0) + (fcdr || 0) + videoMs;
    document.getElementById('lat_c2fc').textContent = fm(c2fc);
    document.getElementById('lat_fcdr').textContent = fm(fcdr);
    document.getElementById('lat_vid').textContent = fm(videoMs);
    const tEl = document.getElementById('lat_total');
    tEl.textContent = Math.round(totalMs) + ' ms';
    // Colour total: <80ms green, <200ms amber, else red
    tEl.style.color = (totalMs < 80) ? '#22c55e'
                    : (totalMs < 200) ? '#fbbf24' : '#ef4444';
    // Auto-push into Position Tracker slider if toggle on
    const auto = document.getElementById('lat_auto_apply');
    if (auto && auto.checked && c2fc != null && fcdr != null) {
      const slider = document.getElementById('pos_latency');
      const lbl = document.getElementById('pos_latency_val');
      if (slider) {
        slider.value = Math.round(totalMs);
        if (lbl) lbl.textContent = Math.round(totalMs);
        // Mirror to server so the positioning pipeline uses this too
        fetch('/proxy/position/config', {method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({latency_ms: Math.round(totalMs)})}).catch(()=>{});
      }
    }
  } catch {}
}
setInterval(latPoll, 2000);
setTimeout(latPoll, 800);

// ── Collapsible panels ───────────────────────────────────────────────
// Wrap a panel's existing content under a click-to-toggle header.
// If the panel already contains an h2/h3 as its first significant
// child, reuse it as the toggle; otherwise inject a new header with
// `defaultTitle`. State persists in localStorage per `storageKey` so
// each operator keeps their preferred layout across page reloads.
function makeCollapsible(el, defaultTitle, storageKey, startCollapsed) {
  if (!el || el.classList.contains('collapsible')) return;
  // Prefer an existing h2/h3 as the toggle header
  let header = el.querySelector(':scope > h2, :scope > h3');
  let body;
  if (header) {
    body = document.createElement('div');
    body.className = 'collapsible-body';
    const siblings = Array.from(el.children).filter(c => c !== header);
    siblings.forEach(c => body.appendChild(c));
    el.appendChild(body);
  } else {
    body = document.createElement('div');
    body.className = 'collapsible-body';
    while (el.firstChild) body.appendChild(el.firstChild);
    header = document.createElement('div');
    header.innerHTML = '<b>' + defaultTitle + '</b>';
    el.appendChild(header);
    el.appendChild(body);
  }
  header.classList.add('collapsible-toggle');
  el.classList.add('collapsible');
  header.addEventListener('click', (e) => {
    // Don't toggle if the click was on an interactive child
    const t = e.target;
    if (t !== header && (t.tagName === 'INPUT' || t.tagName === 'BUTTON' ||
                         t.tagName === 'SELECT' || t.tagName === 'A' ||
                         t.tagName === 'TEXTAREA')) return;
    el.classList.toggle('collapsed');
    try { localStorage.setItem(storageKey,
          el.classList.contains('collapsed') ? '1' : '0'); } catch {}
  });
  // Load persisted state (fall back to startCollapsed default)
  let saved = null;
  try { saved = localStorage.getItem(storageKey); } catch {}
  const shouldCollapse = (saved === null) ? Boolean(startCollapsed)
                                           : (saved === '1');
  if (shouldCollapse) el.classList.add('collapsed');
}

// Find a .panel that contains a <b> with the given exact text.
function _panelByBoldTitle(title) {
  const panels = Array.from(document.querySelectorAll('.panel'));
  for (const p of panels) {
    const b = p.querySelector('b');
    if (b && b.textContent.trim() === title) return p;
  }
  return null;
}

// Run after the DOM is settled — some panels are only fully populated
// after the first polls (e.g. drone bar) but their structure is fixed.
// ── Relocate tuning controls into the new unified Tuning Parameters
// panel. Runs BEFORE makeCollapsible so the panel's starting state is
// set correctly. Keeps existing DOM nodes intact (IDs, handlers, all
// event bindings) — we just re-parent them. The original ArUco Seek
// panel keeps only the live readout; the Position Tracker panel keeps
// only the canvas + coordinate readouts. Everything else moves here.
function relocateTuningControls() {
  const obsSlot = document.getElementById('tuning_observer_slot');
  const posSlot = document.getElementById('tuning_position_slot');
  if (!obsSlot || !posSlot) return;

  // Observer PD — the slider grid + Reload button container
  const arcParamsWrap = document.getElementById('arc_params');
  if (arcParamsWrap && arcParamsWrap.parentElement) {
    // The adjacent 'Live tuning parameters' label + Reload button are
    // siblings of #arc_params inside the same wrapper <div>. Move the
    // whole wrapper for convenience.
    const wrap = arcParamsWrap.parentElement;
    obsSlot.appendChild(wrap);
  }

  // Position Tracker — the .pos-cfg div holding all the fusion sliders.
  // The tracker panel keeps its canvas + numeric readouts; the config
  // rows (Profile / FOV / Latency / IMU blend / Kalman / Marker size /
  // Top-K / Outlier / Apply Config) move over.
  const posCfgWrap = document.querySelector('#pos_panel .pos-cfg') ||
                     Array.from(document.querySelectorAll('.pos-cfg'))
                       .find(el => el.querySelector('#pos_profile'));
  if (posCfgWrap) posSlot.appendChild(posCfgWrap);
}
relocateTuningControls();

setTimeout(() => {
  // id-addressable panels
  makeCollapsible(document.getElementById('tuning_panel'),     'Tuning Parameters',     'collapsed_tuning',           true);
  makeCollapsible(document.getElementById('mission_panel'),    'Mission Planner',       'collapsed_mission_planner',  true);
  makeCollapsible(document.getElementById('anafi_panel'),      'Anafi / Olympe controls','collapsed_anafi',           true);
  makeCollapsible(document.getElementById('video_panel'),      'Video stream',          'collapsed_video',            false);
  makeCollapsible(document.getElementById('aruco_panel'),      'ArUco Seek',            'collapsed_aruco',            true);
  makeCollapsible(document.getElementById('missions_panel'),   'Special Missions',      'collapsed_missions',         true);

  // Panels addressed by their <b>-wrapped title
  makeCollapsible(_panelByBoldTitle('Telemetry'),          'Telemetry',          'collapsed_telemetry',  false);
  makeCollapsible(_panelByBoldTitle('Position Tracker'),   'Position Tracker',   'collapsed_pos_tracker', false);
  makeCollapsible(_panelByBoldTitle('Arena Configuration'),'Arena Configuration','collapsed_arena_cfg',  true);

  // WASD grid panel — first .panel that contains a .grid
  (function wrapKeyPanel() {
    const p = document.querySelector('.panel:has(> .grid)') ||
              Array.from(document.querySelectorAll('.panel')).find(
                x => x.querySelector(':scope > .grid'));
    if (p) makeCollapsible(p, 'WASD Key controls', 'collapsed_keys', false);
  })();

  // Advanced SDK controls — .adv block with the "Advanced SDK controls" label
  (function wrapAdvPanel() {
    const advs = document.querySelectorAll('.adv');
    for (const a of advs) {
      const lbl = a.querySelector(':scope > .small');
      if (lbl && lbl.textContent.trim().startsWith('Advanced SDK controls')) {
        // Replace the tiny 'small' label with a proper header
        lbl.remove();
        makeCollapsible(a, 'Advanced SDK controls', 'collapsed_adv_sdk', true);
        return;
      }
    }
  })();
}, 150);

// Poll video status
async function refreshVideoStatus() {
  try {
    const r = await fetch('/proxy/video/status', {cache:'no-store'});
    const d = await r.json();
    if (!videoActive && d.mode !== 'off') {
      videoActive = true;
      videoToggle.textContent = 'Stop Video';
      videoMode.value = d.mode;
    }
    if (videoActive) {
      videoStatus.textContent = 'Mode: ' + d.mode + ' | has_frame: ' + (d.has_frame||false);
    }
  } catch {}
}
setInterval(refreshVideoStatus, 5000);

// Heartbeat — keeps the drone watchdog alive so it doesn't auto-land
// ── UNIFIED STATUS POLL ─────────────────────────────────────────────
// Replaces /proxy/heartbeat + /proxy/pause_status + /proxy/ws/status +
// /proxy/config/ceiling + /proxy/config/transport + /proxy/missions/status
// with ONE request at 1 Hz. The C2 aggregates them all locally (all are
// in-memory dict reads) — this frees the browser's 6-connection pool so
// CONTROL traffic (WASD, takeoff, rc) goes through without queueing.
//
// Control latency (keypress → drone) is completely unaffected: those go
// via /proxy/key_down | key_up | key_batch which are NOT part of this
// unified poll. Only the low-frequency UI bookkeeping lives here.
let _uiPollSeq = 0;
async function uiStatePoll(){
  const mySeq = ++_uiPollSeq;
  try {
    const r = await fetch('/proxy/ui_state', {cache:'no-store'});
    if (_uiPollSeq !== mySeq) return;   // a newer poll already fired
    const j = await r.json();
    // Distribute to the individual consumers
    if (j.pause && typeof applyPauseUI === 'function') {
      if (!!j.pause.paused !== !!window._globalPaused) applyPauseUI(j.pause.paused, j.pause);
    }
    if (j.ceiling && typeof window._applyCeilingStatus === 'function') {
      window._applyCeilingStatus(j.ceiling);
    }
    if (j.arena_safety && typeof window._applyArenaSafety === 'function') {
      window._applyArenaSafety(j.arena_safety);
    }
    if (j.ws && typeof window._applyWsStatus === 'function') {
      window._applyWsStatus(j.ws);
    }
    if (j.transport && typeof window._applyTransportState === 'function') {
      window._applyTransportState(j.transport);
    }
    if (j.missions && typeof window._applyMissionsStatus === 'function') {
      window._applyMissionsStatus(j.missions);
    }
  } catch {}
}
setInterval(uiStatePoll, 1000);
uiStatePoll();

// ── Telemetry SSE (replaces polling for near-real-time updates) ──
let telEvtSource = null;
function startTelemetrySSE() {
  if (telEvtSource) { telEvtSource.close(); telEvtSource = null; }
  telEvtSource = new EventSource('/proxy/telemetry/stream');
  telEvtSource.onmessage = (e) => {
    try {
      const t = JSON.parse(e.data);
      _handleTelemetryData(t);
    } catch {}
  };
  telEvtSource.onerror = () => {
    telEvtSource.close(); telEvtSource = null;
    // Fast reconnect — 500ms instead of 3s
    setTimeout(startTelemetrySSE, 500);
  };
}
// Extract telemetry UI update into reusable function
function _handleTelemetryData(t) {
  lastTelemetry = t;
  window.lastTelemetry = t;   // expose to standalone scripts (graphs)
  const apiEl = document.getElementById('api_status');
  const droneEl = document.getElementById('drone_status');
  const fmt = v => (v != null && !isNaN(v)) ? Number(v).toFixed(1) : '\\u2014';
  document.getElementById('tel_vx').textContent = fmt(t.vgx);
  document.getElementById('tel_vy').textContent = fmt(t.vgy);
  document.getElementById('tel_vz').textContent = fmt(t.vgz);
  const hasAccel = t.agx != null;
  document.getElementById('tel_ax').textContent = hasAccel ? fmt(t.agx) : 'N/A';
  document.getElementById('tel_ay').textContent = hasAccel ? fmt(t.agy) : 'N/A';
  document.getElementById('tel_az').textContent = hasAccel ? fmt(t.agz) : 'N/A';
  apiEl.textContent = 'connected'; apiEl.style.color = '#22c55e';
  // state_fresh may be absent from older API servers; infer liveness from real data
  const fresh = (t.state_fresh !== undefined) ? Boolean(t.state_fresh) : (t.battery != null || t.yaw != null);
  const live = Boolean(t.connected) && fresh;
  droneEl.textContent = live ? 'live' : 'no live telemetry';
  droneEl.style.color = live ? '#22c55e' : '#f59e0b';
  if (!live) {
    setMeter('battery_bar', 'battery_val', null);
    document.getElementById('telemetry').textContent =
      `no live drone telemetry\\napi reachable: yes\\ndrone connected: ${t.connected}\\nstate age: ${t.state_age_s ?? '-'} s`;
    return;
  }
  const battery = (typeof t.battery === 'number') ? t.battery : null;
  setMeter('battery_bar', 'battery_val', battery, '%');
  document.getElementById('telemetry').textContent =
    `battery: ${t.battery ?? '-'} %\\ntemperature: ${t.temperature ?? '-'} °C\\nheight: ${t.height_cm ?? '-'} cm\\ntof: ${t.tof_cm ?? '-'} cm\\nbarometer: ${t.barometer_cm ?? '-'} cm\\nflight time: ${t.flight_time_s ?? '-'} s\\nspeed: ${t.speed ?? '-'}\\nwifi snr: ${t.wifi_snr ?? '-'}\\nattitude p/r/y: ${t.pitch ?? '-'} / ${t.roll ?? '-'} / ${t.yaw ?? '-'}\\nvelocity xyz: ${t.vgx ?? '-'} / ${t.vgy ?? '-'} / ${t.vgz ?? '-'}\\naccel xyz: ${t.agx ?? '-'} / ${t.agy ?? '-'} / ${t.agz ?? '-'}\\nsdk version: ${t.sdk_version ?? '-'}\\nserial number: ${t.serial_number ?? '-'}\\nmission pad mid/x/y/z/mpry: ${t.mid ?? '-'} / ${t.pad_x ?? '-'} / ${t.pad_y ?? '-'} / ${t.pad_z ?? '-'} / ${t.pad_mpry ?? '-'}\\ngps: ${t.gps_lat ?? '-'}, ${t.gps_lon ?? '-'} alt=${t.gps_alt ?? '-'}m\\ngimbal p/r/y: ${t.gimbal_pitch ?? '-'} / ${t.gimbal_roll ?? '-'} / ${t.gimbal_yaw ?? '-'}\\nstate age: ${t.state_age_s ?? '-'} s\\nflying: ${t.flying}\\nconnected: ${t.connected}`;

  // ── Compass + takeoff-heading tracking ──
  updateCompass(t);
}

// ── Compass widget: tracks magnetic yaw + takeoff-reference heading ──
// Anafi AttitudeChanged.yaw is NED yaw in degrees, range -180..+180
// (positive = nose turned clockwise when viewed from above). Magnetic-north
// referenced, NOT true north (no declination correction without GPS).
// Takeoff heading is captured the first time `flying` transitions false → true
// and on the Reset button. Stored per active drone id.
let _takeoffHeadingByDrone = {};
let _wasFlying = false;
let _lastActiveDroneId = null;

function normDeg(d) {
  // Normalize any degree to -180..+180
  while (d >  180) d -= 360;
  while (d < -180) d += 360;
  return d;
}

function updateCompass(t) {
  const cv = document.getElementById('compass_canvas');
  if (!cv) return;

  // Detect active-drone change — clear stale state so ref doesn't bleed across drones
  const activeId = (window.currentDroneId || t.drone_id || 'default');
  if (activeId !== _lastActiveDroneId) {
    _lastActiveDroneId = activeId;
    _wasFlying = Boolean(t.flying);  // don't auto-capture just from switching
  }

  const yawDeg = (typeof t.yaw === 'number') ? t.yaw : null;
  const flying = Boolean(t.flying);

  // Capture takeoff heading on false→true transition of `flying`
  if (flying && !_wasFlying && yawDeg != null) {
    _takeoffHeadingByDrone[activeId] = yawDeg;
  }
  _wasFlying = flying;

  const takeoffRef = _takeoffHeadingByDrone[activeId];
  const relative = (yawDeg != null && takeoffRef != null)
    ? normDeg(yawDeg - takeoffRef) : null;

  // Numeric labels
  document.getElementById('compass_abs').textContent =
    yawDeg != null ? yawDeg.toFixed(0) : '--';
  document.getElementById('compass_ref').textContent =
    takeoffRef != null ? takeoffRef.toFixed(0) + '°' : 'not captured';
  document.getElementById('compass_rel').textContent =
    relative != null ? (relative >= 0 ? '+' : '') + relative.toFixed(0) : '--';

  // Draw the compass rose
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const cx = W / 2, cy = H / 2;
  const r = Math.min(W, H) / 2 - 6;

  ctx.clearRect(0, 0, W, H);

  // Outer ring
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();

  // Tick marks every 30°
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 1;
  for (let a = 0; a < 360; a += 30) {
    const rad = (a - 90) * Math.PI / 180;  // 0° = up = North
    const isCardinal = (a % 90 === 0);
    const inner = r - (isCardinal ? 8 : 4);
    ctx.beginPath();
    ctx.moveTo(cx + inner * Math.cos(rad), cy + inner * Math.sin(rad));
    ctx.lineTo(cx + (r - 1) * Math.cos(rad), cy + (r - 1) * Math.sin(rad));
    ctx.stroke();
  }

  // Cardinal labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = 'bold 10px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('N', cx, cy - r + 7);
  ctx.fillText('S', cx, cy + r - 7);
  ctx.fillText('E', cx + r - 7, cy);
  ctx.fillText('W', cx - r + 7, cy);

  // Takeoff-heading marker on the rim (small gray triangle)
  if (takeoffRef != null) {
    const radRef = (takeoffRef - 90) * Math.PI / 180;
    const mx = cx + (r - 2) * Math.cos(radRef);
    const my = cy + (r - 2) * Math.sin(radRef);
    const backLen = 6;
    ctx.fillStyle = '#64748b';
    ctx.beginPath();
    ctx.arc(mx, my, 3.5, 0, Math.PI * 2);
    ctx.fill();
    // Small "T" label next to it
    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px sans-serif';
    const tx = cx + (r + 6) * Math.cos(radRef);
    const ty = cy + (r + 6) * Math.sin(radRef);
    ctx.fillText('T', tx, ty);
  }

  // Current heading needle — red/blue dual (red=front, blue=tail)
  if (yawDeg != null) {
    const rad = (yawDeg - 90) * Math.PI / 180;
    const nx = cx + (r - 10) * Math.cos(rad);
    const ny = cy + (r - 10) * Math.sin(rad);
    const tx = cx - (r - 18) * Math.cos(rad);
    const ty = cy - (r - 18) * Math.sin(rad);
    // Tail (blue)
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    // Front (red arrow)
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(nx, ny);
    ctx.stroke();
    // Arrowhead
    const ahLen = 7, ahAngle = 0.35;
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.moveTo(nx, ny);
    ctx.lineTo(
      nx - ahLen * Math.cos(rad - ahAngle),
      ny - ahLen * Math.sin(rad - ahAngle));
    ctx.lineTo(
      nx - ahLen * Math.cos(rad + ahAngle),
      ny - ahLen * Math.sin(rad + ahAngle));
    ctx.closePath();
    ctx.fill();
  }

  // Center pivot
  ctx.fillStyle = '#e2e8f0';
  ctx.beginPath();
  ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
  ctx.fill();
}

// Manual Reset-ref button — re-capture takeoff heading from current yaw
document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('compass_reset');
  if (btn) btn.addEventListener('click', () => {
    const t = lastTelemetry || {};
    const activeId = (activeDroneId || t.drone_id || 'default');
    if (typeof t.yaw === 'number') {
      _takeoffHeadingByDrone[activeId] = t.yaw;
      updateCompass(t);
    } else {
      alert('No yaw data available — drone not connected?');
    }
  });
});

startTelemetrySSE();
// Fallback poll — only triggers if SSE is down (reduced from 700ms to 2000ms since SSE handles real-time)
setInterval(()=>{ if (!telEvtSource) refreshTelemetry(); }, 2000);

setInterval(refreshLogStatus, 5000);
setInterval(refreshSafeTakeoff, 5000);
setInterval(refreshCommandLogStatus, 5000);
loadDrones();
refreshTelemetry();
refreshLogStatus();
refreshSafeTakeoff();
refreshCommandLogStatus();

// --- Magnetometer recalibration wizard ---
// Drives /proxy/magneto/recalibrate/stream (SSE) and updates the step list,
// per-axis panels, and result banner live as the server emits events.
const magModal = document.getElementById('mag_modal');
let magEventSource = null;

function magLog(line){
  const el = document.getElementById('mag_log');
  const t = new Date().toLocaleTimeString();
  el.textContent += `[${t}] ${line}\\n`;
  el.scrollTop = el.scrollHeight;
}

function magResetUI(){
  document.querySelectorAll('#mag_steps .mag-step').forEach(s=>{
    s.classList.remove('active','ok','fail');
    s.querySelector('[data-role="info"]').textContent = '';
  });
  document.querySelectorAll('#mag_axes .mag-axis').forEach(a=>{
    a.classList.remove('active','ok');
    a.querySelector('[data-role="state"]').innerHTML = '&#9675;';
  });
  const res = document.getElementById('mag_result');
  res.className = '';
  res.textContent = '';
  res.style.display = 'none';
  document.getElementById('mag_log').textContent = '';
  document.getElementById('mag_retry_btn').style.display = 'none';
}

function magSetStep(name, state, info){
  const el = document.querySelector(`#mag_steps .mag-step[data-step="${name}"]`);
  if (!el) return;
  document.querySelectorAll('#mag_steps .mag-step').forEach(s=>{
    if (s !== el && s.classList.contains('active')) s.classList.remove('active');
  });
  el.classList.remove('active','ok','fail');
  if (state) el.classList.add(state);
  if (info !== undefined) el.querySelector('[data-role="info"]').textContent = info;
}

function magSetAxes(axes){
  if (!axes) return;
  ['x','y','z'].forEach(ax=>{
    const el = document.querySelector(`#mag_axes .mag-axis[data-axis="${ax}"]`);
    if (!el) return;
    const stateEl = el.querySelector('[data-role="state"]');
    el.classList.remove('active','ok');
    if (axes[ax] === 1) {
      el.classList.add('ok');
      stateEl.innerHTML = '&#10004;';
    } else {
      if (axes[ax] === 0 || axes[ax] === null) {
        if (!axes.done && !axes.failed) el.classList.add('active');
      }
      stateEl.innerHTML = '&#9675;';
    }
  });
}

function magShowResult(ok, msg){
  const res = document.getElementById('mag_result');
  res.className = ok ? 'ok' : 'fail';
  res.textContent = msg || (ok ? 'Calibration complete.' : 'Calibration failed.');
  res.style.display = 'block';
  document.getElementById('mag_start_btn').disabled = false;
  document.getElementById('mag_retry_btn').style.display = ok ? 'none' : '';
}

function openMagnetoWizard(){
  magResetUI();
  magModal.classList.add('show');
  refreshMagStatus();
}

function closeMagnetoWizard(){
  magModal.classList.remove('show');
  if (magEventSource) { magEventSource.close(); magEventSource = null; }
}

async function refreshMagStatus(){
  try {
    const r = await fetch('/proxy/magneto');
    const j = await r.json();
    const txt = j.status || '(no report)';
    document.getElementById('mag_status').textContent = txt;
    document.getElementById('mag_status').style.color =
      j.required ? '#f87171' : (j.status ? '#86efac' : '#94a3b8');
    return j;
  } catch (e) {
    document.getElementById('mag_status').textContent = 'unreachable';
    document.getElementById('mag_status').style.color = '#f87171';
    return null;
  }
}

function runMagnetoWizard(){
  if (magEventSource) { magEventSource.close(); magEventSource = null; }
  magResetUI();
  document.getElementById('mag_start_btn').disabled = true;
  magLog('connecting to recalibration stream…');
  // Use SSE so every step flip shows up the instant the server observes it.
  // timeout_s=60 covers a patient operator; poll_s=1.0 is plenty.
  magEventSource = new EventSource('/proxy/magneto/recalibrate/stream?timeout_s=60&poll_s=1.0');

  magEventSource.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.kind === 'step') {
      const state = msg.ok ? 'ok' : (msg.step === 'poll' ? 'fail' : 'active');
      let info = '';
      if (msg.step === 'heartbeat')
        info = `connected=${msg.connected} flying=${msg.flying}`;
      else if (msg.step === 'pre_status')
        info = msg.status || '(not reported)';
      else if (msg.step === 'start')
        info = msg.message || msg.error || '';
      else if (msg.step === 'poll')
        info = msg.final_status || (msg.timed_out ? 'timed out' : '');
      magSetStep(msg.step, msg.ok ? 'ok' : 'fail', info);
      magLog(`step ${msg.step}: ${msg.ok ? 'OK' : 'FAIL'} ${info}`);
      if (!msg.ok && msg.step !== 'poll') {
        // Hard failure before the dance even started — stop the wizard.
        magShowResult(false,
          (msg.error ? 'Error: ' + msg.error : 'Step failed: ' + msg.step));
        if (magEventSource) { magEventSource.close(); magEventSource = null; }
      }
      if (msg.step === 'start' && msg.ok) magSetStep('poll', 'active');
    } else if (msg.kind === 'status') {
      magSetAxes(msg.axes);
      magLog(`status: ${msg.status || '(no report)'}`);
    } else if (msg.kind === 'final') {
      if (msg.post && msg.post.status) magSetAxes(
        (function(){ const s = (msg.post.status||'').toLowerCase().replace(/\\s+/g,'');
          const m = s.match(/axes=x(\\d)y(\\d)z(\\d)/);
          return { x: m?+m[1]:null, y: m?+m[2]:null, z: m?+m[3]:null,
                   done: s.indexOf('all-axes-ok')>=0,
                   failed: s.indexOf('failed')>=0 }; })()
      );
      magShowResult(msg.ok, msg.message || msg.error || '');
      magLog(msg.ok ? 'DONE.' : 'FINISHED (with errors).');
      refreshMagStatus();
      if (magEventSource) { magEventSource.close(); magEventSource = null; }
    }
  };

  magEventSource.onerror = () => {
    magLog('stream closed.');
    document.getElementById('mag_start_btn').disabled = false;
    if (magEventSource) { magEventSource.close(); magEventSource = null; }
  };
}

document.getElementById('mag_open').onclick = openMagnetoWizard;
document.getElementById('mag_close_btn').onclick = closeMagnetoWizard;
document.getElementById('mag_start_btn').onclick = runMagnetoWizard;
document.getElementById('mag_retry_btn').onclick = runMagnetoWizard;
magModal.addEventListener('click', (e)=>{ if (e.target === magModal) closeMagnetoWizard(); });

// Refresh magneto status in the Anafi panel every 10s (lightweight).
setInterval(refreshMagStatus, 10000);
refreshMagStatus();

// --- Mission Planner ---
let missionRunning = false;
let missionAbort = false;

function missionLog(msg) {
  const el = document.getElementById('mission_log');
  el.textContent += msg + '\\n';
  el.scrollTop = el.scrollHeight;
}

async function runMission() {
  if (missionRunning) return;
  const textarea = document.getElementById('mission_cmds');
  const lines = textarea.value.split('\\n').map(l=>l.trim()).filter(l=>l && !l.startsWith('#'));
  if (!lines.length) { missionLog('No commands to run.'); return; }

  missionRunning = true;
  missionAbort = false;
  document.getElementById('mission_run').style.display = 'none';
  document.getElementById('mission_stop').style.display = '';
  document.getElementById('mission_status').textContent = 'running...';
  document.getElementById('mission_status').style.color = '#22c55e';
  document.getElementById('mission_log').textContent = '';

  for (let i = 0; i < lines.length; i++) {
    if (missionAbort) { missionLog('ABORTED'); break; }
    const line = lines[i];
    document.getElementById('mission_status').textContent = `step ${i+1}/${lines.length}: ${line}`;
    missionLog(`> ${line}`);

    const parts = line.toLowerCase().split(/\\s+/);
    let ok = false;
    let result = '';

    try {
      if (parts[0] === 'takeoff') {
        const r = await post('/proxy/takeoff', {});
        result = 'takeoff sent';
        ok = true;
        await sleep(3000);
      } else if (parts[0] === 'land') {
        const r = await post('/proxy/land', {});
        result = 'land sent';
        ok = true;
        await sleep(3000);
      } else if (parts[0] === 'wait' || parts[0] === 'hover' || parts[0] === 'sleep') {
        const secs = parseFloat(parts[1]) || 2;
        result = `waiting ${secs}s`;
        ok = true;
        await sleep(secs * 1000);
      } else if (parts[0] === 'emergency') {
        await post('/proxy/emergency', {});
        result = 'emergency sent';
        ok = true;
        missionAbort = true;
      } else {
        // Parse as: <value> <direction> or <direction> <value>
        let val = parseFloat(parts[0]);
        let dir = parts[1];
        if (isNaN(val)) {
          dir = parts[0];
          val = parseFloat(parts[1]) || 30;
        }
        // Map directions
        const moveMap = {forward:1, fwd:1, back:1, backward:1, left:1, right:1, up:1, down:1};
        const rotateMap = {cw:1, ccw:1, clockwise:'cw', counterclockwise:'ccw', turn:'cw'};
        if (dir === 'backward') dir = 'back';
        if (dir === 'fwd') dir = 'forward';

        if (moveMap[dir]) {
          const r = await fetch('/proxy/move', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({dir:dir, cm:Math.round(val)})});
          const d = await r.json();
          ok = d.ok;
          result = ok ? `moved ${dir} ${Math.round(val)}cm` : (d.error || 'failed');
          await sleep(1500);
        } else if (rotateMap[dir]) {
          const realDir = typeof rotateMap[dir] === 'string' ? rotateMap[dir] : dir;
          const r = await fetch('/proxy/rotate', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({dir:realDir, deg:Math.round(val)})});
          const d = await r.json();
          ok = d.ok;
          result = ok ? `rotated ${realDir} ${Math.round(val)}°` : (d.error || 'failed');
          await sleep(1500);
        } else {
          result = `unknown command: ${line}`;
        }
      }
    } catch(e) {
      result = `error: ${e.message}`;
    }
    missionLog(ok ? `  OK: ${result}` : `  FAIL: ${result}`);
  }

  missionRunning = false;
  missionAbort = false;
  document.getElementById('mission_run').style.display = '';
  document.getElementById('mission_stop').style.display = 'none';
  document.getElementById('mission_status').textContent = 'done';
  document.getElementById('mission_status').style.color = '#94a3b8';
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

document.getElementById('mission_run').onclick = runMission;
document.getElementById('mission_stop').onclick = () => {
  missionAbort = true;
  missionLog('Abort requested...');
};

// --- Drone Config Editor ---
const modal = document.getElementById('drone_config_modal');
let editDrones = {};

function openDroneConfig() {
  editDrones = JSON.parse(JSON.stringify(drones));
  renderConfigFields();
  modal.style.display = 'flex';
}

function renderConfigFields() {
  const container = document.getElementById('drone_config_fields');
  container.innerHTML = '';
  const ids = Object.keys(editDrones).sort();
  ids.forEach(id => {
    const d = editDrones[id];
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:6px;margin-bottom:8px;align-items:center;';
    row.innerHTML = `
      <span class="small" style="min-width:30px;color:#94a3b8;">#${id}</span>
      <input type="text" value="${d.name}" placeholder="Name" data-id="${id}" data-field="name"
        style="width:120px;padding:4px 6px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:3px;font-size:12px;" />
      <select data-id="${id}" data-field="type"
        style="padding:4px 6px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:3px;font-size:12px;">
        <option value="tello" ${d.type==='tello'?'selected':''}>Tello</option>
        <option value="anafi" ${d.type==='anafi'?'selected':''}>Anafi</option>
      </select>
      <input type="text" value="${d.base}" placeholder="http://IP:port" data-id="${id}" data-field="base"
        style="flex:1;min-width:200px;padding:4px 6px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:3px;font-size:12px;font-family:monospace;" />
      <button onclick="deleteConfigDrone('${id}')" style="padding:2px 8px;background:#7f1d1d;border-color:#dc2626;font-size:11px;">X</button>
    `;
    container.appendChild(row);
  });
  // Bind change handlers
  container.querySelectorAll('input,select').forEach(el => {
    el.addEventListener('change', () => {
      const id = el.dataset.id, field = el.dataset.field;
      if (id && field && editDrones[id]) editDrones[id][field] = el.value;
    });
    el.addEventListener('input', () => {
      const id = el.dataset.id, field = el.dataset.field;
      if (id && field && editDrones[id]) editDrones[id][field] = el.value;
    });
  });
}

function deleteConfigDrone(id) {
  delete editDrones[id];
  renderConfigFields();
}

document.getElementById('drone_config_add').onclick = () => {
  const ids = Object.keys(editDrones).map(Number).filter(n=>!isNaN(n));
  const newId = String((ids.length ? Math.max(...ids) : 0) + 1);
  editDrones[newId] = {name: `Drone ${newId}`, type: 'tello', base: 'http://192.168.1.100:8080'};
  renderConfigFields();
};

document.getElementById('drone_config_save').onclick = async () => {
  // Sync fields from DOM
  document.querySelectorAll('#drone_config_fields input, #drone_config_fields select').forEach(el => {
    const id = el.dataset.id, field = el.dataset.field;
    if (id && field && editDrones[id]) editDrones[id][field] = el.value;
  });
  const st = document.getElementById('drone_config_status');
  try {
    const r = await fetch('/proxy/drones/config', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({drones: editDrones})
    });
    const d = await r.json();
    if (d.ok) {
      drones = d.drones;
      st.textContent = 'Saved!';
      st.style.color = '#22c55e';
      renderDroneBar();
      updatePiLabel();
      setTimeout(()=>{ modal.style.display='none'; st.textContent=''; }, 800);
    } else {
      st.textContent = d.error || 'Save failed';
      st.style.color = '#ef4444';
    }
  } catch(e) {
    st.textContent = 'Error: ' + e.message;
    st.style.color = '#ef4444';
  }
};

document.getElementById('drone_config_cancel').onclick = () => { modal.style.display = 'none'; };
document.getElementById('edit_drones_btn').onclick = openDroneConfig;
modal.onclick = (e) => { if (e.target === modal) modal.style.display = 'none'; };

// ── Position Tracker ─────────────────────────────────────────────────────────
let posEvtSource = null;
let posVideoOn = false;

const arenaCanvas = document.getElementById('arena_canvas');
const arenaCtx = arenaCanvas.getContext('2d');

// World dimensions — updated when arena config loads
let arenaW = 20, arenaD = 10.8, arenaOX = -10, arenaOY = 0;
// Tight 1 m border around the arena — the arena fills most of the view,
// and out-of-bounds positions beyond 1 m get clamped with an OOB arrow.
const VIEW_MARGIN = 1;  // metres
let viewOX = -11, viewOY = -1, viewW = 22, viewD = 12.8;

function _updateView() {
  viewOX = arenaOX - VIEW_MARGIN;
  viewOY = arenaOY - VIEW_MARGIN;
  viewW  = arenaW + 2 * VIEW_MARGIN;
  viewD  = arenaD + 2 * VIEW_MARGIN;
}

const ARENA_PAD = 30;  // pixel padding for axis labels

function arenaToCanvas(ax, ay) {
  const W = arenaCanvas.width, H = arenaCanvas.height;
  const iw = W - 2 * ARENA_PAD, ih = H - 2 * ARENA_PAD;
  return [
    ARENA_PAD + (ax - viewOX) / viewW * iw,
    H - ARENA_PAD - (ay - viewOY) / viewD * ih,
  ];
}

const WALL_COLOR = { front:'#6366f1', back:'#a855f7', left:'#06b6d4', right:'#10b981' };
let _seenMarkers = new Set();   // IDs currently visible in the drone camera (as strings)
let _refMarkers  = new Set();   // IDs actually used as world-frame refs this frame

// Track last drawn position so arena config reload doesn't erase the dot
let _lastPos = null, _lastCompPos = null, _lastDir = null, _lastFrameRes = null;

function drawArena(pos, compPos, dir) {
  // Persist last known position so config reloads don't blank it
  if (pos !== undefined) { _lastPos = pos; _lastCompPos = compPos; _lastDir = dir; window._lastPos = pos; }
  const _pos = _lastPos, _compPos = _lastCompPos, _dir = _lastDir;

  const ctx = arenaCtx;
  const W = arenaCanvas.width, H = arenaCanvas.height;
  const PAD = ARENA_PAD;

  // Background
  ctx.fillStyle = '#0f172a'; ctx.fillRect(0, 0, W, H);

  // Compute arena sub-rect in canvas pixels
  const [ax0, ay0] = arenaToCanvas(arenaOX, arenaOY + arenaD);
  const [ax1, ay1] = arenaToCanvas(arenaOX + arenaW, arenaOY);

  // Dim margin zone outside arena, brighter inside
  ctx.fillStyle = 'rgba(0,0,0,0.4)'; ctx.fillRect(PAD, PAD, W - 2*PAD, H - 2*PAD);
  ctx.fillStyle = '#0f172a'; ctx.fillRect(ax0, ay0, ax1 - ax0, ay1 - ay0);

  // ── Team-zone shading (SDC26: red home zone LEFT, blue home zone RIGHT,
  //    neutral middle). Splits the 20 m length into three equal thirds.
  (function shadeZones() {
    const thirdW = (arenaW) / 3;
    // Red zone: x ∈ [arenaOX, arenaOX + thirdW]
    const [rx0] = arenaToCanvas(arenaOX, arenaOY);
    const [rx1] = arenaToCanvas(arenaOX + thirdW, arenaOY);
    ctx.fillStyle = 'rgba(239,68,68,0.06)';
    ctx.fillRect(rx0, ay0, rx1 - rx0, ay1 - ay0);
    // Blue zone: x ∈ [arenaOX + 2·thirdW, arenaOX + arenaW]
    const [bx0] = arenaToCanvas(arenaOX + 2 * thirdW, arenaOY);
    const [bx1] = arenaToCanvas(arenaOX + arenaW, arenaOY);
    ctx.fillStyle = 'rgba(59,130,246,0.06)';
    ctx.fillRect(bx0, ay0, bx1 - bx0, ay1 - ay0);
  })();

  // Grid lines — fine 1 m grid (subtle) + major 5 m grid (brighter)
  // so the user can read position to the metre at a glance.
  const minorStroke = '#17243b';
  const majorStroke = '#1e3a5f';
  ctx.lineWidth = 1;
  for (let gx = Math.ceil(viewOX); gx <= viewOX + viewW + 0.01; gx += 1) {
    const [cx] = arenaToCanvas(gx, viewOY);
    ctx.strokeStyle = (gx % 5 === 0) ? majorStroke : minorStroke;
    ctx.beginPath(); ctx.moveTo(cx, PAD); ctx.lineTo(cx, H - PAD); ctx.stroke();
  }
  for (let gy = Math.ceil(viewOY); gy <= viewOY + viewD + 0.01; gy += 1) {
    const [, cy] = arenaToCanvas(viewOX, gy);
    ctx.strokeStyle = (gy % 5 === 0) ? majorStroke : minorStroke;
    ctx.beginPath(); ctx.moveTo(PAD, cy); ctx.lineTo(W - PAD, cy); ctx.stroke();
  }

  // Arena border (highlighted)
  ctx.strokeStyle = '#475569'; ctx.lineWidth = 2;
  ctx.strokeRect(ax0, ay0, ax1 - ax0, ay1 - ay0);

  // ── Safety margin rectangle ────────────────────────────────────
  // Draws the Pi-side arena guard's inner boundary as a dashed red
  // outline and shades the restricted zone (between the outline and
  // the arena wall) with a faint red tint. The drone is not allowed
  // to cross the inner boundary during autonomous flight OR manual
  // flight (when the guard is ON). Value comes from the last
  // /proxy/config/arena_safety poll, cached in window._arenaSafety.
  (function drawSafetyMargin() {
    const sa = window._arenaSafety || {};
    const margin = (typeof sa.margin_m === 'number' && sa.margin_m > 0)
                     ? sa.margin_m : null;
    if (margin == null) return;
    // Inner boundary rectangle — arena bounds minus margin.
    const [sx0, sy0] = arenaToCanvas(arenaOX + margin, arenaOY + arenaD - margin);
    const [sx1, sy1] = arenaToCanvas(arenaOX + arenaW - margin, arenaOY + margin);
    // Faint red shading in the restricted band (between margin and wall).
    // We paint four rectangles: top/bottom/left/right edges.
    ctx.fillStyle = 'rgba(239,68,68,0.08)';
    ctx.fillRect(ax0, ay0, ax1 - ax0, sy0 - ay0);                  // top (y=max zone)
    ctx.fillRect(ax0, sy1, ax1 - ax0, ay1 - sy1);                  // bottom
    ctx.fillRect(ax0, sy0, sx0 - ax0, sy1 - sy0);                  // left
    ctx.fillRect(sx1, sy0, ax1 - sx1, sy1 - sy0);                  // right
    // Dashed inner boundary — brighter if the guard is currently engaged.
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = sa.engaged ? 2 : 1.5;
    ctx.strokeStyle = sa.engaged ? '#ef4444' : '#f87171';
    ctx.strokeRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
    ctx.restore();
    // Tiny label so operators know what the dashed rect means
    ctx.font = '9px monospace';
    ctx.fillStyle = '#fca5a5';
    ctx.textAlign = 'left';
    ctx.fillText('safe @ -' + margin.toFixed(1) + 'm', sx0 + 4, sy0 + 11);
  })();

  // Outer view border
  ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1;
  ctx.strokeRect(PAD, PAD, W - 2*PAD, H - 2*PAD);

  // Axis labels every 5 m (brighter inside arena range)
  ctx.font = '9px monospace'; ctx.textAlign = 'center';
  for (let gx = Math.ceil(viewOX / 5) * 5; gx <= viewOX + viewW + 0.01; gx += 5) {
    const [cx] = arenaToCanvas(gx, viewOY);
    ctx.fillStyle = (gx >= arenaOX && gx <= arenaOX + arenaW) ? '#64748b' : '#334155';
    ctx.fillText(gx, cx, H - 2);
  }
  for (let gy = Math.ceil(viewOY / 5) * 5; gy <= viewOY + viewD + 0.01; gy += 5) {
    const [, cy] = arenaToCanvas(viewOX, gy);
    ctx.textAlign = 'right';
    ctx.fillStyle = (gy >= arenaOY && gy <= arenaOY + arenaD) ? '#64748b' : '#334155';
    ctx.fillText(gy, PAD - 3, cy + 3);
  }

  // Wall labels inside arena
  const midAx = (ax0 + ax1) / 2, midAy = (ay0 + ay1) / 2;
  ctx.fillStyle = '#64748b'; ctx.font = 'bold 9px monospace'; ctx.textAlign = 'center';
  ctx.fillText('BACK',  midAx, ay0 + 11);
  ctx.fillText('FRONT', midAx, ay1 - 4);
  ctx.save(); ctx.translate(ax0 + 10, midAy); ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center'; ctx.fillText('LEFT', 0, 0); ctx.restore();
  ctx.save(); ctx.translate(ax1 - 10, midAy); ctx.rotate(Math.PI / 2);
  ctx.textAlign = 'center'; ctx.fillText('RIGHT', 0, 0); ctx.restore();

  // Arena markers — colored square + bold ID pill placed away from the wall
  // Currently-visible markers get a bright halo; markers used as refs get a
  // brighter border; unseen markers are dimmed to ~40% for visual separation.
  ctx.font = 'bold 11px monospace';
  ctx.textBaseline = 'middle';

  // ── Bucket markers that share the same (x,y) top-down projection ──
  // The SDC arena uses stacked pairs (low at z≈2 m, high at z≈4 m) on
  // every wall. On the 2D top-down they collapse to identical pixels.
  // Drawing them individually means the label of the later-iterated ID
  // hides the first. We group by rounded canvas coordinate and render
  // a single combined glyph with a multi-ID label ("1/2") where each
  // individual ID is coloured by its own seen/ref state.
  const _bucketKey = (x, y) => Math.round(x) + ',' + Math.round(y);
  const buckets = new Map();
  for (const [id, m] of Object.entries(arenaMarkers)) {
    if (!m.pos) continue;
    const [mx, my] = arenaToCanvas(m.pos[0], m.pos[1]);
    if (mx < PAD - 4 || mx > W - PAD + 4 || my < PAD - 4 || my > H - PAD + 4) continue;
    const key = _bucketKey(mx, my);
    let b = buckets.get(key);
    if (!b) {
      b = {cx: mx, cy: my, wall: m.wall, entries: []};
      buckets.set(key, b);
    }
    b.entries.push({
      id:   String(id),
      seen: _seenMarkers.has(String(id)),
      ref:  _refMarkers.has(String(id)),
      z:    (m.pos[2] != null) ? Number(m.pos[2]) : 0,
    });
  }

  for (const b of buckets.values()) {
    // Sort HIGH altitude first so the top row of the label is the
    // upper marker (matches the physical stacking on the wall: low ID
    // paints low, high ID paints high in the same pixel column).
    b.entries.sort((a, b) => b.z - a.z);
    const mx = b.cx, my = b.cy;
    const anySeen = b.entries.some(e => e.seen);
    const anyRef  = b.entries.some(e => e.ref);
    const baseColor = WALL_COLOR[b.wall] || '#94a3b8';

    // Halo whenever ANY marker in the stack is seen.
    if (anySeen) {
      ctx.beginPath();
      ctx.arc(mx, my, 14, 0, Math.PI * 2);
      ctx.fillStyle = (anyRef ? 'rgba(34,197,94,0.35)' : 'rgba(251,191,36,0.35)');
      ctx.fill();
      ctx.beginPath();
      ctx.arc(mx, my, 14, 0, Math.PI * 2);
      ctx.strokeStyle = (anyRef ? '#22c55e' : '#fbbf24');
      ctx.lineWidth = 1.5; ctx.stroke();
    }

    // Square marker glyph — same shape as before; brighter border if
    // any of the stacked IDs is currently seen.
    ctx.fillStyle = baseColor;
    ctx.fillRect(mx - 5, my - 5, 10, 10);
    ctx.strokeStyle = anySeen ? '#ffffff' : 'rgba(15,23,42,0.9)';
    ctx.lineWidth = anySeen ? 1.5 : 1;
    ctx.strokeRect(mx - 5.5, my - 5.5, 11, 11);

    // Vertical label stack — one row per ID. Upper-altitude marker sits
    // on the top row; lower marker on the bottom. This mirrors the
    // physical layout of the stacked wall pairs (high above low).
    const LINE_H = 13;
    const pillW  = Math.max(...b.entries.map(e => ctx.measureText(e.id).width)) + 10;
    const pillH  = b.entries.length * LINE_H + 4;

    // Side opposite the BACK/FRONT/LEFT/RIGHT text so the pill doesn't
    // collide with it. Offset includes pillH now since the pill is taller.
    const wall = (b.wall || '').toLowerCase();
    let labelX, labelY;
    if      (wall === 'front') { labelX = mx;                      labelY = my - pillH / 2 - 8; }
    else if (wall === 'back')  { labelX = mx;                      labelY = my + pillH / 2 + 8; }
    else if (wall === 'left')  { labelX = mx + 9 + pillW / 2;      labelY = my; }
    else if (wall === 'right') { labelX = mx - 9 - pillW / 2;      labelY = my; }
    else                        { labelX = mx;                      labelY = my - pillH / 2 - 8; }

    // Pill background + border (brighter border when anything seen)
    ctx.fillStyle = 'rgba(15,23,42,0.9)';
    ctx.fillRect(labelX - pillW / 2, labelY - pillH / 2, pillW, pillH);
    ctx.strokeStyle = baseColor;
    ctx.lineWidth = anySeen ? 1.5 : 1;
    ctx.strokeRect(labelX - pillW / 2 + 0.5, labelY - pillH / 2 + 0.5, pillW - 1, pillH - 1);

    // Render each ID on its own row; seen → bold white, unseen → wall
    // colour. Thin separator line between rows makes the pair visually
    // obvious when both IDs are unseen (same colour otherwise).
    ctx.textAlign = 'center';
    b.entries.forEach((e, i) => {
      const rowY = labelY - pillH / 2 + (i + 0.5) * LINE_H + 2;
      ctx.fillStyle = e.seen ? '#ffffff' : baseColor;
      ctx.font      = e.seen ? 'bold 11px monospace' : '11px monospace';
      ctx.fillText(e.id, labelX, rowY);
      if (i < b.entries.length - 1) {
        ctx.strokeStyle = 'rgba(100,116,139,0.5)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(labelX - pillW / 2 + 2, labelY - pillH / 2 + (i + 1) * LINE_H + 2);
        ctx.lineTo(labelX + pillW / 2 - 2, labelY - pillH / 2 + (i + 1) * LINE_H + 2);
        ctx.stroke();
      }
    });
    ctx.font = 'bold 11px monospace';   // restore default for any later caller
  }
  ctx.textBaseline = 'alphabetic';  // restore default so downstream drawing is unaffected

  // Debug overlay — large text, dark background box, drawn over everything
  const dbgLines = [
    `view: [${viewOX},${viewOX+viewW}] x [${viewOY},${viewOY+viewD}]`,
    `pos: ${_pos ? `${_pos[0].toFixed(2)}, ${_pos[1].toFixed(2)}, ${_pos[2].toFixed(2)}` : 'null'}`,
    `frame: ${_lastFrameRes || 'unknown'}`,
  ];
  ctx.font = 'bold 12px monospace';
  const dbgW = Math.max(...dbgLines.map(l => ctx.measureText(l).width)) + 10;
  ctx.fillStyle = 'rgba(0,0,0,0.75)'; ctx.fillRect(PAD, PAD, dbgW, dbgLines.length * 16 + 6);
  ctx.fillStyle = '#22d3ee'; ctx.textAlign = 'left';
  dbgLines.forEach((l, i) => ctx.fillText(l, PAD + 5, PAD + 15 + i * 16));

  // ── Target boxes (from the capture-targets mission config) ──
  // Drawn on the floor as a labelled square with a diamond marker.
  // Colour indicates capture state when a mission is running:
  //   grey   = not yet visited
  //   yellow = currently claimed by a drone
  //   green  = captured
  const boxes = window._targetBoxes;
  if (Array.isArray(boxes) && boxes.length) {
    const claimed = (window._missionClaimedBoxes || {});
    const captured = new Set(window._missionCapturedBoxes || []);
    boxes.forEach((b, i) => {
      if (b == null || b.x == null || b.y == null) return;
      const [bx, by] = arenaToCanvas(Number(b.x), Number(b.y));
      const idx = (typeof b.idx === 'number') ? b.idx : i;
      const isCap = captured.has(idx);
      const isClaimed = Object.values(claimed).includes(idx);
      const team = (b.home_team || '').toLowerCase();
      // Default colour is the team the box BELONGS to (red/blue start colour
      // from the SDC26 rules). While being approached by us → yellow.
      // After capture → green (ours).
      const fill = isCap     ? 'rgba(34,197,94,0.55)'
                 : isClaimed ? 'rgba(250,204,21,0.55)'
                 : team === 'red'  ? 'rgba(239,68,68,0.55)'
                 : team === 'blue' ? 'rgba(59,130,246,0.55)'
                                   : 'rgba(148,163,184,0.45)';
      const stroke = isCap     ? '#16a34a'
                   : isClaimed ? '#eab308'
                   : team === 'red'  ? '#dc2626'
                   : team === 'blue' ? '#2563eb'
                                     : '#64748b';
      // Box body: 20 px diamond-ish square
      const s = 18;
      ctx.save();
      ctx.translate(bx, by);
      ctx.rotate(Math.PI / 4);
      ctx.fillStyle = fill;
      ctx.fillRect(-s/2, -s/2, s, s);
      ctx.strokeStyle = stroke; ctx.lineWidth = 2;
      ctx.strokeRect(-s/2, -s/2, s, s);
      ctx.restore();
      // Label: box id + status
      const label = `#${b.id ?? idx+1}` +
                    (isCap ? ' ✓' : (isClaimed ? ' ⏱' : ''));
      ctx.font = 'bold 11px monospace'; ctx.textAlign = 'center';
      ctx.fillStyle = 'rgba(0,0,0,0.75)';
      const tw = ctx.measureText(label).width;
      ctx.fillRect(bx - tw/2 - 3, by + 14, tw + 6, 13);
      ctx.fillStyle = stroke;
      ctx.fillText(label, bx, by + 24);
    });
  }

  // Drone positions
  if (!_pos) return;

  // Clamp canvas coords to inner area; returns [cx, cy, wasOutOfBounds, rawPx, rawPy]
  const M = PAD + 8;
  const cp = _compPos || _pos;
  const [rpx, rpy] = arenaToCanvas(cp[0], cp[1]);
  const cx2 = Math.max(M, Math.min(W - M, rpx));
  const cy2 = Math.max(M, Math.min(H - M, rpy));
  const outOfBounds = rpx !== cx2 || rpy !== cy2;

  const pxDbg = `px: raw(${rpx.toFixed(0)},${rpy.toFixed(0)}) clamped(${cx2.toFixed(0)},${cy2.toFixed(0)}) oob:${outOfBounds}`;
  console.log('[POS]', pxDbg);
  ctx.font = 'bold 12px monospace'; ctx.textAlign = 'left';
  const pdW = ctx.measureText(pxDbg).width + 10;
  ctx.fillStyle = 'rgba(0,0,0,0.75)'; ctx.fillRect(PAD, PAD + 56, pdW, 20);
  ctx.fillStyle = '#22d3ee'; ctx.fillText(pxDbg, PAD + 5, PAD + 70);

  const dotColor = outOfBounds ? '#ef4444' : '#f97316';

  // Outer glow ring (always drawn, big and visible)
  ctx.beginPath(); ctx.arc(cx2, cy2, 18, 0, Math.PI * 2);
  ctx.strokeStyle = outOfBounds ? 'rgba(239,68,68,0.5)' : 'rgba(249,115,22,0.4)';
  ctx.lineWidth = 4; ctx.stroke();

  // Solid filled dot
  ctx.beginPath(); ctx.arc(cx2, cy2, 10, 0, Math.PI * 2);
  ctx.fillStyle = dotColor; ctx.fill();
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.stroke();

  // Crosshair
  ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(cx2 - 20, cy2); ctx.lineTo(cx2 + 20, cy2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx2, cy2 - 20); ctx.lineTo(cx2, cy2 + 20); ctx.stroke();

  // Arrow toward true out-of-bounds position
  if (outOfBounds) {
    const ang = Math.atan2(rpy - cy2, rpx - cx2);
    ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 3; ctx.lineCap = 'round';
    const ax = cx2 + Math.cos(ang) * 24, ay = cy2 + Math.sin(ang) * 24;
    ctx.beginPath(); ctx.moveTo(cx2 + Math.cos(ang) * 12, cy2 + Math.sin(ang) * 12);
    ctx.lineTo(ax, ay); ctx.stroke();
    // arrowhead
    const perp = ang + Math.PI / 2;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - Math.cos(ang)*6 + Math.cos(perp)*4, ay - Math.sin(ang)*6 + Math.sin(perp)*4);
    ctx.lineTo(ax - Math.cos(ang)*6 - Math.cos(perp)*4, ay - Math.sin(ang)*6 - Math.sin(perp)*4);
    ctx.closePath(); ctx.fillStyle = '#ef4444'; ctx.fill();
    ctx.lineCap = 'butt';
  }

  // Heading arrow
  if (_dir) {
    const ang = Math.atan2(_dir[0], _dir[1]);
    ctx.strokeStyle = '#facc15'; ctx.lineWidth = 2.5; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(cx2, cy2);
    ctx.lineTo(cx2 + Math.sin(ang) * 28, cy2 - Math.cos(ang) * 28); ctx.stroke();
    ctx.lineCap = 'butt';
  }

  // Coordinate label — "OUT" prefix + background box for readability
  const label = (outOfBounds ? 'OUT ' : '') + `(${_pos[0].toFixed(1)},${_pos[1].toFixed(1)})`;
  ctx.font = 'bold 11px monospace';
  const lw = ctx.measureText(label).width;
  const lx = Math.min(cx2 + 14, W - lw - PAD - 4);
  const ly = cy2 - 6;
  ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(lx - 2, ly - 11, lw + 4, 14);
  ctx.fillStyle = dotColor; ctx.textAlign = 'left';
  ctx.fillText(label, lx, ly);

  // ── Multi-drone overlay (when checkbox enabled) ──────────────────
  // Draw every OTHER drone in the fleet in a distinct colour so the
  // C2 operator can see the whole swarm at once.
  const showAll = document.getElementById('arena_show_all_drones');
  if (showAll && showAll.checked && window._fleetObservers) {
    const FLEET_COLORS = ['#38bdf8', '#a78bfa', '#f472b6', '#fbbf24', '#34d399'];
    let idx = 0;
    for (const [did, st] of Object.entries(window._fleetObservers)) {
      if (did === (window.activeDroneId || activeDroneId)) { idx++; continue; }
      const p = st && (st.pos || st.cam);
      if (!p || p.length < 2) { idx++; continue; }
      const col = FLEET_COLORS[idx % FLEET_COLORS.length];
      const [rx, ry] = arenaToCanvas(p[0], p[1]);
      const cxN = Math.max(M, Math.min(W - M, rx));
      const cyN = Math.max(M, Math.min(H - M, ry));
      const oob = rx !== cxN || ry !== cyN;
      // Smaller dot for non-active drones
      ctx.beginPath(); ctx.arc(cxN, cyN, 7, 0, Math.PI*2);
      ctx.fillStyle = col; ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 1.5; ctx.stroke();
      // heading tick
      const yawDeg = st.yaw;
      if (typeof yawDeg === 'number') {
        const a = yawDeg * Math.PI / 180;
        ctx.strokeStyle = col; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(cxN, cyN);
        ctx.lineTo(cxN + Math.sin(a) * 16, cyN - Math.cos(a) * 16);
        ctx.stroke();
      }
      // label with drone name + position
      ctx.font = 'bold 10px monospace';
      const name = (window._fleetNames && window._fleetNames[did]) || did;
      const tag = `${name} (${p[0].toFixed(1)},${p[1].toFixed(1)})${oob ? ' OOB':''}`;
      const tw = ctx.measureText(tag).width;
      const tx = Math.min(cxN + 10, W - tw - PAD - 4);
      const ty = cyN - 9;
      ctx.fillStyle = 'rgba(0,0,0,0.65)'; ctx.fillRect(tx - 2, ty - 10, tw + 4, 13);
      ctx.fillStyle = col; ctx.textAlign = 'left';
      ctx.fillText(tag, tx, ty);
      idx++;
    }
  }
}

// ── Fleet-wide position poll (drives multi-drone arena view) ──
window._fleetObservers = {};
window._fleetNames = {};
async function fleetPoll() {
  try {
    const r = await fetch('/proxy/aruco/fleet', {cache:'no-store'});
    const d = await r.json();
    if (d && d.observers) {
      window._fleetObservers = d.observers;
      // Grab drone display names from the main drones dict if available
      if (typeof drones === 'object') {
        const names = {};
        for (const [did, info] of Object.entries(drones || {})) {
          names[did] = info?.name || did;
        }
        window._fleetNames = names;
      }
      // Feed positions into the Three.js scene if active
      if (window._arena3d && window._arena3d.updateDrones) {
        window._arena3d.updateDrones(d.observers);
        if (window._arena3d.syncTargetBoxes) window._arena3d.syncTargetBoxes();
        if (window._arena3d.updateDronePositionHUD) {
          window._arena3d.updateDronePositionHUD(d.observers);
        }
        // Aggregate which markers are currently visible / used as refs
        // across the whole fleet so the 3D highlight matches the 2D
        // halo. Prefer the position service's seen_markers field; fall
        // back to ref_markers-only when the observer isn't running.
        if (window._arena3d.updateVisibleMarkers) {
          const seenAll = new Set();
          const refAll  = new Set();
          for (const st of Object.values(d.observers || {})) {
            (st.seen_markers || []).forEach(m => seenAll.add(String(m)));
            (st.ref_markers  || []).forEach(m => {
              refAll.add(String(m));
              seenAll.add(String(m));   // ref implies seen
            });
          }
          window._arena3d.updateVisibleMarkers(
            Array.from(seenAll), Array.from(refAll));
        }
      }
    }
  } catch {}
}
setInterval(fleetPoll, 1000);   // was 500ms/2Hz → 1Hz; fleet poll fans out to all drones, heaviest endpoint
fleetPoll();

function updatePosUI(d) {
  const pos = d.pos;
  const vel = d.vel || [0, 0, 0];
  const lat = (d.latency_ms || 0) / 1000;
  const compPos = pos ? pos.map((v, i) => v + (vel[i] || 0) * lat) : null;

  document.getElementById('pos_x').textContent = pos ? pos[0].toFixed(2) : '\\u2014';
  document.getElementById('pos_y').textContent = pos ? pos[1].toFixed(2) : '\\u2014';
  document.getElementById('pos_z').textContent = pos ? pos[2].toFixed(2) : '\\u2014';

  const dir = d.dir;
  const hdg = dir ? ((Math.atan2(dir[0], dir[1]) * 180 / Math.PI + 360) % 360).toFixed(1) : '\\u2014';
  document.getElementById('pos_hdg').textContent = hdg;
  const spd = vel ? Math.sqrt((vel[0]||0)**2 + (vel[1]||0)**2).toFixed(2) : '\\u2014';
  document.getElementById('pos_vel').textContent = spd + ' m/s';
  document.getElementById('pos_refs').textContent = d.ref_markers ? d.ref_markers.length : '\\u2014';

  // ── Target Boxes table update ────────────────────────────────────
  // Resolves team/colour from the currently-loaded arena_cfg.target_teams
  // (ID-range list) + target_overrides (per-ID explicit). Targets carry
  // a 3 s TTL so a target briefly out of view doesn't flicker off; a
  // \"stale\" status flag is set when we haven't seen it in the latest
  // frame.
  (function updateTargetsPanel() {
    const tbody = document.getElementById('targets_tbody');
    const badge = document.getElementById('targets_badge');
    const table = document.getElementById('targets_table');
    const empty = document.getElementById('targets_empty');
    if (!tbody || !badge) return;
    const targets = d.targets || {};
    const ids = Object.keys(targets);
    // Resolve team+colour+box-number. SDC26 convention:
    //   - Marker ID = 10·(team_code) + box_number
    //   - First digit 3 → Blue team, first digit 4 → Red team
    //   - Box number = id % 10 (1..6)
    function teamFor(tid) {
      const n = Number(tid);
      const ac = window._arenaCfgCache || {};
      // Per-ID overrides always win
      for (const o of (ac.target_overrides || [])) {
        if (Number(o.id) === n) {
          return { team: o.team || 'Override', color: o.color || '#64748b',
                   box_num: (n % 10) || null };
        }
      }
      // Range mapping from arena config
      for (const r of (ac.target_teams || [])) {
        const rng = r.id_range || [0, 0];
        if (n >= Number(rng[0]) && n <= Number(rng[1])) {
          const box = n % 10;
          const label = r.team + (box >= 1 && box <= 9 ? ' Box ' + box : '');
          return { team: label, color: r.color || '#64748b', box_num: box };
        }
      }
      return { team: 'Unknown', color: '#64748b', box_num: null };
    }
    badge.textContent = ids.length + ' visible';
    badge.style.color = ids.length > 0 ? '#fbbf24' : '#64748b';
    if (ids.length === 0) {
      table.style.display = 'none';
      empty.style.display = '';
      return;
    }
    table.style.display = '';
    empty.style.display = 'none';
    // Sort by ID for stability
    ids.sort((a, b) => Number(a) - Number(b));
    const rows = ids.map(tid => {
      const t = targets[tid];
      const p = t.pos || [0, 0, 0];
      const age = t.age_s || 0;
      const fresh = !!t.fresh;
      const team = teamFor(tid);
      const statusHtml = fresh
        ? '<span style=\"color:#22c55e;\">● live</span>'
        : '<span style=\"color:#94a3b8;\">○ held ' + age.toFixed(1) + 's</span>';
      return (
        '<tr style=\"border-bottom:1px solid #1e293b;\">' +
        '<td style=\"padding:4px 6px;font-family:monospace;color:#94a3b8;\">' + tid + '</td>' +
        '<td style=\"padding:4px 6px;\">' +
          '<span style=\"display:inline-block;width:12px;height:12px;border-radius:2px;' +
          'background:' + team.color + ';vertical-align:middle;margin-right:6px;\"></span>' +
          '<span style=\"color:' + team.color + ';font-weight:600;\">' + team.team + '</span>' +
        '</td>' +
        '<td style=\"padding:4px 6px;font-family:monospace;\">' + p[0].toFixed(2) + '</td>' +
        '<td style=\"padding:4px 6px;font-family:monospace;\">' + p[1].toFixed(2) + '</td>' +
        '<td style=\"padding:4px 6px;font-family:monospace;\">' + p[2].toFixed(2) + '</td>' +
        '<td style=\"padding:4px 6px;font-family:monospace;color:#64748b;\">' + age.toFixed(1) + 's</td>' +
        '<td style=\"padding:4px 6px;\">' + statusHtml + '</td>' +
        '</tr>'
      );
    });
    tbody.innerHTML = rows.join('');
  })();

  // ── Safety distance readout ───────────────────────────────────────
  // How close is the drone to the nearest safety boundary? Green =
  // safe zone with margin; amber = within 30 cm of the boundary; red
  // = past the margin (i.e. inside the restricted band between the
  // dashed safety line and the arena wall).
  (function updateSafetyReadout() {
    const el = document.getElementById('pos_safety_readout');
    if (!el) return;
    const sa = window._arenaSafety || {};
    if (!sa.enabled) {
      el.textContent = 'guard OFF — operator owns boundaries';
      el.style.color = '#fbbf24';
      return;
    }
    if (sa.margin_m == null || !Array.isArray(pos) || pos.length < 2) {
      el.textContent = '—';
      el.style.color = '#64748b';
      return;
    }
    const m = sa.margin_m;
    const x = Number(pos[0]), y = Number(pos[1]);
    // Distance to nearest wall, then distance to the SAFE inner boundary.
    // Negative value = drone is inside the restricted band (past the margin).
    const dWall = Math.min(
      (arenaOX + arenaW) - x,      // right wall
      x - arenaOX,                  // left wall
      (arenaOY + arenaD) - y,      // back wall (y_max)
      y - arenaOY,                  // front wall (y_min)
    );
    const dSafe = dWall - m;        // metres from safe boundary
    const wallNames = {
      xr: (arenaOX + arenaW) - x,
      xl: x - arenaOX,
      yb: (arenaOY + arenaD) - y,
      yf: y - arenaOY,
    };
    let closest = 'right';
    let closestD = wallNames.xr;
    if (wallNames.xl < closestD) { closest = 'left';  closestD = wallNames.xl; }
    if (wallNames.yb < closestD) { closest = 'back';  closestD = wallNames.yb; }
    if (wallNames.yf < closestD) { closest = 'front'; closestD = wallNames.yf; }

    if (dSafe < 0) {
      // Inside restricted band — or worse, outside arena.
      el.textContent = '\u26a0 RESTRICTED — ' + Math.abs(dSafe).toFixed(2) +
                        ' m past ' + closest + ' margin  (wall ' +
                        closestD.toFixed(2) + ' m away)';
      el.style.color = '#ef4444';
    } else if (dSafe < 0.3) {
      el.textContent = '\u26a1 approaching ' + closest + ' — ' +
                        dSafe.toFixed(2) + ' m to margin  (' +
                        closestD.toFixed(2) + ' m to wall)';
      el.style.color = '#fbbf24';
    } else {
      el.textContent = '\u2713 safe — ' + dSafe.toFixed(2) +
                        ' m to ' + closest + ' margin  (' +
                        closestD.toFixed(2) + ' m to wall)';
      el.style.color = '#22c55e';
    }
  })();
  document.getElementById('pos_fps').textContent = d.fps != null ? d.fps : '\\u2014';
  document.getElementById('pos_stale').style.display = d.stale ? '' : 'none';

  const enabled = d.enabled !== false;
  const badge = document.getElementById('pos_status_badge');
  const dr = d.dead_reckoning;
  badge.textContent = !enabled ? 'disabled' : pos ? (dr ? 'IMU DR' : d.stale ? 'stale' : 'live') : 'no markers';
  badge.style.color = !enabled ? '#64748b' : (pos && !d.stale && !dr) ? '#22c55e' : dr ? '#06b6d4' : '#f59e0b';

  if (d.frame_w && d.frame_h) _lastFrameRes = `${d.frame_w}x${d.frame_h}`;
  // Stash visible + reference markers so drawArena can highlight them
  _seenMarkers = new Set((d.seen_markers || []).map(String));
  _refMarkers  = new Set((d.ref_markers  || []).map(String));
  if (pos) console.log('[POS] drawArena pos=', pos, 'compPos=', compPos, 'frame=', _lastFrameRes);
  drawArena(pos, compPos, dir);
}

function startPosEvents() {
  if (posEvtSource) { posEvtSource.close(); posEvtSource = null; }
  posEvtSource = new EventSource('/proxy/position/events');
  posEvtSource.onmessage = (e) => { try { updatePosUI(JSON.parse(e.data)); } catch(err) { console.error('POS SSE error:', err, e.data); } };
  posEvtSource.onerror = () => {
    posEvtSource.close(); posEvtSource = null;
    setTimeout(startPosEvents, 500);
  };
}

async function loadPosConfig() {
  try {
    const r = await fetch('/proxy/position/config');
    const d = await r.json();
    // Server returns {config:{...}, ...} but older endpoints are flat — handle both
    const c = d.config || d;
    document.getElementById('pos_enabled').checked = !!c.enabled;
    if (c.detect_profile) document.getElementById('pos_profile').value = c.detect_profile;
    if (c.fov_deg) document.getElementById('pos_fov').value = c.fov_deg;
    if (typeof c.imu_weight === 'number') {
      const pct = Math.round(c.imu_weight * 100);
      document.getElementById('pos_imu_weight').value = pct;
      document.getElementById('pos_imu_weight_val').textContent = pct + '%';
    }
    const latMs = (c.latency_ms != null) ? c.latency_ms
                  : (c.latency_comp_s != null ? Math.round(c.latency_comp_s * 1000) : null);
    if (latMs != null) {
      document.getElementById('pos_latency').value = latMs;
      document.getElementById('pos_latency_val').textContent = Math.round(latMs);
    }
    // ── Populate filter controls ──
    const kCb = document.getElementById('pos_kalman');
    if (kCb) kCb.checked = (c.enable_kalman_filter !== false);  // default ON if missing
    const mSz = document.getElementById('pos_marker_size');
    if (mSz && c.marker_size_m != null) mSz.value = Number(c.marker_size_m).toFixed(2);
    const tK = document.getElementById('pos_top_k');
    if (tK && c.top_k_markers != null) tK.value = c.top_k_markers;
    const out = document.getElementById('pos_outlier');
    if (out && c.outlier_reject_m != null) out.value = c.outlier_reject_m;
    const ds = document.getElementById('pos_distance_scale');
    if (ds && c.distance_scale != null) ds.value = Number(c.distance_scale).toFixed(3);
    // Sync the Auto Positioning master toggle — also re-locks/unlocks
    // the manual tuning sliders accordingly.
    if (typeof window._applyPosAutoUI === 'function') {
      window._applyPosAutoUI(c.auto_positioning !== false);
    }
    // ── Populate precision (advanced) controls ──
    const setIf = (id, val, formatter) => {
      const el = document.getElementById(id);
      if (el && val != null) el.value = formatter ? formatter(val) : val;
    };
    setIf('pos_pose_hold',  c.pose_hold_sec);
    setIf('pos_min_refs',   c.min_ref_count);
    setIf('pos_min_ref_w',  c.min_ref_weight);
    setIf('pos_blend_min',  c.meas_blend_min);
    setIf('pos_blend_max',  c.meas_blend_max);
    setIf('pos_vel_blend',  c.vel_blend);
    setIf('pos_max_dt',     c.max_state_dt);
    setIf('pos_kf_q',       c.kalman_process_var, (v)=>Number(v).toPrecision(3));
    setIf('pos_kf_r',       c.kalman_meas_var,    (v)=>Number(v).toPrecision(3));
    setIf('pos_max_jump',   c.max_pose_jump_m);
    setIf('pos_target_size', c.target_marker_size_m);
    setIf('pos_zupt_speed',  c.zupt_speed_m_s);
    setIf('pos_zupt_hold',   c.zupt_hold_frames);
    // IMU LPF slider + its live-label
    if (c.imu_lowpass_hz != null) {
      const slider = document.getElementById('pos_imu_lpf');
      const label  = document.getElementById('pos_imu_lpf_val');
      if (slider) slider.value = Number(c.imu_lowpass_hz).toFixed(1);
      if (label)  label.textContent = (Number(c.imu_lowpass_hz) > 0
                                        ? Number(c.imu_lowpass_hz).toFixed(1) + ' Hz'
                                        : 'OFF');
    }
    const cs = document.getElementById('pos_calib_status');
    cs.textContent = d.has_calibration ? '\\u2713 calibration loaded' : 'no calibration';
    cs.style.color = d.has_calibration ? '#22c55e' : '#94a3b8';
    if (c.enabled) startPosEvents();
  } catch {}
}

// ── Auto Positioning toggle ─────────────────────────────────────────
// Single master switch that forces the FC to use CLAUDE_AUTO_CONFIG
// instead of whatever the operator has tuned. While on, the manual
// slider/input controls below are disabled + visibly dimmed so the
// operator can't accidentally change a value that isn't being applied.
(function wireAutoPositioning(){
  const tgl = document.getElementById('pos_auto_toggle');
  const status = document.getElementById('pos_auto_status');
  if (!tgl) return;
  // Every ID of a control that belongs to the \"manual tuning\" surface.
  // When auto is ON, they're disabled. They stay visible for reference.
  const MANUAL_IDS = [
    'pos_profile','pos_fov','pos_imu_weight',
    'pos_kalman','pos_marker_size','pos_top_k','pos_outlier','pos_distance_scale',
    'pos_filters_apply','pos_filters_reset',
    'pos_pose_hold','pos_min_refs','pos_min_ref_w',
    'pos_blend_min','pos_blend_max','pos_vel_blend','pos_max_dt',
    'pos_kf_q','pos_kf_r','pos_imu_lpf',
    'pos_precision_apply','pos_precision_reset',
    'pos_preset_apply','pos_preset_name','pos_preset_save','pos_preset_delete','pos_preset_sel',
  ];
  function setManualEnabled(enabled) {
    for (const id of MANUAL_IDS) {
      const el = document.getElementById(id);
      if (!el) continue;
      el.disabled = !enabled;
      // Also visually dim the surrounding <label> containers so it's
      // obvious the whole row is frozen, not just the input field.
      const lbl = el.closest('label');
      if (lbl) lbl.style.opacity = enabled ? '1' : '0.45';
      el.style.opacity = enabled ? '1' : '0.55';
    }
  }
  function flash(msg, col) {
    if (!status) return;
    status.textContent = msg; status.style.color = col || '#86efac';
    setTimeout(() => { if (status.textContent === msg) status.textContent = ''; }, 2500);
  }
  window._applyPosAutoUI = function(auto) {
    tgl.checked = !!auto;
    setManualEnabled(!auto);
    if (status) status.textContent = auto
      ? 'active — manual tuning locked'
      : 'off — manual tuning active';
    if (status) status.style.color = auto ? '#86efac' : '#fbbf24';
  };
  tgl.addEventListener('change', async () => {
    const auto = tgl.checked;
    window._applyPosAutoUI(auto);
    try {
      const r = await fetch('/proxy/position/config', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({auto_positioning: auto}),
      });
      const d = await r.json();
      if (d.ok) {
        flash(auto ? '\\u2713 Claude preset applied' : '\\u2713 manual mode', '#86efac');
        if (typeof loadPosConfig === 'function') setTimeout(loadPosConfig, 300);
      } else {
        flash('\\u2717 ' + (d.error || 'apply failed'), '#ef4444');
      }
    } catch (e) {
      flash('\\u2717 ' + e, '#ef4444');
    }
  });
  // Apply initial UI state (will be overwritten by loadPosConfig once the
  // FC replies with the persisted auto_positioning value).
  window._applyPosAutoUI(true);
})();

// ── Filter controls — live-apply to running positioner ──
(function wireFilterControls(){
  const statusEl = () => document.getElementById('pos_filters_status');
  const flash = (msg, col) => {
    const e = statusEl(); if (!e) return;
    e.textContent = msg; e.style.color = col || '#64748b';
    setTimeout(() => { if (e.textContent === msg) e.textContent = ''; }, 2500);
  };
  const applyBtn = document.getElementById('pos_filters_apply');
  if (applyBtn) applyBtn.onclick = async () => {
    const dsEl = document.getElementById('pos_distance_scale');
    const payload = {
      enable_kalman_filter: document.getElementById('pos_kalman').checked,
      marker_size_m: parseFloat(document.getElementById('pos_marker_size').value),
      top_k_markers: parseInt(document.getElementById('pos_top_k').value, 10),
      outlier_reject_m: parseFloat(document.getElementById('pos_outlier').value),
      distance_scale: dsEl ? parseFloat(dsEl.value) : 1.0,
    };
    try {
      const r = await fetch('/proxy/position/config', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload),
      });
      const d = await r.json();
      if (d.ok) flash('\\u2713 applied', '#22c55e');
      else flash('error: ' + (d.error || 'unknown'), '#ef4444');
    } catch (e) { flash('request failed', '#ef4444'); }
  };
  const resetBtn = document.getElementById('pos_filters_reset');
  if (resetBtn) resetBtn.onclick = () => {
    document.getElementById('pos_kalman').checked = true;
    document.getElementById('pos_marker_size').value = '0.50';
    document.getElementById('pos_top_k').value = '0';
    document.getElementById('pos_outlier').value = '2.5';
    const ds = document.getElementById('pos_distance_scale');
    if (ds) ds.value = '1.000';
    flash('defaults loaded — click Apply', '#94a3b8');
  };
})();

// ── Precision (advanced) controls — pose hold, Kalman variances,
// measurement/velocity blend, min-refs. Live-apply through the same
// endpoint; no restart required. Defaults mirror ctrl_position.py.
(function wirePrecisionControls(){
  const flash = (msg, col) => {
    const e = document.getElementById('pos_precision_status');
    if (!e) return;
    e.textContent = msg; e.style.color = col || '#64748b';
    setTimeout(() => { if (e.textContent === msg) e.textContent = ''; }, 2500);
  };
  const applyBtn = document.getElementById('pos_precision_apply');
  if (applyBtn) applyBtn.onclick = async () => {
    const num = (id) => parseFloat(document.getElementById(id).value);
    const int = (id) => parseInt(document.getElementById(id).value, 10);
    const payload = {
      pose_hold_sec:       num('pos_pose_hold'),
      min_ref_count:       int('pos_min_refs'),
      min_ref_weight:      num('pos_min_ref_w'),
      meas_blend_min:      num('pos_blend_min'),
      meas_blend_max:      num('pos_blend_max'),
      vel_blend:           num('pos_vel_blend'),
      max_state_dt:        num('pos_max_dt'),
      kalman_process_var:  num('pos_kf_q'),
      kalman_meas_var:     num('pos_kf_r'),
      imu_lowpass_hz:      num('pos_imu_lpf'),
      max_pose_jump_m:     num('pos_max_jump'),
      target_marker_size_m: num('pos_target_size'),
      zupt_speed_m_s:      num('pos_zupt_speed'),
      zupt_hold_frames:    int('pos_zupt_hold'),
    };
    try {
      const r = await fetch('/proxy/position/config', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload),
      });
      const d = await r.json();
      if (d.ok) flash('\\u2713 applied', '#22c55e');
      else flash('error: ' + (d.error || 'unknown'), '#ef4444');
    } catch (e) { flash('request failed', '#ef4444'); }
  };
  const resetBtn = document.getElementById('pos_precision_reset');
  if (resetBtn) resetBtn.onclick = () => {
    document.getElementById('pos_pose_hold').value = '0.8';
    document.getElementById('pos_min_refs').value  = '1';
    document.getElementById('pos_min_ref_w').value = '0';
    document.getElementById('pos_blend_min').value = '0.35';
    document.getElementById('pos_blend_max').value = '0.85';
    document.getElementById('pos_vel_blend').value = '0.25';
    document.getElementById('pos_max_dt').value    = '1.0';
    document.getElementById('pos_kf_q').value      = '1e-3';
    document.getElementById('pos_kf_r').value      = '0.1';
    const lpf = document.getElementById('pos_imu_lpf');
    if (lpf) {
      lpf.value = '5';
      const lbl = document.getElementById('pos_imu_lpf_val');
      if (lbl) lbl.textContent = '5.0 Hz';
    }
    const mj = document.getElementById('pos_max_jump');
    if (mj) mj.value = '0';
    const ts = document.getElementById('pos_target_size');
    if (ts) ts.value = '0.19';
    const zs = document.getElementById('pos_zupt_speed');
    if (zs) zs.value = '0.05';
    const zh = document.getElementById('pos_zupt_hold');
    if (zh) zh.value = '3';
    flash('defaults loaded — click Apply', '#94a3b8');
  };
})();

// ── Position-tracker preset management ─────────────────────────────
// Mirrors the mission preset system. Apply fans out to every drone
// via /proxy/position/config; Save gathers current UI values and
// POSTs to /proxy/position/presets.
(function wirePositionPresets(){
  const sel   = document.getElementById('pos_preset_sel');
  const aBtn  = document.getElementById('pos_preset_apply');
  const sBtn  = document.getElementById('pos_preset_save');
  const dBtn  = document.getElementById('pos_preset_delete');
  const nameI = document.getElementById('pos_preset_name');
  const status = document.getElementById('pos_preset_status');
  if (!sel) return;
  function flash(msg, col) {
    if (!status) return;
    status.textContent = msg;
    status.style.color = col || '#64748b';
    setTimeout(() => { if (status.textContent === msg) status.textContent = ''; }, 3000);
  }
  function readCurrentParams() {
    // Gather everything live-tunable on the Position Tracker panel.
    // The server accepts unknown keys gracefully; we send a superset.
    const v = (id, parser) => {
      const el = document.getElementById(id);
      if (!el) return undefined;
      const raw = el.value;
      if (parser === 'float') { const f = parseFloat(raw); return isFinite(f) ? f : undefined; }
      if (parser === 'int')   { const i = parseInt(raw, 10); return isFinite(i) ? i : undefined; }
      if (parser === 'bool')  return !!el.checked;
      return raw;
    };
    return {
      detect_profile:        v('pos_profile'),
      fov_deg:               v('pos_fov', 'float'),
      imu_weight:            (v('pos_imu_weight', 'float') || 0) / 100.0,
      latency_ms:            v('pos_latency', 'float'),
      enable_kalman_filter:  v('pos_kalman', 'bool'),
      marker_size_m:         v('pos_marker_size', 'float'),
      top_k_markers:         v('pos_top_k', 'int'),
      outlier_reject_m:      v('pos_outlier', 'float'),
      distance_scale:        v('pos_distance_scale', 'float'),
      pose_hold_sec:         v('pos_pose_hold', 'float'),
      min_ref_count:         v('pos_min_refs', 'int'),
      min_ref_weight:        v('pos_min_ref_w', 'float'),
      meas_blend_min:        v('pos_blend_min', 'float'),
      meas_blend_max:        v('pos_blend_max', 'float'),
      vel_blend:             v('pos_vel_blend', 'float'),
      max_state_dt:          v('pos_max_dt', 'float'),
      kalman_process_var:    v('pos_kf_q', 'float'),
      kalman_meas_var:       v('pos_kf_r', 'float'),
      imu_lowpass_hz:        v('pos_imu_lpf', 'float'),
      max_pose_jump_m:       v('pos_max_jump', 'float'),
      target_marker_size_m:  v('pos_target_size', 'float'),
      zupt_speed_m_s:        v('pos_zupt_speed', 'float'),
      zupt_hold_frames:      v('pos_zupt_hold', 'int'),
    };
  }
  async function refresh() {
    try {
      const j = await (await fetch('/proxy/position/presets')).json();
      const presets = j.presets || {};
      const names = Object.keys(presets).sort();
      sel.innerHTML = '';
      if (!names.length) {
        const opt = document.createElement('option');
        opt.value = ''; opt.textContent = '(no presets)';
        sel.appendChild(opt);
      } else {
        names.forEach(n => {
          const opt = document.createElement('option');
          opt.value = n; opt.textContent = n;
          sel.appendChild(opt);
        });
        if (names.includes('balanced')) sel.value = 'balanced';
      }
    } catch {}
  }
  if (aBtn) aBtn.onclick = async () => {
    const name = sel.value;
    if (!name) { flash('no preset selected', '#ef4444'); return; }
    try {
      const r = await fetch('/proxy/position/presets/apply', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({name}),
      });
      const j = await r.json();
      if (j.ok) {
        flash('\u2713 "' + name + '" \u2192 ' + (j.applied_to || 0) + '/' + (j.total || 0) + ' drones', '#22c55e');
        if (typeof loadPosConfig === 'function') loadPosConfig();   // pull fresh UI values
      } else flash('\u2717 ' + (j.error || 'apply failed'), '#ef4444');
    } catch (e) { flash('\u2717 ' + e, '#ef4444'); }
  };
  if (sBtn) sBtn.onclick = async () => {
    const name = (nameI.value.trim()) || sel.value;
    if (!name) { flash('enter a preset name', '#ef4444'); return; }
    const params = readCurrentParams();
    try {
      const r = await fetch('/proxy/position/presets', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({name, params}),
      });
      const j = await r.json();
      if (j.ok) { flash('\u2713 saved "' + name + '"', '#22c55e'); await refresh(); }
      else flash('\u2717 ' + (j.error || 'save failed'), '#ef4444');
    } catch (e) { flash('\u2717 ' + e, '#ef4444'); }
  };
  if (dBtn) dBtn.onclick = async () => {
    const name = sel.value;
    if (!name) return;
    if (!confirm('Delete position preset "' + name + '"?')) return;
    try {
      const r = await fetch('/proxy/position/presets?name=' + encodeURIComponent(name),
                             {method:'DELETE'});
      const j = await r.json();
      if (j.ok) { flash('\u2713 deleted', '#22c55e'); await refresh(); }
      else flash('\u2717 ' + (j.error || 'delete failed'), '#ef4444');
    } catch (e) { flash('\u2717 ' + e, '#ef4444'); }
  };
  refresh();
})();

// ── Claude Automatic Calibration ───────────────────────────────────
// Orchestrates the autonomous calibration flight:
//   1. Pre-flight warning (operator must have drone centred in arena)
//   2. POST /proxy/calibration/start to kick off the ~90 s sequence
//   3. Poll /proxy/calibration/status at 500 ms to drive the progress bar
//   4. On completion, show the download hint pointing to Flight Logs
//   5. Provide an Import Preset box so the operator can paste Claude's
//      tuned JSON and save it as a named preset via /proxy/position/presets
// ── Scan & Capture Targets mission ─────────────────────────────
// Kicks off /proxy/missions/scan_and_capture/start and polls
// /proxy/missions/scan_and_capture/status every 500 ms to drive the
// progress readout in the Target Boxes panel.
(function wireScanAndCapture(){
  const startBtn  = document.getElementById('scan_cap_start_btn');
  const abortBtn  = document.getElementById('scan_cap_abort_btn');
  const statusTxt = document.getElementById('scan_cap_status_txt');
  const progWrap  = document.getElementById('scan_cap_progress');
  const progBar   = document.getElementById('scan_cap_progress_bar');
  const phaseEl   = document.getElementById('scan_cap_phase');
  const stepEl    = document.getElementById('scan_cap_step');
  const ndetEl    = document.getElementById('scan_cap_ndet');
  const elapsedEl = document.getElementById('scan_cap_elapsed');
  const hoverI    = document.getElementById('scan_cap_hover');
  const aboveI    = document.getElementById('scan_cap_above');
  if (!startBtn) return;

  let pollTimer = null;
  let active = false;

  function setStatus(msg, col) {
    if (!statusTxt) return;
    statusTxt.textContent = msg;
    statusTxt.style.color = col || '#94a3b8';
  }
  function setActiveUI(on) {
    active = on;
    startBtn.style.display = on ? 'none' : '';
    abortBtn.style.display = on ? '' : 'none';
    progWrap.style.display = on ? '' : 'none';
    if (!on) progBar.style.width = '0%';
  }
  // Phase → percentage (rough progress bar). "scanning" 10-70%,
  // "capture" 70-100% (the capture mission runs after the handoff).
  function phaseProgress(phase, elapsed, n_detected) {
    if (phase === 'takeoff' || phase === 'starting') return 5;
    if (phase === 'scanning') {
      // 10s scan ≈ 70% target → scale with elapsed (after takeoff ~5s)
      return Math.min(70, 10 + elapsed * 6);
    }
    if (phase === 'capture' || phase === 'done')     return 90;
    if (phase === 'error' || phase === 'aborted')    return 100;
    return 0;
  }
  async function refreshStatus() {
    try {
      const r = await fetch('/proxy/missions/scan_and_capture/status');
      const d = await r.json();
      if (!d.ok) { setStatus('status error', '#ef4444'); return; }
      phaseEl.textContent = d.phase || '—';
      stepEl.textContent  = d.step_name || '—';
      ndetEl.textContent  = d.n_detected || 0;
      elapsedEl.textContent = (d.elapsed_s || 0).toFixed(1);
      progBar.style.width = phaseProgress(d.phase, d.elapsed_s || 0, d.n_detected || 0) + '%';
      if (d.active) {
        setActiveUI(true);
        setStatus('in progress — ' + (d.phase || ''), '#fbbf24');
      } else {
        if (active) {
          setActiveUI(false);
          if (d.result === 'ok') {
            setStatus('\u2713 scan done — ' + (d.n_detected || 0) +
                      ' target(s), capture mission launched', '#22c55e');
          } else if (d.result === 'aborted') {
            setStatus('aborted', '#f59e0b');
          } else if (d.result === 'error') {
            setStatus('\u2717 ' + (d.last_error || 'unknown error'), '#ef4444');
          }
          if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        }
      }
    } catch (e) {
      setStatus('network error: ' + e, '#ef4444');
    }
  }

  startBtn.addEventListener('click', async () => {
    if (!confirm(
      'Scan & Capture Targets:\\n\\n' +
      '• The drone will take off (if not already flying) and rotate 360°\\n' +
      '  to discover target boxes via ArUco (IDs 31-36 Blue, 41-46 Red).\\n' +
      '• It then flies exactly over each discovered box, hovering\\n' +
      '  ' + (hoverI.value||'3') + ' s at ' + (aboveI.value||'1.5') + ' m altitude above the marker.\\n' +
      '• Arena boundary guard + ceiling stay active throughout.\\n\\n' +
      'Start now?'
    )) return;
    setStatus('starting...', '#fbbf24');
    try {
      const body = {
        hover_seconds: parseFloat(hoverI.value || '3.0'),
        hover_above_m: parseFloat(aboveI.value || '1.5'),
      };
      const r = await fetch('/proxy/missions/scan_and_capture/start', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!d.ok) {
        setStatus('\u2717 ' + (d.error || 'start failed'), '#ef4444');
        return;
      }
      setActiveUI(true);
      setStatus('in progress', '#fbbf24');
      if (pollTimer) clearInterval(pollTimer);
      pollTimer = setInterval(refreshStatus, 500);
      refreshStatus();
    } catch (e) {
      setStatus('\u2717 ' + e, '#ef4444');
    }
  });

  abortBtn.addEventListener('click', async () => {
    if (!confirm('Abort scan & capture?\\n\\nThe drone will stop rotating/visiting; it stays airborne — land it manually.')) return;
    try {
      const r = await fetch('/proxy/missions/scan_and_capture/abort', {method:'POST'});
      const d = await r.json();
      if (d.ok) setStatus('abort requested...', '#f59e0b');
      else setStatus('\u2717 ' + (d.error || 'abort failed'), '#ef4444');
    } catch (e) {
      setStatus('\u2717 ' + e, '#ef4444');
    }
  });

  // One status poll at page load in case a mission is already running
  refreshStatus();
})();

(function wireCalibrationFlight(){
  const startBtn  = document.getElementById('calib_start_btn');
  const abortBtn  = document.getElementById('calib_abort_btn');
  const statusTxt = document.getElementById('calib_status_txt');
  const progWrap  = document.getElementById('calib_progress_wrap');
  const progBar   = document.getElementById('calib_progress_bar');
  const stepNum   = document.getElementById('calib_step_num');
  const stepTotal = document.getElementById('calib_step_total');
  const stepName  = document.getElementById('calib_step_name');
  const elapsed   = document.getElementById('calib_elapsed');
  const dlHint    = document.getElementById('calib_download_hint');
  const importBtn = document.getElementById('calib_import_btn');
  const presetJsonEl = document.getElementById('calib_preset_json');
  const presetNameEl = document.getElementById('calib_preset_name');
  const presetStatus = document.getElementById('calib_preset_status');
  if (!startBtn) return;

  let pollTimer = null;
  let active = false;

  function setStatus(msg, col) {
    if (!statusTxt) return;
    statusTxt.textContent = msg;
    statusTxt.style.color = col || '#94a3b8';
  }
  function setActiveUI(on) {
    active = on;
    startBtn.style.display = on ? 'none' : '';
    abortBtn.style.display = on ? '' : 'none';
    progWrap.style.display = on ? '' : 'none';
    if (!on) {
      progBar.style.width = '0%';
    }
  }
  async function refreshStatus() {
    try {
      const r = await fetch('/proxy/calibration/status');
      const d = await r.json();
      if (!d.ok) { setStatus('status error', '#ef4444'); return; }
      const total = d.total_steps || 1;
      const cur = d.current_step || 0;
      const pct = Math.round((cur / total) * 100);
      progBar.style.width = pct + '%';
      stepNum.textContent = cur;
      stepTotal.textContent = total;
      stepName.textContent = d.step_name || '—';
      elapsed.textContent = (d.elapsed_s || 0).toFixed(1);
      if (d.active) {
        setActiveUI(true);
        setStatus('in progress — step ' + cur + '/' + total, '#fbbf24');
      } else {
        if (active) {
          // Just finished
          setActiveUI(false);
          if (d.result === 'ok') {
            setStatus('\u2713 completed (' + (d.elapsed_s||0).toFixed(1) + 's)', '#22c55e');
            dlHint.style.display = '';
            // Refresh flight-logs list so the new calibration flight appears
            const fr = document.getElementById('flight_logs_refresh');
            if (fr) setTimeout(() => fr.click(), 800);
          } else if (d.result === 'aborted') {
            setStatus('aborted', '#f59e0b');
            dlHint.style.display = '';
          } else if (d.result === 'error') {
            setStatus('\u2717 error: ' + (d.last_error || 'unknown'), '#ef4444');
            dlHint.style.display = '';
          }
          if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        }
      }
    } catch (e) {
      setStatus('network error: ' + e, '#ef4444');
    }
  }

  startBtn.addEventListener('click', async () => {
    if (!confirm(
      'Calibration Flight:\\n\\n' +
      '• The drone will take off, fly a ~90 s scan pattern, and land.\\n' +
      '• It stays within \u00b11.5 m of its current position — make sure\\n' +
      '  the drone is placed in the arena CENTRE before continuing.\\n' +
      '• Arena boundary guard and ceiling remain active.\\n\\n' +
      'Start now?'
    )) return;
    setStatus('starting...', '#fbbf24');
    try {
      const r = await fetch('/proxy/calibration/start', {method:'POST'});
      const d = await r.json();
      if (!d.ok) {
        setStatus('\u2717 ' + (d.error || 'start failed'), '#ef4444');
        return;
      }
      stepTotal.textContent = d.total_steps || '?';
      setActiveUI(true);
      dlHint.style.display = 'none';
      setStatus('in progress', '#fbbf24');
      // Begin polling
      if (pollTimer) clearInterval(pollTimer);
      pollTimer = setInterval(refreshStatus, 500);
      refreshStatus();
    } catch (e) {
      setStatus('\u2717 ' + e, '#ef4444');
    }
  });

  abortBtn.addEventListener('click', async () => {
    if (!confirm('Abort calibration? The drone will land at the next safe point.')) return;
    try {
      const r = await fetch('/proxy/calibration/abort', {method:'POST'});
      const d = await r.json();
      if (d.ok) setStatus('abort requested...', '#f59e0b');
      else setStatus('\u2717 ' + (d.error || 'abort failed'), '#ef4444');
    } catch (e) {
      setStatus('\u2717 ' + e, '#ef4444');
    }
  });

  // Toggle the paste-JSON textarea when operator clicks Import.
  importBtn.addEventListener('click', async () => {
    if (presetJsonEl.style.display === 'none') {
      presetJsonEl.style.display = '';
      presetJsonEl.focus();
      presetStatus.textContent = 'paste JSON above, then click Import again to save';
      presetStatus.style.color = '#fbbf24';
      return;
    }
    // Actually import
    const raw = presetJsonEl.value.trim();
    const name = (presetNameEl.value || '').trim();
    if (!name) {
      presetStatus.textContent = '\u2717 enter a preset name first';
      presetStatus.style.color = '#ef4444';
      return;
    }
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (e) {
      presetStatus.textContent = '\u2717 invalid JSON: ' + e.message;
      presetStatus.style.color = '#ef4444';
      return;
    }
    if (typeof parsed !== 'object' || Array.isArray(parsed)) {
      presetStatus.textContent = '\u2717 JSON must be an object';
      presetStatus.style.color = '#ef4444';
      return;
    }
    try {
      const r = await fetch('/proxy/position/presets', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({name, params: parsed}),
      });
      const d = await r.json();
      if (d.ok) {
        presetStatus.textContent = '\u2713 saved "' + name + '" — apply from Presets dropdown above';
        presetStatus.style.color = '#22c55e';
        presetJsonEl.style.display = 'none';
        presetJsonEl.value = '';
        // Refresh presets dropdown
        const presetSel = document.getElementById('pos_preset_sel');
        if (presetSel) {
          fetch('/proxy/position/presets').then(r=>r.json()).then(j=>{
            if (!j.presets) return;
            const names = Object.keys(j.presets).sort();
            presetSel.innerHTML = '';
            names.forEach(n => {
              const opt = document.createElement('option');
              opt.value = n; opt.textContent = n;
              presetSel.appendChild(opt);
            });
            presetSel.value = name;
          });
        }
      } else {
        presetStatus.textContent = '\u2717 save failed: ' + (d.error || 'unknown');
        presetStatus.style.color = '#ef4444';
      }
    } catch (e) {
      presetStatus.textContent = '\u2717 ' + e;
      presetStatus.style.color = '#ef4444';
    }
  });

  // One initial status poll — useful if operator loads page mid-flight
  refreshStatus();
})();

// ── IMU LPF slider live-label + debounced apply ─────────────────────
// Updates the 'X.X Hz' / 'OFF' label on every frame of the drag, and
// POSTs the value once the user stops moving it (~200 ms debounce).
(function wireImuLpfSlider(){
  const slider = document.getElementById('pos_imu_lpf');
  const label  = document.getElementById('pos_imu_lpf_val');
  if (!slider) return;
  let _t = null;
  function refreshLabel() {
    if (!label) return;
    const v = parseFloat(slider.value);
    label.textContent = (v > 0 ? v.toFixed(1) + ' Hz' : 'OFF');
  }
  slider.addEventListener('input', () => {
    refreshLabel();
    if (_t) clearTimeout(_t);
    _t = setTimeout(async () => {
      try {
        await fetch('/proxy/position/config', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({imu_lowpass_hz: parseFloat(slider.value)}),
        });
      } catch {}
    }, 200);
  });
  refreshLabel();
})();

// ── Parameter-info popup — click any ⓘ icon for an explanation ──
// One shared map drives every tuning knob. The modal picks up title,
// key, body (explanation), and an optional range/units hint.
window.PARAM_INFO = {
  // ===== Observer PD (visual servo — used by all missions) =====
  hover_distance_m: {title:'Hover distance', units:'metres', body:
    "Target stand-off distance from the marker during hover. The mission "+
    "flies forward until the marker is this far away, then holds.\\n\\n"+
    "Smaller = closer / more camera detail, but less safety margin against "+
    "the net. 2 m is the rules-safe default for SDC26.\\n"+
    "Feeds into: approach phase (distance error) and hover phase (dead-band)."},
  fb_max:           {title:'Approach speed (forward)', units:'% RC', body:
    "Upper clamp for forward throttle when approaching a marker. Scales "+
    "the P gain output — higher = faster approach but larger overshoot.\\n\\n"+
    "Pair with dist_p: the effective command is min(fb_max, dist_p · err_dist)."},
  fb_back_max:      {title:'Retreat speed (backward)', units:'% RC', body:
    "Upper clamp for backward throttle when the drone is too close to the "+
    "target. Set lower than fb_max — retreats tend to happen near the net "+
    "and should be cautious."},
  dist_p:           {title:'Approach aggressiveness', units:'P · dist error (m)', body:
    "Proportional gain turning distance-error (m) into forward RC%. Higher "+
    "values react more sharply to distance error.\\n\\n"+
    "Start low (10-20) and raise until the drone reaches the hover distance "+
    "without overshoot. Tied to fb_max and the IMU D term (d_fb)."},
  ema_alpha:        {title:'EMA smoothing (α)', units:'0..1', body:
    "First-order low-pass on camera-derived errors. 1.0 = no smoothing "+
    "(fastest response, jitteriest), 0.05 = heavy smoothing (laggy).\\n\\n"+
    "Typical 0.25-0.5. Lower when the video is noisy or markers are small."},
  deadband_x:       {title:'Yaw/lateral dead-band (err_x)', units:'normalised', body:
    "Below this threshold, yaw + sideways commands are zero. Stops the "+
    "drone hunting around an already-centred marker.\\n\\n"+
    "Typical 0.03-0.08. Increase if the drone wiggles while hovered."},
  deadband_y:       {title:'Altitude dead-band (err_y)', units:'normalised', body:
    "Below this threshold vertical command is zero. Similar purpose to "+
    "deadband_x but on the vertical axis. 0.05-0.1 typical."},
  deadband_skew:    {title:'Skew dead-band', units:'normalised', body:
    "Below this threshold the perpendicular-alignment (strafe) command is "+
    "zero. Skew measures how tilted the marker appears — a small tilt "+
    "doesn't need correction."},
  deadband_dist_m:  {title:'Distance dead-band', units:'metres', body:
    "Below this distance-error threshold the forward/back command is zero. "+
    "Keeps the drone parked once it's ~within range of hover_distance_m.\\n\\n"+
    "Too small and you get buzzing; too large and the drone drifts."},
  yaw_p:            {title:'Yaw P-gain', units:'per err_x', body:
    "Proportional gain from horizontal image-error to yaw RC%. Higher = "+
    "snappier rotation toward the marker but more overshoot."},
  skew_p:           {title:'Lateral P-gain', units:'per skew', body:
    "Proportional gain from marker-tilt to sideways RC%. Used to strafe "+
    "around the marker for head-on approach."},
  alt_p:            {title:'Altitude P-gain', units:'per err_y', body:
    "Proportional gain from vertical image-error to vertical RC%. Higher = "+
    "snappier climb/descend to marker height."},
  d_yaw:            {title:'Yaw D-damping', units:'per °/s (gyro)', body:
    "Derivative damping on yaw using the gyro. Subtracts a fraction of the "+
    "current yaw-rate from the yaw command — cancels oscillation.\\n\\n"+
    "If the drone visibly orbits around the marker, increase."},
  d_lr:             {title:'Lateral D-damping', units:'per cm/s (vgy)', body:
    "Derivative damping on sideways RC using the body-frame Y velocity "+
    "from the IMU. Prevents over-strafing."},
  d_ud:             {title:'Vertical D-damping', units:'per cm/s (vgz)', body:
    "Derivative damping on vertical RC using the body-frame Z velocity. "+
    "Prevents vertical oscillation as the drone approaches target altitude."},
  d_fb:             {title:'Fwd/back D-damping', units:'per cm/s (vgx)', body:
    "Derivative damping on forward/back RC using the body-frame X velocity. "+
    "The most important D term for not slamming into the net.\\n\\n"+
    "Combined with the boundary guard (arena edge prediction) this is the "+
    "primary brake during fast approaches."},
  yaw_max:          {title:'Clamp · max yaw', units:'% RC', body:
    "Hard upper limit for yaw RC%. Applied after the PD math. Keeps the "+
    "drone from spinning wildly on large errors. 20-30 typical."},
  lr_max:           {title:'Clamp · max lateral', units:'% RC', body:
    "Hard upper limit for sideways RC%. 20-40 typical — too high and the "+
    "drone over-strafes during approach."},
  ud_max:           {title:'Clamp · max vertical', units:'% RC', body:
    "Hard upper limit for vertical RC%. Typical 20-40. Raise for faster "+
    "altitude acquisition if the ceiling guard allows."},
  rc_min:           {title:'RC dead-floor', units:'% RC', body:
    "Below this magnitude the RC output is forced to zero. Anafi ignores "+
    "very small RC values anyway — this prevents buzzing/whine while "+
    "hovered. Usually 2-3."},
  cam_hfov_deg:     {title:'Camera H-FOV (drawing only)', units:'degrees', body:
    "Used ONLY to draw the camera cone in the top-down view — does NOT "+
    "affect PnP or any control. 69° is the Anafi nominal."},
  marker_size_m:    {title:'Observer marker size', units:'metres', body:
    "Physical marker side length for the observer's own PnP. Must match "+
    "the printed markers. SDC26 markers are 0.5 m.\\n\\n"+
    "Tip: keep this in sync with the Position Tracker's marker size."},

  // ===== Position Tracker (arena-frame pose fusion) =====
  detect_profile:     {title:'Detection profile', body:
    "Preset for the ArUco detector parameters (corner refinement, adaptive "+
    "threshold window, min marker size). Profiles:\\n"+
    "  · Balanced  — default, good speed + robustness.\\n"+
    "  · Sensitive — lighter thresholds, catches distant / partially-lit "+
    "markers at higher CPU cost.\\n"+
    "  · Strict    — tighter accept, rejects noisy detections."},
  fov_deg:            {title:'Camera H-FOV', units:'degrees', body:
    "Horizontal field-of-view used to synthesise the intrinsics matrix "+
    "when no calibration file is loaded. Anafi 4K ≈ 69°.\\n\\n"+
    "Uploading a .npz calibration overrides this."},
  latency_ms:         {title:'Video-to-IMU latency', units:'ms', body:
    "How long the camera frame is old relative to the current IMU sample. "+
    "The positioner rewinds the IMU buffer by this amount so the IMU "+
    "velocity used for dead-reckoning corresponds to the same moment as "+
    "the vision measurement.\\n\\n"+
    "Measure: the header Latency row shows c2→fc + fc→drone RTT plus a "+
    "video decode offset; enabling auto-set pushes that total here."},
  imu_weight:         {title:'IMU ↔ ArUco blend', units:'0..1', body:
    "Mix between pure ArUco pose (0) and pure IMU dead-reckoning (1). "+
    "Higher = smoother but drifts more during marker outages.\\n\\n"+
    "30 % is a good default. Raise to 50-60 % if markers flicker in/out, "+
    "lower if the IMU shows bias."},
  enable_kalman_filter:{title:'Kalman filter', body:
    "Per-axis 1-D Kalman filter on x/y/z. Models state as position+velocity "+
    "and fuses ArUco fixes as measurements.\\n\\n"+
    "ON (recommended): smoother, handles brief marker dropouts.\\n"+
    "OFF: pose jumps straight to the last ArUco solution — noisier but "+
    "with zero added latency."},
  top_k_markers:      {title:'Top-K markers', body:
    "Use only the N closest markers in the weighted-mean fusion. 0 = auto "+
    "(picks 4). Smaller K = faster, less robust to outliers. Larger K = "+
    "more samples but includes distant, less-accurate detections."},
  outlier_reject_m:   {title:'Outlier reject distance', units:'metres', body:
    "Per-marker poses further than this from the weighted-mean position "+
    "are rejected as outliers before re-averaging.\\n\\n"+
    "Default 2.5 m. Tighten to 1.0 m for a smaller, tidy arena; loosen to "+
    "3.5 m if markers are far apart."},
  target_marker_size_m: {title:'Target-box marker size', units:'metres', body:
    "Physical side length of the ArUco stickers on SDC26 target boxes.\\n\\n"+
    "The drone's ArUco dictionary (DICT_4X4_100) has 100 IDs. IDs 0-29 are "+
    "reserved for arena wall/reference markers (50 cm) and IDs \\u226530 for "+
    "target boxes (19 cm SDC26 default).\\n\\n"+
    "This matters because solvePnP needs the real marker size to compute "+
    "distance correctly — if we measure 19 cm markers with a 50 cm template, "+
    "we get distances 50/19 \\u2248 2.6\\u00d7 too large, and the target lands far "+
    "from where it really is.\\n\\n"+
    "Default 0.19 m. Adjust only if your arena uses a different sticker size."},
  zupt_speed_m_s:     {title:'ZUPT speed threshold', units:'m/s (0 = disabled)', body:
    "Zero-Velocity Update. When the IMU reports speed below this threshold "+
    "for \\\"hold\\\" consecutive frames, the positioner snaps the fresh ArUco "+
    "measurement to the last valid pose — eliminates the parked-drone drift "+
    "caused by sub-pixel corner noise + IPPE mirror ambiguity.\\n\\n"+
    "Below 0.05 m/s is \\\"not moving\\\" in practice. Raise to 0.10 m/s for "+
    "very noisy indoor flight (it'll also freeze during slow hovers), or "+
    "drop to 0.02 m/s if you want aggressive drift suppression.\\n\\n"+
    "0 disables."},
  zupt_hold_frames:   {title:'ZUPT hold frames', units:'frames', body:
    "How many consecutive slow frames are required before ZUPT engages. "+
    "Stops false-triggering during quick directional changes (where IMU "+
    "speed dips briefly as the drone reverses).\\n\\n"+
    "At 5 Hz positioning: 3 frames \\u2248 600 ms. That's a good balance — long "+
    "enough to ignore brief lulls, short enough to lock in quickly when "+
    "actually parked."},
  max_pose_jump_m:    {title:'Pose-jump gate', units:'metres (0 = disabled)', body:
    "Safety limit on how far a fresh ArUco fix is allowed to disagree with "+
    "the Kalman-predicted state before it's rejected as an outlier.\\n\\n"+
    "Single-marker solvePnP occasionally produces mirror-pose glitches that "+
    "put the drone 10-20 m from where it actually is. The gate silently "+
    "drops those fixes (the filter keeps predicting forward) so the UI and "+
    "missions don't see the spike.\\n\\n"+
    "Default 0 = off. 3.0 m is a good starting value for indoor arenas — "+
    "large enough not to block legitimate fast motion, small enough to "+
    "catch the bad frames. Auto Positioning uses 3.0 m by default."},
  distance_scale:     {title:'Distance correction factor', units:'multiplier (1.0 = no correction)', body:
    "Multiplicative correction applied to the camera\\u2194marker translation "+
    "from solvePnP. Compensates for systematic scale error, usually caused "+
    "by a mismatch between the real marker size and the configured "+
    "marker_size_m, or by an uncalibrated focal length.\\n\\n"+
    "Calibration recipe: put the drone a known distance D_real from a marker, "+
    "read the UI distance D_ui, then set distance_scale = D_real / D_ui.\\n\\n"+
    "Example: UI shows 7 m, tape measure says 9 m \\u2192 9 / 7 \\u2248 1.286.\\n\\n"+
    "The scale applies equally to every pose axis, so it works whether the "+
    "error shows up in x, y, z, or the combined 3-D distance. If the error "+
    "is direction-dependent, fix the camera calibration instead."},
  pose_hold_sec:      {title:'Pose hold (dead-reckon)', units:'seconds', body:
    "After the last valid ArUco fix, keep publishing pose based on IMU "+
    "dead-reckoning for this many seconds.\\n\\n"+
    "Too short → pose vanishes every time the camera blinks.\\n"+
    "Too long  → stale pose during marker outages drifts meters.\\n"+
    "0.5-1.0 s typical. Raise only if the IMU is well-calibrated."},
  min_ref_count:      {title:'Minimum reference markers', body:
    "Require at least this many markers to be visible before a fused pose "+
    "is accepted. 1 = accept a single marker (can be noisy). 2-3 gives a "+
    "much more robust fix by cross-checking."},
  min_ref_weight:     {title:'Minimum reference weight', units:'0..1', body:
    "Require the best-matching marker to have at least this fused weight. "+
    "Rejects very low-confidence fits (tiny / extreme-angle markers).\\n\\n"+
    "0 = accept any detection. 0.2-0.4 is restrictive but clean."},
  meas_blend_min:     {title:'Measurement blend — low', units:'α ∈ [0..1]', body:
    "Minimum EMA α applied to fresh ArUco measurements when fusing with "+
    "the Kalman state. Used when fix quality is high (trust the filter).\\n\\n"+
    "Lower values = more Kalman smoothing, less jitter."},
  meas_blend_max:     {title:'Measurement blend — high', units:'α ∈ [0..1]', body:
    "Maximum EMA α used when fix quality is low (trust the latest fresh "+
    "measurement more). The positioner interpolates between min and max "+
    "based on residual error and ref count."},
  vel_blend:          {title:'Velocity blend', units:'0..1', body:
    "Blend between IMU-measured velocity (0) and Kalman-state velocity "+
    "derivative (1). Raises 0.25 means 25 % Kalman-derived, 75 % raw IMU.\\n\\n"+
    "Higher = smoother vz/vy plots. Lower = faster reaction to real motion."},
  max_state_dt:       {title:'Max state Δt', units:'seconds', body:
    "If more than this amount of time passes between updates, the Kalman "+
    "state is reset instead of extrapolated. Prevents exploding covariance "+
    "during long outages (landing, lost camera link)."},
  kalman_process_var: {title:'Kalman process variance (Q)', body:
    "How much the state is expected to change between steps. Low Q → the "+
    "filter believes its model (smooth but sluggish). High Q → the filter "+
    "expects rapid changes (reacts faster but noisier).\\n\\n"+
    "Default 1e-3. Try 5e-4 for smooth hover, 5e-3 for dynamic missions."},
  imu_lowpass_hz:     {title:'IMU low-pass cut-off', units:'Hz', body:
    "First-order IIR low-pass filter applied to body-frame IMU "+
    "velocity (vgx, vgy, vgz) at telemetry ingestion on the Pi. Smooths "+
    "Anafi's noisy per-sample velocity before it reaches the position "+
    "fusion, the synchronisation buffer, and the arena view.\\n\\n"+
    "Parameter: cut-off frequency (Hz). Lower = more smoothing.\\n"+
    "  • 0       → filter disabled (raw values passed through).\\n"+
    "  • 1–3 Hz  → heavy smoothing, visibly laggy reaction.\\n"+
    "  • 5 Hz    → default; cuts Anafi's ~15 Hz broadband jitter without\\n"+
    "               dulling reaction to real motion.\\n"+
    "  • 10-30 Hz→ minimal smoothing, reacts to faster manoeuvres.\\n\\n"+
    "Implemented as a time-based alpha = dt / (tau + dt), where "+
    "tau = 1 / (2π·fc), so uneven telemetry intervals still yield "+
    "the correct filter response. State is reset when the filter is "+
    "disabled so re-enabling starts clean."},
  kalman_meas_var:    {title:'Kalman measurement variance (R)', body:
    "How noisy the ArUco measurements are. Low R → trust the camera "+
    "more (snaps to detections). High R → trust the model more (smoother "+
    "but may lag).\\n\\n"+
    "Default 1e-1. Large markers at short range can use 1e-2; noisy, "+
    "distant markers may benefit from 3e-1."},

  // ===== Safety =====
  axis_locked: {title:'Axis-locked (Manhattan) autonomous flight', body:
    "When ON, autonomous motion (missions + ArUco LIVE + waypoint nav) "+
    "is constrained to wall-parallel axes only:\\n\\n"+
    "  1. Yaw snaps to the nearest 90° multiple (0° / 90° / 180° / 270° "+
    "in arena frame). The drone aligns to a cardinal heading before "+
    "moving. Rotations happen in 90° increments only.\\n"+
    "  2. Only ONE horizontal axis moves at a time: either strafe (LR) "+
    "or forward/back (FB), whichever has the larger magnitude. The "+
    "drone never flies diagonally.\\n"+
    "  3. Vertical (up/down) is unaffected.\\n\\n"+
    "Manual WASD / Q-E / R-F is unaffected — this constraint applies "+
    "exclusively to autonomous decisions (observer LIVE + waypoints).\\n\\n"+
    "Typical use: structured arena navigation where diagonals invite "+
    "unnecessary marker-tracking jitter. Default OFF."},
  camera_face_center: {title:'Camera → arena centre', body:
    "Fleet-wide toggle: when ON, every observer's waypoint face-target "+
    "defaults to the arena centre (x=0, y=half-depth). During autonomous "+
    "missions, the drone's camera then aims at the centre of the arena "+
    "regardless of which direction it's flying — keeping the maximum "+
    "number of ArUco markers in view at once.\\n\\n"+
    "Why: the position processor fuses poses from every visible marker. "+
    "More visible markers = tighter fused arena-frame pose. Pointing the "+
    "camera at one specific marker (the mission's target) means only a "+
    "handful of markers are in the FOV at a time; pointing at the centre "+
    "typically keeps 4-6 markers visible throughout a flight.\\n\\n"+
    "Missions can still override this on a per-call basis by passing an "+
    "explicit face target. If OFF, missions that don't set a face leave "+
    "the camera aligned with the drone's direction of travel.\\n\\n"+
    "Arena centre defaults to (0, 5.4) for a 20×10.8m arena. The value "+
    "is adjustable via POST /proxy/config/camera_face_center body "+
    "{\\\"xy\\\": [x, y]}."},
  arena_guard_enabled: {title:'Arena guard (manual + auto)', body:
    "When ON (default), the Pi's own RC tick loop runs a boundary "+
    "guard on EVERY command — manual WASD, autonomous missions, "+
    "ArUco LIVE, everything. If the drone is within the safety "+
    "margin of any arena wall AND the command would push it closer, "+
    "that axis of the command is clamped to zero.\\n\\n"+
    "Independent of C2 connection — the guard runs on the flight "+
    "controller itself using the position processor's arena-frame "+
    "pose. If the position fix is stale or missing, the guard "+
    "gracefully falls back (no clamping) and the operator owns the "+
    "boundary decision.\\n\\n"+
    "Turn OFF for test/debug flights where you want raw control."},
  safety_margin_m: {title:'Arena safety margin', units:'metres', body:
    "Minimum distance the drone will maintain from ANY arena wall "+
    "during autonomous flight. The boundary guard runs per-tick at "+
    "20 Hz and overrides any waypoint or PD command that would drive "+
    "the drone closer to a wall than this margin.\\n\\n"+
    "Defaults to 1.5 m per ops policy. Adjustable 0.1–5.0 m.\\n\\n"+
    "The guard uses the latency-aware lookahead (GUARD_LOOKAHEAD_S = "+
    "0.35 s) so it accounts for momentum — the drone brakes / retreats "+
    "BEFORE it would cross the margin, not after. Only active during "+
    "autonomous flight (observer LIVE / waypoint). Manual flight is "+
    "bounded only by the hard altitude ceiling."},
  ceiling_m: {title:'Hard altitude ceiling', units:'metres', body:
    "Maximum altitude above ground. The value is set from this UI, but "+
    "the enforcement is ENTIRELY on each drone's flight-controller Pi — "+
    "in its own 20 Hz RC tick loop, using its own height_cm telemetry. "+
    "Independent of any C2 connection: if this browser or the C2 server "+
    "crashes mid-flight, the Pi keeps clamping upward RC.\\n\\n"+
    "Behaviour:\\n"+
    "  • approaching (within 50 cm): climb RC clamped proportionally to "+
    "remaining clearance.\\n"+
    "  • at ceiling: all climb blocked.\\n"+
    "  • above ceiling (+20 cm): forced active descent regardless of any "+
    "input — manual WASD or autonomous mission, no difference.\\n\\n"+
    "Persistence: the Pi writes the chosen value to flight_config.json "+
    "and reloads it on restart, so power-cycling the Pi still leaves "+
    "your last-set ceiling active. The firmware MaxAltitude is also "+
    "pushed to the Anafi as a second-line guard on every connect.\\n\\n"+
    "Default 5 m."},
};

window.showParamInfo = function(key, ev) {
  if (ev) { ev.stopPropagation(); ev.preventDefault(); }
  const info = window.PARAM_INFO[key];
  const m = document.getElementById('param_info_modal');
  if (!m) return;
  if (!info) {
    // Still show the modal with a graceful note — useful while adding new params.
    document.getElementById('pim_title').textContent = key;
    document.getElementById('pim_key').textContent   = key;
    document.getElementById('pim_body').textContent  = 'No description registered for this parameter yet.';
    document.getElementById('pim_range').textContent = '';
  } else {
    document.getElementById('pim_title').textContent = info.title || key;
    document.getElementById('pim_key').textContent   = key;
    document.getElementById('pim_body').textContent  = info.body || '';
    document.getElementById('pim_range').textContent = info.units ? ('Units: ' + info.units) : '';
  }
  m.style.display = 'flex';
};

// Event delegation — any element with class .info-icon and data-info="<key>"
// opens the modal. Works for icons injected later (observer PD rows) too.
document.addEventListener('click', function(ev){
  const el = ev.target.closest && ev.target.closest('.info-icon');
  if (!el) return;
  const key = el.dataset.info || el.getAttribute('data-info');
  if (!key) return;
  window.showParamInfo(key, ev);
});

// ── Light / dark theme toggle ──────────────────────────────────────
// Persists via localStorage. Default = dark (matches the original UI).
// The data-theme attribute drives the CSS overrides at the top of the
// <style> block. Using !important there lets the light theme defeat
// the many hard-coded inline style="" colours without rewriting every
// DOM element.
(function wireThemeToggle(){
  const KEY = 'sdc_theme';
  const btn = document.getElementById('theme_toggle');
  function applyTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    if (btn) btn.innerHTML = (t === 'light') ? '\\u2600\\ufe0f Light' : '\\ud83c\\udf19 Dark';
  }
  const saved = (function(){ try { return localStorage.getItem(KEY) || 'dark'; } catch { return 'dark'; } })();
  applyTheme(saved === 'light' ? 'light' : 'dark');
  if (btn) btn.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem(KEY, next); } catch {}
    applyTheme(next);
  });
})();

document.getElementById('pos_enabled').onchange = async function() {
  await post('/proxy/position/config', { enabled: this.checked });
  if (this.checked) startPosEvents();
  else { if (posEvtSource) { posEvtSource.close(); posEvtSource = null; } _lastPos = null; drawArena(); }
};

document.getElementById('pos_latency').oninput = function() {
  document.getElementById('pos_latency_val').textContent = this.value;
};

// IMU blend slider — debounced write-through so dragging doesn't spam
let _imuWeightTimer = null;
document.getElementById('pos_imu_weight').oninput = function() {
  document.getElementById('pos_imu_weight_val').textContent = this.value + '%';
  if (_imuWeightTimer) clearTimeout(_imuWeightTimer);
  const v = parseFloat(this.value) / 100;
  _imuWeightTimer = setTimeout(() => {
    post('/proxy/position/config', { imu_weight: v });
  }, 150);
};

document.getElementById('pos_cfg_save').onclick = async () => {
  const profile = document.getElementById('pos_profile').value;
  const fov = parseFloat(document.getElementById('pos_fov').value);
  const lat = parseFloat(document.getElementById('pos_latency').value);
  await post('/proxy/position/config', { detect_profile: profile, fov_deg: fov, latency_ms: lat });
  setTimeout(loadPosConfig, 400);
};

document.getElementById('pos_calib_file').onchange = async function() {
  if (!this.files.length) return;
  const fd = new FormData();
  fd.append('file', this.files[0]);
  const cs = document.getElementById('pos_calib_status');
  cs.textContent = 'uploading...'; cs.style.color = '#94a3b8';
  try {
    const r = await fetch('/proxy/position/calibration', { method: 'POST', body: fd });
    const d = await r.json();
    cs.textContent = d.ok ? '\\u2713 calibration saved' : ('error: ' + d.error);
    cs.style.color = d.ok ? '#22c55e' : '#ef4444';
    if (d.ok) loadPosConfig();
  } catch { cs.textContent = 'upload error'; cs.style.color = '#ef4444'; }
  this.value = '';
};

document.getElementById('pos_video_toggle').onclick = () => {
  posVideoOn = !posVideoOn;
  document.getElementById('pos_video_toggle').textContent = posVideoOn ? 'Hide ArUco Video' : 'Show ArUco Video';
  const container = document.getElementById('pos_video_container');
  const img = document.getElementById('pos_video_img');
  if (posVideoOn) { img.src = '/proxy/position/video?' + Date.now(); container.style.display = ''; }
  else { img.src = ''; container.style.display = 'none'; }
};

let _recActive = false;
const recBtn = document.getElementById('rec_btn');
const recStatus = document.getElementById('rec_status');

async function refreshRecStatus() {
  try {
    const d = await (await fetch('/proxy/video/record/status')).json();
    _recActive = d.recording;
    recBtn.textContent = _recActive ? '\\u25a0 Stop Rec' : '\\u25cf Record';
    recBtn.style.borderColor = _recActive ? '#ef4444' : '#22c55e';
    recBtn.style.color = _recActive ? '#ef4444' : '#22c55e';
    recBtn.style.background = _recActive ? '#3b0f0f' : '#1e3a2e';
    document.getElementById('rec_raw').disabled = _recActive;
    if (_recActive) recStatus.textContent = `${d.frames} frames \u2022 ${d.raw ? 'raw' : 'ann'} \u2022 ${d.path.split('/').pop()}`;
    else recStatus.textContent = d.frames ? `saved ${d.frames} frames` : '';
  } catch {}
}

recBtn.onclick = async () => {
  if (_recActive) {
    const d = await (await fetch('/proxy/video/record/stop', {method:'POST'})).json();
    recStatus.textContent = d.ok ? `saved ${d.frames} frames: ${d.path.split('/').pop()}` : ('error: ' + d.error);
  } else {
    const raw = document.getElementById('rec_raw').checked;
    const d = await (await fetch('/proxy/video/record/start', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({raw})})).json();
    recStatus.textContent = d.ok ? `recording... ${d.path.split('/').pop()}` : ('error: ' + d.error);
  }
  refreshRecStatus();
};

refreshRecStatus();
setInterval(refreshRecStatus, 5000);

loadPosConfig();
// ── Arena Configuration ───────────────────────────────────────────────────────
let arenaMarkers = {};   // {id_str: {pos:[x,y,z], wall:'front'}}

loadArenaConfig();   // pre-load markers so canvas shows them before panel is opened
drawArena();

const WALLS = ['front','back','left','right'];

function renderMarkerTable() {
  const tbody = document.getElementById('arena_marker_table');
  const sorted = Object.keys(arenaMarkers).sort((a,b) => Number(a)-Number(b));
  tbody.innerHTML = '';
  const rowStyle = 'display:flex;gap:4px;align-items:center;margin-bottom:3px;';
  const iStyle = 'height:26px;border-radius:4px;border:1px solid #334155;background:#0f172a;color:#e2e8f0;padding:0 4px;';
  sorted.forEach(id => {
    const m = arenaMarkers[id];
    const row = document.createElement('div');
    row.style.cssText = rowStyle;
    const wallOpts = WALLS.map(w => `<option value="${w}" ${m.wall===w?'selected':''}>${w}</option>`).join('');
    row.innerHTML = `
      <input type="number" step="1" min="0" value="${id}" data-oldid="${id}" data-rename="1" style="${iStyle}width:52px;text-align:right;" title="Marker ID (editable)" />
      <input type="number" step="0.001" value="${m.pos[0]}" data-id="${id}" data-f="0" style="${iStyle}width:58px;" title="X" />
      <input type="number" step="0.001" value="${m.pos[1]}" data-id="${id}" data-f="1" style="${iStyle}width:58px;" title="Y" />
      <input type="number" step="0.001" value="${m.pos[2]}" data-id="${id}" data-f="2" style="${iStyle}width:52px;" title="Z" />
      <select data-id="${id}" data-f="wall" style="${iStyle}width:70px;">${wallOpts}</select>
      <button data-del="${id}" style="padding:1px 6px;font-size:10px;background:#7f1d1d;border-color:#dc2626;">✕</button>
    `;
    tbody.appendChild(row);
  });
  // Bind change events
  tbody.querySelectorAll('input[data-f]').forEach(el => {
    el.addEventListener('change', () => {
      const id = el.dataset.id, f = parseInt(el.dataset.f);
      if (arenaMarkers[id]) arenaMarkers[id].pos[f] = parseFloat(el.value) || 0;
    });
  });
  tbody.querySelectorAll('select[data-f]').forEach(el => {
    el.addEventListener('change', () => {
      if (arenaMarkers[el.dataset.id]) arenaMarkers[el.dataset.id].wall = el.value;
    });
  });
  tbody.querySelectorAll('input[data-rename]').forEach(el => {
    el.addEventListener('change', () => {
      const oldId = el.dataset.oldid;
      const n = parseInt(el.value, 10);
      if (isNaN(n) || n < 0) { alert('Marker ID must be a non-negative integer.'); el.value = oldId; return; }
      const newId = String(n);
      if (newId === oldId) return;
      if (arenaMarkers[newId]) {
        alert('Marker ID ' + newId + ' already exists.');
        el.value = oldId;
        return;
      }
      arenaMarkers[newId] = arenaMarkers[oldId];
      delete arenaMarkers[oldId];
      renderMarkerTable();
    });
  });
  tbody.querySelectorAll('button[data-del]').forEach(btn => {
    btn.addEventListener('click', () => {
      delete arenaMarkers[btn.dataset.del];
      renderMarkerTable();
    });
  });
}

async function loadArenaConfig() {
  try {
    const r = await fetch('/proxy/arena/config');
    const d = await r.json();
    if (d.arena) {
      document.getElementById('ac_width').value = d.arena.width_m ?? 20;
      document.getElementById('ac_depth').value = d.arena.depth_m ?? 10;
      document.getElementById('ac_hmin').value  = d.arena.height_min_m ?? -1;
      document.getElementById('ac_hmax').value  = d.arena.height_max_m ?? 1;
    }
    if (d.marker_size_m != null) document.getElementById('ac_msize').value = d.marker_size_m;
    if (d.markers) { arenaMarkers = JSON.parse(JSON.stringify(d.markers)); renderMarkerTable(); }
    // Cache target-box metadata for the Target Boxes panel — team &
    // colour come from here, keyed by marker ID.
    window._arenaCfgCache = {
      target_marker_size_m: d.target_marker_size_m,
      target_teams:      d.target_teams || [],
      target_overrides:  d.target_overrides || [],
    };
    // Update arena canvas world dimensions
    if (d.arena) {
      arenaW = d.arena.width_m  || 20;
      arenaD = d.arena.depth_m  || 10;
      arenaOX = -(arenaW / 2);
      arenaOY = 0;
      _updateView();
      drawArena();  // redraw with last known position (no args = use saved state)
    }
  } catch {}
}
// Load arena config once at startup so the Target Boxes panel has
// team/colour metadata immediately (operator doesn't need to click
// \"Show\" on Arena Configuration first).
window.addEventListener('load', () => {
  try { loadArenaConfig(); } catch (e) {}
});

document.getElementById('arena_cfg_toggle').onclick = function() {
  const body = document.getElementById('arena_cfg_body');
  const hidden = body.style.display === 'none';
  body.style.display = hidden ? '' : 'none';
  this.textContent = hidden ? 'Hide' : 'Show';
  if (hidden) loadArenaConfig();
};

document.getElementById('arena_add_marker').onclick = () => {
  const input = document.getElementById('arena_new_marker_id');
  const typed = (input.value || '').trim();
  let newId;
  if (typed !== '') {
    const n = parseInt(typed, 10);
    if (isNaN(n) || n < 0) { alert('Marker ID must be a non-negative integer.'); return; }
    newId = String(n);
    if (arenaMarkers[newId]) {
      alert('Marker ID ' + newId + ' already exists. Pick a different number or clear the field to auto-assign.');
      return;
    }
  } else {
    const ids = Object.keys(arenaMarkers).map(Number).filter(n => !isNaN(n));
    newId = String(ids.length ? Math.max(...ids) + 1 : 1);
  }
  arenaMarkers[newId] = {pos: [0, 0, 0], wall: 'front'};
  // Auto-increment the input so repeated clicks add sequential IDs
  input.value = String(parseInt(newId, 10) + 1);
  renderMarkerTable();
  // Scroll to bottom
  const tb = document.getElementById('arena_marker_table');
  tb.scrollTop = tb.scrollHeight;
};

document.getElementById('arena_save').onclick = async () => {
  // Sync any un-committed inputs
  document.querySelectorAll('#arena_marker_table input[data-f]').forEach(el => {
    const id = el.dataset.id, f = parseInt(el.dataset.f);
    if (arenaMarkers[id]) arenaMarkers[id].pos[f] = parseFloat(el.value) || 0;
  });
  document.querySelectorAll('#arena_marker_table select[data-f]').forEach(el => {
    if (arenaMarkers[el.dataset.id]) arenaMarkers[el.dataset.id].wall = el.value;
  });
  const payload = {
    arena: {
      width_m: parseFloat(document.getElementById('ac_width').value),
      depth_m: parseFloat(document.getElementById('ac_depth').value),
      height_min_m: parseFloat(document.getElementById('ac_hmin').value),
      height_max_m: parseFloat(document.getElementById('ac_hmax').value),
    },
    marker_size_m: parseFloat(document.getElementById('ac_msize').value),
    markers: arenaMarkers,
  };
  const st = document.getElementById('arena_cfg_status');
  st.textContent = 'saving...'; st.style.color = '#94a3b8';
  try {
    const r = await fetch('/proxy/arena/config', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    st.textContent = d.ok ? `\\u2713 saved (${d.marker_count} markers)` : ('error: ' + (d.error||'?'));
    st.style.color = d.ok ? '#22c55e' : '#ef4444';
  } catch(e) { st.textContent = 'error: ' + e.message; st.style.color = '#ef4444'; }
};

document.getElementById('arena_reset').onclick = async () => {
  if (!confirm('Reset arena config to built-in defaults?')) return;
  const st = document.getElementById('arena_cfg_status');
  st.textContent = 'resetting...'; st.style.color = '#94a3b8';
  try {
    const r = await fetch('/proxy/arena/config/reset', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    const d = await r.json();
    if (d.ok) { await loadArenaConfig(); st.textContent = '\\u2713 reset to defaults'; st.style.color = '#22c55e'; }
    else { st.textContent = 'error'; st.style.color = '#ef4444'; }
  } catch(e) { st.textContent = 'error: ' + e.message; st.style.color = '#ef4444'; }
};

// Dynamic arena dimensions (updated when config is loaded)
let ARENA_W_dyn = 20, ARENA_D_dyn = 10;

// ── Live Telemetry Graphs (DISABLED — moved to standalone script block earlier in HTML) ─
/* (function(){
  const WINDOW_S = 10;          // rolling window in seconds
  const SAMPLE_HZ = 20;         // sampling rate (polls lastTelemetry/_lastPos)
  const CANVAS_W = 340, CANVAS_H = 130;
  const GROUPS = [
    {title:'Altitude (cm)',     keys:['height_cm','tof_cm','barometer_cm'], colors:['#22d3ee','#f472b6','#a78bfa']},
    {title:'Attitude (°)',      keys:['pitch','roll','yaw'],                colors:['#22d3ee','#f472b6','#a78bfa']},
    {title:'Velocity (cm/s)',   keys:['vgx','vgy','vgz'],                   colors:['#22d3ee','#f472b6','#a78bfa']},
    {title:'Acceleration',      keys:['agx','agy','agz'],                   colors:['#22d3ee','#f472b6','#a78bfa']},
    {title:'Speed',             keys:['speed'],                              colors:['#22d3ee']},
    {title:'Battery (%)',       keys:['battery'],                            colors:['#34d399']},
    {title:'Temperature (°C)',  keys:['temperature'],                        colors:['#fb923c']},
    {title:'Position (m)',      keys:['pos_x','pos_y','pos_z'],              colors:['#22d3ee','#f472b6','#a78bfa']},
  ];
  const graphs = [];
  let graphsVisible = false;
  let rafId = null;
  let sampleTimer = null;

  function initGraphs() {
    const container = document.getElementById('graphs_container');
    if (!container) { console.error('[graphs] graphs_container not found'); return; }
    if (graphs.length) return;
    GROUPS.forEach(g => {
      const wrap = document.createElement('div');
      wrap.style.cssText = 'background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:6px;';
      const hdr = document.createElement('div');
      hdr.style.cssText = 'font-size:11px;color:#94a3b8;margin-bottom:4px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;';
      let html = '<b style="color:#e2e8f0;">' + g.title + '</b>';
      g.keys.forEach((k,i) => { html += '<span style="color:'+g.colors[i]+';">'+k+'</span>'; });
      hdr.innerHTML = html;
      wrap.appendChild(hdr);
      const canvas = document.createElement('canvas');
      canvas.width = CANVAS_W; canvas.height = CANVAS_H;
      canvas.style.cssText = 'width:100%;height:auto;display:block;background:#020617;border-radius:4px;';
      wrap.appendChild(canvas);
      container.appendChild(wrap);
      graphs.push({keys: g.keys, colors: g.colors, samples: [], canvas, ctx: canvas.getContext('2d')});
    });
    console.log('[graphs] initialized', graphs.length, 'graphs');
  }

  // Sample current telemetry + position into all graphs
  function takeSample() {
    const ts = performance.now();
    // Build a combined sample object from globals
    const t = (typeof lastTelemetry === 'object' && lastTelemetry) ? Object.assign({}, lastTelemetry) : {};
    if (typeof _lastPos !== 'undefined' && Array.isArray(_lastPos)) {
      t.pos_x = _lastPos[0]; t.pos_y = _lastPos[1]; t.pos_z = _lastPos[2];
    }
    graphs.forEach(g => {
      const vals = {};
      let hasAny = false;
      g.keys.forEach(k => {
        const v = t[k];
        if (v != null && !isNaN(v)) { vals[k] = Number(v); hasAny = true; }
        else { vals[k] = null; }
      });
      if (hasAny) g.samples.push({t: ts, vals});
      const cutoff = ts - WINDOW_S * 1000;
      while (g.samples.length > 0 && g.samples[0].t < cutoff) g.samples.shift();
    });
  }

  function drawAll() {
    if (!graphsVisible) { rafId = null; return; }
    const now = performance.now();
    graphs.forEach(g => {
      const ctx = g.ctx, W = g.canvas.width, H = g.canvas.height;
      ctx.fillStyle = '#020617'; ctx.fillRect(0,0,W,H);
      if (g.samples.length < 2) {
        ctx.fillStyle = '#475569'; ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('waiting for data...', W/2, H/2);
        ctx.textAlign = 'left';
        return;
      }
      const tMin = now - WINDOW_S*1000, tMax = now;
      let yMin = Infinity, yMax = -Infinity;
      g.samples.forEach(s => { g.keys.forEach(k => {
        if (s.vals[k]!=null) { yMin=Math.min(yMin,s.vals[k]); yMax=Math.max(yMax,s.vals[k]); }
      }); });
      if (!isFinite(yMin) || !isFinite(yMax)) return;
      if (yMin === yMax) { yMin -= 1; yMax += 1; }
      const pad = (yMax - yMin) * 0.1 || 1;
      yMin -= pad; yMax += pad;
      // grid
      ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 0.5;
      for (let i=0;i<=4;i++) {
        const y = (i/4)*H;
        ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke();
      }
      // y labels
      ctx.fillStyle = '#64748b'; ctx.font = '9px monospace';
      for (let i=0;i<=4;i++) {
        const v = yMin + ((4-i)/4)*(yMax-yMin);
        ctx.fillText(v.toFixed(1), 2, (i/4)*H + 9);
      }
      // each series
      g.keys.forEach((k, ki) => {
        ctx.strokeStyle = g.colors[ki]; ctx.lineWidth = 1.5; ctx.beginPath();
        let started = false;
        g.samples.forEach(s => {
          if (s.vals[k] == null) { started = false; return; }
          const x = ((s.t - tMin) / (tMax - tMin)) * W;
          const y = H - ((s.vals[k] - yMin) / (yMax - yMin)) * H;
          if (!started) { ctx.moveTo(x,y); started = true; } else { ctx.lineTo(x,y); }
        });
        ctx.stroke();
        // last value label
        const last = g.samples[g.samples.length - 1];
        if (last && last.vals[k] != null) {
          ctx.fillStyle = g.colors[ki]; ctx.font = '10px monospace';
          ctx.textAlign = 'right';
          ctx.fillText(last.vals[k].toFixed(1), W - 2, 10 + ki * 11);
          ctx.textAlign = 'left';
        }
      });
    });
    rafId = requestAnimationFrame(drawAll);
  }

  function startGraphs() {
    initGraphs();
    if (!sampleTimer) sampleTimer = setInterval(takeSample, 1000 / SAMPLE_HZ);
    if (!rafId) rafId = requestAnimationFrame(drawAll);
  }
  function stopGraphs() {
    if (sampleTimer) { clearInterval(sampleTimer); sampleTimer = null; }
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  }

  // Wire toggle button — wait for DOM if needed
  function wireToggle() {
    const btn = document.getElementById('graphs_toggle');
    const panel = document.getElementById('graphs_panel');
    if (!btn || !panel) {
      console.warn('[graphs] button or panel not found, retrying...');
      setTimeout(wireToggle, 100);
      return;
    }
    btn.addEventListener('click', () => {
      graphsVisible = !graphsVisible;
      btn.textContent = graphsVisible ? 'Hide Graphs' : 'Show Graphs';
      panel.style.display = graphsVisible ? 'block' : 'none';
      console.log('[graphs] toggled', graphsVisible);
      if (graphsVisible) startGraphs(); else stopGraphs();
    });
    console.log('[graphs] toggle wired');
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wireToggle);
  } else {
    wireToggle();
  }
})(); */
</script>

<!-- ── Optional 3D arena view via Three.js ─────────────────────────── -->
<script type=\"module\">
  import * as THREE from 'three';
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

  let scene, camera, renderer, controls, droneMeshes = {}, markerMeshes = {}, rafId = 0;
  const ARENA_W = 20.0, ARENA_D = 10.8, ARENA_H = 6.0;
  const ARENA_OX = -10.0, ARENA_OY = 0.0;

  function init3D(container) {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1220);

    camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 200);
    camera.position.set(15, 12, 15);

    renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.5, 5.4);
    controls.update();

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.45);
    scene.add(ambient);
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(5, 12, 8);
    scene.add(dir);

    // Arena floor — 1 m grid (matches the 2D overlay)
    const grid = new THREE.GridHelper(20, 20, 0x475569, 0x1e3a5f);
    grid.position.set(0, 0, ARENA_D / 2);
    scene.add(grid);
    // Depth-direction grid (10.8 m, we round to 11 cells)
    const grid2 = new THREE.GridHelper(12, 12, 0x475569, 0x1e3a5f);
    grid2.rotation.x = Math.PI / 2;
    grid2.position.set(0, 3, 0);
    grid2.material.opacity = 0.0;
    // keep subtle — 1 grid is enough; the floor grid is what matters

    // Arena box (wireframe)
    const boxGeom = new THREE.BoxGeometry(ARENA_W, ARENA_H, ARENA_D);
    const boxMat = new THREE.LineBasicMaterial({color: 0x3b82f6, transparent: true, opacity: 0.4});
    const box = new THREE.LineSegments(new THREE.EdgesGeometry(boxGeom), boxMat);
    box.position.set(0, ARENA_H / 2, ARENA_D / 2);
    scene.add(box);

    // Arena origin marker (green corner cube)
    const originGeom = new THREE.BoxGeometry(0.3, 0.3, 0.3);
    const origin = new THREE.Mesh(originGeom, new THREE.MeshStandardMaterial({color: 0x10b981}));
    origin.position.set(0, 0.15, 0);
    scene.add(origin);

    // Helper — build a floating canvas-sprite label with a given text.
    // Canvas resolution is fixed; sprite.scale controls world size.
    function _makeLabelSprite(text, bgRGBA, fgRGB, fontPx) {
      const canvas = document.createElement('canvas');
      canvas.width = 128; canvas.height = 56;
      const c = canvas.getContext('2d');
      c.fillStyle = bgRGBA;
      // Rounded rect for a nicer badge look
      const r = 8;
      c.beginPath();
      c.moveTo(r, 0);
      c.lineTo(canvas.width - r, 0);
      c.quadraticCurveTo(canvas.width, 0, canvas.width, r);
      c.lineTo(canvas.width, canvas.height - r);
      c.quadraticCurveTo(canvas.width, canvas.height, canvas.width - r, canvas.height);
      c.lineTo(r, canvas.height);
      c.quadraticCurveTo(0, canvas.height, 0, canvas.height - r);
      c.lineTo(0, r);
      c.quadraticCurveTo(0, 0, r, 0);
      c.closePath();
      c.fill();
      c.fillStyle = fgRGB;
      c.font = 'bold ' + (fontPx || 32) + 'px monospace';
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText(String(text), canvas.width / 2, canvas.height / 2 + 2);
      const tex = new THREE.CanvasTexture(canvas);
      tex.minFilter = THREE.LinearFilter;
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: tex, transparent: true, depthTest: false}));
      sprite.renderOrder = 100;   // float on top of everything
      return sprite;
    }

    // Fetch + plot arena markers (from the arena_config we already fetched).
    // Each marker gets a coloured face cube AND a floating ID label so the
    // 3D scene is immediately readable without hovering / guessing.
    if (window.arenaMarkers && Object.keys(window.arenaMarkers).length) {
      for (const [id, m] of Object.entries(window.arenaMarkers)) {
        if (!m.pos) continue;
        const g = new THREE.BoxGeometry(0.5, 0.5, 0.05);
        const col = (m.wall === 'front') ? 0x6366f1
                 : (m.wall === 'back')  ? 0xa855f7
                 : (m.wall === 'left')  ? 0x06b6d4
                 : (m.wall === 'right') ? 0x10b981 : 0x94a3b8;
        // Wrap in a group so we can carry a label sprite alongside the cube.
        const grp = new THREE.Group();
        const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({color: col}));
        grp.add(mesh);
        const label = _makeLabelSprite(id, 'rgba(15,23,42,0.85)', '#e2e8f0', 34);
        label.scale.set(0.55, 0.24, 1);     // world units — ~55 cm wide
        label.position.set(0, 0.4, 0);      // above the cube
        grp.add(label);
        grp.position.set(m.pos[0], m.pos[2] || 2, m.pos[1]);
        scene.add(grp);
        // Keep the Mesh in markerMeshes (updateVisibleMarkers mutates the
        // material on the cube, not the group).
        mesh.userData._label = label;       // so updateVisibleMarkers can tint
        mesh.userData._group = grp;
        markerMeshes[id] = mesh;
      }
    }

    window.addEventListener('resize', () => {
      if (!renderer) return;
      const w = container.clientWidth, h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });

    // Render the initial set of target boxes as soon as the scene opens
    syncTargetBoxes();

    function loop() {
      controls.update();
      renderer.render(scene, camera);
      rafId = requestAnimationFrame(loop);
    }
    loop();
  }

  // ── Target-box 3D rendering ────────────────────────────────────
  // Kept separately so we can call it both at scene-init AND every
  // time the textarea / mission status updates the list.
  const targetBoxMeshes = {};   // index → THREE.Mesh
  function syncTargetBoxes() {
    if (!scene) return;
    const boxes = window._targetBoxes || [];
    const claimed = window._missionClaimedBoxes || {};
    const captured = new Set(window._missionCapturedBoxes || []);
    const seen = new Set();
    boxes.forEach((b, i) => {
      if (!b || b.x == null || b.y == null) return;
      const idx = (typeof b.idx === 'number') ? b.idx : i;
      seen.add(idx);
      const isCap = captured.has(idx);
      const isClaimed = Object.values(claimed).includes(idx);
      const team = (b.home_team || '').toLowerCase();
      const col = isCap     ? 0x22c55e
                : isClaimed ? 0xfacc15
                : team === 'red'  ? 0xef4444
                : team === 'blue' ? 0x3b82f6
                                  : 0x94a3b8;
      let mesh = targetBoxMeshes[idx];
      if (!mesh) {
        const group = new THREE.Group();
        // SDC box dimensions (rules §1.2): 57.5×37.5×55 cm closed,
        // up to 73 cm when open. We render 0.575×0.55×0.375 as an
        // approximation sitting on the floor.
        const body = new THREE.Mesh(
          new THREE.BoxGeometry(0.575, 0.55, 0.375),
          new THREE.MeshStandardMaterial({color: col, transparent:true, opacity:0.85}));
        body.position.y = 0.275;
        group.add(body);
        const ring = new THREE.Mesh(
          new THREE.TorusGeometry(0.42, 0.04, 8, 24),
          new THREE.MeshStandardMaterial({color: col, transparent:true, opacity:0.55}));
        ring.rotation.x = Math.PI / 2;
        ring.position.y = 0.02;
        group.add(ring);
        // Label
        const canvas = document.createElement('canvas');
        canvas.width = 128; canvas.height = 36;
        const c = canvas.getContext('2d');
        c.fillStyle = 'rgba(0,0,0,0.7)'; c.fillRect(0,0,128,36);
        c.fillStyle = '#fff'; c.font = 'bold 20px monospace';
        c.fillText('BOX ' + (b.id ?? idx+1), 6, 25);
        const tex = new THREE.CanvasTexture(canvas);
        const sprite = new THREE.Sprite(
          new THREE.SpriteMaterial({map: tex, transparent: true}));
        sprite.scale.set(1.1, 0.32, 1);
        sprite.position.set(0, 0.95, 0);
        group.add(sprite);
        group.userData.body = body;
        group.userData.ring = ring;
        scene.add(group);
        targetBoxMeshes[idx] = group;
        mesh = group;
      }
      // Update position + colour every frame in case they moved or state changed
      mesh.position.set(Number(b.x), 0, Number(b.y));
      if (mesh.userData.body) {
        mesh.userData.body.material.color.setHex(col);
        mesh.userData.body.material.opacity = isCap ? 0.55 : 0.85;
      }
      if (mesh.userData.ring) {
        mesh.userData.ring.material.color.setHex(col);
      }
    });
    // Remove meshes for boxes no longer in the list
    for (const idx of Object.keys(targetBoxMeshes)) {
      if (!seen.has(Number(idx))) {
        scene.remove(targetBoxMeshes[idx]);
        delete targetBoxMeshes[idx];
      }
    }
  }

  function updateDrones(observers) {
    if (!scene) return;
    const DRONE_COLORS = [0xf97316, 0x38bdf8, 0xa78bfa, 0xf472b6, 0xfbbf24];
    let idx = 0;
    const seen = new Set();
    for (const [did, st] of Object.entries(observers || {})) {
      seen.add(did);
      const p = st && (st.pos || st.cam);
      if (!p || p.length < 2) { idx++; continue; }
      let mesh = droneMeshes[did];
      if (!mesh) {
        const col = DRONE_COLORS[idx % DRONE_COLORS.length];
        // Drone = small sphere with a forward nose-cone
        const group = new THREE.Group();
        const body = new THREE.Mesh(
          new THREE.SphereGeometry(0.18, 16, 12),
          new THREE.MeshStandardMaterial({color: col}));
        group.add(body);
        const nose = new THREE.Mesh(
          new THREE.ConeGeometry(0.08, 0.3, 8),
          new THREE.MeshStandardMaterial({color: 0xffffff}));
        nose.rotation.x = Math.PI / 2;
        nose.position.set(0, 0, 0.2);
        group.add(nose);
        // Label (using a sprite)
        const canvas = document.createElement('canvas');
        canvas.width = 128; canvas.height = 32;
        const c = canvas.getContext('2d');
        c.fillStyle = 'rgba(0,0,0,0.7)'; c.fillRect(0,0,128,32);
        c.fillStyle = '#fff'; c.font = 'bold 18px monospace';
        c.fillText(did, 8, 22);
        const tex = new THREE.CanvasTexture(canvas);
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: tex, transparent: true}));
        sprite.scale.set(0.9, 0.22, 1);
        sprite.position.set(0, 0.4, 0);
        group.add(sprite);
        scene.add(group);
        mesh = group;
        droneMeshes[did] = mesh;
      }
      // Map ArUco-arena coords → Three.js world coords:
      //   ArUco  x = arena horizontal  →  Three.js X
      //   ArUco  y = arena depth       →  Three.js Z (forward)
      //   ArUco  z = altitude          →  Three.js Y (up)
      // Use a typeof check instead of `||` so a legit z=0 reading isn't
      // silently replaced by the 1.5 m fallback (which was hiding grounded
      // or pre-takeoff drones on the cover of the arena).
      const ax = Number(p[0]) || 0;
      const ay = Number(p[1]) || 0;
      const az = (typeof p[2] === 'number' && isFinite(p[2]))
                   ? p[2]
                   : (Number(st.altitude_m) || 1.5);
      mesh.position.set(ax, Math.max(0, az), ay);
      // Heading: prefer the ArUco-derived direction vector (dx,dy) in the
      // arena frame because it matches the pose we just plotted. Fall back
      // to the compass yaw from drone telemetry if the positioner is stale.
      if (Array.isArray(st.dir) && st.dir.length >= 2 &&
          (st.dir[0]*st.dir[0] + st.dir[1]*st.dir[1]) > 1e-6) {
        const hdg = Math.atan2(st.dir[0], st.dir[1]);   // rad from +Y axis
        mesh.rotation.y = -hdg;
      } else if (typeof st.yaw === 'number') {
        mesh.rotation.y = -st.yaw * Math.PI / 180;
      }
      // Flash opacity if the pose is stale so operators see the drone is
      // running on IMU dead-reckoning rather than live vision.
      const staleAlpha = (st.pos_stale === true) ? 0.55 : 1.0;
      mesh.traverse(o => {
        if (o.material && 'opacity' in o.material) {
          o.material.transparent = staleAlpha < 1.0;
          o.material.opacity = staleAlpha;
        }
      });
      idx++;
    }
    // Remove meshes for drones that disappeared
    for (const did of Object.keys(droneMeshes)) {
      if (!seen.has(did)) {
        scene.remove(droneMeshes[did]);
        delete droneMeshes[did];
      }
    }
  }

  function teardown3D() {
    if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
    if (renderer) {
      renderer.dispose();
      if (renderer.domElement && renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
      renderer = null;
    }
    scene = null; camera = null; controls = null;
    droneMeshes = {}; markerMeshes = {};
    // Remove the HUD overlay DIV so it doesn't show stale coordinates
    // when the 3D view gets re-enabled later.
    if (_hudEl && _hudEl.parentNode) {
      _hudEl.parentNode.removeChild(_hudEl);
    }
    _hudEl = null;
  }

  // ── Visible-marker highlight for the 3D arena ─────────────────────
  // Mirrors the 2D halo/white-border on seen markers. We cache each
  // marker's base colour + emissive on first create and mutate only
  // on state changes — cheaper than rebuilding materials per frame.
  function updateVisibleMarkers(seenIds, refIds) {
    if (!scene) return;
    const seen = new Set((seenIds || []).map(String));
    const refs = new Set((refIds  || []).map(String));
    for (const [id, mesh] of Object.entries(markerMeshes)) {
      if (!mesh || !mesh.material) continue;
      if (mesh.userData._baseColor == null) {
        // Cache originals the first time we touch this mesh.
        mesh.userData._baseColor    = mesh.material.color.getHex();
        mesh.userData._baseEmissive = mesh.material.emissive
                                        ? mesh.material.emissive.getHex() : 0;
        mesh.userData._baseScale    = mesh.scale.x;
      }
      const isSeen = seen.has(String(id));
      const isRef  = refs.has(String(id));
      const lbl = mesh.userData._label;
      if (isSeen) {
        // Bright yellow (or green if actually used for pose fusion).
        const highlightColor = isRef ? 0x22c55e : 0xfbbf24;
        mesh.material.color.setHex(highlightColor);
        if (mesh.material.emissive) {
          mesh.material.emissive.setHex(highlightColor);
          mesh.material.emissiveIntensity = 0.6;
        }
        mesh.scale.setScalar(mesh.userData._baseScale * 1.35);
        if (lbl) lbl.scale.set(0.75, 0.32, 1);  // grow label too
      } else {
        mesh.material.color.setHex(mesh.userData._baseColor);
        if (mesh.material.emissive) {
          mesh.material.emissive.setHex(mesh.userData._baseEmissive);
          mesh.material.emissiveIntensity = 0;
        }
        mesh.scale.setScalar(mesh.userData._baseScale);
        if (lbl) lbl.scale.set(0.55, 0.24, 1);
      }
    }
  }

  // ── Drone position HUD — floating DIV above the 3D container ──────
  // Three.js scenes lack an obvious "show me the active drone's xyz"
  // readout. Add a small top-left overlay that shows per-drone
  // coordinates; falls out of the container's bottom if there are many
  // drones, which is fine for up to ~5.
  let _hudEl = null;
  function _ensureHUD() {
    if (_hudEl) return _hudEl;
    const wrap = document.getElementById('arena3d_wrap');
    if (!wrap) return null;
    _hudEl = document.createElement('div');
    _hudEl.id = 'arena3d_hud';
    _hudEl.style.cssText = [
      'position:absolute', 'top:6px', 'left:6px', 'z-index:10',
      'padding:4px 8px',
      'background:rgba(15,23,42,0.8)',
      'color:#e2e8f0', 'font-family:monospace', 'font-size:11px',
      'line-height:1.4', 'border:1px solid #334155', 'border-radius:4px',
      'pointer-events:none', 'max-width:240px',
    ].join(';') + ';';
    wrap.appendChild(_hudEl);
    return _hudEl;
  }
  function updateDronePositionHUD(observers) {
    const el = _ensureHUD();
    if (!el) return;
    const lines = [];
    const DRONE_COLORS = ['#f97316', '#38bdf8', '#a78bfa', '#f472b6', '#fbbf24'];
    let idx = 0;
    for (const [did, st] of Object.entries(observers || {})) {
      const p = st && (st.pos || st.cam);
      const col = DRONE_COLORS[idx % DRONE_COLORS.length];
      idx++;
      if (!Array.isArray(p) || p.length < 2) {
        lines.push('<span style="color:#64748b;">drone ' + did + ': no fix</span>');
        continue;
      }
      const x = Number(p[0]).toFixed(2);
      const y = Number(p[1]).toFixed(2);
      const z = (typeof p[2] === 'number' && isFinite(p[2])) ? Number(p[2]).toFixed(2) : '–';
      const stale = st.pos_stale ? ' <span style="color:#fbbf24;">stale</span>' : '';
      lines.push(
        '<span style="color:' + col + ';font-weight:700;">drone ' + did + '</span> ' +
        '<span style="color:#38bdf8;">x=' + x + '</span> ' +
        '<span style="color:#4ade80;">y=' + y + '</span> ' +
        '<span style="color:#fb923c;">z=' + z + '</span>' + stale
      );
    }
    if (!lines.length) lines.push('<span style="color:#64748b;">no drones</span>');
    el.innerHTML = lines.join('<br/>');
  }

  // ── Safety-margin box in the 3D arena ─────────────────────────────
  // A translucent red wireframe cuboid indicating the Pi-side arena
  // guard boundary. Created lazily the first time updateSafetyMargin
  // is called; subsequent calls just rescale it.
  let safetyBoxMesh = null;
  function updateSafetyMargin(marginM, engaged) {
    if (!scene) return;
    if (marginM == null || marginM <= 0) {
      if (safetyBoxMesh) { scene.remove(safetyBoxMesh); safetyBoxMesh = null; }
      return;
    }
    // Dimensions: arena minus 2×margin on each horizontal axis; height
    // matches the ceiling (MAX_ALTITUDE_M comes from the safety bar).
    const w = Math.max(0.2, ARENA_W - 2 * marginM);
    const d = Math.max(0.2, ARENA_D - 2 * marginM);
    const ceilM = (window._arenaSafety && window._arenaSafety.ceiling_m)
                    || parseFloat((document.getElementById('ceiling_input') || {}).value)
                    || 5.0;
    const h = Math.max(0.5, ceilM);
    if (!safetyBoxMesh) {
      const geom = new THREE.BoxGeometry(w, h, d);
      const edges = new THREE.EdgesGeometry(geom);
      const mat = new THREE.LineBasicMaterial({
        color: 0xef4444, transparent: true, opacity: 0.35,
      });
      safetyBoxMesh = new THREE.LineSegments(edges, mat);
      scene.add(safetyBoxMesh);
    } else {
      // Rebuild geometry on dimension change — simpler than scaling.
      safetyBoxMesh.geometry.dispose();
      const geom = new THREE.BoxGeometry(w, h, d);
      safetyBoxMesh.geometry = new THREE.EdgesGeometry(geom);
    }
    // Centre of arena: x=0, depth=arena_depth/2, y=height/2.
    safetyBoxMesh.position.set(0, h / 2, ARENA_D / 2);
    // Brighter + thicker line when guard is actively clamping.
    if (safetyBoxMesh.material) {
      safetyBoxMesh.material.opacity = engaged ? 0.85 : 0.35;
      safetyBoxMesh.material.color.setHex(engaged ? 0xfca5a5 : 0xef4444);
    }
  }

  // Expose for fleetPoll()
  window._arena3d = {updateDrones, syncTargetBoxes, updateVisibleMarkers,
                     updateDronePositionHUD, updateSafetyMargin};

  // Toggle wiring — 3D view is shown BY DEFAULT below the 2D canvas.
  // The 2D canvas stays visible the whole time so operators have the
  // top-down reference; unchecking simply tears down the 3D scene.
  const cb = document.getElementById('arena_show_3d');
  const wrap = document.getElementById('arena3d_wrap');
  const container = document.getElementById('arena3d_container');
  function apply3DState() {
    if (cb.checked) {
      wrap.style.display = '';
      if (!scene) {
        try { init3D(container); }
        catch (e) { console.error('[3D] init failed:', e); cb.checked = false; wrap.style.display = 'none'; }
      }
    } else {
      wrap.style.display = 'none';
      teardown3D();
    }
  }
  if (cb && wrap && container) {
    cb.addEventListener('change', apply3DState);
    // Start in the default state — 3D active alongside the 2D canvas.
    // Defer a tick so the container has layout dimensions before THREE initialises.
    setTimeout(apply3DState, 0);
  }
</script>
</body>
</html>
"""


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
    return Response(HTML, mimetype="text/html")


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


def _fetch_takeoff_failure_reason(base_url: str) -> dict:
    """After a takeoff timeout/error, best-effort fetch the FC's live
    diagnostic state so we can tell the operator WHAT went wrong instead
    of just "timed out". Short 1.5 s timeout so an unreachable FC
    doesn't compound the problem."""
    out: dict = {}
    base = base_url.rstrip("/")
    try:
        r = _http_session.get(f"{base}/api/telemetry", timeout=1.5)
        if r.ok:
            tel = r.json() or {}
            for k in ("magneto_status", "magneto_required", "battery",
                      "flying", "connected"):
                if k in tel:
                    out[k] = tel[k]
    except Exception:
        pass
    return out


def _format_takeoff_error(raw_error: str, diag: dict) -> tuple[str, str, str]:
    """Map a backend (reason_code|detail) or plain string into
    (reason_code, short_title, long_hint) for the UI.

    Returns ("magnetometer_calibration_required", "Magnetometer needs
    calibration", "Lay the drone flat ... FreeFlight ...") etc."""
    # Parse reason_code from the FC's "code|message" wire format
    code = None
    detail = raw_error or ""
    if isinstance(detail, str) and "|" in detail:
        code, _, detail = detail.partition("|")
    # Magneto is the single most common cause — detect from diagnostic too
    mag_bad = bool(diag.get("magneto_required")) or (
        isinstance(diag.get("magneto_status"), str)
        and "required" in diag["magneto_status"].lower().split(",")
    )
    if code == "magnetometer_calibration_required" or mag_bad:
        return (
            "magnetometer_calibration_required",
            "Magnetometer needs calibration",
            "Pick up the drone and rotate it through a figure-8 on each "
            "axis (roll, pitch, yaw) until the Parrot FreeFlight app "
            "reports all axes OK. Then retry takeoff.",
        )
    if code == "alert_state":
        return ("alert_state", f"Drone alert: {detail}",
                "Clear the alert state and retry. Common causes: low "
                "battery (swap pack), excessive tilt (level the drone), "
                "motor cut-out (wait 5 s then retry).")
    if code == "motor_error":
        return ("motor_error", f"Motor fault: {detail}",
                "Power-cycle the drone, check props aren't obstructed.")
    if code == "sensor_fault":
        return ("sensor_fault", f"Sensor fault: {detail}",
                "Restart the drone and let it re-initialise on a level surface.")
    if code == "takeoff_timeout":
        return ("takeoff_timeout", "Takeoff command timed out",
                f"The FC didn't respond within {TIMEOUT_SLOW} s. The drone "
                f"may still be arming; check telemetry for flying=true "
                f"before retrying.")
    return (code or "takeoff_failed", detail or "takeoff failed",
            "Check the FC log — no specific fault flagged by Olympe.")


@app.post("/proxy/takeoff")
def proxy_takeoff():
    """Relay takeoff to the active drone's Pi.

    Uses TIMEOUT_SLOW (default 15s) because Anafi takeoff takes 3-5 s
    for armament + motor spin-up before /api/takeoff returns. On
    timeout or FC-side failure we best-effort fetch the drone's live
    diagnostic state (magnetometer, alert, battery) so the operator
    sees the ACTUAL reason instead of just "timed out".
    """
    log_command("takeoff")
    base = PI_BASE
    try:
        r = pi_post("/api/takeoff", timeout=TIMEOUT_SLOW)
    except Exception as e:
        import requests as _rq
        diag = _fetch_takeoff_failure_reason(base)
        # Did we find a magneto-calibration-needed flag even though the
        # call timed out? Surface that as the real reason.
        code, title, hint = _format_takeoff_error(
            "takeoff_timeout|" + (
                f"takeoff timed out after {TIMEOUT_SLOW}s — FC did not respond"
                if isinstance(e, _rq.exceptions.ReadTimeout)
                else f"network error: {e}"
            ),
            diag,
        )
        return jsonify(
            ok=False, error=title, reason_code=code, hint=hint,
            diagnostic=diag,
        ), (504 if isinstance(e, _rq.exceptions.ReadTimeout) else 502)
    # FC responded — parse the body so we can normalise the error shape
    try:
        body = r.json() if r.content else {}
    except Exception:
        body = {}
    if r.ok and body.get("ok") is not False:
        return (r.text, r.status_code,
                {"Content-Type": r.headers.get("Content-Type", "application/json")})
    # FC-side refusal → surface structured reason
    diag = body.get("diagnostic") or {}
    if not diag:
        diag = _fetch_takeoff_failure_reason(base)
    raw_err = body.get("error") or ""
    # If the FC's error string is in the "code|message" format, it's
    # already structured. Otherwise, let the formatter infer from diag.
    code, title, hint = _format_takeoff_error(raw_err, diag)
    return jsonify(
        ok=False, error=title, reason_code=code, hint=hint,
        diagnostic=diag,
    ), r.status_code if r.status_code >= 400 else 500


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
      "target_boxes": [{"id":1,"x":-5.0,"y":2.0}, ...],
      "home_xy":      [x, y],                    # team home coord
      "arena_face_xy":[x, y],                    # where the camera aims
      "hover_above_m": 1.5,
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
    Returns [{id, x, y, home_team}, ...] in visit order."""
    remaining = {int(tid): [float(p[0]), float(p[1])] for tid, p in targets.items()}
    cx, cy = float(start_xy[0]), float(start_xy[1])
    order = []
    while remaining:
        best_tid = min(
            remaining,
            key=lambda t: (remaining[t][0] - cx) ** 2 + (remaining[t][1] - cy) ** 2,
        )
        bx, by = remaining.pop(best_tid)
        order.append({
            "id":        best_tid,
            "x":         round(bx, 3),
            "y":         round(by, 3),
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
    drone_id: str,
    rotation_deg: int,
    rotation_steps: int,
    dwell_s: float,
    hover_seconds: float,
    hover_above_m: float,
    home_xy: tuple,
    arena_face_xy: tuple,
    nav_tol_xy_m: float,
):
    info = DRONES.get(drone_id) or {}
    base = (info or {}).get("base")
    if not base:
        _scan_cap_set(active=False, phase="error", result="error",
                      last_error=f"drone {drone_id} has no base URL",
                      ended_at=time.time())
        return
    base = base.rstrip("/")
    try:
        # ── Phase 1: ensure airborne ─────────────────────────────
        _scan_cap_set(phase="takeoff", step_name="Checking flight state")
        tel = {}
        try:
            r = _http_session.get(f"{base}/api/telemetry", timeout=TIMEOUT_FAST)
            if r.ok:
                tel = r.json() or {}
        except Exception:
            pass
        if not tel.get("flying"):
            _scan_cap_set(step_name="Taking off")
            r = _http_session.post(f"{base}/api/takeoff", json={}, timeout=TIMEOUT_SLOW)
            if not r.ok:
                raise RuntimeError(f"takeoff failed: HTTP {r.status_code}")
            body = r.json() if r.content else {}
            if not body.get("ok"):
                raise RuntimeError(f"takeoff failed: {body.get('error','unknown')}")
            time.sleep(4.0)  # settle — same as safe_takeoff_s
        else:
            _scan_cap_set(step_name="Already airborne")

        # Brief settle hover before we start rotating — lets the
        # positioner latch onto the arena references cleanly.
        for _ in range(20):
            if _scan_cap_abort.is_set(): break
            time.sleep(0.1)

        # ── Phase 2: scan (rotate + accumulate targets) ──────────
        _scan_cap_set(phase="scanning", step_name=f"Rotating 360° in {rotation_steps}×{rotation_deg}° steps")
        accumulated: dict = {}
        accept_ids = set(range(31, 37)) | set(range(41, 47))  # Blue 1-6, Red 1-6
        for step in range(rotation_steps):
            if _scan_cap_abort.is_set():
                raise RuntimeError("aborted during scan")
            _scan_cap_set(step_name=f"Rotation {step+1}/{rotation_steps} — CW {rotation_deg}°")
            try:
                r = _http_session.post(
                    f"{base}/api/rotate",
                    json={"dir": "cw", "deg": rotation_deg},
                    timeout=TIMEOUT_SLOW,
                )
                if not r.ok:
                    print(f"[SCAN_CAP] rotate {step+1} HTTP {r.status_code}: {r.text[:120]}")
            except Exception as e:
                print(f"[SCAN_CAP] rotate {step+1} failed: {e}")
            # Observation dwell — poll targets at 3-4 Hz, latch positions
            t_end = time.time() + dwell_s
            while time.time() < t_end:
                if _scan_cap_abort.is_set():
                    raise RuntimeError("aborted during dwell")
                try:
                    r = _http_session.get(f"{base}/api/position", timeout=TIMEOUT_FAST)
                    if r.ok:
                        state = r.json() or {}
                        targets = state.get("targets") or {}
                        for tid_str, tinfo in targets.items():
                            try:
                                tid = int(tid_str)
                            except Exception:
                                continue
                            if tid not in accept_ids:
                                continue
                            # Only accept a FRESH observation (seen in the
                            # latest frame) so we don't latch onto a stale
                            # TTL entry from before the rotation started.
                            if not tinfo.get("fresh"):
                                continue
                            pos = tinfo.get("pos")
                            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                                accumulated[tid] = [float(pos[0]), float(pos[1]), float(pos[2])]
                except Exception:
                    pass
                time.sleep(0.28)
            _scan_cap_set(n_detected=len(accumulated), targets_found=dict(accumulated))

        if not accumulated:
            raise RuntimeError("no target boxes detected during scan — nothing to visit")

        # ── Phase 3: plan visit order + start capture mission ────
        _scan_cap_set(phase="capture", step_name="Building visit plan")
        # Start from current drone XY (from /api/position), fall back to home_xy
        plan_start = home_xy
        try:
            r = _http_session.get(f"{base}/api/position", timeout=TIMEOUT_FAST)
            if r.ok:
                pos = (r.json() or {}).get("pos")
                if pos and len(pos) >= 2:
                    plan_start = (float(pos[0]), float(pos[1]))
        except Exception:
            pass
        target_boxes = _scan_cap_nn_order(plan_start, accumulated)
        _scan_cap_set(target_boxes=target_boxes,
                      step_name=f"Visiting {len(target_boxes)} targets")

        ok, msg = mission_manager.start_capture_all_targets(
            drone_ids=[drone_id],
            target_boxes=target_boxes,
            home_xy=tuple(home_xy),
            arena_face_xy=tuple(arena_face_xy),
            hover_above_m=hover_above_m,
            hover_seconds=hover_seconds,
            nav_tol_xy_m=nav_tol_xy_m,
            auto_takeoff=False,   # drone is already flying
        )
        if not ok:
            raise RuntimeError(f"capture mission refused to start: {msg}")
        _scan_cap_set(phase="done", result="ok", step_name="Handed off to capture mission",
                      ended_at=time.time())
        log_command("scan_and_capture_done", {
            "drone_id": drone_id,
            "n_detected": len(accumulated),
            "target_boxes": target_boxes,
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
    """Scan-and-capture: drone takes off (if needed), rotates to discover
    SDC26 target boxes via ArUco, then visits each using the standard
    CaptureAllTargetsMission. Body:
      {
        "drone_id":       "2",              # optional; active drone if omitted
        "rotation_deg":   60,               # degrees per rotate step
        "rotation_steps": 6,                # full 360° by default
        "dwell_s":        2.0,              # observation window per step
        "hover_seconds":  3.0,              # dwell over each target box
        "hover_above_m":  1.5,              # altitude above the target marker
        "home_xy":        [0.0, 1.5],       # landing / return point
        "arena_face_xy":  [0.0, 5.4],       # where camera points during capture
        "nav_tol_xy_m":   0.3               # arrival tolerance
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
    drone_id = str(data.get("drone_id") or active_drone_id)
    if not DRONES.get(drone_id):
        return jsonify(ok=False, error=f"unknown drone_id {drone_id}"), 400
    rotation_deg   = max(10, min(180, int(data.get("rotation_deg", 60))))
    rotation_steps = max(1, min(24, int(data.get("rotation_steps", 6))))
    dwell_s        = max(0.5, min(10.0, float(data.get("dwell_s", 2.0))))
    hover_seconds  = max(1.0, min(20.0, float(data.get("hover_seconds", 3.0))))
    hover_above_m  = max(0.5, min(5.0, float(data.get("hover_above_m", 1.5))))
    home_xy        = data.get("home_xy") or [0.0, 1.5]
    arena_face_xy  = data.get("arena_face_xy") or [0.0, 5.4]
    nav_tol_xy_m   = max(0.1, min(2.0, float(data.get("nav_tol_xy_m", 0.3))))
    with _scan_cap_lock:
        _scan_cap_state.update({
            "active":        True,
            "phase":         "starting",
            "drone_id":      drone_id,
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
        args=(drone_id, rotation_deg, rotation_steps, dwell_s,
              hover_seconds, hover_above_m,
              tuple(home_xy), tuple(arena_face_xy), nav_tol_xy_m),
        daemon=True, name="scan-and-capture",
    ).start()
    log_command("scan_and_capture_start", {
        "drone_id": drone_id, "rotation_steps": rotation_steps,
        "hover_seconds": hover_seconds,
    })
    return jsonify(ok=True, message="scan-and-capture started",
                   drone_id=drone_id)


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
