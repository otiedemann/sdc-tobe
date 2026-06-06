"""In-process implementation of the DroneApi surface.

Same public methods as :class:`marker_mission.drone_api.DroneApi`, but
every call dispatches into :mod:`controller_unified.drone_core` and
shares the unified-server's globals in the same Python process — no
HTTP, no JSON encode/decode, no socket round-trip. Used when the
combined Flask app (``marker_mission.app``) runs marker_mission and
the unified controller side-by-side in one process.

The HTTP variant (:mod:`marker_mission.drone_api`) stays the default
for deployments where marker_mission talks to a remote unified-server.
Pick between them via ``MissionConfig.in_process`` or the
``--in-process`` CLI flag in ``mission.py``.

Frame consumption
-----------------
MJPEG over HTTP is replaced by :func:`drone_core.iter_video_jpegs`,
which yields raw JPEG bytes from the backend's frame ring buffer.
:class:`InProcMjpegReader` wraps that iterator in the same lock /
latest-frame interface as :class:`drone_api.MjpegStreamReader` so
existing callers (``vision_worker``, ``recorder``, ``ui.py``)
don't need to know which transport they're on.
"""
from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Make ``import drone_core`` resolvable when marker_mission runs from
# its own directory. The combined entry point (marker_mission/app.py)
# arranges this too, but importing this module standalone (e.g. from
# tests) should also work.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DRONE_CORE_DIR = _REPO_ROOT / "controller_unified"
if str(_DRONE_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_DRONE_CORE_DIR))

import drone_core  # noqa: E402 — sys.path tweak above must precede the import

from .drone_api import DroneApiError, TelemetrySnapshot  # noqa: E402


class DroneApiInProc:
    """Same surface as ``DroneApi``, but every call routes through
    ``drone_core`` instead of HTTP. Drop-in replacement: mission.py
    and the controller don't care which one they hold.

    The ``timeout`` constructor argument is accepted for API parity
    but ignored — in-proc calls don't have a network round-trip to
    time out on. Long-running discrete commands (takeoff, land) still
    block on the backend SDK exactly as the HTTP variant did."""

    def __init__(self, base_url: Optional[str] = None,
                 timeout_s: float = 2.0):
        self.base_url = base_url  # informational, never dialed
        self.timeout = timeout_s  # ignored, kept for API parity

    @staticmethod
    def _unwrap(payload: dict, status: int) -> dict:
        """Raise on >= 400 to match DroneApi's behaviour where HTTP
        4xx/5xx become ``DroneApiError``. Successful payloads are
        returned unchanged."""
        if status >= 400:
            err = payload.get("error") if isinstance(payload, dict) else None
            raise DroneApiError(f"HTTP {status}: {err or payload}")
        return payload

    # ----------------------------------------------------------- status
    def telemetry(self) -> TelemetrySnapshot:
        payload, status = drone_core.get_telemetry_payload()
        return TelemetrySnapshot(
            raw=self._unwrap(payload, status), received_at=time.time()
        )

    def is_connected(self) -> bool:
        try:
            return self.telemetry().connected
        except DroneApiError:
            return False

    # ------------------------------------------------------------ video
    def video_start_mjpeg(self) -> tuple:
        payload, status = drone_core.do_video_start(mode="mjpeg")
        ok = status < 400 and bool(payload.get("ok", True) if isinstance(payload, dict) else True)
        msg = (payload.get("message") or payload.get("error") or "ok") if isinstance(payload, dict) else str(payload)
        return ok, msg

    def video_stop(self) -> dict:
        return self._unwrap(*drone_core.do_video_stop())

    def video_url(self) -> str:
        # No URL; in-proc consumers read frames via InProcMjpegReader
        # which calls drone_core.iter_video_jpegs() directly. Returned
        # as a marker string so the rare caller that logs the URL gets
        # an obvious "you're in-proc" hint instead of a broken HTTP URL.
        return "inproc://video"

    # --------------------------------------------------------- piloting
    def takeoff(self) -> dict:
        return self._unwrap(*drone_core.do_takeoff())

    def land(self) -> dict:
        return self._unwrap(*drone_core.do_land())

    def emergency(self) -> dict:
        return self._unwrap(*drone_core.do_emergency())

    def rotate(self, direction: str, degrees: int) -> dict:
        if direction not in ("cw", "ccw"):
            raise ValueError("direction must be cw or ccw")
        return self._unwrap(*drone_core.do_rotate(
            direction=direction,
            deg=int(max(1, min(360, degrees))),
        ))

    def move(self, direction: str, cm: int) -> dict:
        """Closed-loop position-relative move (Olympe moveBy under the
        hood). Used by mission's FB_IMU / LR_IMU / UD_IMU steps.
        Bypasses the HTTP route's [20, 500] cm safety clamp so the
        operator can request sub-20cm moves if they really want them;
        a SCRIPT-level clamp lives in the parser instead."""
        if direction not in ("forward", "back", "left", "right",
                             "up", "down"):
            raise ValueError("direction must be one of "
                             "forward/back/left/right/up/down")
        return self._unwrap(*drone_core.do_move(
            direction=direction, cm=int(cm),
        ))

    def rc(self, lr: int = 0, fb: int = 0, ud: int = 0, yaw: int = 0,
           duration_ms: int = 300) -> dict:
        return self._unwrap(*drone_core.do_rc(
            lr=lr, fb=fb, ud=ud, yaw=yaw, duration_ms=duration_ms,
        ))

    def rc_zero(self) -> dict:
        return self.rc(0, 0, 0, 0, duration_ms=200)

    def set_ceiling(self, ceiling_m: float) -> dict:
        """Push the soft altitude ceiling to the FC. mission.py calls
        this on startup so the FC's MAX_ALTITUDE_M (which clamps every
        climb stick at the wire) matches MissionConfig.max_height_m
        — without it the FC's persisted flight_config.json ceiling
        wins, and mission can't reach markers mounted higher than
        whatever the operator last saved on /tune."""
        return self._unwrap(*drone_core.do_set_ceiling(ceiling_m))

    def camera_config_get(self) -> dict:
        # The /api/camera/config endpoint isn't migrated to drone_core
        # yet — fall through to the Flask test_client so we keep the
        # surface identical. Once the route body is extracted in the
        # bulk migration this becomes a do_* call too.
        import unified_api_server as _srv
        with _srv.app.test_client() as c:
            r = c.get("/api/camera/config")
            return self._unwrap(r.get_json() or {}, r.status_code)

    def camera_config_set(self, **fields) -> dict:
        import unified_api_server as _srv
        with _srv.app.test_client() as c:
            r = c.post("/api/camera/config", json=fields)
            return self._unwrap(r.get_json() or {}, r.status_code)


# ---------------------------------------------------------------------------
# In-proc MJPEG reader — drop-in for drone_api.MjpegStreamReader
# ---------------------------------------------------------------------------

class InProcMjpegReader:
    """Same interface as :class:`drone_api.MjpegStreamReader`, but
    pulls JPEG bytes from :func:`drone_core.iter_video_jpegs` instead
    of an HTTP MJPEG stream. The frame-decode + lock-protected
    latest-frame pattern is identical, so callers using ``latest()`` /
    ``stats`` don't need to change."""

    def __init__(self, video_url: Optional[str] = None,
                 timeout_s: float = 5.0):
        # ``video_url`` is accepted for parity with MjpegStreamReader
        # but never used — frames come from the in-proc iterator.
        self.url = video_url or "inproc://video"
        self.timeout = timeout_s
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._frames_received: int = 0
        self._last_error: Optional[str] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="mjpeg-reader-inproc")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def restart(self) -> None:
        self.stop()
        self._stop = threading.Event()
        self._thread = None
        self.start()

    def latest(self) -> tuple[Optional[np.ndarray], Optional[bytes], float]:
        with self._lock:
            return self._latest_frame, self._latest_jpeg, self._latest_ts

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "frames_received": self._frames_received,
                "age_s": (time.monotonic() - self._latest_ts) if self._latest_ts else None,
                "last_error": self._last_error,
            }

    # Cap reader at this rate so vision_worker (ArUco + annotation) has
    # time to process each frame. Anafi delivers ~30fps; detection takes
    # ~30-50ms, so 20fps (50ms/frame) is the sustainable ceiling.
    _MAX_FPS = 20.0

    def _run(self) -> None:
        # Read raw BGR frames directly from the callback-driven condition
        # variable in unified_api_server. This avoids the JPEG round-trip
        # (encode→decode) that caused image distortion.
        import unified_api_server as _srv
        last_ts = 0.0
        last_publish = 0.0
        min_interval = 1.0 / self._MAX_FPS
        while not self._stop.is_set():
            try:
                if _srv._video_mode != "mjpeg":
                    self._stop.wait(0.3)
                    continue
                with _srv._video_bgr_cond:
                    # Wait up to 0.5 s for a new frame notification.
                    notified = _srv._video_bgr_cond.wait_for(
                        lambda: _srv._video_bgr_ts != last_ts
                                or self._stop.is_set(),
                        timeout=0.5,
                    )
                if self._stop.is_set():
                    return
                if not notified:
                    continue  # timeout — loop back and check mode/stop
                with _srv._video_bgr_cond:
                    bgr = _srv._video_bgr_latest
                    ts  = _srv._video_bgr_ts
                if bgr is None or ts == last_ts:
                    continue
                last_ts = ts
                # Rate-cap: skip frames that arrive faster than _MAX_FPS.
                # Always takes the LATEST frame (not queued), so latency
                # stays minimal while the vision pipeline isn't overloaded.
                now = time.monotonic()
                if now - last_publish < min_interval:
                    continue
                last_publish = now
                self._publish_bgr(bgr, ts)
            except Exception as e:
                with self._lock:
                    self._last_error = str(e)
                self._stop.wait(0.5)

    def _publish_bgr(self, frame: np.ndarray, ts: float) -> None:
        with self._lock:
            self._latest_frame = frame
            self._latest_jpeg = None   # encoded on demand by callers that need it
            self._latest_ts = ts
            self._frames_received += 1
            self._last_error = None


@contextmanager
def open_video_stream_inproc(api: DroneApiInProc):
    """Same as :func:`drone_api.open_video_stream` but uses the in-proc
    reader. Returns an :class:`InProcMjpegReader`."""
    api.video_start_mjpeg()
    reader = InProcMjpegReader()
    reader.start()
    try:
        yield reader
    finally:
        reader.stop()
        try:
            api.video_stop()
        except Exception:
            pass
