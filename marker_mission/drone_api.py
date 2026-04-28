"""
HTTP client wrapper around the unified API server.

The server exposes the drone over a simple JSON REST API. We do NOT talk
to the drone SDK directly -- the server already handles connection,
auto-reconnect, watchdog, and ceiling/arena safety. We just call its
endpoints.

Endpoints used (see ``unified_api_server.py`` for full details):

* GET  /api/telemetry                    -- battery, yaw, height, serial, ...
* GET  /api/video                        -- MJPEG stream
* POST /api/video/start                  -- start MJPEG stream
* POST /api/video/stop                   -- stop streaming
* POST /api/takeoff                      -- arm + take off
* POST /api/land                         -- land
* POST /api/emergency                    -- cut motors immediately
* POST /api/rotate {dir, deg}            -- discrete rotation
* POST /api/rc {lr, fb, ud, yaw, duration_ms}  -- continuous control (PRIMARY)
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import requests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DroneApiError(Exception):
    pass


@dataclass
class TelemetrySnapshot:
    """Whatever the server returned in /api/telemetry, plus a fetch timestamp."""
    raw: dict = field(default_factory=dict)
    received_at: float = 0.0

    @property
    def battery(self) -> Optional[int]:
        v = self.raw.get("battery")
        return int(v) if v is not None else None

    @property
    def yaw_deg(self) -> Optional[float]:
        v = self.raw.get("yaw")
        return float(v) if v is not None else None

    @property
    def height_cm(self) -> Optional[float]:
        v = self.raw.get("height_cm")
        return float(v) if v is not None else None

    @property
    def flying(self) -> bool:
        return bool(self.raw.get("flying"))

    @property
    def connected(self) -> bool:
        return bool(self.raw.get("connected"))

    @property
    def serial_number(self) -> Optional[str]:
        sn = self.raw.get("serial_number")
        if sn:
            return str(sn).strip()
        return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class DroneApi:
    """Thin client over the API server."""

    def __init__(self, base_url: str = "http://127.0.0.1:5050",
                 timeout_s: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_s
        self._session = requests.Session()
        # Reuse connections, keep request log small.
        self._session.headers["Connection"] = "keep-alive"

    # --------------------------------------------------------------- helpers
    def _post(self, path: str, json_body: Optional[dict] = None,
              timeout_s: Optional[float] = None) -> dict:
        try:
            r = self._session.post(self.base_url + path,
                                   json=json_body or {},
                                   timeout=timeout_s if timeout_s is not None
                                   else self.timeout)
        except requests.RequestException as e:
            raise DroneApiError(f"POST {path} failed: {e}") from e
        if not r.ok:
            raise DroneApiError(f"POST {path} -> HTTP {r.status_code}: {r.text}")
        try:
            return r.json()
        except ValueError:
            return {"ok": True, "raw": r.text}

    def _get(self, path: str) -> dict:
        try:
            r = self._session.get(self.base_url + path, timeout=self.timeout)
        except requests.RequestException as e:
            raise DroneApiError(f"GET {path} failed: {e}") from e
        if not r.ok:
            raise DroneApiError(f"GET {path} -> HTTP {r.status_code}: {r.text}")
        try:
            return r.json()
        except ValueError:
            return {"ok": True, "raw": r.text}

    # ---------------------------------------------------------------- status
    def telemetry(self) -> TelemetrySnapshot:
        data = self._get("/api/telemetry")
        return TelemetrySnapshot(raw=data, received_at=time.time())

    def is_connected(self) -> bool:
        try:
            return self.telemetry().connected
        except DroneApiError:
            return False

    # ----------------------------------------------------------------- video
    def video_start_mjpeg(self) -> dict:
        return self._post("/api/video/start", {"mode": "mjpeg"})

    def video_stop(self) -> dict:
        return self._post("/api/video/stop")

    def video_url(self) -> str:
        return self.base_url + "/api/video"

    # --------------------------------------------------------------- piloting
    def takeoff(self) -> dict:
        # Server blocks on the SDK until the drone acks takeoff; this can
        # take several seconds, so allow a generous timeout.
        return self._post("/api/takeoff", timeout_s=20.0)

    def land(self) -> dict:
        return self._post("/api/land", timeout_s=20.0)

    def emergency(self) -> dict:
        return self._post("/api/emergency")

    def rotate(self, direction: str, degrees: int) -> dict:
        if direction not in ("cw", "ccw"):
            raise ValueError("direction must be cw or ccw")
        return self._post("/api/rotate", {"dir": direction,
                                          "deg": int(max(1, min(360, degrees)))})

    def rc(self, lr: int = 0, fb: int = 0, ud: int = 0, yaw: int = 0,
           duration_ms: int = 300) -> dict:
        return self._post("/api/rc", {
            "lr": int(lr), "fb": int(fb), "ud": int(ud), "yaw": int(yaw),
            "duration_ms": int(duration_ms),
        })

    def rc_zero(self) -> dict:
        return self.rc(0, 0, 0, 0, duration_ms=200)


# ---------------------------------------------------------------------------
# Video stream reader
# ---------------------------------------------------------------------------

class MjpegStreamReader:
    """Background thread that pulls JPEG frames from the server's MJPEG feed
    and exposes the latest one. Designed so that consumers (detector,
    recorder, UI) can each grab the most recent frame without blocking the
    network read.
    """

    BOUNDARY = b"--frame"

    def __init__(self, video_url: str, timeout_s: float = 5.0):
        self.url = video_url
        self.timeout = timeout_s
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._frames_received: int = 0
        self._last_error: Optional[str] = None

    # --------------------------------------------------------------- control
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="mjpeg-reader")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ read
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

    # ------------------------------------------------------------------- run
    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                with requests.get(self.url, stream=True, timeout=self.timeout) as r:
                    if not r.ok:
                        raise DroneApiError(f"video stream HTTP {r.status_code}")
                    self._consume(r)
                    backoff = 0.5
            except Exception as e:
                with self._lock:
                    self._last_error = str(e)
                # Back off on repeat errors but cap so we don't sleep forever.
                if not self._stop.is_set():
                    self._stop.wait(min(backoff, 5.0))
                    backoff = min(backoff * 2, 5.0)

    def _consume(self, response) -> None:
        """Walk the multipart MJPEG stream, handing off complete JPEGs."""
        buf = bytearray()
        for chunk in response.iter_content(chunk_size=4096):
            if self._stop.is_set():
                return
            if not chunk:
                continue
            buf += chunk
            # Each frame ends with the boundary; we look for JPEG markers
            # (FFD8 .. FFD9) and slice them out. This is more robust than
            # parsing multipart headers because Flask's chunk boundaries
            # don't align with frame boundaries.
            while True:
                start = buf.find(b"\xff\xd8")
                if start < 0:
                    if len(buf) > 1_000_000:
                        # Drop garbage if no JPEG marker for a long time.
                        buf.clear()
                    break
                end = buf.find(b"\xff\xd9", start + 2)
                if end < 0:
                    # No complete frame yet, wait for more bytes.
                    if start > 0:
                        del buf[:start]
                    break
                jpg = bytes(buf[start:end + 2])
                del buf[:end + 2]
                self._publish(jpg)

    def _publish(self, jpg: bytes) -> None:
        arr = np.frombuffer(jpg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        with self._lock:
            self._latest_jpeg = jpg
            self._latest_frame = frame
            self._latest_ts = time.monotonic()
            self._frames_received += 1
            self._last_error = None


# ---------------------------------------------------------------------------
# Convenience: context manager that takes care of stream startup / teardown
# ---------------------------------------------------------------------------

@contextmanager
def open_video_stream(api: DroneApi):
    api.video_start_mjpeg()
    reader = MjpegStreamReader(api.video_url())
    reader.start()
    try:
        yield reader
    finally:
        reader.stop()
        try:
            api.video_stop()
        except Exception:
            pass
