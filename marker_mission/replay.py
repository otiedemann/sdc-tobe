"""
Flight replay -- drive a recorded flight through the live web UI.

A :class:`FlightReplay` reads ``<flight_dir>/flight_log.csv`` and
``<flight_dir>/annotated.mp4`` and walks them in sync, populating a
:class:`MissionState` and a :class:`LatestFrame` so the existing UI
templates render the playback. Supports play / pause / seek / speed via
the methods on the instance.

Designed to coexist with a live mission: the live ``UiServer`` keeps
its own ``state`` / ``frame`` and never sees these.
"""

from __future__ import annotations

import bisect
import csv
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

# Force OpenCV's ffmpeg backend into single-threaded decoding. The
# multi-threaded H.264 decoder asserts on libavcodec/pthread_frame.c:174
# when a Python-driven loop interleaves cap.set(POS_MSEC) and cap.read()
# faster than its async-lock invariants tolerate. Single-threaded mode
# is plenty fast for our ~25 fps inputs and avoids the abort.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")

import cv2  # noqa: E402  -- after the env-var override above

from .controller import MissionState, Phase
from .drone_api import TelemetrySnapshot


class FlightReplay:
    """Reads one flight's recorded artefacts and exposes a paused
    playback driven by a daemon thread.

    State and frame are public attributes so the UI can mount its
    existing templates against them without any changes.
    """

    def __init__(self, flight_dir: Path):
        from .ui import LatestFrame  # local import: avoid circular
        self.dir = Path(flight_dir)
        self.flight_id = self.dir.name

        # Load the per-tick CSV. Anything that fails to parse is fatal --
        # the operator should pick a different flight.
        rows: List[dict] = []
        with (self.dir / "flight_log.csv").open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
        if not rows:
            raise ValueError(f"flight_log.csv in {flight_dir} is empty")
        self.rows = rows
        t0 = float(rows[0]["monotonic"])
        self.timeline: List[float] = [float(r["monotonic"]) - t0
                                       for r in rows]
        self.duration_s = self.timeline[-1]

        # Annotated.mp4 already has the HUD baked in -- prefer it.
        # Fall back to raw.mp4 if annotated is missing for some reason.
        # NOTE: don't open the cv2.VideoCapture here. libavcodec's
        # threaded H.264 decoder asserts (pthread_frame.c:174) when a
        # capture opened in one thread is read from another at our
        # cadence. Defer opening to inside the worker thread.
        for name in ("annotated.mp4", "raw.mp4"):
            p = self.dir / name
            if p.exists() and p.stat().st_size > 0:
                self.video_path: Optional[Path] = p
                break
        else:
            self.video_path = None
        self.cap = None  # opened lazily inside _run

        # Public state for the UI to read.
        self.state = MissionState()
        self.frame = LatestFrame()
        self._last_pos_ms = -1.0

        # Playback controls.
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._paused = True            # always start paused
        self._speed = 1.0
        self._playhead_idx = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"replay-{self.flight_id}")
        self._thread.start()

    # ------------------------------------------------------------ controls
    def play(self) -> None:
        with self._lock:
            self._paused = False
        self._wakeup.set()

    def pause(self) -> None:
        with self._lock:
            self._paused = True
        self._wakeup.set()

    def toggle_play(self) -> None:
        with self._lock:
            self._paused = not self._paused
        self._wakeup.set()

    def seek(self, t_seconds: float) -> None:
        """Move the playhead to absolute time ``t_seconds`` (clamped)."""
        with self._lock:
            t = max(0.0, min(float(t_seconds), self.duration_s))
            new_idx = bisect.bisect_left(self.timeline, t)
            if new_idx >= len(self.timeline):
                new_idx = len(self.timeline) - 1
            self._playhead_idx = new_idx
        self._wakeup.set()

    def set_speed(self, rate: float) -> None:
        with self._lock:
            self._speed = max(0.1, min(10.0, float(rate)))
        self._wakeup.set()

    @property
    def status(self) -> dict:
        with self._lock:
            idx = self._playhead_idx
            return {
                "flight_id": self.flight_id,
                "duration_s": round(self.duration_s, 3),
                "playhead_s": round(self.timeline[idx], 3)
                              if idx < len(self.timeline)
                              else round(self.duration_s, 3),
                "paused": self._paused,
                "speed": self._speed,
                "row_count": len(self.rows),
                "video_path": str(self.video_path)
                              if self.video_path else None,
            }

    def close(self) -> None:
        self._stop.set()
        self._wakeup.set()
        # Don't release self.cap here -- it was opened in the worker
        # thread and should be released there. The worker exits its
        # loop on _stop and releases its own cap.

    # ---------------------------------------------------- per-row dispatch
    @staticmethod
    def _fnum(row: dict, key: str) -> Optional[float]:
        v = row.get(key, "")
        if v in ("", "nan", "NaN", None):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    def _ensure_cap(self) -> None:
        """Open the VideoCapture in the calling thread the first time
        we need it. Used so the cap is owned by the playback worker
        rather than the constructor's thread."""
        if self.cap is None and self.video_path is not None:
            self.cap = cv2.VideoCapture(str(self.video_path))

    def _apply_row(self, idx: int) -> None:
        row = self.rows[idx]
        # Rebuild a TelemetrySnapshot from the recorded tel_* columns.
        tel_raw: dict = {}
        for src in ("battery", "yaw", "pitch", "roll", "height_cm",
                    "flight_time_s", "vgx", "vgy", "vgz"):
            v = self._fnum(row, f"tel_{src}")
            if v is not None:
                tel_raw[src] = v
        tel_raw["flying"] = (row.get("tel_flying") == "1")
        tel_raw["connected"] = (row.get("tel_connected") == "1")
        tel_snap = TelemetrySnapshot(raw=tel_raw, received_at=time.time())

        # Pose tuple (or None if marker wasn't seen this row).
        d = self._fnum(row, "distance_m")
        y = self._fnum(row, "yaw_to_marker_deg")
        h = self._fnum(row, "relative_heading_deg")
        smoothed = (d, y, h) if d is not None else None

        try:
            phase = Phase(row.get("phase", "init"))
        except ValueError:
            phase = Phase.INIT

        with self.state.lock:
            self.state.phase = phase
            self.state.last_telemetry = tel_snap
            self.state.smoothed = smoothed
            self.state.target_distance_m = (
                self._fnum(row, "target_distance_m") or 1.0)
            self.state.target_relative_heading_deg = (
                self._fnum(row, "target_relative_heading_deg") or 0.0)
            self.state.last_rc = (
                int(self._fnum(row, "rc_lr") or 0),
                int(self._fnum(row, "rc_fb") or 0),
                int(self._fnum(row, "rc_ud") or 0),
                int(self._fnum(row, "rc_yaw") or 0),
            )
            seen = (row.get("marker_seen") == "1")
            if seen:
                self.state.last_marker_seen_at = time.monotonic()
            self.state.note = (
                f"REPLAY {self.flight_id} -- t={self.timeline[idx]:.1f}s "
                f"/ {self.duration_s:.1f}s"
            )
            self.state.phase_started_at = time.monotonic()

        # Seek + read the matching video frame. Only issue an explicit
        # seek when the target is far from the last read position;
        # otherwise let the decoder advance sequentially. The cap was
        # opened lazily by _ensure_cap() inside the worker thread (see
        # __init__'s note).
        if self.cap is not None:
            t_ms = self.timeline[idx] * 1000.0
            drift_threshold_ms = 600.0
            if (self._last_pos_ms < 0
                    or abs(t_ms - self._last_pos_ms) > drift_threshold_ms
                    or t_ms < self._last_pos_ms):
                try:
                    self.cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
                except Exception:
                    return
            try:
                ok, frame = self.cap.read()
            except Exception:
                return
            if ok and frame is not None:
                self._last_pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                self.frame.set(frame)

    # ---------------------------------------------------- playback loop
    def _wait(self, timeout: float) -> None:
        """Sleep up to ``timeout`` seconds; wake immediately on a state
        change or stop. Caller clears the event after waking."""
        self._wakeup.wait(timeout=timeout)
        self._wakeup.clear()

    def _run(self) -> None:
        # Open the VideoCapture in this thread to keep libavcodec's
        # internal lock invariants happy (see __init__ note).
        self._ensure_cap()
        # Show the first frame so the UI isn't blank while paused.
        if self.rows:
            self._apply_row(0)
            last_dispatched = 0
        else:
            return

        # Wall-clock anchor for advancing through the timeline.
        base_wall = time.monotonic()
        base_t = self.timeline[0]

        while not self._stop.is_set():
            with self._lock:
                idx = self._playhead_idx
                paused = self._paused
                speed = self._speed

            # Seek (or initial entry) -- redraw + reset the wall anchor.
            if idx != last_dispatched:
                self._apply_row(idx)
                last_dispatched = idx
                base_wall = time.monotonic()
                base_t = self.timeline[idx]

            # End of timeline -- auto-pause and wait.
            if idx >= len(self.rows) - 1:
                if not paused:
                    with self._lock:
                        self._paused = True
                self._wait(0.5)
                continue

            if paused:
                # Keep the anchor pinned to the current row so unpause
                # resumes at exactly the right wall time.
                base_wall = time.monotonic()
                base_t = self.timeline[idx]
                self._wait(0.5)
                continue

            # Advance to the new row whose timeline timestamp is
            # <= base_t + (wall_elapsed * speed).
            now = time.monotonic()
            target_t = base_t + (now - base_wall) * speed
            new_idx = idx
            while (new_idx + 1 < len(self.rows)
                   and self.timeline[new_idx + 1] <= target_t):
                new_idx += 1

            if new_idx > idx:
                with self._lock:
                    if self._playhead_idx == idx:
                        self._playhead_idx = new_idx
                # Apply outside the lock so cv2.VideoCapture I/O doesn't
                # serialize against the controls.
                self._apply_row(new_idx)
                last_dispatched = new_idx
            else:
                # Sleep until the next row's relative timestamp.
                next_dt_real = (self.timeline[idx + 1] - target_t) / max(speed, 0.001)
                self._wait(max(0.005, min(next_dt_real, 0.1)))

        # Loop exited (close() was called). Release the cap from the
        # same thread that opened it.
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
