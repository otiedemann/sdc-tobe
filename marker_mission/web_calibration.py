"""
Web-driven camera calibration.

A small recorder that taps the existing :class:`MjpegStreamReader`,
writes the live frames to a temp video, and -- on demand -- runs the
existing :func:`calibrate_from_video` pipeline against the result.
Surfaced through the operator UI's ``/calibrate`` tab so the operator
can re-calibrate without dropping to the CLI.

Lifecycle::

    idle -> capturing -> captured -> running -> { done | failed }

The capture and the run are both done on background threads -- the
Flask request handlers just kick them off and poll status.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2

from .calibration_store import (Calibration, CalibrationStore,
                                calibrate_from_video)


class CalibrationCapture:
    STATE_IDLE       = "idle"
    STATE_CAPTURING  = "capturing"
    STATE_CAPTURED   = "captured"
    STATE_RUNNING    = "running"
    STATE_DONE       = "done"
    STATE_FAILED     = "failed"

    _MIN_FRAMES = 30  # require at least this many frames before allowing run

    def __init__(self, reader, calib_dir: Path, fps: int = 25):
        self.reader = reader
        self.calib_dir = Path(calib_dir)
        self.fps = int(fps)
        self.video_path = self.calib_dir / "_capture_tmp.mp4"

        self._lock = threading.Lock()
        self.state: str = self.STATE_IDLE
        self.frames_captured: int = 0
        self.error: Optional[str] = None
        self.last_calibration: Optional[Calibration] = None
        self.last_save_path: Optional[Path] = None

        self._capture_stop = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

    # --------------------------------------------------------------- status
    @property
    def status(self) -> dict:
        with self._lock:
            cal_summary = None
            if self.last_calibration is not None:
                c = self.last_calibration
                rms = float(c.rms_error)
                # NaN -> None for JSON compatibility.
                if rms != rms:
                    rms = None
                cal_summary = {
                    "serial": c.serial,
                    "resolution": c.resolution,
                    "rms_error": rms,
                    "fx": float(c.fx), "fy": float(c.fy),
                    "cx": float(c.cx), "cy": float(c.cy),
                    "image_size": list(c.image_size),
                    "calibrated_at": c.calibrated_at,
                    "is_default": bool(c.is_default),
                }
            return {
                "state": self.state,
                "frames_captured": int(self.frames_captured),
                "min_frames": int(self._MIN_FRAMES),
                "error": self.error,
                "video_path": (str(self.video_path)
                               if self.video_path.exists() else None),
                "saved_path": (str(self.last_save_path)
                               if self.last_save_path else None),
                "calibration": cal_summary,
            }

    # ------------------------------------------------------------- capture
    def start_capture(self) -> bool:
        with self._lock:
            if self.state in (self.STATE_CAPTURING, self.STATE_RUNNING):
                return False
            self.state = self.STATE_CAPTURING
            self.frames_captured = 0
            self.error = None
        self.calib_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.video_path.unlink()
        except FileNotFoundError:
            pass
        self._capture_stop.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True,
            name="calibration-capture")
        self._capture_thread.start()
        return True

    def stop_capture(self) -> bool:
        with self._lock:
            if self.state != self.STATE_CAPTURING:
                return False
        self._capture_stop.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
        return True

    def _capture_loop(self) -> None:
        last_ts = 0.0
        writer: Optional[cv2.VideoWriter] = None
        local_error: Optional[str] = None
        try:
            while not self._capture_stop.is_set():
                frame, _jpg, ts = self.reader.latest()
                if frame is None or ts == last_ts:
                    time.sleep(0.02)
                    continue
                last_ts = ts
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(self.video_path),
                                             fourcc, self.fps, (w, h))
                    if not writer.isOpened():
                        local_error = "VideoWriter failed to open"
                        break
                writer.write(frame)
                with self._lock:
                    self.frames_captured += 1
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
        with self._lock:
            if local_error is not None:
                self.state = self.STATE_FAILED
                self.error = local_error
            elif self.frames_captured >= self._MIN_FRAMES:
                self.state = self.STATE_CAPTURED
            else:
                self.state = self.STATE_FAILED
                self.error = (f"only {self.frames_captured} frame(s) "
                              f"captured (need at least {self._MIN_FRAMES}). "
                              f"Is the drone connected and streaming video?")

    # ------------------------------------------------------------ calibrate
    def run_calibration(self, serial: str,
                        resolution: str = "720p",
                        pattern: tuple[int, int] = (9, 6),
                        square_size_m: float = 0.025) -> tuple[bool, str]:
        with self._lock:
            if self.state in (self.STATE_CAPTURING, self.STATE_RUNNING):
                return (False, f"busy ({self.state})")
            if self.state not in (self.STATE_CAPTURED, self.STATE_DONE,
                                   self.STATE_FAILED):
                return (False, f"no captured video to calibrate from "
                               f"(state={self.state})")
            if not self.video_path.exists():
                return (False, "captured video file is gone -- recapture")
            if not serial or serial == "unknown":
                return (False, "drone serial is unknown -- can't save")
            self.state = self.STATE_RUNNING
            self.error = None
        threading.Thread(
            target=self._run_calibration_sync,
            args=(serial, resolution, pattern, square_size_m),
            daemon=True, name="calibration-runner",
        ).start()
        return (True, "started")

    def _run_calibration_sync(self, serial: str, resolution: str,
                              pattern: tuple[int, int],
                              square_size_m: float) -> None:
        try:
            cal = calibrate_from_video(
                self.video_path, pattern_size=pattern,
                square_size_m=square_size_m, frame_stride=10)
            cal.serial = serial
            cal.resolution = resolution
            store = CalibrationStore(self.calib_dir)
            saved_path = store.save(cal)
            with self._lock:
                self.state = self.STATE_DONE
                self.last_calibration = cal
                self.last_save_path = saved_path
        except Exception as e:
            with self._lock:
                self.state = self.STATE_FAILED
                self.error = f"calibration failed: {e}"
