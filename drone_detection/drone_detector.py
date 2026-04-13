#!/usr/bin/env python3
"""
DroneDetector — real-time drone detection from MJPEG video stream.

Follows the same pattern as VideoMarkerTracker in tools/aruco_seek.py:
  - Background thread consumes MJPEG stream
  - Runs YOLO inference (OpenCV DNN, no PyTorch needed)
  - Provides thread-safe access to latest detections

Usage:
    # Standalone — runs detection on live stream and prints results
    python drone_detector.py --api http://flightctrl1:8080

    # As a module — import and use in your own code
    from drone_detection.drone_detector import DroneDetector

    detector = DroneDetector("http://flightctrl1:8080", "path/to/best.onnx")
    detector.start()

    # Later, in your control loop:
    drones = detector.get_detections()
    for d in drones:
        print(f"Drone at pixel ({d['cx']:.0f}, {d['cy']:.0f}), "
              f"size={d['bbox_area']:.0f}px, conf={d['confidence']:.2f}")

    detector.stop()
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import cv2
import numpy as np


class DroneDetector:
    """
    Background thread that grabs JPEG frames from an MJPEG stream
    and runs YOLO drone detection. Provides thread-safe access to
    latest detections.

    Similar architecture to VideoMarkerTracker — designed to run
    alongside ArUco detection without interference.
    """

    def __init__(
        self,
        api_base: str,
        model_path: str,
        video_endpoint: str = "/api/video",
        input_size: int = 640,
        conf_thresh: float = 0.4,
        nms_thresh: float = 0.45,
        max_fps: float = 10.0,
    ):
        """
        Args:
            api_base: Base URL of the unified API server (e.g. "http://flightctrl1:8080")
            model_path: Path to ONNX model file
            video_endpoint: MJPEG stream endpoint (default: /api/video)
            input_size: YOLO input resolution (default: 640)
            conf_thresh: Detection confidence threshold (default: 0.4)
            nms_thresh: NMS IoU threshold (default: 0.45)
            max_fps: Max inference rate to limit CPU usage (default: 10)
        """
        self._stream_url = f"{api_base}{video_endpoint}"
        self._model_path = model_path
        self._input_size = input_size
        self._conf_thresh = conf_thresh
        self._nms_thresh = nms_thresh
        self._min_interval = 1.0 / max_fps

        self._lock = threading.Lock()
        self._detections: list[dict] = []
        self._frame_w: int = 1280
        self._frame_h: int = 720
        self._last_ts: float = 0.0
        self._fps: float = 0.0
        self._inference_ms: float = 0.0

        self._thread = threading.Thread(target=self._run, daemon=True, name="drone-detect")
        self._running = True
        self._net = None

    def start(self):
        """Start the detection background thread."""
        self._thread.start()

    def stop(self):
        """Signal the background thread to stop."""
        self._running = False

    @property
    def is_active(self) -> bool:
        """True if detector is running and has produced results."""
        with self._lock:
            return self._last_ts > 0

    @property
    def age(self) -> float:
        """Seconds since last detection result."""
        with self._lock:
            return time.time() - self._last_ts if self._last_ts > 0 else float("inf")

    @property
    def fps(self) -> float:
        """Current detection FPS."""
        with self._lock:
            return self._fps

    @property
    def inference_ms(self) -> float:
        """Latest inference time in milliseconds."""
        with self._lock:
            return self._inference_ms

    def get_detections(self) -> list[dict]:
        """
        Return list of current drone detections.

        Each detection dict:
            {
                "confidence": float,     # 0.0-1.0
                "cx": float,             # center x in pixels
                "cy": float,             # center y in pixels
                "bbox": [x, y, w, h],    # top-left x,y + width,height
                "bbox_area": float,      # w*h in pixels (proxy for distance)
                "rel_x": float,          # normalised x: -1.0 (left) to +1.0 (right)
                "rel_y": float,          # normalised y: -1.0 (top) to +1.0 (bottom)
            }
        """
        with self._lock:
            return list(self._detections)

    def get_nearest(self) -> Optional[dict]:
        """Return the largest (nearest) drone detection, or None."""
        with self._lock:
            if not self._detections:
                return None
            return max(self._detections, key=lambda d: d["bbox_area"])

    @property
    def frame_size(self) -> tuple[int, int]:
        with self._lock:
            return self._frame_w, self._frame_h

    # ─── YOLO inference via OpenCV DNN ──────────────────────────────────

    def _init_model(self) -> bool:
        """Load ONNX model into OpenCV DNN."""
        model_path = Path(self._model_path)
        if not model_path.exists():
            print(f"[drone-detect] ERROR: Model not found: {model_path}")
            return False

        try:
            self._net = cv2.dnn.readNetFromONNX(str(model_path))
            print(f"[drone-detect] Model loaded: {model_path} "
                  f"(input={self._input_size}x{self._input_size})")
            return True
        except Exception as e:
            print(f"[drone-detect] Failed to load model: {e}")
            return False

    def _detect(self, frame: np.ndarray) -> list[dict]:
        """Run YOLO inference on a single BGR frame."""
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self._input_size, self._input_size),
            swapRB=True, crop=False
        )
        self._net.setInput(blob)
        outputs = self._net.forward()

        # YOLO output: (1, 5+nc, N) → transpose to (N, 5+nc)
        preds = outputs[0].T

        scores = preds[:, 4:]
        max_scores = scores.max(axis=1)
        mask = max_scores > self._conf_thresh
        preds = preds[mask]
        max_scores = max_scores[mask]

        if len(preds) == 0:
            return []

        # Scale boxes from input_size to original frame
        boxes = preds[:, :4].copy()
        sx, sy = w / self._input_size, h / self._input_size
        x_tl = (boxes[:, 0] - boxes[:, 2] / 2) * sx
        y_tl = (boxes[:, 1] - boxes[:, 3] / 2) * sy
        bw = boxes[:, 2] * sx
        bh = boxes[:, 3] * sy

        rects = np.column_stack([x_tl, y_tl, bw, bh])
        indices = cv2.dnn.NMSBoxes(
            rects.tolist(), max_scores.tolist(),
            self._conf_thresh, self._nms_thresh
        )

        img_cx, img_cy = w / 2.0, h / 2.0
        results = []
        for i in indices:
            idx = i if isinstance(i, int) else i[0]
            x, y, bwidth, bheight = rects[idx]
            cx = x + bwidth / 2
            cy = y + bheight / 2
            results.append({
                "confidence": float(max_scores[idx]),
                "cx": float(cx),
                "cy": float(cy),
                "bbox": [float(x), float(y), float(bwidth), float(bheight)],
                "bbox_area": float(bwidth * bheight),
                "rel_x": float((cx - img_cx) / img_cx),  # -1..+1
                "rel_y": float((cy - img_cy) / img_cy),  # -1..+1
            })
        return results

    # ─── Background thread ──────────────────────────────────────────────

    def _run(self):
        import http.client
        from urllib.parse import urlparse

        if not self._init_model():
            return

        print(f"[drone-detect] Starting detection on {self._stream_url}")
        _frames = 0
        _t_start = time.time()

        while self._running:
            try:
                parsed = urlparse(self._stream_url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=10)
                conn.request("GET", parsed.path)
                resp = conn.getresponse()
                print(f"[drone-detect] Connected (status {resp.status})")

                buf = b""
                while self._running:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    buf += chunk

                    # Parse MJPEG frames (same as VideoMarkerTracker)
                    while True:
                        hdr_idx = buf.find(b"\r\n\r\n")
                        if hdr_idx < 0:
                            break
                        jpeg_start = hdr_idx + 4
                        next_boundary = buf.find(b"--frame", jpeg_start)
                        if next_boundary < 0:
                            break

                        jpeg_data = buf[jpeg_start:next_boundary]
                        buf = buf[next_boundary:]

                        try:
                            arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if frame is None:
                                continue

                            # Rate limiting
                            now = time.time()
                            if now - self._last_ts < self._min_interval:
                                continue

                            h, w = frame.shape[:2]

                            t0 = time.monotonic()
                            dets = self._detect(frame)
                            inf_ms = (time.monotonic() - t0) * 1000

                            _frames += 1
                            elapsed = now - _t_start
                            current_fps = _frames / elapsed if elapsed > 0 else 0

                            with self._lock:
                                self._detections = dets
                                self._frame_w = w
                                self._frame_h = h
                                self._last_ts = now
                                self._fps = current_fps
                                self._inference_ms = inf_ms

                            if _frames == 1:
                                print(f"[drone-detect] First frame: {w}x{h}, "
                                      f"{len(dets)} drone(s), {inf_ms:.0f}ms")

                        except Exception as e:
                            if _frames == 0:
                                print(f"[drone-detect] Frame error: {e}")

                conn.close()
            except Exception as e:
                if self._running:
                    print(f"[drone-detect] Connection error: {e} — retrying in 2s")
                    time.sleep(2.0)


# ─── Standalone mode ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Drone Detector — YOLO on MJPEG stream")
    parser.add_argument("--api", default="http://flightctrl1:8080",
                        help="API server base URL")
    parser.add_argument("--model", required=True,
                        help="Path to ONNX model file")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")
    parser.add_argument("--max-fps", type=float, default=10.0,
                        help="Max detection FPS (default: 10)")
    parser.add_argument("--endpoint", default="/api/video",
                        help="MJPEG stream endpoint (default: /api/video)")
    args = parser.parse_args()

    detector = DroneDetector(
        api_base=args.api,
        model_path=args.model,
        video_endpoint=args.endpoint,
        conf_thresh=args.conf,
        max_fps=args.max_fps,
    )
    detector.start()

    print(f"\nDrone detector running on {args.api}{args.endpoint}")
    print(f"  Model: {args.model}")
    print(f"  Confidence: {args.conf}")
    print(f"  Max FPS: {args.max_fps}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(0.5)
            dets = detector.get_detections()
            if dets:
                for d in dets:
                    print(f"  DRONE: conf={d['confidence']:.2f}  "
                          f"pos=({d['cx']:.0f},{d['cy']:.0f})  "
                          f"size={d['bbox_area']:.0f}px  "
                          f"rel=({d['rel_x']:+.2f},{d['rel_y']:+.2f})")
            else:
                age = detector.age
                if age < 2.0:
                    print(f"  No drones detected  (age={age:.1f}s, "
                          f"fps={detector.fps:.1f}, "
                          f"inf={detector.inference_ms:.0f}ms)")
                elif detector.is_active:
                    print(f"  No recent frames (age={age:.1f}s)")
                else:
                    print(f"  Waiting for first frame...")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.stop()


if __name__ == "__main__":
    main()
