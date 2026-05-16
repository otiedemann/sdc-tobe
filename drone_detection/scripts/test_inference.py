#!/usr/bin/env python3
"""
Test drone detection inference — on a single image, a folder, or a live MJPEG stream.

Usage:
    # Test on a single image
    python test_inference.py --model best.onnx --source image.jpg

    # Test on a folder of images
    python test_inference.py --model best.onnx --source test_images/

    # Test on live MJPEG stream (from unified API server)
    python test_inference.py --model best.onnx --source http://flightctrl1:8080/api/video

    # Test against the MacBook / built-in webcam (device 0)
    python test_inference.py --model best.onnx --source 0 --show

    # Use Ultralytics .pt model (requires PyTorch)
    python test_inference.py --model best.pt --source image.jpg

    # Use OpenCV DNN only (no PyTorch needed)
    python test_inference.py --model best.onnx --source image.jpg --opencv-dnn

    # Benchmark FPS on MJPEG stream
    python test_inference.py --model best.onnx --source http://flightctrl1:8080/api/video --benchmark 100
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ─── OpenCV DNN inference (no PyTorch needed) ───────────────────────────────

class YOLOv11_OpenCV:
    """Lightweight YOLO inference using only OpenCV DNN module."""

    def __init__(self, onnx_path: str, input_size: int = 640,
                 conf_thresh: float = 0.4, nms_thresh: float = 0.45):
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.class_names = ["drone"]

    def detect(self, frame: np.ndarray) -> list:
        """
        Run inference on a BGR frame.

        Returns list of dicts:
            [{"class": "drone", "confidence": 0.92, "bbox": [x, y, w, h]}, ...]
        """
        h, w = frame.shape[:2]

        # Preprocess: letterbox to input_size x input_size
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward()

        # YOLO11 output shape: (1, 5+num_classes, num_predictions)
        # Transpose to (num_predictions, 5+num_classes)
        preds = outputs[0].T  # shape: (8400, 5+nc)

        # Filter by confidence
        scores = preds[:, 4:]  # class scores (nc columns)
        max_scores = scores.max(axis=1)
        mask = max_scores > self.conf_thresh
        preds = preds[mask]
        max_scores = max_scores[mask]
        class_ids = scores[mask].argmax(axis=1)

        if len(preds) == 0:
            return []

        # Convert center-xywh to top-left xywh, scale to original image
        boxes = preds[:, :4].copy()
        scale_x = w / self.input_size
        scale_y = h / self.input_size
        # xywh center → top-left xy + wh
        x_tl = (boxes[:, 0] - boxes[:, 2] / 2) * scale_x
        y_tl = (boxes[:, 1] - boxes[:, 3] / 2) * scale_y
        bw = boxes[:, 2] * scale_x
        bh = boxes[:, 3] * scale_y

        # NMS
        rects = np.column_stack([x_tl, y_tl, bw, bh])
        indices = cv2.dnn.NMSBoxes(
            rects.tolist(), max_scores.tolist(),
            self.conf_thresh, self.nms_thresh
        )

        # NMSBoxes return shape varies by OpenCV version:
        #   ≤ 4.5: list[list[int]]  → each `i` is `[idx]`
        #   ≥ 4.7: np.ndarray[int32], 1-D → each `i` is a numpy scalar
        # np.atleast_1d + .ravel() flattens both shapes to a 1-D int array.
        indices = np.atleast_1d(np.asarray(indices)).ravel()
        results = []
        for i in indices:
            idx = int(i)
            cid = int(class_ids[idx])
            results.append({
                "class": self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}",
                "confidence": float(max_scores[idx]),
                "bbox": [float(x_tl[idx]), float(y_tl[idx]),
                         float(bw[idx]), float(bh[idx])],
            })
        return results


# ─── Visualization ──────────────────────────────────────────────────────────

def draw_detections(frame, detections):
    """Draw bounding boxes on frame."""
    for det in detections:
        x, y, w, h = [int(v) for v in det["bbox"]]
        conf = det["confidence"]
        label = f"{det['class']} {conf:.2f}"
        color = (0, 0, 255)  # red
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test drone detection inference")
    parser.add_argument("--model", required=True, help="Model path (.onnx or .pt)")
    parser.add_argument("--source", required=True,
                        help="Image path, folder, or MJPEG URL")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input size (default: 640)")
    parser.add_argument("--opencv-dnn", action="store_true",
                        help="Force OpenCV DNN backend (ONNX only, no PyTorch)")
    parser.add_argument("--benchmark", type=int, default=0,
                        help="Benchmark N frames and report FPS")
    parser.add_argument("--show", action="store_true",
                        help="Show detections in window (requires display)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save annotated output to file/folder")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    # Choose inference backend
    use_opencv = args.opencv_dnn or model_path.suffix == ".onnx"
    if use_opencv and model_path.suffix != ".onnx":
        print("ERROR: --opencv-dnn requires .onnx model")
        sys.exit(1)

    if use_opencv:
        print(f"Using OpenCV DNN backend: {model_path}")
        detector = YOLOv11_OpenCV(str(model_path), args.imgsz, args.conf)
    else:
        # Use Ultralytics (needs PyTorch)
        try:
            from ultralytics import YOLO
        except ImportError:
            print("ERROR: ultralytics not installed. Use --opencv-dnn with .onnx model")
            sys.exit(1)
        print(f"Using Ultralytics backend: {model_path}")
        model = YOLO(str(model_path))

    source = args.source

    # ── Webcam (numeric device index) or MJPEG stream ──
    is_webcam = source.isdigit()
    if is_webcam or source.startswith("http"):
        if is_webcam:
            dev_index = int(source)
            # On macOS the default backend (AVFoundation) is usually correct
            # but it can be slow to open; passing CAP_AVFOUNDATION explicitly
            # skips OpenCV's backend autodetect and shaves ~2 s off startup.
            if sys.platform == "darwin":
                print(f"Opening macOS webcam device {dev_index} via AVFoundation…")
                cap = cv2.VideoCapture(dev_index, cv2.CAP_AVFOUNDATION)
            else:
                print(f"Opening webcam device {dev_index}…")
                cap = cv2.VideoCapture(dev_index)
        else:
            print(f"Connecting to MJPEG stream: {source}")
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("ERROR: Cannot open stream")
            sys.exit(1)

        frame_count = 0
        total_time = 0.0
        max_frames = args.benchmark if args.benchmark > 0 else float("inf")

        try:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.monotonic()
                if use_opencv:
                    dets = detector.detect(frame)
                else:
                    results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
                    dets = []
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            dets.append({
                                "class": r.names[int(box.cls)],
                                "confidence": float(box.conf),
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                            })
                elapsed = time.monotonic() - t0
                total_time += elapsed
                frame_count += 1

                n_drones = len(dets)
                fps = frame_count / total_time if total_time > 0 else 0
                print(f"  Frame {frame_count}: {n_drones} drone(s) detected  "
                      f"{elapsed*1000:.0f}ms  avg FPS={fps:.1f}", end="\r")

                if args.show:
                    draw_detections(frame, dets)
                    cv2.imshow("Drone Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            pass

        cap.release()
        if args.show:
            cv2.destroyAllWindows()

        if frame_count > 0:
            avg_ms = total_time / frame_count * 1000
            avg_fps = frame_count / total_time
            print(f"\n\nBenchmark: {frame_count} frames, "
                  f"avg {avg_ms:.1f}ms/frame, {avg_fps:.1f} FPS")

    # ── Single image or folder ──
    else:
        source_path = Path(source)
        if source_path.is_dir():
            images = sorted(source_path.glob("*.jpg")) + sorted(source_path.glob("*.png"))
        elif source_path.exists():
            images = [source_path]
        else:
            print(f"ERROR: Source not found: {source}")
            sys.exit(1)

        print(f"Processing {len(images)} image(s)...")
        total_time = 0.0

        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  SKIP: Cannot read {img_path}")
                continue

            t0 = time.monotonic()
            if use_opencv:
                dets = detector.detect(frame)
            else:
                results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
                dets = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        dets.append({
                            "class": r.names[int(box.cls)],
                            "confidence": float(box.conf),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                        })
            elapsed = time.monotonic() - t0
            total_time += elapsed

            n_drones = len(dets)
            print(f"  {img_path.name}: {n_drones} drone(s) detected  {elapsed*1000:.0f}ms")
            for d in dets:
                print(f"    {d['class']} conf={d['confidence']:.2f} "
                      f"bbox=[{d['bbox'][0]:.0f},{d['bbox'][1]:.0f},"
                      f"{d['bbox'][2]:.0f},{d['bbox'][3]:.0f}]")

            if args.show:
                draw_detections(frame, dets)
                cv2.imshow("Drone Detection", frame)
                cv2.waitKey(0)

            if args.save:
                save_path = Path(args.save)
                if save_path.is_dir():
                    out_path = save_path / f"det_{img_path.name}"
                else:
                    out_path = save_path
                draw_detections(frame, dets)
                cv2.imwrite(str(out_path), frame)
                print(f"    Saved: {out_path}")

        if len(images) > 0 and total_time > 0:
            avg_ms = total_time / len(images) * 1000
            print(f"\nAverage: {avg_ms:.1f}ms/image ({1000/avg_ms:.1f} FPS)")

        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
