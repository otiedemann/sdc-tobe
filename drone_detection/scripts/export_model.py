#!/usr/bin/env python3
"""
Export trained YOLO11n drone detector to edge-optimized formats.

Usage:
    # Export all formats (ONNX + NCNN)
    python drone_detection/scripts/export_model.py

    # Export specific format
    python drone_detection/scripts/export_model.py --format onnx
    python drone_detection/scripts/export_model.py --format ncnn
    python drone_detection/scripts/export_model.py --format tflite

    # Custom model path
    python drone_detection/scripts/export_model.py --model path/to/best.pt

Outputs:
    drone_detection/models/train/weights/best.onnx      (OpenCV DNN, ONNX Runtime)
    drone_detection/models/train/weights/best_ncnn_model/ (RPi 5 — fastest)
    drone_detection/models/train/weights/best_saved_model/ (TFLite, optional)

Performance targets (640x640 input):
    RPi 5 + NCNN:     ~67ms  → 15 FPS
    RPi 5 + ONNX:     ~128ms →  8 FPS
    Jetson + TensorRT: ~20ms  → 50 FPS
    OpenCV DNN (any):  ~100-150ms → 7-10 FPS
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "train" / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="Export YOLO11n drone model for edge deployment")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help=f"Path to trained .pt model (default: {DEFAULT_MODEL})")
    parser.add_argument("--format", type=str, default="all",
                        choices=["all", "onnx", "ncnn", "tflite", "engine"],
                        help="Export format (default: all = onnx + ncnn)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--half", action="store_true",
                        help="Export FP16 (TensorRT only)")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.  pip install ultralytics")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        print(f"  Train first: python drone_detection/scripts/train.py")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    formats = [args.format] if args.format != "all" else ["onnx", "ncnn"]

    for fmt in formats:
        print(f"\n{'='*50}")
        print(f"Exporting to {fmt.upper()} ...")
        print(f"{'='*50}")

        try:
            kwargs = {
                "format": fmt,
                "imgsz": args.imgsz,
            }
            if fmt == "onnx":
                kwargs["simplify"] = True
                kwargs["opset"] = 12
            elif fmt == "engine":
                kwargs["half"] = args.half

            export_path = model.export(**kwargs)
            print(f"  Exported: {export_path}")
        except Exception as e:
            print(f"  FAILED: {e}")
            if fmt == "ncnn":
                print("  Install ncnn: pip install ncnn")
            elif fmt == "engine":
                print("  TensorRT only available on Jetson/NVIDIA GPU")

    # Print deployment summary
    model_dir = model_path.parent
    print(f"\n{'='*60}")
    print(f"Export complete!  Files in: {model_dir}")
    print()
    print("Deployment options:")
    print(f"  RPi 5 (fastest): Copy *_ncnn_model/ folder")
    print(f"  Any (simplest):  Copy *.onnx file")
    print(f"  Jetson:          Export with --format engine --half on the Jetson itself")
    print()
    print("Test inference:")
    print(f"  python drone_detection/scripts/test_inference.py --model {model_dir}/best.onnx")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
