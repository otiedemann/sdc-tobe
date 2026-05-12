# Drone Detection Module

Real-time drone detection using YOLO11n for the Swarm Drone Challenge arena.

**Pre-trained model is ready to use** in [`pretrained/`](pretrained/) —
mAP50 = 0.967 on the Seraphim test set. See
[`pretrained/README.md`](pretrained/README.md) for metrics, file format
guide, and code snippets. The rest of this README is the training +
deployment pipeline.

## Quick Start

### 1. Install dependencies

```bash
# Training (on your workstation with GPU)
pip install ultralytics huggingface_hub

# Inference (on the flight controller Pi — only OpenCV needed)
pip install opencv-python-headless numpy
```

### 2. Download the Seraphim dataset (83k images)

```bash
python drone_detection/scripts/download_dataset.py
```

This downloads ~9 GB from HuggingFace and creates the dataset YAML config.

### 3. Train YOLO11n

```bash
# Quick test (5 epochs, 10% data — verify everything works)
python drone_detection/scripts/train.py --quick

# Full training (~2-3 hours on RTX 3060)
python drone_detection/scripts/train.py

# Resume interrupted training
python drone_detection/scripts/train.py --resume
```

### 4. Export for edge deployment

```bash
python drone_detection/scripts/export_model.py
```

Creates `best.onnx` (works everywhere) and `best_ncnn_model/` (fastest on RPi 5).

### 5. Test inference

```bash
# On a single image
python drone_detection/scripts/test_inference.py \
    --model drone_detection/models/train/weights/best.onnx \
    --source test_image.jpg

# On live MJPEG stream
python drone_detection/scripts/test_inference.py \
    --model drone_detection/models/train/weights/best.onnx \
    --source http://flightctrl1:8080/api/video \
    --benchmark 100

# Standalone detector
python drone_detection/drone_detector.py \
    --api http://flightctrl1:8080 \
    --model drone_detection/models/train/weights/best.onnx
```

## Architecture

```
┌─────────────────────┐
│ MJPEG Video Stream  │  /api/video or /api/position/video
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │ DroneDetector│  Background thread (like VideoMarkerTracker)
    │             │  YOLO11n via OpenCV DNN
    └──────┬──────┘
           │
    get_detections()  →  [{confidence, cx, cy, bbox, rel_x, rel_y}, ...]
           │
    ┌──────┴──────┐
    │  Evasion    │  C2 strategy or local flight controller
    │  Module     │  Artificial Potential Fields + altitude layering
    └─────────────┘
```

## Performance

| Platform | Format | Inference | FPS |
|----------|--------|-----------|-----|
| RPi 5 | NCNN | ~67ms | 15 |
| RPi 5 | ONNX (OpenCV DNN) | ~128ms | 8 |
| Jetson Orin Nano | TensorRT FP16 | ~27ms | 37 |
| x86 laptop (CPU) | ONNX | ~50ms | 20 |

## Files

```
drone_detection/
├── drone_detector.py          # Main module — import this
├── README.md
├── scripts/
│   ├── download_dataset.py    # Download Seraphim dataset
│   ├── train.py               # Train YOLO11n
│   ├── export_model.py        # Export to ONNX/NCNN
│   └── test_inference.py      # Test on images or streams
├── models/                    # Training output (gitignored)
│   └── train/weights/best.pt
└── data/                      # Dataset (gitignored)
    ├── drone_dataset.yaml
    └── seraphim/
```
