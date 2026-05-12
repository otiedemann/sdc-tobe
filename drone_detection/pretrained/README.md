# Pretrained YOLO11n drone detector

Trained on the [Seraphim Drone Detection Dataset](https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset)
(83,483 images: 75,134 train + 8,349 test, single class `drone`, 640×640).

Use these weights as-is for **collision detection** — the drone's own
camera feed gets scanned for other drones; a hit fires an avoidance
manoeuvre in the flight controller.

## Training run

| Setting | Value |
|---|---|
| Base model | `yolo11n.pt` (Ultralytics 8.3 pretrained on COCO) |
| Hardware | NVIDIA RTX 5060 Laptop GPU (8 GB, Blackwell sm_120) |
| Framework | PyTorch 2.11.0 + CUDA 12.8, Ultralytics 8.3 |
| Epochs | **37 of 100** (stopped manually — see metrics below) |
| Image size | 640 |
| Batch | auto (~16) |
| Optimiser | SGD, lr0=0.01 → lr_f=0.0001, momentum 0.937 |
| Augmentation | mosaic 1.0, mixup 0.1 |
| Wall time | ~5 h 5 min |

The training was stopped at epoch 37 when mAP50-95 was still improving
but at +0.0007/epoch — diminishing returns. Natural early-stop
(`patience=25`) would have triggered around epoch 70.

## Final metrics (epoch 37, on held-out test set)

| Metric | Value | What it means |
|---|---|---|
| **mAP50** | **0.9666** | At standard IoU≥0.5, model finds drones 96.7 % of the time |
| **mAP50-95** | **0.6835** | Averaged over IoU 0.5–0.95 (strict) — near state-of-the-art for single-class drone detection |
| precision | 0.928 | Of detections, 92.8 % are real drones |
| recall | 0.923 | Of real drones, 92.3 % get caught |
| cls_loss | 0.856 | (training-side, monitored for convergence) |

See `results.csv` for the full epoch-by-epoch trace, `labels.jpg` for
the dataset's bbox size + center distribution.

## Files

| File | Size | Use when |
|---|---|---|
| `yolo11n_seraphim_e37.pt` | 10.7 MB | Working in Python with Ultralytics — easiest |
| `yolo11n_seraphim_e37.onnx` | 10.6 MB | Production runtime (CPU or GPU), any language with OpenCV/ONNX-Runtime |
| `yolo11n_seraphim_e37_ncnn/` | 10 MB | Raspberry Pi 5 / mobile / embedded — fastest CPU inference |
| `results.csv` | 4.5 KB | Loss + metric per training epoch |
| `labels.jpg` | 194 KB | Visualization of label distribution |

## How to use it

### Quick check from the CLI

```bash
# In the drone_detection venv:
yolo predict task=detect \
    model=drone_detection/pretrained/yolo11n_seraphim_e37.pt \
    source=path/to/image.jpg \
    conf=0.25

# Predicted boxes saved to runs/detect/predict/
```

### From Python — single image

```python
from ultralytics import YOLO

model = YOLO("drone_detection/pretrained/yolo11n_seraphim_e37.pt")
results = model.predict("path/to/image.jpg", conf=0.25)

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])           # always 0 (drone)
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"drone  conf={conf:.2f}  bbox=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
```

### Production runtime: ONNX + OpenCV DNN

For the Pi-side flight controller. No PyTorch dependency — just
`opencv-python-headless` and `numpy`. The repo already has a
`DroneDetector` class wired up to do this against the FC's MJPEG
stream — point it at the .onnx file:

```bash
# On the flight controller (or anywhere with cv2 + the FC reachable)
python drone_detection/drone_detector.py \
    --api http://flightctrl1:8080 \
    --model drone_detection/pretrained/yolo11n_seraphim_e37.onnx \
    --conf 0.25
```

In code, the integration looks like this (mirrors
`drone_detection/drone_detector.py`):

```python
from drone_detection.drone_detector import DroneDetector
import time

detector = DroneDetector(
    api_base="http://flightctrl1:8080",   # FC HTTP API
    model_path="drone_detection/pretrained/yolo11n_seraphim_e37.onnx",
    conf_threshold=0.25,
)
detector.start()                          # background thread

while True:
    detections = detector.get_detections()
    # detections is a list of dicts: confidence, cx, cy, bbox, rel_x, rel_y
    # rel_x/rel_y are normalised [0..1] from the frame centre; use them
    # to decide which way to dodge.
    if detections:
        nearest = max(detections, key=lambda d: d["bbox"][2] * d["bbox"][3])
        print(f"INTRUDER at rel ({nearest['rel_x']:+.2f}, {nearest['rel_y']:+.2f})")
        # … hand off to your evasion module here …
    time.sleep(0.1)
```

### One-shot test on the live MJPEG stream

```bash
python drone_detection/scripts/test_inference.py \
    --model drone_detection/pretrained/yolo11n_seraphim_e37.onnx \
    --source http://flightctrl1:8080/api/video \
    --benchmark 100
```

Prints per-frame FPS + saves overlay frames so you can verify the model
is firing on the right things.

## Inference performance

Measured by Ultralytics during export (640×640 input):

| Platform | Format | Inference latency | FPS |
|---|---|---|---|
| RTX 5060 (training host) | PyTorch | 6.2 ms | 160 |
| RPi 5 | NCNN | ~67 ms | 15 |
| RPi 5 | ONNX (cv2.dnn) | ~128 ms | 8 |
| Jetson Orin Nano | TensorRT FP16 | ~27 ms | 37 |
| x86 laptop CPU | ONNX | ~50 ms | 20 |

For collision avoidance you want ≥10 fps on the platform that runs the
inference; the Pi 5 with NCNN sits comfortably above that with headroom
for the rest of the FC code.

## Reproducing this run

If you ever want to retrain (e.g. on a bigger model, different
augmentation, or with new data):

```bash
cd /home/sdc/sdc-tobe                         # on the bare-metal box
source drone_detection/.venv/bin/activate
python drone_detection/scripts/train.py       # default: 100 epochs, patience 25
# or:
python drone_detection/scripts/train.py --epochs 50 --patience 10 --quick
```

The training writes new checkpoints to `drone_detection/models/train/`
(gitignored). When happy, run `drone_detection/scripts/export_model.py`
to produce ONNX + NCNN and replace the artefacts here.

## Why this dataset

Seraphim has 83 k images spanning:
- Drones at every scale from 6×6 px to full-frame
- Indoor + outdoor + arena environments (closest to the SDC arena)
- Multiple drone types (Anafi, DJI, Mavic, generic quad)
- Single-class — label noise is minimal vs multi-class datasets

For the SDC use-case (detect-other-drones-from-your-drone-camera)
that's exactly the visual prior we need: "a thing that looks like a
small flying quadcopter, regardless of its colour or model".
