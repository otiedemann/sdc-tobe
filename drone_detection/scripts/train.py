#!/usr/bin/env python3
"""
Train YOLO11n for drone detection.

Prerequisites:
    pip install ultralytics
    python drone_detection/scripts/download_dataset.py   # download + prepare data

Usage:
    # Full training (GPU recommended)
    python drone_detection/scripts/train.py

    # Quick test run (5 epochs, small subset)
    python drone_detection/scripts/train.py --quick

    # Resume interrupted training
    python drone_detection/scripts/train.py --resume

    # Custom parameters
    python drone_detection/scripts/train.py --epochs 200 --batch 32 --imgsz 640

After training:
    Best model:  drone_detection/models/train/weights/best.pt
    Last model:  drone_detection/models/train/weights/last.pt

Export for deployment:
    python drone_detection/scripts/export_model.py
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_YAML = PROJECT_DIR / "data" / "drone_dataset.yaml"
OUTPUT_DIR = PROJECT_DIR / "models"


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO11n for drone detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Base model (default: yolo11n.pt, pretrained on COCO)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size (-1 = auto 60%% GPU, default: -1)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--device", default=None,
                        help="Device: 0 (GPU), cpu, mps (Apple Silicon). Auto-detected if omitted.")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience (default: 25)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Dataloader workers (default: 8)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 5 epochs, fraction=0.1 (fast validation)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--data", type=str, default=None,
                        help=f"Dataset YAML path (default: {DATASET_YAML})")
    args = parser.parse_args()

    # Check dependencies
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        print("  pip install ultralytics")
        sys.exit(1)

    # Determine dataset config
    data_yaml = Path(args.data) if args.data else DATASET_YAML
    if not data_yaml.exists():
        print(f"ERROR: Dataset YAML not found: {data_yaml}")
        print(f"  Run first: python drone_detection/scripts/download_dataset.py")
        sys.exit(1)

    # Determine device
    device = args.device
    if device is None:
        import torch
        if torch.cuda.is_available():
            device = 0
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"GPU detected: {gpu_name} ({vram:.1f} GB)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("Apple Silicon MPS detected")
        else:
            device = "cpu"
            print("No GPU detected — training on CPU (will be slow)")

    # Quick mode overrides
    epochs = args.epochs
    fraction = 1.0
    if args.quick:
        epochs = 5
        fraction = 0.1
        print("\n*** QUICK MODE: 5 epochs, 10% data ***\n")

    # Resume or fresh training
    if args.resume:
        last_pt = OUTPUT_DIR / "train" / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"ERROR: No checkpoint to resume from: {last_pt}")
            sys.exit(1)
        print(f"Resuming from {last_pt}")
        model = YOLO(str(last_pt))
    else:
        print(f"Loading base model: {args.model}")
        model = YOLO(args.model)

    print(f"\nTraining config:")
    print(f"  Dataset:    {data_yaml}")
    print(f"  Model:      {args.model}")
    print(f"  Epochs:     {epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch:      {args.batch} {'(auto)' if args.batch == -1 else ''}")
    print(f"  Device:     {device}")
    print(f"  Patience:   {args.patience}")
    print(f"  Output:     {OUTPUT_DIR}")
    if fraction < 1.0:
        print(f"  Fraction:   {fraction} (subset)")
    print()

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        workers=args.workers,
        project=str(OUTPUT_DIR),
        name="train",
        exist_ok=True,
        # Augmentation
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        # Training params
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,       # final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # Data
        fraction=fraction,
        # Logging
        verbose=True,
        plots=True,
    )

    # Print results
    best_pt = OUTPUT_DIR / "train" / "weights" / "best.pt"
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best model: {best_pt}")
    print(f"  Results:    {OUTPUT_DIR / 'train'}")
    print(f"\nNext step — export for edge deployment:")
    print(f"  python drone_detection/scripts/export_model.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
