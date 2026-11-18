#!/usr/bin/env python3
"""
Export YOLOv8/v11 segmentation model to TensorRT engine for Jetson.

Usage:
    python3 scripts/export_tensorrt.py --model models/best225v3.pt --imgsz 960 --half

The resulting .engine file is saved alongside the .pt file.
Load it in main.py with --model models/best225v3.engine

Notes:
    - Must be run ON the target Jetson device (TensorRT engines are device-specific).
    - First export takes 5–20 minutes depending on imgsz and Jetson model.
    - Subsequent runs reuse the cached engine (much faster).
    - FP16 (--half) recommended for Orin Nano: ~2× throughput vs FP32.
    - dynamic=False gives best latency (fixed batch=1).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export YOLO .pt → TensorRT .engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",   required=True, help="Path to .pt model file")
    p.add_argument("--imgsz",   type=int, default=960, choices=[640, 800, 960, 1024],
                   help="Inference image size")
    # [multicam] Model camera lược là object-detection; khai báo rõ để không bị
    # ultralytics suy luận sai khi .pt không mang task.
    p.add_argument("--task",    default=None, choices=["detect", "segment"],
                   help="Override model task (comb camera dùng detect)")
    p.add_argument("--half",    action="store_true", help="FP16 precision")
    p.add_argument("--dynamic", action="store_true", help="Dynamic batch size (slower)")
    p.add_argument("--workspace", type=int, default=4,
                   help="TensorRT workspace size in GB")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. TensorRT export requires a GPU.", file=sys.stderr)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Model: {model_path}")
    print(f"imgsz: {args.imgsz}  half(FP16): {args.half}  dynamic: {args.dynamic}")
    print(f"Workspace: {args.workspace} GB")
    print()

    # Import after CUDA check to avoid confusion
    from ultralytics import YOLO

    model = YOLO(str(model_path), task=args.task) if args.task else YOLO(str(model_path))

    engine_path = model_path.with_suffix(".engine")
    print(f"Exporting to: {engine_path}")
    print("This may take 5–20 minutes on first run…")

    t0 = time.time()
    exported = model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        device=0,
        workspace=args.workspace,
        verbose=True,
    )
    elapsed = time.time() - t0

    print()
    print(f"Export complete in {elapsed:.1f}s")
    print(f"Engine saved to: {exported}")
    print()
    print("To use in main.py:")
    print(f"  python3 main.py --model {engine_path} --imgsz {args.imgsz} --device 0")


if __name__ == "__main__":
    main()
