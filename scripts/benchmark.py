#!/usr/bin/env python3
"""
Benchmark script for YarnBreak detection pipeline on Jetson.

Measures:
  - Per-inference latency (ms)
  - End-to-end FPS
  - UI drawing time (optional)

Usage:
    # Benchmark a video file at different sizes
    python3 scripts/benchmark.py --video ./videos/YB3.mp4 --model ./models/best225v3.pt

    # Compare .pt vs .engine
    python3 scripts/benchmark.py --video ./videos/YB3.mp4 \
        --models ./models/best225v3.pt ./models/best225v3.engine

    # Benchmark imgsz sweep
    python3 scripts/benchmark.py --video ./videos/YB3.mp4 --model ./models/best225v3.pt \
        --imgsz 640 800 960
"""

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YarnBreak inference benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",   required=True, help="Path to video file")
    p.add_argument("--model",   default=None,  help="Single model path")
    p.add_argument("--models",  nargs="+",     help="Multiple model paths to compare")
    p.add_argument("--imgsz",   nargs="+", type=int, default=[960], help="Image size(s)")
    p.add_argument("--conf",    type=float, default=0.20)
    p.add_argument("--device",  default="auto", help="auto / 0 / cpu")
    p.add_argument("--half",    action="store_true", help="FP16")
    p.add_argument("--warmup",  type=int, default=10, help="Warmup frames")
    p.add_argument("--frames",  type=int, default=200, help="Benchmark frames")
    p.add_argument("--with-ui", action="store_true", dest="with_ui",
                   help="Include OpenCV drawing time in measurement")
    p.add_argument("--target-class", default="YarnBreak", dest="target_class")
    return p.parse_args()


def _normalize(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _find_class_id(names, target: str) -> int:
    key = _normalize(target)
    mapping = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
    for cid, cname in mapping.items():
        if _normalize(cname) == key:
            return int(cid)
    raise ValueError(f"Class '{target}' not found in {names}")


def _load_frames(video_path: str, n: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from: {video_path}")
    # Tile frames if video is shorter than requested
    while len(frames) < n:
        frames.extend(frames[: n - len(frames)])
    return frames[:n]


def _run_benchmark(
    model_path: str,
    frames: List[np.ndarray],
    imgsz: int,
    conf: float,
    device: str,
    half: bool,
    warmup: int,
    target_class: str,
    with_ui: bool,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    class_id = _find_class_id(model.names, target_class)

    # Warmup
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(warmup):
        with torch.inference_mode():
            model.predict(source=dummy, conf=conf, imgsz=imgsz,
                          classes=[class_id], device=device, half=half, verbose=False)

    infer_times: List[float] = []
    ui_times:    List[float] = []
    detections = 0

    for frame in frames:
        t0 = time.perf_counter()
        with torch.inference_mode():
            results = model.predict(source=frame, conf=conf, imgsz=imgsz,
                                    classes=[class_id], device=device, half=half, verbose=False)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        infer_times.append(infer_ms)

        r = results[0]
        if r.boxes and len(r.boxes) > 0:
            detections += 1

        if with_ui:
            t_ui = time.perf_counter()
            out = frame.copy()
            if r.boxes and len(r.boxes) > 0:
                x1, y1, x2, y2 = map(int, r.boxes.xyxy[0].tolist())
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(out, "YarnBreak", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ui_times.append((time.perf_counter() - t_ui) * 1000.0)

    return {
        "model":     Path(model_path).name,
        "imgsz":     imgsz,
        "n_frames":  len(frames),
        "det_rate":  detections / len(frames) * 100.0,
        "infer_avg": statistics.mean(infer_times),
        "infer_p50": statistics.median(infer_times),
        "infer_p95": sorted(infer_times)[int(len(infer_times) * 0.95)],
        "infer_min": min(infer_times),
        "infer_max": max(infer_times),
        "fps_infer": 1000.0 / statistics.mean(infer_times),
        "ui_avg":    statistics.mean(ui_times) if ui_times else 0.0,
    }


def _print_result(r: dict):
    print(f"\n  Model   : {r['model']}")
    print(f"  imgsz   : {r['imgsz']}")
    print(f"  Frames  : {r['n_frames']}   Detect rate: {r['det_rate']:.1f}%")
    print(f"  Infer   : avg={r['infer_avg']:.1f}ms  p50={r['infer_p50']:.1f}ms  p95={r['infer_p95']:.1f}ms  min={r['infer_min']:.1f}ms  max={r['infer_max']:.1f}ms")
    print(f"  FPS     : {r['fps_infer']:.1f} (inference only)")
    if r['ui_avg'] > 0:
        total_fps = 1000.0 / (r['infer_avg'] + r['ui_avg'])
        print(f"  UI draw : {r['ui_avg']:.1f}ms   End-to-end FPS: {total_fps:.1f}")


def main():
    args = parse_args()

    # Device
    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    half = args.half and device != "cpu"

    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device={device}  half={half}")
    print(f"Loading {args.warmup + args.frames} frames from: {args.video}")

    try:
        frames = _load_frames(args.video, args.warmup + args.frames)
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    bench_frames = frames[args.warmup:]  # warmup frames are passed to each run separately

    models = args.models or ([args.model] if args.model else [])
    if not models:
        print("[ERROR] Provide --model or --models", file=sys.stderr)
        sys.exit(1)

    results = []
    for model_path in models:
        for imgsz in args.imgsz:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {model_path}  imgsz={imgsz}")
            print(f"{'='*60}")
            try:
                r = _run_benchmark(
                    model_path=model_path,
                    frames=bench_frames,
                    imgsz=imgsz,
                    conf=args.conf,
                    device=device,
                    half=half,
                    warmup=args.warmup,
                    target_class=args.target_class,
                    with_ui=args.with_ui,
                )
                results.append(r)
                _print_result(r)
            except Exception as exc:
                print(f"[FAILED] {exc}")

    # Summary table
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"{'Model':<28} {'imgsz':>6} {'avg ms':>8} {'FPS':>7} {'p95 ms':>8}")
        print(f"{'-'*60}")
        for r in results:
            print(f"{r['model']:<28} {r['imgsz']:>6} {r['infer_avg']:>8.1f} {r['fps_infer']:>7.1f} {r['infer_p95']:>8.1f}")


if __name__ == "__main__":
    main()
