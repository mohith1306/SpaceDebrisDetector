#!/usr/bin/env python3
"""JarvisLabs entrypoint: rebuild pipeline if needed, then train p1p2-n.

Usage (via jl run):
    jl run . --script train_jarvis.py --gpu L4 --requirements requirements-gpu.txt \\
      -- --epochs 30 --imgsz 640 --batch 16

Or on an existing instance:
    jl run train_jarvis.py --on <machine_id> -- --epochs 30
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Project root = directory containing this file
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("CAPSTONE_DIR", str(ROOT))
sys.path.insert(0, str(ROOT))


def counts() -> dict[str, int]:
    from space_debris.config import PROCESSED_DIR, YOLO_LABEL_DIR, YOLO_BASE

    def n(p: Path) -> int:
        try:
            return sum(1 for x in p.iterdir() if x.is_file())
        except FileNotFoundError:
            return 0

    return {
        "processed": n(PROCESSED_DIR),
        "labels_yolo": n(YOLO_LABEL_DIR),
        "train_img": n(YOLO_BASE / "images" / "train"),
        "val_img": n(YOLO_BASE / "images" / "val"),
    }


def ensure_pipeline(force_rebuild: bool = False) -> None:
    from space_debris.config import verify_paths, PROCESSED_DIR, YOLO_LABEL_DIR, YOLO_BASE
    from space_debris.data.preprocess import preprocess_dataset
    from space_debris.data.convert_labels import convert_labels
    from space_debris.data.split_dataset import split_dataset

    if not verify_paths():
        raise SystemExit(
            "Missing train/ or labels/train.csv under CAPSTONE_DIR. "
            f"CAPSTONE_DIR={os.environ.get('CAPSTONE_DIR')}"
        )

    c = counts()
    print(f"[jarvis] counts before: {c}")

    if force_rebuild or c["processed"] < 1000:
        print("[jarvis] running preprocess...")
        # preprocess_dataset uses num_images as a cap; pass a huge number for full set
        failed = preprocess_dataset(num_images=10**9, force=force_rebuild)
        if failed:
            print(f"[jarvis] preprocess failures: {len(failed)}")

    if force_rebuild or c["labels_yolo"] < 1000:
        print("[jarvis] running convert_labels...")
        convert_labels()

    if force_rebuild or c["train_img"] < 1000 or c["val_img"] < 100:
        print("[jarvis] running split_dataset (seed 42, 90/10)...")
        split_dataset(train_ratio=0.9)

    c2 = counts()
    print(f"[jarvis] counts after: {c2}")
    if c2["train_img"] < 1000 or c2["val_img"] < 100:
        raise SystemExit(f"Dataset incomplete after rebuild: {c2}")


def main() -> None:
    p = argparse.ArgumentParser(description="JarvisLabs train entrypoint")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    p.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"])
    p.add_argument("--arch", default="p1p2", choices=["standard", "p1p2"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--skip-rebuild", action="store_true", help="Assume yolo_dataset already built")
    p.add_argument("--force-rebuild", action="store_true")
    args = p.parse_args()

    print(f"[jarvis] ROOT={ROOT}")
    print(f"[jarvis] CAPSTONE_DIR={os.environ.get('CAPSTONE_DIR')}")

    import torch

    if torch.cuda.is_available():
        print(f"[jarvis] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[jarvis] WARNING: CUDA not available")

    if not args.skip_rebuild:
        ensure_pipeline(force_rebuild=args.force_rebuild)

    from space_debris.models.train_resilient import train_with_recovery

    train_with_recovery(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        resume=args.resume,
        workers=args.workers,
        max_retries=args.max_retries,
        model_size=args.model_size,
        arch=args.arch,
    )


if __name__ == "__main__":
    main()
