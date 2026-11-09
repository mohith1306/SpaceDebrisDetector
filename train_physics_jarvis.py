#!/usr/bin/env python3
"""JarvisLabs entrypoint for the physics-motivated preprocessing experiment.

Mirrors train_jarvis.py (baseline) but drives the NEW experiment:
  1. ensures processed_dataset/ exists (runs scripts/run_preprocessing.py
     if it does not, including the automatic Drive upload),
  2. trains the SAME baseline model (p1p2-n) on the 3-channel dataset with
     HSV augmentation disabled (scripts/train_physics_model.py).

Usage (via jl):
    jl run . --script train_physics_jarvis.py --gpu L4 \
      --requirements requirements-gpu.txt -- --epochs 30 --batch 16

    # existing instance:
    jl run train_physics_jarvis.py --on <machine_id> -- --epochs 30

Flags are forwarded to scripts/train_physics_model.py; add
--skip-preprocess to assume processed_dataset/ is already built.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("CAPSTONE_DIR", str(ROOT))
sys.path.insert(0, str(ROOT))


def dataset_ready() -> bool:
    yaml_path = ROOT / "processed_dataset" / "space_debris.yaml"
    if not yaml_path.exists():
        return False
    images = ROOT / "processed_dataset" / "images" / "train"
    return images.is_dir() and any(images.iterdir())


def run_preprocessing(args: argparse.Namespace) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_preprocessing.py")]
    if args.source:
        cmd += ["--source", args.source]
    if args.sync:
        cmd += ["--sync"]
    if args.workers is not None:
        cmd += ["--workers", str(args.workers)]
    if args.no_upload:
        cmd += ["--no-upload"]
    print(f"[physics-jarvis] building processed_dataset: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="JarvisLabs physics-experiment entrypoint")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="assume processed_dataset/ already exists")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="build processed_dataset/ then exit (no training)")
    parser.add_argument("--source", help="forwarded to run_preprocessing.py")
    parser.add_argument("--sync", action="store_true",
                        help="forwarded to run_preprocessing.py (download from Drive)")
    parser.add_argument("--workers", type=int, default=None,
                        help="preprocessing workers (default: config/CPU count)")
    parser.add_argument("--no-upload", action="store_true",
                        help="forwarded to run_preprocessing.py")
    args, forward = parser.parse_known_args()

    print(f"[physics-jarvis] ROOT={ROOT}")
    print(f"[physics-jarvis] CAPSTONE_DIR={os.environ.get('CAPSTONE_DIR')}")

    import torch

    if torch.cuda.is_available():
        print(f"[physics-jarvis] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[physics-jarvis] WARNING: CUDA not available")

    # ---- 1. preprocessing -------------------------------------------------
    if args.preprocess_only or not dataset_ready():
        if args.skip_preprocess and not dataset_ready():
            raise SystemExit(
                "processed_dataset/ is missing but --skip-preprocess was given. "
                "Run scripts/run_preprocessing.py first."
            )
        rc = run_preprocessing(args)
        if rc != 0:
            raise SystemExit(f"preprocessing failed (exit {rc})")
        if args.preprocess_only:
            print("[physics-jarvis] preprocessing only - done.")
            return
    else:
        print("[physics-jarvis] processed_dataset/ present (no rebuild)")

    # ---- 2. training (only after preprocessing succeeds) ------------------
    cmd = [sys.executable, str(ROOT / "scripts" / "train_physics_model.py"), *forward]
    print(f"[physics-jarvis] training: {' '.join(cmd)}")
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
