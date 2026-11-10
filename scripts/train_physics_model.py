#!/usr/bin/env python3
"""Train the detector on the 3-channel physics-motivated dataset.

CONTROLLED-EXPERIMENT GUARANTEES (identical to the baseline run):
  * Model      : p1p2 - custom space_debris/models/yolov8_p1p2.yaml (P1/P2
                 heads for small objects), scale n, trained FROM SCRATCH
                 exactly like train_jarvis.py -> train_with_recovery() did.
                 (create_custom_model is imported from the baseline module.)
  * Training   : epochs=30, imgsz=640, batch=16, workers=4, patience=30,
                 save_period=1, ultralytics defaults for all other knobs.
  * Data       : same 11-class taxonomy (space_debris.config.CLASS_NAMES),
                 same annotation files (byte-identical copies), only the
                 IMAGES differ (3-channel preprocessing).
  * Logging    : baseline TrainingLogger (reused), own directory so baseline
                 logs are never touched.

THE ONLY DELIBERATE DIFFERENCES:
  1. Input images come from processed_dataset/ (3-channel pipeline).
  2. HSV colour augmentation is DISABLED (hsv_h=hsv_s=hsv_v=0) because the
     three channels are engineered features, not RGB colours. Geometric
     augmentations that keep bounding boxes aligned are retained.
  3. Outputs go to runs/physics_detect/ (baseline runs/detect/ is untouched).

Usage (JarvisLabs, GPU):
    python scripts/train_physics_model.py                  # full training
    python scripts/train_physics_model.py --epochs 30 --batch 16
    python scripts/train_physics_model.py --resume          # after a crash
    python scripts/train_physics_model.py --preflight-only  # sanity check, no training

This script NEVER starts on its own - run it explicitly after reviewing the
preprocessing results (README.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("CAPSTONE_DIR", str(REPO_ROOT))


def _latest_weights(weights_dir: Path) -> str | None:
    if not weights_dir.is_dir():
        return None
    pts = list(weights_dir.glob("*.pt"))
    if not pts:
        return None
    return str(max(pts, key=lambda p: p.stat().st_mtime))


def _find_sample_image(data_yaml: Path) -> Path | None:
    """One processed image for the 3-channel forward-pass preflight."""
    import yaml

    meta = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(str(meta.get("path", data_yaml.parent)))
    train_dir = root / str(meta.get("train", "images/train"))
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        found = sorted(train_dir.glob(ext))
        if found:
            return found[0]
    return None


def preflight(model, data_yaml: Path, device) -> None:
    """Verify the dataset YAML and that the model accepts 3-channel input.

    Runs a single forward pass on one processed image and checks the tensor
    the network actually sees is (H, W, 3) uint8-compatible. Raises on failure
    so training never starts on a broken setup.
    """
    import cv2

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Run: python scripts/run_preprocessing.py --source <drive_folder>"
        )
    import yaml

    meta = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(str(meta.get("path", data_yaml.parent)))
    for split_key in ("train", "val"):
        split_dir = root / str(meta.get(split_key, ""))
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset split dir missing: {split_dir} (yaml key '{split_key}')"
            )
    print(f"[preflight] dataset yaml OK: {data_yaml}")
    print(f"[preflight] classes: nc={meta.get('nc')} names={len(meta.get('names') or [])}")

    sample = _find_sample_image(data_yaml)
    if sample is None:
        raise FileNotFoundError("No processed image found for preflight")

    img = cv2.imread(str(sample), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise RuntimeError(
            f"Processed image is not 3-channel: {sample} "
            f"(shape={None if img is None else img.shape})"
        )
    print(f"[preflight] sample image OK: {sample.name} shape={img.shape} dtype={img.dtype}")

    results = model.predict(source=str(sample), imgsz=640, device=device, verbose=False)
    orig = results[0].orig_img
    if orig is None or orig.ndim != 3 or orig.shape[2] != 3:
        raise RuntimeError(f"Model input is not 3-channel: {getattr(orig, 'shape', None)}")
    print(f"[preflight] forward pass OK: model input {orig.shape} "
          f"(3 engineered channels accepted)")


def main() -> int:
    from preprocessing.experiment_config import cfg_get, load_physics_config

    cfg = load_physics_config()

    parser = argparse.ArgumentParser(
        description="Train p1p2-n on the 3-channel physics dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=str(REPO_ROOT / "processed_dataset" / "space_debris.yaml"),
        help="dataset YAML written by run_preprocessing.py",
    )
    parser.add_argument("--epochs", type=int, default=int(cfg_get(cfg, "training.epochs", 30)))
    parser.add_argument("--imgsz", type=int, default=int(cfg_get(cfg, "training.imgsz", 640)))
    parser.add_argument("--batch", type=int, default=int(cfg_get(cfg, "training.batch", 16)))
    parser.add_argument("--workers", type=int, default=int(cfg_get(cfg, "training.workers", 4)))
    parser.add_argument("--device", default=None, help="cuda index / 'cpu' / auto")
    parser.add_argument("--model-size", default=str(cfg_get(cfg, "model.model_size", "n")),
                        choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--arch", default=str(cfg_get(cfg, "model.arch", "p1p2")),
                        choices=["standard", "p1p2", "yolo11"],
                        help="baseline used p1p2 - do not change for the comparison")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest physics_detect checkpoint")
    parser.add_argument("--max-retries", type=int,
                        default=int(cfg_get(cfg, "training.max_retries", 3)))
    parser.add_argument("--name", default="physics_detect",
                        help="run directory name under runs/ (baseline uses detect)")
    parser.add_argument("--preflight-only", action="store_true",
                        help="run dataset + forward-pass checks, then exit (no training)")
    args = parser.parse_args()

    data_yaml = Path(args.data).expanduser().resolve()

    import torch
    from ultralytics import YOLO

    # Reuse the baseline's own utilities - same model builder, device logic,
    # crash-recovery logging. Nothing in space_debris/ is modified.
    from space_debris.models.train_resilient import (
        TrainingLogger,
        create_custom_model,
        detect_device,
    )

    device, device_info = detect_device(args.device)
    logger = TrainingLogger(log_dir=REPO_ROOT / "logs" / "physics_training")
    logger.log(f"Device: {device_info}")
    logger.log(
        f"Experiment: 3-channel physics preprocessing | model={args.arch}-{args.model_size} "
        f"| epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} "
        f"workers={args.workers} | HSV augmentation DISABLED"
    )
    logger.log(f"Dataset yaml: {data_yaml}")

    runs_dir = REPO_ROOT / "runs" / args.name
    weights_dir = runs_dir / "weights"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot of this run's configuration (audit trail for the report)
    snapshot = {
        "experiment": "physics_3channel",
        "model": {"arch": args.arch, "model_size": args.model_size,
                  "initialization": "scratch" if args.arch == "p1p2" else "pretrained"},
        "training": {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
                     "workers": args.workers, "patience": int(cfg_get(cfg, "training.patience", 30)),
                     "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
                     "geometric_augmentations": "ultralytics defaults (box-aligned)"},
        "data": str(data_yaml),
        "device": device,
        "split": cfg_get(cfg, "split", {}),
        "baseline_run_dir": "runs/detect (untouched)",
    }
    (runs_dir / "train_config.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    # ---------------- model: EXACTLY the baseline architecture --------------
    checkpoint = _latest_weights(weights_dir) if args.resume else None
    if checkpoint:
        logger.log(f"Resuming weights from: {checkpoint}")
        model = YOLO(checkpoint)
    else:
        logger.log(f"Creating model from scratch: arch={args.arch} size={args.model_size}")
        model = create_custom_model(model_size=args.model_size, arch=args.arch)

    # ---------------- preflight --------------------------------------------
    try:
        preflight(model, data_yaml, device)
    except Exception as exc:
        logger.log(f"PREFLIGHT FAILED: {exc}", "ERROR")
        traceback.print_exc()
        return 1
    if args.preflight_only:
        logger.log("Preflight only - stopping before training.")
        return 0

    # ---------------- training with crash recovery --------------------------
    # Mirrors train_resilient.train_with_recovery(): a single train() call for
    # all remaining epochs; on crash the newest weights are reloaded and
    # training continues (JarvisLabs spot interruptions).
    state = {"done": 0, "base": 0, "last_t": time.time()}
    total_epochs = args.epochs
    patience = int(cfg_get(cfg, "training.patience", 30))
    save_period = int(cfg_get(cfg, "training.save_period", 1))

    def on_fit_epoch_end(*cb_args, **cb_kwargs):
        trainer = None
        for a in cb_args:
            if hasattr(a, "epoch"):
                trainer = a
                break
        if trainer is None:
            return
        try:
            now = time.time()
            m = {}
            for attr in ("metrics", "results_dict", "results"):
                v = getattr(trainer, attr, None)
                if isinstance(v, dict) and v:
                    m = v
                    break
                rd = getattr(v, "results_dict", None) if v is not None else None
                if isinstance(rd, dict) and rd:
                    m = rd
                    break
            state["done"] = state["base"] + int(trainer.epoch) + 1
            logger.log_metrics(state["done"], {
                "epoch_time_sec": now - state["last_t"],
                "metrics": m,
            })
            state["last_t"] = now
            shown = ", ".join(
                f"{k}={v:.4f}" for k, v in list(m.items())[:4] if isinstance(v, float)
            )
            logger.log(f"Epoch {state['done']}/{total_epochs} completed ({shown})")
        except Exception:
            pass

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    train_started = time.time()
    results = None
    attempts = 0
    while True:
        remaining = total_epochs - state["done"]
        if remaining <= 0:
            break
        state["base"] = state["done"]
        state["last_t"] = time.time()
        try:
            logger.log(f"Training {remaining} epoch(s) ({state['done']}/{total_epochs} done)")
            results = model.train(
                data=str(data_yaml),
                epochs=remaining,
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                device=device,
                resume=False,
                save=True,
                save_period=save_period,
                val=True,
                project=str(runs_dir.parent),
                name=args.name,
                patience=patience,
                verbose=True,
                exist_ok=True,
                # --- the only augmentation change vs baseline ---------------
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
            )
            state["done"] = total_epochs
            break
        except Exception as exc:
            attempts += 1
            logger.log_error(state["done"], exc, traceback.format_exc())
            logger.log(
                f"Training interrupted after {state['done']} epochs "
                f"(attempt {attempts}): {exc}",
                "ERROR",
            )
            if attempts > args.max_retries:
                logger.log("Max retries reached. Giving up.", "ERROR")
                break
            wait = min(30 * attempts, 300)
            logger.log(f"Waiting {wait}s before retry...")
            time.sleep(wait)
            latest = _latest_weights(weights_dir)
            if latest:
                logger.log(f"Reloading model from: {latest}")
                model = YOLO(latest)
                model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    duration = time.time() - train_started
    peak_gb = (
        round(torch.cuda.max_memory_allocated(0) / 1e9, 3)
        if device == "cuda" and torch.cuda.is_available()
        else None
    )

    final_metrics = {}
    if results is not None:
        for attr in ("results_dict", "metrics"):
            v = getattr(results, attr, None)
            if isinstance(v, dict) and v:
                final_metrics = v
                break
        shown = ", ".join(f"{k}={v:.4f}" for k, v in final_metrics.items() if isinstance(v, float))
        if shown:
            logger.log(f"Final metrics: {shown}")

    summary = {
        "experiment": "physics_3channel",
        "epochs_completed": state["done"],
        "epochs_requested": total_epochs,
        "duration_sec": round(duration, 1),
        "device": device,
        "peak_gpu_mem_gb": peak_gb,
        "final_metrics": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                          for k, v in final_metrics.items()},
        "weights": str(weights_dir / "best.pt"),
        "hsv_augmentation": "disabled",
        "model": f"{args.arch}-{args.model_size}",
    }
    (runs_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.log(f"\nTraining finished: {state['done']}/{total_epochs} epochs, "
               f"{round(duration, 1)}s, peak GPU={peak_gb} GB")
    logger.log(f"Weights: {weights_dir / 'best.pt'}")
    logger.log(f"Summary: {runs_dir / 'train_summary.json'}")
    logger.log("Next: python scripts/evaluate_model.py --tag physics")
    return 0 if state["done"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
