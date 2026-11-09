#!/usr/bin/env python3
"""Run the 3-Channel Physics-Motivated preprocessing on the full dataset.

Reads the pre-built YOLO dataset from the Google Drive-synced source, re-splits
it 80/20 (seed 42, per configs/physics_config.yaml), preprocesses every image
with the 3-channel pipeline, copies the original annotations byte-for-byte,
writes processed_dataset/space_debris.yaml - and automatically uploads
processed_dataset/ to Google Drive (resumable; use --no-upload to skip).

Usage (local or JarvisLabs):
    # Dataset already synced / available locally:
    python scripts/run_preprocessing.py --source /path/to/Capstone

    # Sync from Google Drive first (uses existing pydrive2 OAuth flow):
    python scripts/run_preprocessing.py --sync

    # Full run, 8 workers, upload to Drive enabled (default):
    python scripts/run_preprocessing.py --source ~/data/Capstone --workers 8

    # Quick trial on 50 images, no upload:
    python scripts/run_preprocessing.py --source ~/data/Capstone --limit 50 --no-upload

    # Upload an already-processed tree (resume/repair):
    python scripts/run_preprocessing.py --upload-only --out processed_dataset

Re-run safety: already-processed images are skipped (checkpointing), labels are
re-copied idempotently, and the Drive manifest prevents duplicate uploads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("CAPSTONE_DIR", str(REPO_ROOT))

from preprocessing.experiment_config import cfg_get, load_physics_config  # noqa: E402
from preprocessing.physics_pipeline import PhysicsPipelineConfig  # noqa: E402


def _pipeline_from_config(cfg: dict) -> PhysicsPipelineConfig:
    p = cfg_get(cfg, "preprocessing", {}) or {}
    return PhysicsPipelineConfig(
        percentiles=tuple(p.get("percentiles", (1.0, 99.0))),
        denoise_kernel=tuple(p.get("denoise_kernel", (3, 3))),
        denoise_sigma=float(p.get("denoise_sigma", 0.0)),
        clahe_clip_limit=float(p.get("clahe_clip_limit", 1.5)),
        clahe_tile_grid=tuple(p.get("clahe_tile_grid", (8, 8))),
        background_sigma=float(p.get("background_sigma", 25.0)),
        residual_shift=int(p.get("residual_shift", 128)),
        residual_blur_kernel=tuple(p.get("residual_blur_kernel", (3, 3))),
    )


def _default_source(cfg: dict) -> Path | None:
    """Find a locally available dataset without inventing paths."""
    base = Path(os.environ.get("CAPSTONE_DIR", REPO_ROOT))
    candidates = [
        base / cfg_get(cfg, "dataset.dataset_dir_name", "yolo_dataset"),
        base,
        base / "gdrive_cache" / "capstone_data",
        base / "gdrive_cache" / "Capstone",
    ]
    from preprocessing.dataset_processor import (  # local import: cheap check
        _is_yolo_root,
    )

    for cand in candidates:
        if not cand.is_dir():
            continue
        if _is_yolo_root(cand):
            return cand
        nested = cand / cfg_get(cfg, "dataset.dataset_dir_name", "yolo_dataset")
        if nested.is_dir() and _is_yolo_root(nested):
            return nested
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="3-channel physics-motivated preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        help="Local path to the synced Drive dataset (folder containing "
        "yolo_dataset/, or the yolo_dataset directory itself)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync the dataset from Google Drive first (pydrive2 OAuth)",
    )
    parser.add_argument(
        "--out",
        help="Output directory (default: <CAPSTONE_DIR>/processed_dataset)",
    )
    parser.add_argument("--config", help="Experiment YAML (default: configs/physics_config.yaml)")
    parser.add_argument("--train-ratio", type=float, help="override config split.train_ratio")
    parser.add_argument("--seed", type=int, help="override config split.seed")
    parser.add_argument("--workers", type=int, help="parallel preprocessing workers")
    parser.add_argument("--limit", type=int, help="cap number of pooled images (quick trials)")
    parser.add_argument("--force", action="store_true", help="reprocess already-cached images")
    parser.add_argument("--no-upload", action="store_true", help="skip the automatic Drive upload")
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="do not preprocess; just upload an existing --out tree to Drive",
    )
    parser.add_argument("--credentials", default="credentials.json", help="pydrive2 OAuth client secrets")
    args = parser.parse_args()

    cfg = load_physics_config(args.config)
    pipeline_cfg = _pipeline_from_config(cfg)
    train_ratio = args.train_ratio if args.train_ratio is not None else float(cfg_get(cfg, "split.train_ratio", 0.8))
    seed = args.seed if args.seed is not None else int(cfg_get(cfg, "split.seed", 42))
    out_root = Path(args.out) if args.out else Path(os.environ.get("CAPSTONE_DIR", REPO_ROOT)) / cfg_get(cfg, "dataset.output_dir_name", "processed_dataset")

    from preprocessing.drive_upload import DriveUploader, upload_tree

    # ------------------------------------------------------------------ upload-only
    if args.upload_only:
        print(f"[physics] upload-only mode: {out_root}")
        stats = upload_tree(
            out_root,
            folder_name=cfg_get(cfg, "upload.folder_name", "processed_dataset"),
            parent_folder_name=cfg_get(cfg, "dataset.source_drive_folder", "Capstone"),
            credentials_file=args.credentials,
        )
        print(json.dumps(stats, indent=2))
        return 0 if stats.get("failed", 0) == 0 else 1

    # ------------------------------------------------------------------ resolve source
    if args.sync:
        from space_debris.data.gdrive_sync import sync_capstone_data

        print("[physics] syncing dataset from Google Drive (first run needs OAuth)...")
        source = sync_capstone_data(
            cache_dir=str(Path(os.environ.get("CAPSTONE_DIR", REPO_ROOT)) / "gdrive_cache"),
            folder_name=cfg_get(cfg, "dataset.source_drive_folder", "Capstone"),
        )
    elif args.source:
        source = args.source
    else:
        source = _default_source(cfg)
        if source is None:
            print(
                "[physics] ERROR: no local dataset found. Use --source PATH "
                "or --sync to download from Google Drive."
            )
            return 2
    print(f"[physics] source: {source}")
    print(f"[physics] output: {out_root}")

    # ------------------------------------------------------------------ upload hook
    uploader = None
    on_output = None
    if not args.no_upload and bool(cfg_get(cfg, "upload.enabled", True)):
        uploader = DriveUploader(
            out_root,
            folder_name=cfg_get(cfg, "upload.folder_name", "processed_dataset"),
            parent_folder_name=cfg_get(cfg, "dataset.source_drive_folder", "Capstone"),
            credentials_file=args.credentials,
        )
        try:
            uploader.start()  # auth + remote folder + background thread
            on_output = uploader.put
        except Exception as exc:  # noqa: BLE001 - processing must still run
            print(
                f"[physics] WARNING: Drive upload disabled ({exc}).\n"
                "           Preprocessing continues locally; resume uploads later with:\n"
                f"           python scripts/run_preprocessing.py --upload-only --out {out_root}"
            )
            uploader = None
            on_output = None
    else:
        print("[physics] Drive upload disabled (--no-upload or config)")

    # ------------------------------------------------------------------ process
    from preprocessing.dataset_processor import run_preprocessing

    started = time.time()
    summary = run_preprocessing(
        source=source,
        out_root=out_root,
        train_ratio=train_ratio,
        seed=seed,
        force=args.force,
        limit=args.limit,
        workers=args.workers,
        pipeline_cfg=pipeline_cfg,
        on_output=on_output,
    )

    # ------------------------------------------------------------------ finish upload
    if uploader is not None:
        upload_stats = uploader.finish()
        summary["upload"] = upload_stats
        # rewrite summary so it records the upload outcome
        summary_path = Path(summary["summary_path"])
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # keep a copy in the repo logs/ directory (gitignored)
        logs_dir = REPO_ROOT / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_copy = logs_dir / f"preprocess_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
        log_copy.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[physics] summary copy: {log_copy}")

    totals = summary["totals"]
    print(
        f"\n[physics] finished in {round(time.time() - started, 1)}s | "
        f"processed={totals['processed']} cached={totals['skipped_existing']} "
        f"failed={totals['failed_or_corrupt']} "
        f"labels={totals['labels_copied']}"
    )
    print(f"[physics] dataset yaml: {summary['dataset_yaml']}")
    print(
        "[physics] next: python scripts/verify_dataset.py "
        f"--processed {out_root} --source {source}"
    )
    return 0 if totals["failed_or_corrupt"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
