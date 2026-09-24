#!/usr/bin/env python
"""
Space Debris Detection - Main Pipeline

Usage:
    python -m space_debris.main sync          # Sync data from Google Drive
    python -m space_debris.main preprocess
    python -m space_debris.main convert_labels
    python -m space_debris.main split_dataset
    python -m space_debris.main train
    python -m space_debris.main train --resume
    python -m space_debris.main train_resilient  # P1/P2 model with logging & recovery
    python -m space_debris.main predict
    python -m space_debris.main logs            # View training logs
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from space_debris.config import verify_paths, DEVICE, ensure_data_synced, GDRIVE_CACHE_DIR
from space_debris.data.preprocess import preprocess_dataset, preview_preprocessing
from space_debris.data.convert_labels import convert_labels
from space_debris.data.split_dataset import split_dataset
from space_debris.data.gdrive_sync import sync_capstone_data
from space_debris.models.train import train
from space_debris.models.train_resilient import train_with_recovery
from space_debris.models.inference import predict, visualize_predictions
from space_debris.utils.visualize import show_preprocessing_comparison

def main():
    parser = argparse.ArgumentParser(description="Space Debris Detection Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Sync from Google Drive
    sync_parser = subparsers.add_parser("sync", help="Sync dataset from Google Drive (Shared with me)")
    sync_parser.add_argument("--folder", default="Capstone", help="Folder name in Shared with me")
    sync_parser.add_argument("--cache-dir", default=str(GDRIVE_CACHE_DIR), help="Local cache directory")

    # Preprocess
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess images")
    preprocess_parser.add_argument("--preview", action="store_true", help="Show preview only")
    preprocess_parser.add_argument("--num", type=int, default=2000, help="Number of images")
    preprocess_parser.add_argument("--force", action="store_true", help="Reprocess cached images")

    # Convert labels
    subparsers.add_parser("convert_labels", help="Convert CSV labels to YOLO format")

    # Split dataset
    split_parser = subparsers.add_parser("split_dataset", help="Split train/val")
    split_parser.add_argument("--val-ratio", type=float, default=0.1)

    # Train (standard)
    train_parser = subparsers.add_parser("train", help="Train YOLO model (standard)")
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=16)
    train_parser.add_argument("--workers", type=int, default=4, help="DataLoader workers; use 0 on Windows if workers crash")
    train_parser.add_argument("--device", default=DEVICE)
    train_parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    train_parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"],
                              help="YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), x(large)")
    train_parser.add_argument("--arch", default="standard", choices=["standard", "p1p2"],
                              help="Model architecture: standard (YOLOv8) or p1p2 (custom P1/P2 feature maps)")

    # Train (resilient with P1/P2 model and logging)
    train_r_parser = subparsers.add_parser("train_resilient", help="Train with P1/P2 feature maps, epoch logging & error recovery")
    train_r_parser.add_argument("--epochs", type=int, default=30)
    train_r_parser.add_argument("--imgsz", type=int, default=640)
    train_r_parser.add_argument("--batch", type=int, default=16)
    train_r_parser.add_argument("--workers", type=int, default=4)
    train_r_parser.add_argument("--device", default=DEVICE)
    train_r_parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    train_r_parser.add_argument("--max-retries", type=int, default=3, help="Max retries per epoch")
    train_r_parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"],
                                help="YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), x(large)")
    train_r_parser.add_argument("--arch", default="p1p2", choices=["standard", "p1p2"],
                                help="Model architecture: standard (YOLOv8) or p1p2 (custom P1/P2 feature maps)")

    # Predict
    pred_parser = subparsers.add_parser("predict", help="Run inference")
    pred_parser.add_argument("--model", help="Model path")
    pred_parser.add_argument("--source", help="Source directory")
    pred_parser.add_argument("--conf", type=float, default=0.25)
    pred_parser.add_argument("--device", default=DEVICE)
    pred_parser.add_argument("--visualize", action="store_true", help="Show predictions")

    # View logs
    log_parser = subparsers.add_parser("logs", help="View training logs")
    log_parser.add_argument("--session", help="Session ID (default: latest)")
    log_parser.add_argument("--follow", action="store_true", help="Follow log file (tail -f)")
    log_parser.add_argument("--metrics", action="store_true", help="Show metrics summary")

    args = parser.parse_args()

    # Auto-sync for commands that need data
    if args.command in ("preprocess", "convert_labels", "split_dataset", "train", "train_resilient"):
        if not ensure_data_synced():
            print("❌ Failed to prepare data. Run 'python -m space_debris.main sync' first.")
            sys.exit(1)

    if args.command == "sync":
        print(f"🔄 Syncing '{args.folder}' from Google Drive...")
        path = sync_capstone_data(cache_dir=args.cache_dir, folder_name=args.folder)
        print(f"✅ Done: {path}")

    elif args.command == "preprocess":
        if args.preview:
            preview_preprocessing()
        else:
            preprocess_dataset(args.num, force=args.force)

    elif args.command == "convert_labels":
        convert_labels()

    elif args.command == "split_dataset":
        split_dataset(train_ratio=1 - args.val_ratio)

    elif args.command == "train":
        train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
              device=args.device, resume=args.resume, workers=args.workers,
              model_size=args.model_size, arch=args.arch)

    elif args.command == "train_resilient":
        train_with_recovery(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            resume=args.resume,
            workers=args.workers,
            max_retries=args.max_retries,
            model_size=args.model_size,
            arch=args.arch
        )

    elif args.command == "predict":
        results = predict(model_path=args.model, source=args.source, conf=args.conf, device=args.device)
        if args.visualize:
            visualize_predictions()

    elif args.command == "logs":
        from space_debris.config import RUNS_DIR
        logs_dir = RUNS_DIR / "training_logs"
        
        if args.metrics:
            # Show metrics summary
            import json
            metrics_files = sorted(logs_dir.glob("metrics_*.jsonl"))
            if not metrics_files:
                print("No metrics files found")
            else:
                latest = metrics_files[-1] if args.session is None else logs_dir / f"metrics_{args.session}.jsonl"
                if latest.exists():
                    print(f"Metrics from: {latest}")
                    with open(latest) as f:
                        for line in f:
                            data = json.loads(line)
                            epoch = data.get('epoch', '?')
                            metrics = data.get('metrics', {})
                            print(f"  Epoch {epoch}: {metrics}")
                else:
                    print(f"Metrics file not found: {latest}")
        elif args.follow:
            import subprocess
            log_files = sorted(logs_dir.glob("training_*.log"))
            if not log_files:
                print("No log files found")
            else:
                latest = log_files[-1] if args.session is None else logs_dir / f"training_{args.session}.log"
                print(f"Following: {latest} (Ctrl+C to stop)")
                if sys.platform == "win32":
                    subprocess.run(["powershell", "-Command", f"Get-Content -Wait -Tail 50 {latest}"])
                else:
                    subprocess.run(["tail", "-f", "-n", "50", str(latest)])
        else:
            # List log sessions
            log_files = sorted(logs_dir.glob("training_*.log"))
            if not log_files:
                print("No training logs found")
            else:
                print("Available training sessions:")
                for lf in log_files:
                    session = lf.stem.replace("training_", "")
                    size = lf.stat().st_size
                    mtime = datetime.fromtimestamp(lf.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"  {session}  ({size} bytes, {mtime})")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()