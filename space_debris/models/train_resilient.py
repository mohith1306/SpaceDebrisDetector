import os
import sys
import time
import traceback
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import torch

from ..config import CHECKPOINT_DIR, YOLO_BASE, RUNS_DIR, DEVICE

VALID_MODEL_SIZES = ["n", "s", "m", "l", "x"]
VALID_ARCHITECTURES = ["standard", "p1p2"]

LOGS_DIR = RUNS_DIR / "training_logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class TrainingLogger:
    def __init__(self, log_dir=LOGS_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / "training.log"
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.error_file = self.log_dir / "errors.log"
        
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{level}] {message}"
        print(line, flush=True)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")
            f.flush()
            
    def log_metrics(self, epoch, metrics):
        entry = {"epoch": epoch, "timestamp": datetime.now().isoformat(), **metrics}
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            
    def log_error(self, epoch, error, traceback_str):
        entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "traceback": traceback_str
        }
        with open(self.error_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


def get_latest_checkpoint(runs_dir):
    weights_dir = runs_dir / "detect" / "weights"
    if not weights_dir.exists():
        return None
    checkpoints = list(weights_dir.glob("*.pt"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def create_custom_model(model_size="n", arch="p1p2"):
    """Create YOLO model with chosen architecture, falling back to standard on failure."""
    if arch == "p1p2":
        model_yaml = Path(__file__).parent / "yolov8_p1p2.yaml"
        if model_yaml.exists():
            try:
                # Ultralytics guesses scale from filename (yolov8n...); materialize a
                # scale-tagged copy so --model-size is honored.
                import tempfile
                import yaml as pyyaml
                d = pyyaml.safe_load(model_yaml.read_text())
                d["scale"] = model_size
                tmp = Path(tempfile.mkdtemp()) / f"yolov8{model_size}_p1p2.yaml"
                tmp.write_text(pyyaml.safe_dump(d, sort_keys=False))
                return YOLO(str(tmp))
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Warning: Failed to load custom P1/P2 model: {e}")
                print(f"Falling back to standard yolov8{model_size}.pt")
    return YOLO(f"yolov8{model_size}.pt")


def detect_device(requested_device=None):
    """Auto-detect best available device (NVIDIA GPU > Apple MPS > CPU)."""
    if requested_device is not None:
        dev = str(requested_device).strip().lower()
        if dev in ("cuda", "0", "gpu"):
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return "cuda", f"NVIDIA GPU: {name} ({mem:.1f} GB VRAM)"
            else:
                print("WARNING: CUDA requested but not available, falling back to CPU")
                return "cpu", "CPU (CUDA unavailable)"
        elif dev == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps", "Apple Silicon GPU (MPS)"
            else:
                print("WARNING: MPS requested but not available, falling back to CPU")
                return "cpu", "CPU (MPS unavailable)"
        elif dev == "cpu":
            return "cpu", "CPU"
    # Auto-detect: CUDA > MPS > CPU
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return "cuda", f"NVIDIA GPU: {name} ({mem:.1f} GB VRAM)"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "Apple Silicon GPU (MPS)"
    return "cpu", "CPU"


def train_with_recovery(
    epochs=30,
    imgsz=640,
    batch=16,
    device=None,
    resume=False,
    workers=4,
    max_retries=3,
    model_size="n",
    arch="p1p2"
):
    device, device_info = detect_device(device)
    
    logger = TrainingLogger()
    logger.log(f"Device detected: {device_info}")
    logger.log(f"Starting training on device: {device}")
    logger.log(f"Epochs: {epochs}, Image size: {imgsz}, Batch: {batch}")
    logger.log(f"Logs saved to: {logger.log_dir}")
    
    yaml_path = YOLO_BASE / "space_debris.yaml"
    yaml_content = f"""path: {YOLO_BASE}
train: images/train
val: images/val

nc: 1
names: ['space_object']
"""
    yaml_path.write_text(yaml_content)
    logger.log(f"Dataset config: {yaml_path}")
    
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    if resume:
        checkpoint = get_latest_checkpoint(RUNS_DIR)
        if checkpoint:
            logger.log(f"Resuming from: {checkpoint}")
            model = YOLO(checkpoint)
        else:
            logger.log(f"No checkpoint found, starting fresh (arch: {arch}, size: {model_size})")
            model = create_custom_model(model_size, arch)
    else:
        logger.log(f"Starting fresh (arch: {arch}, size: {model_size})")
        model = create_custom_model(model_size, arch)
    
    # Training loop with error recovery
    completed_epochs = 0
    start_epoch = 0
    
    if resume and checkpoint:
        # Determine completed epochs from saved epoch checkpoints
        epoch_checkpoints = list(CHECKPOINT_DIR.glob("epoch_*.pt"))
        if epoch_checkpoints:
            completed = []
            for ckpt in epoch_checkpoints:
                try:
                    epoch_num = int(ckpt.stem.split("_")[1])
                    completed.append(epoch_num)
                except (IndexError, ValueError):
                    pass
            if completed:
                start_epoch = max(completed)
                completed_epochs = start_epoch
                logger.log(f"Resuming from epoch {start_epoch + 1} (found {len(completed)} completed epoch checkpoints)")
    
    for epoch in range(start_epoch, epochs):
        epoch_num = epoch + 1
        logger.log(f"\n{'='*60}")
        logger.log(f"Starting epoch {epoch_num}/{epochs}")
        logger.log(f"{'='*60}")
        
        epoch_start = time.time()
        retry_count = 0
        epoch_success = False
        
        while retry_count <= max_retries and not epoch_success:
            try:
                if retry_count > 0:
                    logger.log(f"Retry {retry_count}/{max_retries} for epoch {epoch_num}", "WARNING")
                
                # Train single epoch
                results = model.train(
                    data=str(yaml_path),
                    epochs=epoch_num,  # Train up to this epoch
                    imgsz=imgsz,
                    batch=batch,
                    workers=workers,
                    device=device,
                    resume=True,  # Resume from last checkpoint
                    save=True,
                    save_period=1,
                    val=False,  # Skip validation — torchvision NMS has CUDA backend errors
                    project=str(RUNS_DIR),
                    name="detect",
                    patience=epochs,  # Disable early stopping
                    verbose=True,
                    exist_ok=True,
                )
                
                epoch_time = time.time() - epoch_start
                completed_epochs = epoch_num
                epoch_success = True
                
                # Extract metrics - try multiple attribute names
                metrics = {}
                if results is not None:
                    # Try common attribute names for metrics
                    for attr in ['results_dict', 'metrics', 'loss', 'results']:
                        if hasattr(results, attr):
                            val = getattr(results, attr)
                            if isinstance(val, dict):
                                metrics = val
                                break
                            elif isinstance(val, (int, float)):
                                metrics = {attr: val}
                                break
                
                # Log metrics to JSONL file (may be empty if results None)
                try:
                    logger.log_metrics(epoch_num, {
                        "epoch_time_sec": epoch_time,
                        "metrics": metrics
                    })
                except Exception:
                    pass
                
                # Write epoch progress summary to training log file
                logger.log(f"\nEpoch {epoch_num}/{epochs} completed in {epoch_time:.1f}s")
                
                if metrics:
                    metric_parts = []
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            metric_parts.append(f"{k}:{v:.4f}" if isinstance(v, float) else f"{k}:{v}")
                    metric_str = " | ".join(metric_parts[:4])
                    logger.log(f"Metrics: {metric_str}")
                else:
                    logger.log("Metrics: None available")
                
                # Also print to console for real-time monitoring
                if metrics:
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}", end="  |  ")
                    print()
                else:
                    print(f"  Epoch {epoch_num} completed in {epoch_time:.1f}s (no metrics)")
                
            except Exception as e:
                retry_count += 1
                tb_str = traceback.format_exc()
                logger.log_error(epoch_num, e, tb_str)
                logger.log(f"Epoch {epoch_num} failed (attempt {retry_count}): {e}", "ERROR")
                
                if retry_count > max_retries:
                    logger.log(f"Max retries reached for epoch {epoch_num}. Skipping...", "ERROR")
                    break
                
                # Wait before retry
                wait_time = min(30 * retry_count, 300)
                logger.log(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                
                # Try to reload model from last checkpoint
                try:
                    latest = get_latest_checkpoint(RUNS_DIR)
                    if latest:
                        logger.log(f"Reloading model from: {latest}")
                        model = YOLO(latest)
                except Exception as reload_e:
                    logger.log(f"Failed to reload model: {reload_e}", "ERROR")
        
        # Save progress checkpoint after each epoch
        if epoch_success:
            latest = get_latest_checkpoint(RUNS_DIR)
            if latest:
                import shutil
                progress_ckpt = CHECKPOINT_DIR / f"epoch_{epoch_num}.pt"
                shutil.copy2(latest, progress_ckpt)
                logger.log(f"Saved epoch checkpoint: {progress_ckpt}")
    
    logger.log(f"\nTraining finished. Completed {completed_epochs}/{epochs} epochs.")
    logger.log(f"Best model: {RUNS_DIR}/detect/weights/best.pt")
    logger.log(f"Last checkpoint: {RUNS_DIR}/detect/weights/last.pt")
    logger.log(f"Logs: {logger.log_dir}")
    
    return completed_epochs


if __name__ == "__main__":
    train_with_recovery()