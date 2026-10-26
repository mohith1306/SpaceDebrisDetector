import os
import sys
import time
import traceback
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import torch

# oneDNN (mkldnn) SIGFPEs on virtualized CPUs (JarvisLabs VMs); CUDA training unaffected.
torch.backends.mkldnn.enabled = False

from ..config import CHECKPOINT_DIR, YOLO_BASE, RUNS_DIR, DEVICE, NUM_CLASSES, get_yaml_config

VALID_MODEL_SIZES = ["n", "s", "m", "l", "x"]
VALID_ARCHITECTURES = ["standard", "p1p2", "yolo11"]

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
    if arch == "yolo11":
        weights = f"yolo11{model_size}.pt"
        print(f"Loading COCO-pretrained {weights}")
        return YOLO(weights)
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
                name = "Apple Silicon GPU (MPS)"
                return "mps", name
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


def extract_metrics(results):
    """Best-effort metric dict from an ultralytics results object."""
    metrics = {}
    if results is None:
        return metrics
    for attr in ("results_dict", "metrics", "loss", "results"):
        val = getattr(results, attr, None)
        if isinstance(val, dict) and val:
            return val
        if isinstance(val, (int, float)):
            metrics[attr] = val
    return metrics


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
    """One continuous ultralytics run with weight-carry crash recovery.

    The old design called model.train(epochs=N, resume=True) once per outer
    epoch. ultralytics strips optimizer state from best.pt/last.pt in
    final_eval(), so resume always failed, each call trained N fresh epochs
    from scratch counter (1+2+3+... = quadratic blowup), and the LR schedule
    restarted every call. This version makes a single train() call for all
    remaining epochs; on crash it reloads the newest checkpoint (weights)
    and continues with the remaining count.
    """
    device, device_info = detect_device(device)

    logger = TrainingLogger()
    logger.log(f"Device detected: {device_info}")
    logger.log(f"Starting training on device: {device}")
    logger.log(f"Epochs: {epochs}, Image size: {imgsz}, Batch: {batch}, Resume weights: {resume}")
    logger.log(f"Logs saved to: {logger.log_dir}")

    yaml_path = YOLO_BASE / "space_debris.yaml"
    yaml_path.write_text(get_yaml_config())
    logger.log(f"Dataset config: {yaml_path} (nc={NUM_CLASSES})")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = get_latest_checkpoint(RUNS_DIR) if resume else None
    if checkpoint:
        logger.log(f"Loading weights from: {checkpoint}")
        model = YOLO(checkpoint)
    else:
        logger.log(f"Starting fresh (arch: {arch}, size: {model_size})")
        model = create_custom_model(model_size, arch)

    state = {"done": 0, "base": 0, "last_t": time.time()}

    def on_fit_epoch_end(*args, **kwargs):
        trainer = None
        for a in args:
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
            logger.log(f"Epoch {state['done']}/{epochs} completed ({shown})")
        except Exception:
            pass

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    results = None
    attempts = 0
    while True:
        remaining = epochs - state["done"]
        if remaining <= 0:
            break
        state["base"] = state["done"]
        state["last_t"] = time.time()
        try:
            logger.log(f"Training {remaining} epoch(s) ({state['done']}/{epochs} done)")
            results = model.train(
                data=str(yaml_path),
                epochs=remaining,
                imgsz=imgsz,
                batch=batch,
                workers=workers,
                device=device,
                resume=False,
                save=True,
                save_period=1,
                val=True,
                project=str(RUNS_DIR),
                name="detect",
                patience=epochs,
                verbose=True,
                exist_ok=True,
            )
            state["done"] = epochs
            break
        except Exception as e:
            attempts += 1
            tb_str = traceback.format_exc()
            logger.log_error(state["done"], e, tb_str)
            logger.log(
                f"Training interrupted after {state['done']} epochs (attempt {attempts}): {e}",
                "ERROR",
            )
            if attempts > max_retries:
                logger.log("Max retries reached. Giving up.", "ERROR")
                break
            wait_time = min(30 * attempts, 300)
            logger.log(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            latest = get_latest_checkpoint(RUNS_DIR)
            if latest:
                logger.log(f"Reloading model from: {latest}")
                model = YOLO(latest)
                model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    if results is not None:
        final = extract_metrics(results)
        if final:
            shown = ", ".join(
                f"{k}={v:.4f}" for k, v in final.items() if isinstance(v, float)
            )
            logger.log(f"Final metrics: {shown}")

    logger.log(f"\nTraining finished. {state['done']}/{epochs} epochs completed.")
    logger.log(f"Best model: {RUNS_DIR}/detect/weights/best.pt")
    logger.log(f"Last checkpoint: {RUNS_DIR}/detect/weights/last.pt")
    logger.log(f"Logs: {logger.log_dir}")

    return state["done"]


if __name__ == "__main__":
    train_with_recovery()
