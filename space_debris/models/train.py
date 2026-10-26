import os
from pathlib import Path
import torch
from ultralytics import YOLO

from ..config import CHECKPOINT_DIR, YOLO_BASE, RUNS_DIR, DEVICE, NUM_CLASSES, get_yaml_config

VALID_MODEL_SIZES = ["n", "s", "m", "l", "x"]
VALID_ARCHITECTURES = ["standard", "p1p2", "yolo11"]

def get_latest_checkpoint():
    weights_dir = RUNS_DIR / "detect" / "weights"
    if not weights_dir.exists():
        return None
    checkpoints = list(weights_dir.glob("*.pt"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)

def train(epochs=30, imgsz=640, batch=4, device=None, resume=False, workers=0, model_size="n", arch="standard"):
    if model_size not in VALID_MODEL_SIZES:
        raise ValueError(f"Invalid model size '{model_size}'. Choose from: {VALID_MODEL_SIZES}")
    if arch not in VALID_ARCHITECTURES:
        raise ValueError(f"Invalid architecture '{arch}'. Choose from: {VALID_ARCHITECTURES}")
    
    if device is None:
        device = DEVICE
    if str(device).lower() in ("cuda", "0") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this Python environment cannot access CUDA. "
            "Install a CUDA-enabled PyTorch build, then rerun with --device 0."
        )
    print(f"Training on device: {device}")
    yaml_path = YOLO_BASE / "space_debris.yaml"
    yaml_path.write_text(get_yaml_config())
    print(f"Dataset config saved to: {yaml_path} (nc={NUM_CLASSES})")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if resume:
        checkpoint = get_latest_checkpoint()
        if checkpoint:
            print(f"Resuming from: {checkpoint}")
            model = YOLO(checkpoint)
        else:
            if arch == "yolo11":
                base_model = f"yolo11{model_size}.pt"
                print(f"No checkpoint found, starting fresh with {base_model}")
                model = YOLO(base_model)
            elif arch == "p1p2":
                model_yaml = Path(__file__).parent / "yolov8_p1p2.yaml"
                if model_yaml.exists():
                    print(f"No checkpoint found, starting fresh with P1/P2 architecture")
                    model = YOLO(str(model_yaml))
                else:
                    base_model = f"yolov8{model_size}.pt"
                    print(f"No checkpoint found, P1/P2 yaml missing, starting with {base_model}")
                    model = YOLO(base_model)
            else:
                base_model = f"yolov8{model_size}.pt"
                print(f"No checkpoint found, starting fresh with {base_model}")
                model = YOLO(base_model)
    else:
        if arch == "yolo11":
            base_model = f"yolo11{model_size}.pt"
            print(f"Starting training with {base_model}")
            model = YOLO(base_model)
        elif arch == "p1p2":
            model_yaml = Path(__file__).parent / "yolov8_p1p2.yaml"
            if model_yaml.exists():
                print(f"Starting training with P1/P2 architecture")
                model = YOLO(str(model_yaml))
            else:
                base_model = f"yolov8{model_size}.pt"
                print(f"P1/P2 yaml not found, falling back to {base_model}")
                model = YOLO(base_model)
        else:
            base_model = f"yolov8{model_size}.pt"
            print(f"Starting training with {base_model}")
            model = YOLO(base_model)

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        resume=resume,
        save=True,
        save_period=1,
        project=str(RUNS_DIR),
        name="detect",
        patience=5,
    )

    print(f"\nTraining completed!")
    print(f"Best model: {RUNS_DIR}/detect/weights/best.pt")
    print(f"Last checkpoint: {RUNS_DIR}/detect/weights/last.pt")

    return results

if __name__ == "__main__":
    train()