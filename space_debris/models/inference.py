import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

from ..config import YOLO_BASE, RUNS_DIR, DEVICE

def predict(model_path=None, source=None, conf=0.25, save=True, show=False, device=None):
    if device is None:
        device = DEVICE
    
    if model_path is None:
        model_path = RUNS_DIR / "detect" / "weights" / "best.pt"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    if source is None:
        source = YOLO_BASE / "images" / "val"

    print(f"Running inference on: {source} (device: {device})")
    results = model.predict(
        source=str(source),
        conf=conf,
        save=save,
        show=show,
        device=device
    )
    return results

def visualize_predictions(pred_dir=None, num_samples=5):
    if pred_dir is None:
        pred_dirs = sorted((RUNS_DIR / "detect").glob("predict*"))
        if not pred_dirs:
            print("No prediction directory found")
            return
        pred_dir = pred_dirs[-1]

    pred_images = list(Path(pred_dir).glob("*.jpg")) + list(Path(pred_dir).glob("*.png"))
    
    if not pred_images:
        print(f"No predicted images found in {pred_dir}")
        return

    plt.figure(figsize=(15, 3 * num_samples))
    for i, img_path in enumerate(pred_images[:num_samples]):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(img_rgb)
        plt.title(f"Prediction: {img_path.name}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict()
    visualize_predictions()