import os
import shutil
import random
from pathlib import Path

from ..config import PROCESSED_DIR, YOLO_LABEL_DIR, YOLO_BASE

def split_dataset(train_ratio=0.8):
    """Split processed images and YOLO labels into train/val (80/20)."""
    IMG_TRAIN = YOLO_BASE / "images/train"
    LBL_TRAIN = YOLO_BASE / "labels/train"
    IMG_VAL = YOLO_BASE / "images/val"
    LBL_VAL = YOLO_BASE / "labels/val"

    for directory in (IMG_TRAIN, LBL_TRAIN, IMG_VAL, LBL_VAL):
        for path in directory.glob("*"):
            if path.is_file():
                path.unlink()

    IMG_TRAIN.mkdir(parents=True, exist_ok=True)
    LBL_TRAIN.mkdir(parents=True, exist_ok=True)
    IMG_VAL.mkdir(parents=True, exist_ok=True)
    LBL_VAL.mkdir(parents=True, exist_ok=True)

    # Get all processed images that have corresponding YOLO labels
    processed_images = list(PROCESSED_DIR.glob("*.jpg")) + list(PROCESSED_DIR.glob("*.png"))
    labeled_images = []

    for img_path in processed_images:
        label_path = YOLO_LABEL_DIR / f"{img_path.stem}.txt"
        if label_path.exists():
            labeled_images.append(img_path)

    print(f"Total labeled images: {len(labeled_images)}")
    random.Random(42).shuffle(labeled_images)

    train_count = int(train_ratio * len(labeled_images))
    train_images = labeled_images[:train_count]
    val_images = labeled_images[train_count:]

    # Copy train images and labels
    for img_path in train_images:
        base = img_path.stem
        shutil.copy2(str(img_path), str(IMG_TRAIN / img_path.name))
        shutil.copy2(str(YOLO_LABEL_DIR / f"{base}.txt"), str(LBL_TRAIN / f"{base}.txt"))

    # Copy val images and labels
    for img_path in val_images:
        base = img_path.stem
        shutil.copy2(str(img_path), str(IMG_VAL / img_path.name))
        shutil.copy2(str(YOLO_LABEL_DIR / f"{base}.txt"), str(LBL_VAL / f"{base}.txt"))

    print(f"Train: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")

if __name__ == "__main__":
    split_dataset()