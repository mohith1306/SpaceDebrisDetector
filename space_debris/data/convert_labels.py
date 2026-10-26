import ast
import json
import os
import pandas as pd
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from ..config import (
    PROCESSED_DIR,
    LABEL_CSV,
    YOLO_LABEL_DIR,
    CLASS_NAMES,
    LEGACY_COLLAPSE_TO_SINGLE_CLASS,
    class_id,
)

SIZE_CACHE = YOLO_LABEL_DIR / ".image_sizes.json"


def spark_bbox_to_yolo(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height


def _image_size(path: Path, cache: dict) -> tuple:
    """Read (w, h), memoising across runs so re-conversion is fast."""
    key = path.name
    if key in cache:
        return tuple(cache[key])
    with Image.open(path) as im:
        size = im.size
    cache[key] = list(size)
    return size


def convert_labels(collapse: bool | None = None):
    """Write YOLO label files from labels/train.csv.

    collapse=True reproduces the original single-class behaviour (every box -> id 0)
    so the old nc:1 checkpoints stay reproducible; collapse=None uses
    config.LEGACY_COLLAPSE_TO_SINGLE_CLASS (False = multi-class ids).
    """
    if collapse is None:
        collapse = LEGACY_COLLAPSE_TO_SINGLE_CLASS
    df = pd.read_csv(LABEL_CSV)
    print(f"Total label rows: {len(df)}")
    print(f"Mode: {'legacy single-class (nc:1)' if collapse else 'multi-class (nc:%d)' % len(CLASS_NAMES)}")

    unknown = sorted(set(df["class"].dropna().unique()) - set(CLASS_NAMES))
    if unknown and not collapse:
        raise ValueError(f"CSV contains classes missing from config.CLASS_NAMES: {unknown}")

    processed_images = [
        f for f in PROCESSED_DIR.iterdir()
        if f.suffix.lower() in ('.jpg', '.png')
    ]
    processed_basenames = set(f.stem for f in processed_images)
    print(f"Processed images found: {len(processed_basenames)}")

    cache = json.loads(SIZE_CACHE.read_text()) if SIZE_CACHE.exists() else {}
    distribution: Counter = Counter()
    converted = 0
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting labels"):
        csv_name = row["filename"]
        base_name = Path(csv_name).stem

        if base_name not in processed_basenames:
            skipped += 1
            continue

        img_path = None
        for ext in ['.jpg', '.png']:
            p = PROCESSED_DIR / f"{base_name}{ext}"
            if p.exists():
                img_path = p
                break

        if img_path is None:
            skipped += 1
            continue

        try:
            w, h = _image_size(img_path, cache)
        except Exception:
            skipped += 1
            continue
        # ast.literal_eval: bbox is a literal list like "[183, 311, 657, 415]"
        bbox = ast.literal_eval(row["bbox"])
        x_min, y_min, x_max, y_max = bbox
        # Clamp to image bounds so YOLO never sees OOB boxes
        x_min = max(0.0, min(float(x_min), w - 1))
        y_min = max(0.0, min(float(y_min), h - 1))
        x_max = max(0.0, min(float(x_max), w - 1))
        y_max = max(0.0, min(float(y_max), h - 1))
        if x_max <= x_min or y_max <= y_min:
            skipped += 1
            continue
        x_c, y_c, bw, bh = spark_bbox_to_yolo((x_min, y_min, x_max, y_max), w, h)

        cls = 0 if collapse else class_id(row["class"])
        distribution[CLASS_NAMES[cls] if not collapse else "space_object"] += 1

        label_path = YOLO_LABEL_DIR / f"{base_name}.txt"
        with open(label_path, "w") as f:
            f.write(f"{cls} {x_c} {y_c} {bw} {bh}\n")

        converted += 1

    SIZE_CACHE.write_text(json.dumps(cache))
    print(f"Labels converted: {converted}")
    print(f"Labels skipped: {skipped}")
    if collapse:
        print("Class distribution: {'space_object': %d}" % converted)
    else:
        print(f"Class distribution ({len(CLASS_NAMES)} classes):")
        for name in CLASS_NAMES:
            print(f"  {name:26s} {distribution[name]:,}")
    return converted

if __name__ == "__main__":
    convert_labels()