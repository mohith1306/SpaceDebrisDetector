import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from ..config import PROCESSED_DIR, LABEL_CSV, YOLO_LABEL_DIR

def spark_bbox_to_yolo(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height

def convert_labels():
    df = pd.read_csv(LABEL_CSV)
    print(f"Total label rows: {len(df)}")

    processed_images = [
        f for f in PROCESSED_DIR.iterdir()
        if f.suffix.lower() in ('.jpg', '.png')
    ]
    processed_basenames = set(f.stem for f in processed_images)
    print(f"Processed images found: {len(processed_basenames)}")

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
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            skipped += 1
            continue
        # ast.literal_eval: bbox is a literal list like "[183, 311, 657, 415]"
        import ast
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

        label_path = YOLO_LABEL_DIR / f"{base_name}.txt"
        with open(label_path, "w") as f:
            f.write(f"0 {x_c} {y_c} {bw} {bh}\n")

        converted += 1

    print(f"Labels converted: {converted}")
    print(f"Labels skipped: {skipped}")

if __name__ == "__main__":
    convert_labels()