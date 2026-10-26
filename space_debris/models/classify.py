"""Second-stage classification on top of detector crops.

The detector answers "where is an object". This stage answers "what is it,
really?" by re-classifying the cropped box with a dedicated classification
model -- an independent opinion that is fused with the detector's class.

Typical flow:
    python -m space_debris.main build_crops            # make crop dataset from train.csv
    python -m space_debris.main train_classifier       # train yolo11n-cls on the crops
    python -m space_debris.main stream --classifier ...  # use it in the pipeline

Fusion rule (see ``fuse``): the classifier wins only when it disagrees with the
detector and is confident; otherwise the detector label stands. ``is_debris``
comes from config.DEBRIS_CLASSES so callers can answer the debris question
without caring about the 11-way taxonomy.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from ..config import (
    CHECKPOINT_DIR,
    CLASS_NAMES,
    DEBRIS_CLASSES,
    DEVICE,
    LABEL_CSV,
    PROCESSED_DIR,
    class_id,
)

CROP_ROOT = CHECKPOINT_DIR / "crops"
DEFAULT_CLS_WEIGHTS = CHECKPOINT_DIR / "classifier" / "weights" / "best.pt"


def _find_image(stem: str) -> Path | None:
    for ext in (".jpg", ".png", ".jpeg"):
        p = PROCESSED_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def build_crop_dataset(val_ratio: float = 0.2, pad_frac: float = 0.10,
                       min_size: int = 32, out_dir: Path | None = None,
                       seed: int = 42) -> dict:
    """Cut one crop per CSV row into ``<out>/<split>/<class>/<stem>.jpg``."""
    import ast
    import csv

    import cv2

    out_dir = Path(out_dir) if out_dir else CROP_ROOT
    for split in ("train", "val"):
        for name in CLASS_NAMES:
            (out_dir / split / name).mkdir(parents=True, exist_ok=True)

    rows = []
    with open(LABEL_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    stems = [Path(r["filename"]).stem for r in rows]
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    n_val = int(val_ratio * len(order))
    val_set = set(order[:n_val])

    written = {"train": 0, "val": 0}
    skipped = 0
    for i, row in enumerate(rows):
        stem = stems[i]
        img_path = _find_image(stem)
        if img_path is None:
            skipped += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        x_min, y_min, x_max, y_max = (float(v) for v in ast.literal_eval(row["bbox"]))
        pad = pad_frac * max(x_max - x_min, y_max - y_min)
        x1 = max(0, int(x_min - pad))
        y1 = max(0, int(y_min - pad))
        x2 = min(w, int(x_max + pad))
        y2 = min(h, int(y_max + pad))
        if x2 - x1 < min_size or y2 - y1 < min_size:
            skipped += 1
            continue

        split = "val" if i in val_set else "train"
        dst = out_dir / split / row["class"] / f"{stem}.jpg"
        if not cv2.imwrite(str(dst), img[y1:y2, x1:x2]):
            skipped += 1
            continue
        written[split] += 1

    return {
        "out_dir": str(out_dir),
        "train": written["train"],
        "val": written["val"],
        "skipped": skipped,
        "classes": len(CLASS_NAMES),
    }


def train_classifier(epochs: int = 10, imgsz: int = 224, device: str | None = None,
                     model: str = "yolo11n-cls.pt", data: str | Path | None = None,
                     batch: int = 64, workers: int = 4) -> str:
    """Train the second-stage classifier on the crop dataset."""
    from ultralytics import YOLO

    data = Path(data) if data else CROP_ROOT
    if not (data / "train").exists():
        raise FileNotFoundError(
            f"crop dataset not found at {data}; run `main.py build_crops` first"
        )
    model_ = YOLO(model)
    results = model_.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device or DEVICE,
        project=str(CHECKPOINT_DIR),
        name="classifier",
        exist_ok=True,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    return str(best if best.exists() else Path(results.save_dir) / "weights" / "last.pt")


def load_classifier(path: str | Path | None = None):
    from ultralytics import YOLO

    path = Path(path) if path else DEFAULT_CLS_WEIGHTS
    if not path.exists():
        raise FileNotFoundError(
            f"classifier weights not found: {path} (run `main.py train_classifier`)"
        )
    return YOLO(str(path))


def classify_crops(classifier, image_bgr: np.ndarray, boxes_xyxy: np.ndarray,
                   imgsz: int = 224) -> list[dict]:
    """Classify each crop; returns one dict per box (order preserved)."""
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return []
    h, w = image_bgr.shape[:2]
    crops = []
    keep = []
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes_xyxy, dtype=float)):
        xi1, yi1 = max(0, int(x1)), max(0, int(y1))
        xi2, yi2 = min(w, int(x2)), min(h, int(y2))
        crop = image_bgr[yi1:yi2, xi1:xi2]
        if crop.size == 0:
            continue
        crops.append(crop)
        keep.append(i)
    if not crops:
        return []

    preds = classifier.predict(crops, imgsz=imgsz, verbose=False)
    out = [None] * len(boxes_xyxy)
    for slot, src_i in enumerate(keep):
        pred = preds[slot]
        probs = pred.probs
        if probs is None:
            continue
        names = pred.names or {}
        raw = probs.data
        prob_values = (raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw))
        top = int(probs.top1)
        order = np.argsort(prob_values)[::-1][:2]
        out[src_i] = {
            "cls_name": names.get(top, str(top)),
            "cls_conf": round(float(prob_values[top]), 4),
            "top2": [
                {"class": names.get(int(j), str(int(j))), "p": round(float(prob_values[int(j)]), 4)}
                for j in order
            ],
        }
    return [o if o is not None else {"cls_name": None, "cls_conf": 0.0, "top2": []}
            for o in out]


def fuse(det_class_name: str | None, cls_pred: dict | None,
         min_cls_conf: float = 0.45,
         agreement_boost: float = 0.10) -> dict:
    """Combine detector class and second-stage classifier into one decision."""
    det = det_class_name
    cls_name = (cls_pred or {}).get("cls_name")
    cls_conf = float((cls_pred or {}).get("cls_conf") or 0.0)

    if cls_name is None:
        final = det
        source = "detector_only"
    elif det is None:
        final = cls_name
        source = "classifier_only"
    elif det == cls_name:
        final = det
        source = "agree"
    elif cls_conf >= min_cls_conf:
        final = cls_name
        source = "classifier_overrode"
    else:
        final = det
        source = "detector_kept"

    is_debris = final in DEBRIS_CLASSES if final else None
    agreement = det is not None and cls_name is not None and det == cls_name
    conf = cls_conf if agreement else max(cls_conf - agreement_boost, 0.0)
    if source in ("detector_only", "detector_kept"):
        conf = max(conf, 0.0)
    return {
        "final_class": final,
        "final_class_id": class_id(final) if final in CLASS_NAMES else None,
        "detector_class": det,
        "classifier_class": cls_name,
        "agreement": bool(agreement),
        "source": source,
        "is_debris": is_debris,
        "classifier_conf": round(cls_conf, 4),
        "confidence": round(conf, 4),
    }
