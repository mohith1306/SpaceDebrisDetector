"""Dataset handling for the 3-channel physics-motivated experiment.

Responsibilities
----------------
1. Locate a YOLO dataset (images/{train,val} + labels/{train,val}) inside a
   Google Drive-synced source directory.
2. Pool all image/label pairs and re-split deterministically at 80/20
   (seed 42) - the split decision recorded in configs/physics_config.yaml.
3. Preprocess every image with the 3-channel physics pipeline, processing the
   train and val partitions separately, batch-wise and multiprocess, with
   progress display, checkpointing (skip already-processed files) and a log.
4. Copy the ORIGINAL YOLO annotation files byte-for-byte into the output
   tree (never modified, never regenerated) so image<->label correspondence
   is exactly preserved.
5. Write processed_dataset/space_debris.yaml pointing at the new images.

Output layout (mirrors the source, split re-assigned at 80/20):

    processed_dataset/
    ├── images/train/   (3-channel processed, same filenames as source)
    ├── images/val/
    ├── labels/train/   (byte-identical copies of the original .txt files)
    ├── labels/val/
    ├── space_debris.yaml
    └── processing_summary.json

Nothing under the source directory is ever written to or overwritten.
"""
from __future__ import annotations

import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
from tqdm import tqdm

from .physics_pipeline import (
    DEFAULT_PIPELINE_CONFIG,
    PhysicsPipelineConfig,
    preprocess_image_3ch,
    save_processed_image,
)

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
CHUNK_SIZE = 256  # futures per submission wave -> bounded memory at 66k images


@dataclass(frozen=True)
class Pair:
    """One image plus its original YOLO annotation file."""

    image_path: Path
    label_path: Path
    source_split: str  # "train" / "val" / "all" as found in the source


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------
def _is_yolo_root(path: Path) -> bool:
    images = path / "images"
    if not images.is_dir():
        return False
    if (images / "train").is_dir() or (images / "val").is_dir():
        return True
    # flat layout: images/ holds files directly
    return any(p.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS for p in images.iterdir())


def locate_dataset_root(source: str | Path) -> Path:
    """Find the YOLO dataset root inside a synced Drive directory.

    Accepts either the yolo_dataset directory itself, a directory containing
    `yolo_dataset/`, or any parent whose children include a YOLO layout.
    """
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    candidates = [source, source / "yolo_dataset"]
    for cand in candidates:
        if cand.is_dir() and _is_yolo_root(cand):
            return cand

    if source.is_dir():
        for child in sorted(p for p in source.iterdir() if p.is_dir()):
            if _is_yolo_root(child):
                return child
            nested = child / "yolo_dataset"
            if nested.is_dir() and _is_yolo_root(nested):
                return nested

    raise FileNotFoundError(
        f"No YOLO dataset (images/{{train,val}} + labels/{{train,val}}) found "
        f"under: {source}"
    )


# ---------------------------------------------------------------------------
# Pair collection (annotation-preserving)
# ---------------------------------------------------------------------------
def collect_pairs(
    dataset_root: str | Path,
    image_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> tuple[list[Pair], dict]:
    """Enumerate image/label pairs. Original labels are only read, never changed.

    Returns (pairs, report) where report records images without labels and
    labels without images so nothing fails silently.
    """
    root = Path(dataset_root)
    exts = tuple(e.lower() for e in image_extensions)
    pairs: list[Pair] = []
    report: dict = {
        "images_without_label": [],
        "labels_without_image": [],
        "per_source_split": {},
    }

    images_dir = root / "images"
    labels_dir = root / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    split_names = [s for s in ("train", "val") if (images_dir / s).is_dir()]
    if not split_names:
        split_names = ["all"]  # flat layout: images/ holds files directly

    for split in split_names:
        img_dir = images_dir if split == "all" else images_dir / split
        lbl_dir = labels_dir if split == "all" else labels_dir / split

        split_pairs: list[Pair] = []
        image_stems: set[str] = set()
        for img in sorted(p for p in img_dir.iterdir() if p.is_file()):
            if img.suffix.lower() not in exts:
                continue
            image_stems.add(img.stem)
            label = lbl_dir / f"{img.stem}.txt"
            if label.exists():
                split_pairs.append(Pair(img, label, split))
            else:
                report["images_without_label"].append(
                    str(img.relative_to(root))
                )

        if lbl_dir.is_dir():
            for lbl in sorted(lbl_dir.glob("*.txt")):
                if lbl.stem not in image_stems:
                    report["labels_without_image"].append(
                        str(lbl.relative_to(root))
                    )

        pairs.extend(split_pairs)
        report["per_source_split"][split] = len(split_pairs)

    return pairs, report


def split_pairs(
    pairs: list[Pair],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[Pair], list[Pair]]:
    """Deterministic 80/20 re-split (same shuffle style as split_dataset.py)."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n_train = int(train_ratio * len(shuffled))
    return shuffled[:n_train], shuffled[n_train:]


# ---------------------------------------------------------------------------
# Worker (top-level so it is picklable under the spawn multiprocessing context)
# ---------------------------------------------------------------------------
def _process_file(task: tuple) -> dict:
    """Preprocess one image -> 3-channel output. Returns a status dict."""
    src, dst, force = task
    dst = Path(dst)
    if not force and dst.exists():
        return {"status": "skipped_existing"}

    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return {"status": "corrupt", "file": str(src)}

    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed = preprocess_image_3ch(rgb)
        dst.parent.mkdir(parents=True, exist_ok=True)
        save_processed_image(dst, processed)
    except Exception as exc:  # noqa: BLE001 - reported, never crashes the run
        return {"status": "error", "file": str(src), "reason": str(exc)}

    return {"status": "processed"}


def _iter_chunks(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# Partition processing
# ---------------------------------------------------------------------------
def process_partition(
    pairs: list[Pair],
    out_root: Path,
    split: str,
    force: bool = False,
    workers: int | None = None,
    on_output: Callable[[Path], None] | None = None,
    desc: str | None = None,
) -> dict:
    """Preprocess one split (train or val) separately.

    Batch-based, multiprocess (spawn context - safe on Windows and Linux),
    never loads the dataset into RAM (one image at a time per worker),
    checkpoints by skipping outputs that already exist unless force=True.
    """
    stats = {
        "total": len(pairs),
        "processed": 0,
        "skipped_existing": 0,
        "corrupt": 0,
        "error": 0,
        "errors": [],
    }
    if not pairs:
        return stats

    img_out = out_root / "images" / split
    img_out.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            str(p.image_path),
            str(img_out / p.image_path.name),
            force,
        )
        for p in pairs
    ]

    ctx = get_context("spawn")  # safe cross-platform; avoids fork+OpenCV issues
    n_workers = workers or max(1, min(8, (cv2.getNumberOfCPUs() or 2) - 1))

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = []
        bar = tqdm(
            total=len(tasks),
            desc=desc or f"preprocess [{split}]",
            unit="img",
            dynamic_ncols=True,
        )
        try:
            for chunk in _iter_chunks(tasks, CHUNK_SIZE):
                futures.extend(pool.submit(_process_file, t) for t in chunk)
                for fut in as_completed(futures):
                    res = fut.result()
                    status = res.get("status", "error")
                    if status == "processed":
                        stats["processed"] += 1
                    elif status == "skipped_existing":
                        stats["skipped_existing"] += 1
                    elif status == "corrupt":
                        stats["corrupt"] += 1
                        if len(stats["errors"]) < 100:
                            stats["errors"].append(
                                {"file": res.get("file"), "reason": "unreadable image"}
                            )
                    else:
                        stats["error"] += 1
                        if len(stats["errors"]) < 100:
                            stats["errors"].append(
                                {"file": res.get("file"), "reason": res.get("reason")}
                            )
                    bar.update(1)
                futures = []
        finally:
            bar.close()

    # Second pass for upload hook: every local output that exists is a candidate
    # (the uploader's manifest decides what actually needs pushing).
    if on_output:
        for p in pairs:
            out = img_out / p.image_path.name
            if out.exists():
                on_output(out)

    return stats


# ---------------------------------------------------------------------------
# Labels: byte-for-byte copies of the original annotations
# ---------------------------------------------------------------------------
def copy_labels(
    pairs: list[Pair],
    out_root: Path,
    split: str,
) -> dict:
    """Copy original YOLO .txt files for the successfully kept images.

    Files are copied verbatim (shutil.copy2 preserves content and metadata):
    bounding boxes, class ids and filenames are never modified or regenerated.
    """
    lbl_out = out_root / "labels" / split
    lbl_out.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    for p in pairs:
        dst = lbl_out / p.label_path.name
        # Only ship labels whose image actually made it into the output.
        if not (out_root / "images" / split / p.image_path.name).exists():
            continue
        try:
            shutil.copy2(str(p.label_path), str(dst))
            copied += 1
        except OSError:
            missing += 1
    return {"copied": copied, "failed": missing}


# ---------------------------------------------------------------------------
# Dataset YAML
# ---------------------------------------------------------------------------
def write_dataset_yaml(out_root: Path, class_names: Sequence[str]) -> Path:
    """Write processed_dataset/space_debris.yaml (same 11 classes, same order)."""
    names_yaml = "\n".join(f"  - {name}" for name in class_names)
    text = (
        f"path: {out_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        f"{names_yaml}\n"
    )
    yaml_path = out_root / "space_debris.yaml"
    yaml_path.write_text(text, encoding="utf-8")
    return yaml_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_preprocessing(
    source: str | Path,
    out_root: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    force: bool = False,
    limit: int | None = None,
    workers: int | None = None,
    image_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
    class_names: Sequence[str] | None = None,
    pipeline_cfg: PhysicsPipelineConfig = DEFAULT_PIPELINE_CONFIG,
    on_output: Callable[[Path], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Full preprocessing run: locate -> collect -> split -> process -> labels -> yaml.

    Returns the summary dict (also written to <out_root>/processing_summary.json).
    """
    started = time.time()

    if class_names is None:
        # Reuse the baseline class taxonomy (identical ids/order).
        from space_debris.config import CLASS_NAMES as class_names  # noqa: PLC0415

    source = Path(source).expanduser().resolve()
    out_root = Path(out_root).expanduser().resolve()

    if out_root.exists() and out_root.resolve() == source.resolve():
        raise ValueError("Output directory must differ from the source directory")

    log(f"[physics] locating YOLO dataset under: {source}")
    dataset_root = locate_dataset_root(source)
    log(f"[physics] dataset root: {dataset_root}")

    pairs, collect_report = collect_pairs(dataset_root, image_extensions)
    log(
        f"[physics] collected {len(pairs)} image/label pairs "
        f"(per source split: {collect_report['per_source_split']})"
    )
    if collect_report["images_without_label"]:
        log(
            f"[physics] WARNING: {len(collect_report['images_without_label'])} "
            "image(s) without an annotation file were skipped "
            f"(first: {collect_report['images_without_label'][:3]})"
        )
    if collect_report["labels_without_image"]:
        log(
            f"[physics] WARNING: {len(collect_report['labels_without_image'])} "
            "orphan label file(s) found (no matching image)"
        )
    if not pairs:
        raise RuntimeError("No usable image/label pairs found in the source dataset")

    # Deterministic shuffle + optional cap (keeps both splits populated)
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    if limit is not None and limit < len(shuffled):
        log(f"[physics] --limit {limit}: capping pooled pairs (of {len(shuffled)})")
        shuffled = shuffled[: max(limit, 1)]
    n_train = int(train_ratio * len(shuffled))
    train_pairs, val_pairs = shuffled[:n_train], shuffled[n_train:]
    log(
        f"[physics] re-split (seed={seed}): "
        f"train={len(train_pairs)} val={len(val_pairs)} "
        f"({train_ratio:.0%}/{1 - train_ratio:.0%})"
    )

    out_root.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "experiment": "physics_3channel_preprocessing",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(source),
        "dataset_root": str(dataset_root),
        "output": str(out_root),
        "pipeline": pipeline_cfg.describe(),
        "split": {
            "train_ratio": train_ratio,
            "seed": seed,
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
        },
        "source_report": {
            "images_without_label": len(collect_report["images_without_label"]),
            "labels_without_image": len(collect_report["labels_without_image"]),
            "images_without_label_examples": collect_report[
                "images_without_label"
            ][:50],
        },
        "splits": {},
        "labels": {},
        "errors": [],
    }

    # Process train and val separately (separate progress + stats).
    for split, split_pairs_ in (("train", train_pairs), ("val", val_pairs)):
        log(f"[physics] processing split '{split}' ({len(split_pairs_)} images)...")
        stats = process_partition(
            split_pairs_,
            out_root,
            split,
            force=force,
            workers=workers,
            on_output=on_output,
            desc=f"preprocess [{split}]",
        )
        label_stats = copy_labels(split_pairs_, out_root, split)
        summary["splits"][split] = stats
        summary["labels"][split] = label_stats
        summary["errors"].extend(stats.pop("errors", []))
        log(
            f"[physics] '{split}': processed={stats['processed']} "
            f"cached={stats['skipped_existing']} corrupt={stats['corrupt']} "
            f"failed={stats['error']} labels_copied={label_stats['copied']}"
        )

    yaml_path = write_dataset_yaml(out_root, class_names)
    summary["dataset_yaml"] = str(yaml_path)

    total_processed = sum(s["processed"] for s in summary["splits"].values())
    total_cached = sum(s["skipped_existing"] for s in summary["splits"].values())
    total_bad = sum(
        s["corrupt"] + s["error"] for s in summary["splits"].values()
    )
    summary["totals"] = {
        "processed": total_processed,
        "skipped_existing": total_cached,
        "failed_or_corrupt": total_bad,
        "labels_copied": sum(l["copied"] for l in summary["labels"].values()),
        "elapsed_sec": round(time.time() - started, 1),
    }

    summary_path = out_root / "processing_summary.json"
    import json

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)

    log(
        f"[physics] DONE: processed={total_processed} cached={total_cached} "
        f"failed={total_bad} elapsed={summary['totals']['elapsed_sec']}s "
        f"-> {summary_path}"
    )
    return summary
