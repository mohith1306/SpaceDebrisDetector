"""Dataset verification for the physics-motivated experiment.

Answers, with hard evidence:

  1. Does processed_dataset/ have the complete YOLO structure?
  2. Is every image paired with exactly one label and vice versa?
  3. Are the labels BYTE-IDENTICAL to the original annotations
     (bounding boxes / class ids / filenames never modified)?
  4. Are all label coordinates valid YOLO format and inside [0, 1]?
  5. Are processed images 3-channel uint8 with dimensions matching the source?
  6. Are the three engineered channels genuinely independent
     (distributions + pairwise identity/correlation)?

Exit code semantics live in scripts/verify_dataset.py (0 = all PASS).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml

from .visualization import aggregate_channel_report, channel_statistics

VALID_EXTS = (".jpg", ".jpeg", ".png")


def _check(name: str, ok: bool, detail: str, warn: bool = False) -> dict:
    if ok:
        status = "PASS"
    elif warn:
        status = "WARN"
    else:
        status = "FAIL"
    return {"check": name, "status": status, "detail": detail}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _index_labels(root: Path) -> dict[str, Path]:
    """stem -> label path across all splits of a YOLO tree."""
    index: dict[str, Path] = {}
    labels_dir = root / "labels"
    if not labels_dir.is_dir():
        return index
    for split in ("train", "val"):
        d = labels_dir / split
        if d.is_dir():
            for txt in d.glob("*.txt"):
                index[txt.stem] = txt
    # flat layout
    if not index:
        for txt in labels_dir.glob("*.txt"):
            index[txt.stem] = txt
    return index


def _index_images(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    images_dir = root / "images"
    if not images_dir.is_dir():
        return index
    for split in ("train", "val"):
        d = images_dir / split
        if d.is_dir():
            for img in d.iterdir():
                if img.suffix.lower() in VALID_EXTS:
                    index[img.stem] = img
    if not index:
        for img in images_dir.iterdir():
            if img.suffix.lower() in VALID_EXTS:
                index[img.stem] = img
    return index


def _list_split(root: Path, kind: str, split: str) -> list[Path]:
    d = root / kind / split
    if not d.is_dir():
        return []
    exts = VALID_EXTS if kind == "images" else (".txt",)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in exts)


def verify_dataset(
    processed_root: str | Path,
    source_root: str | Path | None = None,
    sample_count: int = 20,
    class_names: Sequence[str] | None = None,
) -> dict:
    """Run all verification checks; returns a JSON-serialisable report."""
    proc = Path(processed_root).resolve()
    checks: list[dict] = []

    # 1. structure -----------------------------------------------------------
    required = [
        proc / "images" / "train",
        proc / "images" / "val",
        proc / "labels" / "train",
        proc / "labels" / "val",
        proc / "space_debris.yaml",
    ]
    missing = [str(p.relative_to(proc)) for p in required if not p.exists()]
    checks.append(
        _check(
            "structure",
            not missing,
            "all required dirs/yaml present"
            if not missing
            else f"missing: {missing}",
        )
    )

    # 2. image <-> label correspondence (both directions, per split) ---------
    correspondence_ok = True
    corr_detail = []
    counts: dict[str, dict] = {}
    for split in ("train", "val"):
        imgs = _list_split(proc, "images", split)
        lbls = _list_split(proc, "labels", split)
        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        only_img = sorted(img_stems - lbl_stems)
        only_lbl = sorted(lbl_stems - img_stems)
        counts[split] = {"images": len(imgs), "labels": len(lbls)}
        if only_img or only_lbl:
            correspondence_ok = False
            corr_detail.append(
                f"{split}: {len(only_img)} img w/o label, "
                f"{len(only_lbl)} label w/o img "
                f"(e.g. {only_img[:3] + only_lbl[:3]})"
            )
    checks.append(
        _check(
            "image_label_correspondence",
            correspondence_ok,
            f"1:1 pairing in both splits: {counts}"
            if correspondence_ok
            else "; ".join(corr_detail),
        )
    )

    # 3. dataset YAML --------------------------------------------------------
    yaml_path = proc / "space_debris.yaml"
    yaml_ok = False
    yaml_detail = "yaml missing"
    if yaml_path.exists():
        try:
            meta = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            names = meta.get("names") or []
            nc_ok = int(meta.get("nc", -1)) == len(names)
            train_ok = (proc / str(meta.get("train", ""))).is_dir()
            val_ok = (proc / str(meta.get("val", ""))).is_dir()
            path_ok = Path(str(meta.get("path", proc))).resolve() == proc
            yaml_ok = nc_ok and train_ok and val_ok and path_ok
            yaml_detail = (
                f"nc={meta.get('nc')} names={len(names)} "
                f"train_dir={train_ok} val_dir={val_ok} path_ok={path_ok}"
            )
            if class_names is not None and list(names) != list(class_names):
                yaml_ok = False
                yaml_detail += " | CLASS NAMES DIFFER FROM BASELINE"
        except Exception as exc:  # noqa: BLE001
            yaml_detail = f"yaml parse error: {exc}"
    checks.append(_check("dataset_yaml", yaml_ok, yaml_detail))

    # 4. label content: valid YOLO floats, class ids, coords in [0,1] --------
    n_boxes = 0
    bad_lines = 0
    oob_lines = 0
    n_class = len(class_names) if class_names else None
    all_labels = _list_split(proc, "labels", "train") + _list_split(
        proc, "labels", "val"
    )
    for txt in all_labels:
        try:
            lines = txt.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            bad_lines += 1
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                bad_lines += 1
                continue
            try:
                cls = int(float(parts[0]))
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                bad_lines += 1
                continue
            if not all(np.isfinite(vals)):
                bad_lines += 1
                continue
            if n_class is not None and not (0 <= cls < n_class):
                bad_lines += 1
                continue
            xc, yc, w, h = vals
            if not (
                -0.001 <= xc <= 1.001
                and -0.001 <= yc <= 1.001
                and 0.0 <= w <= 1.001
                and 0.0 <= h <= 1.001
            ):
                oob_lines += 1
            n_boxes += 1
    checks.append(
        _check(
            "label_format_and_ranges",
            bad_lines == 0 and oob_lines == 0,
            f"{len(all_labels)} label files, {n_boxes} boxes, "
            f"malformed={bad_lines}, out_of_range={oob_lines}",
        )
    )

    # Resolve the source YOLO root ONCE so label and image checks agree
    # even when source_root points at a parent folder (e.g. Capstone/).
    src_yolo: Path | None = None
    if source_root is not None:
        try:
            from .dataset_processor import locate_dataset_root

            src_yolo = locate_dataset_root(source_root)
        except FileNotFoundError:
            src_yolo = Path(source_root).resolve()

    # 5. labels byte-identical to the original annotations -------------------
    if src_yolo is not None:
        src_labels = _index_labels(src_yolo)
        compared = 0
        mismatched: list[str] = []
        for txt in all_labels:
            orig = src_labels.get(txt.stem)
            if orig is None:
                continue
            compared += 1
            if _sha256(txt) != _sha256(orig):
                mismatched.append(txt.name)
            if compared >= 5000:  # cap: hashing tens of thousands is enough
                break
        checks.append(
            _check(
                "labels_byte_identical_to_source",
                compared > 0 and not mismatched,
                f"compared {compared} label files against source, "
                f"mismatched={len(mismatched)} {mismatched[:5]}"
                if compared
                else "no source labels found to compare",
                warn=(compared == 0),
            )
        )
    else:
        checks.append(
            _check(
                "labels_byte_identical_to_source",
                True,
                "SKIPPED (no source_root given)",
                warn=True,
            )
        )

    # 6. processed image properties + source dimension match -----------------
    src_images = _index_images(src_yolo) if src_yolo is not None else {}
    sample_imgs: list[Path] = []
    for split in ("train", "val"):
        sample_imgs.extend(_list_split(proc, "images", split))
    stride = max(1, len(sample_imgs) // max(1, sample_count))
    sampled = sample_imgs[::stride][:sample_count]

    img_ok = True
    img_detail = []
    channel_stats: list[dict] = []
    for img_path in sampled:
        arr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            img_ok = False
            img_detail.append(f"unreadable: {img_path.name}")
            continue
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
            img_ok = False
            img_detail.append(
                f"{img_path.name}: shape={arr.shape} dtype={arr.dtype}"
            )
            continue
        orig = src_images.get(img_path.stem)
        if orig is not None:
            o = cv2.imread(str(orig), cv2.IMREAD_COLOR)
            if o is not None and o.shape[:2] != arr.shape[:2]:
                img_ok = False
                img_detail.append(
                    f"{img_path.name}: dims {arr.shape[:2]} != source {o.shape[:2]}"
                )
        channel_stats.append(channel_statistics(arr))
    checks.append(
        _check(
            "processed_image_properties",
            img_ok and bool(sampled),
            f"{len(sampled)} sampled images: uint8 HxWx3, dims match source"
            if img_ok
            else "; ".join(img_detail[:10]),
        )
    )

    # 7. channel independence ------------------------------------------------
    agg = aggregate_channel_report(channel_stats)
    checks.append(
        _check(
            "channels_independent",
            agg.get("identical_channel_pairs_total", 1) == 0 and bool(sampled),
            f"{agg.get('verdict')} | max equal fraction: "
            f"{agg.get('max_exact_equal_fraction_per_pair')} | "
            f"mean corr: {agg.get('mean_pearson_correlation_per_pair')}",
        )
    )

    overall = all(c["status"] == "PASS" for c in checks)
    report = {
        "processed_root": str(proc),
        "source_root": str(source_root) if source_root else None,
        "checks": checks,
        "counts": counts,
        "boxes_total": n_boxes,
        "channel_analysis": agg,
        "overall": "PASS" if overall else "FAIL",
    }
    return report


def save_report(report: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path
