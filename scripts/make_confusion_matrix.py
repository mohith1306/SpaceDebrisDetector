#!/usr/bin/env python3
"""Rebuild an honest confusion matrix for a trained checkpoint.

Why: ultralytics column-normalizes its matrix. True negatives are never counted
in detection, so for a single-class model the "true background" column is a lone
FP entry divided by its own sum = 1.00 no matter how few false positives there
are. That cell is structurally forced and carries no information.

This script runs inference once (conf=0.001), keeps raw counts, and reports
rates each cell can actually support:
    pred=class / true=class  -> TP   (recall = TP / GT)
    pred=bg    / true=class  -> FN   (miss rate)
    pred=class / true=bg     -> FP   (per image, not per column)
    pred=bg    / true=bg     -> TN   -> undefined for detection

Usage:
    .venv/bin/python scripts/make_confusion_matrix.py \
        --weights results/v8nano_12ep/detect/weights/best.pt \
        --name v8nano_12ep
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def raise_nms_time_limit(seconds: float = 600.0) -> None:
    """Stop ultralytics from silently discarding detections.

    nms.non_max_suppression() gives up on a batch after 2.0 + 0.05*bs seconds
    and `break`s out of the per-image loop, leaving every unprocessed image in
    that batch with zero detections -- no exception, just missing boxes. At
    conf=0.001 this model produces thousands of candidates per image, so a
    6,600-image MPS val pass trips the limit constantly (45% of images came
    back empty). Raise the per-image allowance; accuracy of the counts matters
    more than saving a few seconds here.
    """
    import inspect

    import ultralytics.utils.nms as nms

    fn = nms.non_max_suppression
    params = [p for p in inspect.signature(fn).parameters][1:]  # all defaulted
    defaults = list(fn.__defaults__)
    if len(defaults) != len(params):
        return  # unexpected signature; leave ultralytics alone
    defaults[params.index("max_time_img")] = seconds
    fn.__defaults__ = tuple(defaults)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, help="path to best.pt")
    p.add_argument("--name", required=True, help="run name, used for output files")
    p.add_argument("--data", default=str(ROOT / "yolo_dataset/space_debris.yaml"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="mps", help="mps, cpu, or 0")
    p.add_argument("--iou", type=float, default=0.5, help="match IoU threshold")
    p.add_argument("--confs", type=float, nargs="+",
                   default=[0.10, 0.15, 0.19, 0.25, 0.50])
    p.add_argument("--fresh", action="store_true",
                   help="re-run inference even if cached labels exist")
    p.add_argument("--mode", choices=("predict", "val"), default="predict",
                   help="predict: chunked model.predict dump (robust); "
                        "val: full model.val (writes metrics too, but a long "
                        "MPS val pass has been seen to drop predictions)")
    return p.parse_args()


def xyxy(c: np.ndarray) -> np.ndarray:
    """YOLO normalized cxcywh -> xyxy."""
    out = np.empty_like(c)
    out[..., 0] = c[..., 0] - c[..., 2] / 2
    out[..., 1] = c[..., 1] - c[..., 3] / 2
    out[..., 2] = c[..., 0] + c[..., 2] / 2
    out[..., 3] = c[..., 1] + c[..., 3] / 2
    return out


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (N,4) vs b (M,4) -> (N,M)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None).prod(-1)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(-1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(-1)
    inter = wh
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def read_labels(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cls, xyxy boxes, conf) for one label file.

    GT rows are `cls x y w h`; prediction rows from save_txt+save_conf are
    `cls x y w h conf`. Missing conf -> NaN (GT files).
    """
    if not path.exists():
        return np.zeros(0, int), np.zeros((0, 4), float), np.zeros(0, float)
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4), float), np.zeros(0, float)
    arr = np.array([[float(x) for x in r[:6]] for r in rows])
    if arr.shape[1] < 6:
        arr = np.hstack([arr, np.full((len(arr), 6 - arr.shape[1]), np.nan)])
    return arr[:, 0].astype(int), xyxy(arr[:, 1:5]), arr[:, 5]


def dump_predict(args: argparse.Namespace, dest: Path) -> Path:
    """Chunked model.predict dump -> `<dest>/labels/<stem>.txt`.

    Used instead of model.val on long runs: a 6,600-image MPS val pass was
    observed to emit zero detections for 45% of images (smaller val runs on
    the very same images are fine), which silently guts the counts.
    """
    from ultralytics import YOLO

    raise_nms_time_limit()
    label_dir = dest / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    images_dir = ROOT / "yolo_dataset" / "images" / "val"
    files = sorted(images_dir.glob("*.jpg"))
    if not files:
        raise SystemExit(f"no images under {images_dir}")

    chunk = max(args.batch * 5, 64)
    done = 0
    pending = [f for f in files
               if not (label_dir / f"{f.stem}.txt").exists()]
    if len(pending) < len(files):
        print(f"[cm] {len(files) - len(pending):,} predictions already cached, "
              f"{len(pending):,} to run", flush=True)
    for i in range(0, len(pending), chunk):
        batch_files = pending[i:i + chunk]
        results = model.predict(
            batch_files,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=0.001,
            iou=0.7,
            max_det=300,
            save=False,
            verbose=False,
        )
        for res in results:
            out = label_dir / f"{Path(res.path).stem}.txt"
            boxes = res.boxes
            lines = []
            if boxes is not None and len(boxes):
                cls = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy()
                xywh = boxes.xywhn.cpu().numpy()
                for c, cf, (x, y, w, h) in zip(cls, conf, xywh):
                    lines.append(f"{c} {x:g} {y:g} {w:g} {h:g} {cf:g}")
            out.write_text("\n".join(lines) + ("\n" if lines else ""))
        done += len(batch_files)
        print(f"[cm] {done:,}/{len(files):,} images", flush=True)
    return label_dir


def run_inference(args: argparse.Namespace, dest: Path) -> Path:
    from ultralytics import YOLO

    raise_nms_time_limit()
    model = YOLO(args.weights)
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        conf=0.001,          # keep every box so the sweep below is exact
        iou=0.7,
        max_det=300,
        save_txt=True,
        save_conf=True,
        save_json=False,
        plots=False,
        project=str(dest.parent),
        name=dest.name,
        exist_ok=True,
        verbose=False,
    )
    print(f"[cm] ultralytics mAP50={results.box.map50:.4f} "
          f"mAP50-95={results.box.map:.4f}  "
          f"P={results.box.mp:.4f} R={results.box.mr:.4f}")
    return dest / "labels"


def sweep(pred_dir: Path, labels_dir: Path, confs: list[float],
          iou_thr: float) -> dict:
    gt_files = sorted(labels_dir.glob("*.txt"))
    if not gt_files:
        raise SystemExit(f"no labels under {labels_dir}")

    per_conf = {
        c: dict(tp=0, fp=0, fn=0, gt=0, preds=0) for c in confs
    }
    total_images = 0
    gt_boxes = 0

    for gtf in gt_files:
        cls_g, box_g, _ = read_labels(gtf)
        pf = pred_dir / gtf.name
        cls_p, box_p, conf_p = read_labels(pf)
        if cls_p.size and np.isnan(conf_p).any():
            raise SystemExit(f"prediction file missing conf column: {pf}")

        total_images += 1
        gt_boxes += len(cls_g)

        for c in confs:
            keep = conf_p >= c if conf_p.size else np.zeros(0, bool)
            b_p, s_p = box_p[keep], conf_p[keep]
            # greedy: highest confidence first, each GT matched at most once
            order = np.argsort(-s_p)
            matched_gt: set[int] = set()
            tp = fp = 0
            if len(b_p) and len(box_g):
                ious = iou(b_p[order], box_g)
                for i in range(len(order)):
                    j = int(np.argmax(ious[i])) if ious.shape[1] else -1
                    if j >= 0 and ious[i, j] >= iou_thr and j not in matched_gt:
                        matched_gt.add(j)
                        tp += 1
                    else:
                        fp += 1
            else:
                fp = len(b_p)
            fn = len(cls_g) - len(matched_gt)
            d = per_conf[c]
            d["tp"] += tp
            d["fp"] += fp
            d["fn"] += fn
            d["gt"] += len(cls_g)
            d["preds"] += len(b_p)

    out = {"images": total_images, "gt_boxes": gt_boxes, "iou": iou_thr, "levels": {}}
    for c, d in per_conf.items():
        tp, fp, fn, gt = d["tp"], d["fp"], d["fn"], d["gt"]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out["levels"][str(c)] = {
            **d,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "fp_per_image": fp / total_images,
        }
    return out


def plot(name: str, summary: dict, weights: Path, out_png: Path,
         iou_thr: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_c = max(summary["levels"],
                 key=lambda k: summary["levels"][k]["f1"])
    lv = summary["levels"][best_c]
    conf = float(best_c)
    n = summary["images"]

    # rows = predicted, cols = true (matches ultralytics storage order)
    counts = np.array([[lv["tp"], lv["fp"]],
                       [lv["fn"], 0]], dtype=float)
    # col 0 normalized by GT count; col 1 normalized by image count (FP rate
    # per image) because true negatives do not exist for detection
    rates = np.array([[lv["tp"] / lv["gt"], lv["fp_per_image"]],
                      [lv["fn"] / lv["gt"], np.nan]], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    names = summary.get("names") or ["space_object"]
    cls = names[0] if len(names) == 1 else "object"
    row_labels = [f"Predicted\n{cls}", "Predicted\nbackground"]
    col_labels = [f"True\n{cls}", "True\nbackground"]

    im0 = axes[0].imshow(counts, cmap="Blues")
    for i in range(2):
        for j in range(2):
            v = counts[i, j]
            txt = f"{int(v):,}" if not (i == 1 and j == 1) else "n/a\n(TN not\n counted)"
            axes[0].text(j, i, txt, ha="center", va="center",
                         color="black", fontsize=13,
                         fontweight="bold" if (i, j) != (1, 1) else "normal")
    axes[0].set_title(f"Raw counts  (conf = {conf}, IoU = {iou_thr})")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(np.nan_to_num(rates, nan=0.0), cmap="Blues",
                         vmin=0, vmax=max(1.0, np.nanmax(rates)))
    notes = [
        f"recall\n{lv['recall']*100:.1f}% of GT",
        f"{lv['fp_per_image']:.2f} FP\nper image",
        f"missed\n{lv['fn']/lv['gt']*100:.1f}% of GT",
        "undefined\n(no TN in\ndetection)",
    ]
    k = 0
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, notes[k], ha="center", va="center",
                         fontsize=11,
                         color="black" if np.isfinite(rates[i, j]) else "#444444")
            k += 1
    axes[1].set_title("Meaningful rates (col 0 / GT, col 1 / images)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    for ax in axes:
        ax.set_xticks([0, 1], col_labels)
        ax.set_yticks([0, 1], row_labels)
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)

    fig.suptitle(
        f"{name}  |  P={lv['precision']:.3f}  R={lv['recall']:.3f}  "
        f"F1={lv['f1']:.3f} @ conf {conf}  |  {n:,} val images, "
        f"{lv['gt']:,} GT boxes\n{weights}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise SystemExit(f"missing weights: {weights}")

    out_dir = weights.parent.parent  # results/<run>/detect
    pred_dest = out_dir / "preds"
    pred_dir = pred_dest / "labels"

    if args.fresh and pred_dest.exists():
        shutil.rmtree(pred_dest, ignore_errors=True)
    if pred_dir.exists() and any(pred_dir.glob("*.txt")):
        n = len(list(pred_dir.glob("*.txt")))
        print(f"[cm] reusing {n:,} cached prediction files in {pred_dir}")
    elif args.mode == "predict":
        pred_dir = dump_predict(args, pred_dest)
    else:
        pred_dir = run_inference(args, pred_dest)

    labels_dir = ROOT / "yolo_dataset" / "labels" / "val"
    n_gt = len(list(labels_dir.glob("*.txt")))
    n_pred = len(list(pred_dir.glob("*.txt")))
    if n_pred < n_gt - 10:
        print(f"[cm] WARNING: only {n_pred:,} of {n_gt:,} images have a "
              f"prediction file; counts below are under-reported. Re-run with "
              f"--fresh --mode predict.", flush=True)
    summary = sweep(pred_dir, labels_dir, args.confs, args.iou)

    summary["weights"] = str(weights)
    summary["name"] = args.name
    try:
        from ultralytics import YOLO
        summary["names"] = list(YOLO(str(weights)).names.values())
    except Exception:
        summary["names"] = ["space_object"]

    out_json = out_dir / "confusion_matrix_fixed.json"
    out_png = out_dir / "confusion_matrix_fixed.png"
    out_json.write_text(json.dumps(summary, indent=2))
    plot(args.name, summary, weights, out_png, args.iou)

    print(f"\n=== {args.name} ===")
    print(f"{summary['images']:,} val images, {summary['gt_boxes']:,} GT boxes, "
          f"IoU>={summary['iou']}")
    print(f"{'conf':>6} {'TP':>7} {'FP':>7} {'FN':>7} {'P':>7} {'R':>7} "
          f"{'F1':>7} {'FP/img':>7}")
    for c, lv in sorted(summary["levels"].items(), key=lambda kv: float(kv[0])):
        print(f"{float(c):6.2f} {lv['tp']:7,d} {lv['fp']:7,d} {lv['fn']:7,d} "
              f"{lv['precision']:7.3f} {lv['recall']:7.3f} {lv['f1']:7.3f} "
              f"{lv['fp_per_image']:7.2f}")
    print(f"\nwrote {out_png}\nwrote {out_json}")


if __name__ == "__main__":
    main()
