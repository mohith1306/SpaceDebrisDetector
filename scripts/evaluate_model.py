#!/usr/bin/env python3
"""Evaluate a trained detector and record comparable experiment metrics.

Records exactly the metrics required for the baseline-vs-physics comparison:

    precision, recall, F1-score, mAP@50, mAP@50-95,
    false positives, false negatives,
    training time, peak GPU memory (when a train_summary.json exists),
    evaluation device + wall time.

Confidence and IoU thresholds default to the values in
configs/physics_config.yaml (conf=0.25, iou=0.5) and MUST stay identical
between the baseline and the physics run for a fair comparison.

Usage:
    # physics experiment (default model/data/tag)
    python scripts/evaluate_model.py --tag physics

    # baseline experiment (original preprocessing, same thresholds)
    python scripts/evaluate_model.py \
        --model runs/detect/weights/best.pt \
        --data  yolo_dataset/space_debris.yaml \
        --tag   baseline

Outputs: logs/metrics_<tag>.json (feed both files to compare_experiments.py)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("CAPSTONE_DIR", str(REPO_ROOT))


def _extract_box_metrics(results) -> dict:
    """Precision / recall / mAP from an ultralytics validation result."""
    out: dict = {}
    rd = getattr(results, "results_dict", None) or {}
    key_map = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "map50",
        "metrics/mAP50-95(B)": "map50_95",
    }
    for src, dst in key_map.items():
        if src in rd:
            out[dst] = float(rd[src])

    # F1 at the operating point implied by the reported precision/recall
    p, r = out.get("precision"), out.get("recall")
    out["f1"] = round(2 * p * r / (p + r), 6) if (p and r) else None

    # Per-class breakdown (best effort)
    try:
        box = results.box
        names = getattr(results, "names", {}) or {}
        per_class = {}
        for i, name in names.items():
            per_class[name] = {
                "precision": round(float(box.p[i]), 5),
                "recall": round(float(box.r[i]), 5),
                "map50": round(float(box.ap50[i]), 5),
                "map50_95": round(float(box.ap[i]), 5),
            }
        out["per_class"] = per_class
    except Exception:  # noqa: BLE001 - per-class is informational only
        pass
    return out


def _extract_fp_fn(results, nc: int) -> dict:
    """False positives / negatives from the validation confusion matrix.

    Ultralytics layout: matrix[actual, predicted], with index nc = background:
      matrix[i, nc] = ground-truth i missed      -> FN
      matrix[nc, j] = background predicted as j  -> FP
    Falls back to nulls if the matrix is unavailable (never invents numbers).
    """
    try:
        cm = getattr(results, "confusion_matrix", None)
        matrix = getattr(cm, "matrix", None)
        if matrix is None:
            return {"fp": None, "fn": None, "tp": None, "source": "unavailable"}
        import numpy as np

        m = np.asarray(matrix)
        if m.shape[0] < nc + 1:
            return {"fp": None, "fn": None, "tp": None, "source": "shape_mismatch"}
        tp = int(m[:nc, :nc].sum())
        fn = int(m[:nc, nc].sum())   # missed detections
        fp = int(m[nc, :nc].sum())   # spurious detections
        return {"fp": fp, "fn": fn, "tp": tp, "source": "confusion_matrix"}
    except Exception as exc:  # noqa: BLE001
        return {"fp": None, "fn": None, "tp": None, "source": f"error: {exc}"}


def main() -> int:
    from preprocessing.experiment_config import cfg_get, load_physics_config

    cfg = load_physics_config()

    parser = argparse.ArgumentParser(description="Evaluate a trained detector")
    parser.add_argument("--model", default=str(REPO_ROOT / "runs" / "physics_detect" / "weights" / "best.pt"))
    parser.add_argument("--data", default=str(REPO_ROOT / "processed_dataset" / "space_debris.yaml"))
    parser.add_argument("--tag", default="physics", help="metrics file tag (physics|baseline|...)")
    parser.add_argument("--conf", type=float, default=float(cfg_get(cfg, "evaluation.conf", 0.25)))
    parser.add_argument("--iou", type=float, default=float(cfg_get(cfg, "evaluation.iou", 0.5)))
    parser.add_argument("--imgsz", type=int, default=int(cfg_get(cfg, "evaluation.imgsz", 640)))
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--out", help="default: logs/metrics_<tag>.json")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    if not model_path.exists():
        print(f"[eval] ERROR: model not found: {model_path}")
        print("[eval] Train first: python scripts/train_physics_model.py")
        return 2
    if not data_path.exists():
        print(f"[eval] ERROR: dataset yaml not found: {data_path}")
        return 2

    import torch
    from ultralytics import YOLO

    from space_debris.models.train_resilient import detect_device

    device, device_info = detect_device(args.device)
    print(f"[eval] model:  {model_path}")
    print(f"[eval] data:   {data_path} (split={args.split})")
    print(f"[eval] thresholds: conf={args.conf} iou={args.iou} imgsz={args.imgsz}")
    print(f"[eval] device: {device_info}")

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    model = YOLO(str(model_path))
    started = time.time()
    results = model.val(
        data=str(data_path),
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        verbose=False,
        save_json=False,
        save_conf=False,
    )
    eval_duration = round(time.time() - started, 1)
    peak_eval_gb = (
        round(torch.cuda.max_memory_allocated(0) / 1e9, 3)
        if device == "cuda" and torch.cuda.is_available()
        else None
    )

    nc = int(getattr(results, "nc", 0) or 0)
    metrics = _extract_box_metrics(results)
    counts = _extract_fp_fn(results, nc)

    # Training-time / GPU-memory records (written by train_physics_model.py
    # or the baseline's train_resilient.py logs) - reported only if present.
    train_info: dict = {"duration_sec": None, "peak_gpu_mem_gb": None, "source": None}
    summary_candidate = model_path.parent.parent / "train_summary.json"
    if summary_candidate.exists():
        try:
            s = json.loads(summary_candidate.read_text(encoding="utf-8"))
            train_info = {
                "duration_sec": s.get("duration_sec"),
                "peak_gpu_mem_gb": s.get("peak_gpu_mem_gb"),
                "epochs_completed": s.get("epochs_completed"),
                "source": str(summary_candidate),
            }
        except (json.JSONDecodeError, OSError):
            pass

    payload = {
        "tag": args.tag,
        "experiment": "physics_3channel" if args.tag == "physics" else args.tag,
        "model": str(model_path),
        "data": str(data_path),
        "split": args.split,
        "thresholds": {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz},
        "metrics": {k: v for k, v in metrics.items() if k != "per_class"},
        "per_class": metrics.get("per_class", {}),
        "counts": counts,
        "training": train_info,
        "evaluation": {
            "device": device_info,
            "duration_sec": eval_duration,
            "peak_gpu_mem_gb": peak_eval_gb,
        },
        "note": (
            "Same model architecture, split, thresholds and metrics are used "
            "for baseline and physics experiments; no improvement is claimed "
            "until compare_experiments.py shows measured results."
        ),
    }

    out_path = Path(args.out) if args.out else REPO_ROOT / "logs" / f"metrics_{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n================ EVALUATION ================")
    for key in ("precision", "recall", "f1", "map50", "map50_95"):
        val = payload["metrics"].get(key)
        print(f"  {key:10s}: {val if val is None else round(val, 5)}")
    print(f"  {'TP':10s}: {counts.get('tp')}")
    print(f"  {'FP':10s}: {counts.get('fp')}  (source: {counts.get('source')})")
    print(f"  {'FN':10s}: {counts.get('fn')}")
    print(f"  {'train_s':10s}: {train_info.get('duration_sec')}")
    print(f"  {'gpu_gb':10s}: {train_info.get('peak_gpu_mem_gb')}")
    print("===========================================")
    print(f"[eval] saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
