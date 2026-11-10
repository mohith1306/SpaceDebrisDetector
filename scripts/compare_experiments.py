#!/usr/bin/env python3
"""Compare baseline vs physics-motivated preprocessing experiment metrics.

Reads the two JSON files produced by scripts/evaluate_model.py, prints a
delta table and renders grouped bar charts.

NEUTRAL BY DESIGN: this script only reports measured numbers and deltas.
It never labels a result as "better"/"worse" - that judgement belongs to the
experiment report after the actual runs exist.

Usage:
    python scripts/compare_experiments.py \
        --baseline logs/metrics_baseline.json \
        --new      logs/metrics_physics.json

    # custom output image
    python scripts/compare_experiments.py --baseline ... --new ... \
        --out logs/comparison.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# metric key -> (human label, higher_is_better_or_None)
DISPLAY_METRICS = [
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("f1", "F1-score", True),
    ("map50", "mAP@50", True),
    ("map50_95", "mAP@50-95", True),
]
DISPLAY_COUNTS = [
    ("fp", "False positives", False),
    ("fn", "False negatives", False),
]
DISPLAY_TRAIN = [
    ("duration_sec", "Training time (s)", False),
    ("peak_gpu_mem_gb", "Peak GPU mem (GB)", False),
]


def _load(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metrics file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.5f}" if abs(v) < 1000 else f"{v:.1f}"
    return str(v)


def _delta(a, b):
    if a is None or b is None:
        return None
    try:
        return float(b) - float(a)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two experiment metric files")
    parser.add_argument("--baseline", default=str(REPO_ROOT / "logs" / "metrics_baseline.json"),
                        help="metrics JSON of the baseline (original preprocessing)")
    parser.add_argument("--new", default=str(REPO_ROOT / "logs" / "metrics_physics.json"),
                        help="metrics JSON of the physics-motivated run")
    parser.add_argument("--out", help="comparison plot path (default: logs/comparison_<ts>.png)")
    parser.add_argument("--no-plot", action="store_true", help="print the table only")
    args = parser.parse_args()

    base = _load(args.baseline)
    new = _load(args.new)

    print("\n==================== EXPERIMENT COMPARISON ====================")
    print(f"baseline : {base.get('model')}  [tag={base.get('tag')}]")
    print(f"           data={base.get('data')}")
    print(f"new      : {new.get('model')}  [tag={new.get('tag')}]")
    print(f"           data={new.get('data')}")
    bt, nt = base.get("thresholds", {}), new.get("thresholds", {})
    print(f"thresholds: baseline conf={bt.get('conf')} iou={bt.get('iou')} | "
          f"physics conf={nt.get('conf')} iou={nt.get('iou')}")
    if bt != nt:
        print("WARNING: thresholds differ between runs - metrics are NOT comparable!")
    print("----------------------------------------------------------------")

    header = f"{'metric':22s} {'baseline':>14s} {'physics':>14s} {'delta':>12s}"
    print(header)
    print("-" * len(header))

    table_rows = []
    sections = [
        ("metrics", DISPLAY_METRICS),
        ("counts", DISPLAY_COUNTS),
        ("training", DISPLAY_TRAIN),
    ]
    for section, specs in sections:
        for key, label, _higher in specs:
            bv = (base.get(section) or {}).get(key)
            nv = (new.get(section) or {}).get(key)
            d = _delta(bv, nv)
            print(f"{label:22s} {_fmt(bv):>14s} {_fmt(nv):>14s} "
                  f"{('-' if d is None else f'{d:+.5f}'):>12s}")
            table_rows.append({"metric": label, "baseline": bv, "physics": nv, "delta": d})
        print("-" * len(header))

    print("NOTE: deltas are raw arithmetic differences of measured values.")
    print("      No improvement is claimed until the experiments are reviewed.")
    print("================================================================\n")

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_file": str(args.baseline),
        "physics_file": str(args.new),
        "rows": table_rows,
    }
    table_path = REPO_ROOT / "logs" / "comparison_table.json"
    table_path.parent.mkdir(exist_ok=True)
    table_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[compare] table saved: {table_path}")

    if args.no_plot:
        return 0

    # ------------------------- grouped bar chart ---------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_specs = [
        ("metrics", DISPLAY_METRICS, "Detection metrics (higher = ... measured)"),
        ("counts", DISPLAY_COUNTS, "Error counts"),
        ("training", DISPLAY_TRAIN, "Cost"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange
    for ax, (section, specs, title) in zip(axes, plot_specs):
        labels = [s[1] for s in specs]
        bvals = [(base.get(section) or {}).get(s[0]) for s in specs]
        nvals = [(new.get(section) or {}).get(s[0]) for s in specs]
        idx = x(len(labels))
        width = 0.36
        bvals_f = [0 if v is None else float(v) for v in bvals]
        nvals_f = [0 if v is None else float(v) for v in nvals]
        ax.bar(idx - width / 2, bvals_f, width, label="baseline", color="#888888")
        ax.bar(idx + width / 2, nvals_f, width, label="physics", color="#2b6cb0")
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        # annotate missing values
        for i, (bv, nv) in enumerate(zip(bvals, nvals)):
            if bv is None:
                ax.text(i - width / 2, 0, "n/a", ha="center", va="bottom", fontsize=7, color="red")
            if nv is None:
                ax.text(i + width / 2, 0, "n/a", ha="center", va="bottom", fontsize=7, color="red")

    fig.suptitle(
        "Baseline (duplicated-grayscale) vs 3-Channel Physics-Motivated Preprocessing",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "logs" / f"comparison_{time.strftime('%Y%m%d_%H%M%S')}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[compare] plot saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
