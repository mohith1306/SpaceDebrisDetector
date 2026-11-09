#!/usr/bin/env python3
"""Verify the processed 3-channel dataset (annotations + channel independence).

Checks (see preprocessing/validation.py for details):
  - YOLO structure, 1:1 image<->label correspondence, dataset YAML
  - label format, class ids, coordinates inside [0,1]
  - labels BYTE-IDENTICAL to the original annotation files
  - processed images are uint8 HxWx3 with source dimensions
  - the three channels are genuinely independent (never byte-identical)

Also renders 6-panel sample comparisons (default 20 images) and writes the
channel statistics report.

Usage:
    python scripts/verify_dataset.py --processed processed_dataset \
        --source /path/to/Capstone
    python scripts/verify_dataset.py --processed processed_dataset --samples 40
    python scripts/verify_dataset.py --processed processed_dataset --no-visualize

Exit code: 0 = all checks PASS (warnings allowed), 1 = at least one FAIL.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("CAPSTONE_DIR", str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the physics-processed dataset")
    parser.add_argument("--processed", default=str(REPO_ROOT / "processed_dataset"),
                        help="processed_dataset directory")
    parser.add_argument("--source", help="original dataset root (enables byte-identity "
                                         "and dimension checks against the source)")
    parser.add_argument("--samples", type=int, default=20,
                        help="number of sample images for visualization/statistics")
    parser.add_argument("--no-visualize", action="store_true",
                        help="skip the 6-panel comparison rendering")
    parser.add_argument("--out", help="report JSON path "
                                      "(default: logs/verification_report.json)")
    args = parser.parse_args()

    from preprocessing.validation import save_report, verify_dataset

    print(f"[verify] processed: {args.processed}")
    print(f"[verify] source:    {args.source or '(not given)'}")
    report = verify_dataset(
        processed_root=args.processed,
        source_root=args.source,
        sample_count=args.samples,
    )

    # ---- visualization ----------------------------------------------------
    if not args.no_visualize:
        from preprocessing.visualization import visualize_samples

        proc = Path(args.processed)
        images = sorted((proc / "images" / "train").glob("*.jpg")) + sorted(
            (proc / "images" / "val").glob("*.jpg")
        )
        if not images:
            images = sorted(proc.rglob("*.png"))[: args.samples]
        chosen = images[:: max(1, len(images) // max(1, args.samples))][
            : args.samples
        ]
        if chosen:
            vis_dir = REPO_ROOT / "logs" / "visualizations"
            print(f"[verify] rendering {len(chosen)} sample comparisons -> {vis_dir}")
            vis = visualize_samples(chosen, vis_dir, source_root=args.source)
            report["visualization"] = {
                "dir": str(vis_dir),
                "images_rendered": len(vis.get("images", [])),
                "statistics": vis.get("report_path"),
            }
            print(f"[verify] channel statistics: {vis.get('report_path')}")
        else:
            print("[verify] WARNING: no images found to visualize")

    # ---- print results ----------------------------------------------------
    # ASCII markers: Windows consoles (cp1252) cannot encode emoji.
    icons = {"PASS": "[OK]", "FAIL": "[!!]", "WARN": "[--]"}
    print("\n================ VERIFICATION REPORT ================")
    for check in report["checks"]:
        icon = icons.get(check["status"], "[??]")
        print(f"{icon} [{check['status']}] {check['check']}: {check['detail']}")
    print("----------------------------------------------------")
    print(f"Overall: {report['overall']}")
    print("====================================================\n")

    out = Path(args.out) if args.out else REPO_ROOT / "logs" / "verification_report.json"
    save_report(report, out)
    print(f"[verify] report saved: {out}")

    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
