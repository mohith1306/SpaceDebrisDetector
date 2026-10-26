#!/usr/bin/env python
"""Build a synthetic video with known object motion, for validating track+orbit.

The dataset is a set of unrelated still images, so tracking/orbit cannot be
validated against it. This script takes one image, translates it by a known
pixel shift per frame, and writes the video plus the ground-truth motion so
the estimated angular rate can be checked.

    python scripts/make_demo_video.py --image yolo_dataset/images/val/img000023.jpg \
        --frames 30 --dx 6 --dy -3 --fps 5 --out /tmp/demo.mp4 --gt /tmp/demo_gt.json

Then:
    python -m space_debris.main stream --source /tmp/demo.mp4 \
        --orbit-report /tmp/orbit.json --fov 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--dx", type=float, default=6.0, help="pixel shift per frame (x)")
    ap.add_argument("--dy", type=float, default=-3.0, help="pixel shift per frame (y)")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--fov", type=float, default=60.0, help="camera FOV used by orbit stage")
    ap.add_argument("--label", help="YOLO label file for the image (normalised GT centre)")
    ap.add_argument("--out", default="/tmp/demo.mp4")
    ap.add_argument("--gt", default="/tmp/demo_gt.json")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    h, w = img.shape[:2]

    # GT object centre: from the label file if given, else the image centre
    cx, cy = 0.5, 0.5
    if args.label:
        line = Path(args.label).read_text().split()
        cx, cy = float(line[1]), float(line[2])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))

    centres = []
    for i in range(args.frames):
        tx = args.dx * i
        ty = args.dy * i
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        frame = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))
        writer.write(frame)
        centres.append([cx + tx / w, cy + ty / h])
    writer.release()

    expected = {
        "fov_deg": args.fov,
        "fps": args.fps,
        "frames": args.frames,
        "shift_px_per_frame": [args.dx, args.dy],
        # deg/s: pixel shift -> normalised shift -> degrees via FOV, times fps
        "angular_rate_deg_s": [args.dx / w * args.fov * args.fps,
                               args.dy / h * (args.fov * h / w) * args.fps],
        "centres_norm": centres,
        "video": str(out),
    }
    Path(args.gt).write_text(json.dumps(expected, indent=2))
    print(json.dumps({k: expected[k] for k in
                      ("video", "frames", "fps", "shift_px_per_frame",
                       "angular_rate_deg_s")}, indent=2))
    print(f"ground truth: {args.gt}")


if __name__ == "__main__":
    main()
