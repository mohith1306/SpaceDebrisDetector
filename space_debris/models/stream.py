"""Streaming pipeline: detect -> track -> classify -> orbit -> emit.

One pass over a video / image sequence / camera, emitting one JSON record per
frame (JSONL) plus an optional annotated video and a final per-track orbit
report.

    python -m space_debris.main stream --source yolo_dataset/images/val \
        --max-frames 300 --out runs/stream.jsonl --annotate runs/stream.mp4

Stage toggles:
    --tracker / --no-tracker     association across frames (ByteTrack default)
    --classifier PATH            second-stage classification of each crop
    --orbit                      per-track Kalman + trajectory estimate
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import DEVICE, RUNS_DIR
from ..utils.orbit import TrajectoryBank, summarize_track
from .classify import classify_crops, fuse
from .track import DEFAULT_TRACKER, probe_fps, resolve_model, resolve_source

# tracks need at least this many observations before we report a trajectory
MIN_POINTS_FOR_ORBIT = 3


def _annotate(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    import cv2

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["xyxy"])
        color = (0, 64, 255) if det.get("is_debris") else (80, 200, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tid = det.get("track_id")
        label = f"#{tid} " if tid is not None else ""
        label += f"{det.get('final_class') or '-'} {det.get('conf', 0):.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return frame


def run_stream(
    model_path: str | None = None,
    source: str | None = None,
    conf: float = 0.25,
    device: str | None = None,
    imgsz: int = 640,
    tracker: str | None = DEFAULT_TRACKER,
    classifier_path: str | None = None,
    orbit: bool = True,
    fov_deg: float = 60.0,
    range_km: float | None = None,
    fps: float | None = None,
    out_jsonl: str | None = None,
    annotate_path: str | None = None,
    orbit_report_path: str | None = None,
    max_frames: int | None = None,
    save: bool = False,
) -> dict:
    """Run the full pipeline. Returns a summary dict."""
    import cv2
    from ultralytics import YOLO

    weights = resolve_model(model_path)
    src = resolve_source(source)
    device = device or DEVICE
    names = YOLO(weights).names
    model = YOLO(weights)

    classifier = None
    if classifier_path:
        from .classify import load_classifier
        classifier = load_classifier(classifier_path)

    call_kwargs = dict(
        source=src, conf=conf, device=device, imgsz=imgsz,
        stream=True, save=save, verbose=False,
    )
    if tracker:
        results = model.track(tracker=tracker, persist=True, **call_kwargs)
    else:
        results = model.predict(**call_kwargs)

    bank = TrajectoryBank()
    writer = open(out_jsonl, "w", encoding="utf-8") if out_jsonl else None
    video_writer = None
    frames = 0
    class_counts: Counter = Counter()
    per_track_frames: Counter = Counter()
    rate = fps or probe_fps(src)

    try:
        for result in results:
            if rate is None:
                rate = float(getattr(result, "orig_fps", 0) or 0) or None
            rate = rate or 1.0
            t = frames / rate

            img = result.orig_img
            height, width = img.shape[:2]
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and len(boxes) else np.zeros((0, 4))
            confs = boxes.conf.cpu().numpy() if boxes is not None and len(boxes) else np.zeros(0)
            cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes is not None and len(boxes) else np.zeros(0, int)
            track_ids = (boxes.id.cpu().numpy().astype(int)
                         if boxes is not None and boxes.id is not None and len(boxes)
                         else [None] * len(xyxy))
            xywhn = boxes.xywhn.cpu().numpy() if boxes is not None and len(boxes) else np.zeros((0, 4))

            cls_preds = (classify_crops(classifier, img, xyxy)
                         if classifier is not None else [None] * len(xyxy))

            detections = []
            for i in range(len(xyxy)):
                det_name = names.get(int(cls_ids[i]), str(int(cls_ids[i])))
                fused = fuse(det_name, cls_preds[i])
                tid = None if track_ids[i] is None else int(track_ids[i])
                if orbit and tid is not None:
                    bank.update(tid, float(xywhn[i, 0]), float(xywhn[i, 1]), t)
                record = {
                    "track_id": tid,
                    "class_id": int(cls_ids[i]),
                    "detector_class": det_name,
                    "conf": round(float(confs[i]), 4),
                    "xyxy": [round(float(v), 2) for v in xyxy[i]],
                    **fused,
                }
                if tid is not None:
                    filt = bank.filters.get(tid)
                    if filt is not None:
                        record["motion"] = {
                            "vx_norm_s": round(float(filt.velocity[0]), 6),
                            "vy_norm_s": round(float(filt.velocity[1]), 6),
                            "speed_norm_s": round(float(filt.speed), 6),
                        }
                detections.append(record)
                class_counts[fused["final_class"] or "unknown"] += 1
                if tid is not None:
                    per_track_frames[tid] += 1

            event = {
                "frame": frames,
                "t": round(t, 4),
                "source": getattr(result, "path", None),
                "image_size": [width, height],
                "detections": detections,
            }
            if writer:
                writer.write(json.dumps(event) + "\n")

            if annotate_path:
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(annotate_path), fourcc, rate, (width, height)
                    )
                video_writer.write(_annotate(img.copy(), detections))

            frames += 1
            if max_frames and frames >= max_frames:
                break
    finally:
        if writer:
            writer.close()
        if video_writer is not None:
            video_writer.release()

    # ---- orbit report (end of stream) -------------------------------------
    report = []
    if orbit and bank.ids():
        for tid in sorted(bank.ids()):
            times, points = bank.points(tid)
            if len(points) < MIN_POINTS_FOR_ORBIT:
                continue
            report.append(summarize_track(
                tid, times, points, width=width, height=height,
                fov_deg=fov_deg, range_km=range_km,
            ))
        if orbit_report_path:
            Path(orbit_report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(orbit_report_path).write_text(json.dumps(report, indent=2))

    summary = {
        "model": weights,
        "source": src,
        "frames": frames,
        "fps_assumed": rate or 1.0,
        "tracker": tracker,
        "classifier": str(classifier_path) if classifier_path else None,
        "orbit_enabled": bool(orbit),
        "num_tracks": len(per_track_frames),
        "detections": sum(class_counts.values()),
        "class_counts": dict(class_counts),
        "jsonl": out_jsonl,
        "annotated_video": annotate_path,
        "orbit_report": orbit_report_path if report else None,
        "orbit_tracks": len(report),
    }
    return summary


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run_stream(), indent=2))
