"""Multi-object tracking over a detector (ByteTrack / BoT-SORT).

Wraps ultralytics' built-in trackers so detections keep a stable id across
frames, which is what the orbit/trajectory stage needs as input.

Usage:
    python -m space_debris.main track --source yolo_dataset/images/val --max-frames 200
    python -m space_debris.main track --source scene.mp4 --tracker botsort.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import DEVICE, RUNS_DIR, YOLO_BASE

DEFAULT_TRACKER = "bytetrack.yaml"


def resolve_model(model_path: str | None) -> str:
    """Pick weights: explicit path > runs/ > newest results/* run."""
    if model_path:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"model not found: {p}")
        return str(p)

    for cand in (RUNS_DIR / "detect" / "weights" / "best.pt",
                 RUNS_DIR / "detect" / "weights" / "last.pt"):
        if cand.exists():
            return str(cand)

    runs = sorted(Path(RUNS_DIR).parent.glob("results/*/detect/weights/best.pt"))
    if runs:
        return str(runs[-1])
    raise FileNotFoundError(
        "no weights found; pass --model (e.g. results/yolo11x_15ep/detect/weights/best.pt)"
    )


def resolve_source(source: str | None) -> str | int:
    """Default source is the validation image sequence (sorted order).

    A numeric string ("0") is treated as a camera index for live streaming.
    """
    if source:
        return int(source) if source.isdigit() else source
    return str(YOLO_BASE / "images" / "val")


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mjpeg", ".gif",
              ".ts", ".mpg", ".mpeg", ".m4v"}


def probe_fps(source) -> float | None:
    """Frame rate of a video file, if ``source`` is one (else None)."""
    if isinstance(source, int):
        return None
    p = Path(str(source))
    if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
        import cv2
        cap = cv2.VideoCapture(str(p))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
        cap.release()
        if fps > 0:
            return fps
    return None


def _track_record(result, names: dict, frame_idx: int, t: float) -> dict:
    boxes = result.boxes
    detections = []
    ids = []
    height, width = result.orig_img.shape[:2]
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        xywhn = boxes.xywhn.cpu().numpy()
        track_ids = (boxes.id.cpu().numpy().astype(int)
                     if boxes.id is not None else [None] * len(boxes))
        for i in range(len(boxes)):
            tid = None if track_ids[i] is None else int(track_ids[i])
            if tid is not None:
                ids.append(tid)
            detections.append({
                "track_id": tid,
                "class_id": int(cls[i]),
                "class_name": names.get(int(cls[i]), str(int(cls[i]))),
                "conf": round(float(conf[i]), 4),
                "xyxy": [round(float(v), 2) for v in xyxy[i]],
                # normalised centre (0-1); orbit/trajectory stage expects this
                "cx": round(float(xywhn[i, 0]), 4),
                "cy": round(float(xywhn[i, 1]), 4),
            })
    return {
        "frame": frame_idx,
        "t": round(t, 4),
        "source": getattr(result, "path", None),
        "image_size": [width, height],
        "detections": detections,
        "track_ids": sorted(set(ids)),
    }


def track(
    model_path: str | None = None,
    source: str | None = None,
    conf: float = 0.25,
    device: str | None = None,
    tracker: str = DEFAULT_TRACKER,
    imgsz: int = 640,
    max_frames: int | None = None,
    fps: float | None = None,
    out_jsonl: str | None = None,
    save: bool = False,
    persist: bool = True,
) -> dict:
    """Run detection + tracking and stream per-frame records.

    Returns a summary dict; if ``out_jsonl`` is given every frame is also
    written there (one JSON object per line, in frame order).
    """
    from ultralytics import YOLO

    weights = resolve_model(model_path)
    src = resolve_source(source)
    device = device or DEVICE
    names = YOLO(weights).names

    model = YOLO(weights)
    kwargs = dict(
        source=src,
        conf=conf,
        device=device,
        imgsz=imgsz,
        tracker=tracker,
        persist=persist,
        stream=True,
        save=save,
        verbose=False,
    )
    results = model.track(**kwargs)

    writer = open(out_jsonl, "w", encoding="utf-8") if out_jsonl else None
    per_track: dict[int, int] = {}
    frames = 0
    first_fps = fps or probe_fps(src)
    try:
        for result in results:
            if first_fps is None:
                first_fps = float(getattr(result, "orig_fps", 0) or 0) or None
            rate = first_fps or 1.0
            t = frames / rate
            record = _track_record(result, names, frames, t)
            for tid in record["track_ids"]:
                per_track[tid] = per_track.get(tid, 0) + 1
            if writer:
                writer.write(json.dumps(record) + "\n")
            frames += 1
            if max_frames and frames >= max_frames:
                break
    finally:
        if writer:
            writer.close()

    summary = {
        "model": weights,
        "source": src,
        "tracker": tracker,
        "conf": conf,
        "frames": frames,
        "fps_assumed": first_fps or 1.0,
        "num_tracks": len(per_track),
        "track_lengths": {str(k): v for k, v in sorted(per_track.items())},
        "jsonl": out_jsonl,
    }
    return summary


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(track(), indent=2))
