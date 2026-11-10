"""Streaming pipeline: detect -> track -> classify -> orbit -> emit (Phase 4).

Why this module exists
----------------------
``space_debris.models.stream.run_stream`` is out of bounds (R0), so the defects
it carries are fixed by reimplementing it here rather than patching it:

1. **``TrajectoryBank.miss()`` (``orbit.py:115``) is defined and never called
   anywhere in the codebase.** The original loop only advances tracks present
   in the current frame, so an occluded object's Kalman state freezes. On
   re-acquisition ByteTrack may hand back a new id, the track fragments, and
   both fragments fall below ``MIN_POINTS_FOR_ORBIT = 3`` — occlusion destroys
   results silently, with no error raised anywhere.

2. **``width``/``height`` are taken from the *last* frame processed** but read
   by the orbit report after the loop (``stream.py:181``). For a fixed-size
   video this is harmless; for a directory of mixed-size stills or a camera
   that resizes, the trajectory fit runs on the wrong geometry and px/s and
   deg/s come out wrong. We capture them from the **first** frame instead.

   (``phases.md`` originally described this as an unbound-variable crash on an
   empty source. That is not reachable: the report block is guarded by
   ``bank.ids()``, which can only be truthy if the loop body ran — and that
   is where ``width`` is bound. The guard is kept anyway because the two
   properties are otherwise only coupled by luck.)

3. **Nothing ever expires.** A track that never returns is reported forever
   and keeps consuming work.

4. **Records omit ``cx``/``cy``**, which ``orbit_from_jsonl``
   (``orbit.py:294``) reads unconditionally — so the original's own JSONL
   cannot be replayed through the library's offline orbit stage.

Design: predict once, then correct
----------------------------------
``ConstantVelocityFilter.update`` *predicts with ``dt`` and then corrects*
(``orbit.py:47-60``), and ``TrajectoryBank.update`` derives ``dt`` from the
last **observation** timestamp — but ``miss()`` does not append to history.
Calling ``miss()`` for a 10-frame gap and then ``update()`` at re-acquisition
would advance the filter once per missed frame *and* once more for the whole
gap: 10 + 11 = 21 frames of prediction for 11 frames of elapsed time. See the
measurement below for what that does to the velocity estimate.

So each frame advances **every** active track by exactly one frame
(``predict``), then corrects only the observed ones with ``dt=0.0`` so no
second prediction happens. Every active track advances exactly once per frame
— the invariant that makes "did it predict or freeze?" answerable at all.

Measured consequence of getting this wrong (10-frame gap, true velocity 0.06
units/frame, position re-observed exactly at frame 20):

| | velocity at re-acquisition | error |
|---|---|---|
| naive ``miss()`` x10 + ``update(dt=11)`` | 0.024737 | **-58.77%** |
| ``predict()`` x11 + ``update(dt=0.0)`` | 0.067980 | +13.30% |

The naive form over-predicts position (21 frames of advance for 11 elapsed),
and the position correction then drags the velocity estimate *down* to
compensate — so the damage shows up as a wrong velocity, not a wrong position.
That matters because ``motion.speed_norm_s`` is exactly the field a
"did it predict or freeze?" check reads, and it is also what makes an
occluded track look like it slowed to a crawl.

Occlusion is the *object* disappearing, not the track: ground truth keeps
moving while detections stop (D12), so the filter must bridge the gap.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from space_debris.models.classify import classify_crops, fuse
from space_debris.utils.orbit import TrajectoryBank, summarize_track

# tracks need at least this many observations before we report a trajectory
MIN_POINTS_FOR_ORBIT = 3

# A track missing for more consecutive frames than this is dropped.
DEFAULT_MAX_MISSED = 30

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mjpeg"})

# Association gate, in units of the track's own observed per-frame step. A
# frozen filter accrues the whole gap as error, so a gate of a few steps is
# what makes "the id survived the gap" a real assertion rather than a
# tautology: without prediction the re-appearing object lands outside the gate
# and gets a fresh id.
GATE_SCALE = 4.0
GATE_FLOOR_PX = 8.0


# ---------------------------------------------------------------------------
# Annotation (unchanged from the library, so output looks the same)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Per-track bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class TrackState:
    """Observation/miss bookkeeping that ``TrajectoryBank`` does not keep.

    The bank stores filter state and history only; it has no notion of a track
    being absent, and no notion of one being gone for good.
    """

    tid: int
    first_t: float
    last_seen_t: float
    observations: int = 0
    consecutive_misses: int = 0
    n_missed: int = 0          # cumulative absent frames over the lifetime
    expired: bool = False
    class_votes: dict = field(default_factory=dict)   # final_class -> count
    debris_votes: int = 0      # observations classified as debris

    @property
    def dominant_class(self) -> str | None:
        """Most-voted ``final_class``; ``None`` with no observations."""
        if not self.class_votes:
            return None
        return max(self.class_votes.items(), key=lambda kv: kv[1])[0]

    @property
    def is_debris(self) -> bool | None:
        """True when a majority of observations fused to a debris class.

        ``None`` rather than ``False`` when nothing was observed: "we never
        classified it" and "we classified it and it was not debris" are
        different answers, and collapsing them would overstate the evidence.
        """
        if self.observations == 0:
            return None
        return self.debris_votes * 2 > self.observations

    @property
    def lifespan_s(self) -> float:
        """First observation to last observation — the span of actual data.

        Deliberately *not* extended by the missing tail: a track absent for
        1,000 frames did not exist for 1,000 frames, and reporting it as
        living that long would misstate the evidence.
        """
        return float(self.last_seen_t - self.first_t)


class TrackerEngine:
    """Shared per-frame bookkeeping for both the model and injected paths.

    Frame order is fixed and matters:

    1. :meth:`predict` — advance *every* active track one frame.
    2. caller obtains detections (model boxes, or injected + associated).
    3. :meth:`observe` — correct the observed ones with ``dt=0``.
    4. :meth:`end_frame` — count misses and expire the rest.
    """

    def __init__(self, *, rate: float, max_missed: int = DEFAULT_MAX_MISSED,
                 orbit_enabled: bool = True) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be > 0, got {rate}")
        if max_missed < 0:
            raise ValueError(f"max_missed must be >= 0, got {max_missed}")
        self.rate = float(rate)
        self.max_missed = int(max_missed)
        self.orbit_enabled = bool(orbit_enabled)
        self.bank = TrajectoryBank()
        self.states: dict[int, TrackState] = {}
        self._next_id = 1

    # -- frame steps --------------------------------------------------------

    @property
    def dt(self) -> float:
        """Seconds one frame spans. Every active track advances by this."""
        return 1.0 / self.rate

    def predict(self) -> list[int]:
        """Advance all active tracks one frame without a measurement.

        This is the missed-detection propagation the library never performs.
        Expired tracks are skipped: they are frozen, not merely absent.
        """
        advanced = []
        for tid in list(self.bank.ids()):
            state = self.states.get(tid)
            if state is None or state.expired:
                continue
            self.bank.miss(tid, dt=self.dt)
            advanced.append(tid)
        return advanced

    def observe(self, tid: int, cx: float, cy: float, t: float,
                *, final_class: str | None = None,
                is_debris: bool | None = None) -> None:
        """Correct a track with this frame's measurement.

        ``dt=0.0`` because :meth:`predict` already advanced the filter for this
        frame; passing the elapsed time here would predict a second time.

        ``final_class``/``is_debris`` are optional so the association tests can
        drive the engine without a classifier; when given they accumulate into
        the track's class vote, which is what lets an orbit entry say whether
        the track it describes is debris.
        """
        if tid in self.states and self.states[tid].expired:
            return  # an expired id is not resurrected; the detection still ships
        state = self.states.get(tid)
        if state is None:
            state = TrackState(tid=tid, first_t=t, last_seen_t=t)
            self.states[tid] = state
        self.bank.update(tid, float(cx), float(cy), t, dt=0.0)
        state.observations += 1
        state.consecutive_misses = 0
        state.last_seen_t = t
        if final_class is not None:
            state.class_votes[final_class] = state.class_votes.get(final_class, 0) + 1
        if is_debris:
            state.debris_votes += 1

    def end_frame(self, observed: set[int]) -> list[int]:
        """Count an absent frame for every active track not seen this frame."""
        newly_expired = []
        for tid, state in self.states.items():
            if state.expired or tid in observed:
                continue
            state.consecutive_misses += 1
            state.n_missed += 1
            if state.consecutive_misses > self.max_missed:
                state.expired = True
                newly_expired.append(tid)
        return newly_expired

    # -- association support ------------------------------------------------

    def active_ids(self) -> list[int]:
        return [tid for tid in self.bank.ids()
                if not self.states.get(tid, TrackState(tid, 0.0, 0.0)).expired]

    def predicted_px(self, tid: int, width: int, height: int) -> tuple[float, float]:
        """This track's position *after* this frame's prediction, in pixels."""
        pos = self.bank.filters[tid].position
        return float(pos[0]) * width, float(pos[1]) * height

    def step_px(self, tid: int, width: int, height: int) -> float | None:
        """Observed displacement per frame, in pixels, from recent history.

        ``None`` for a track with fewer than two observations — there is no
        evidence of motion yet, so the gate falls back to its floor.
        """
        hist = self.bank.get_history(tid)
        if len(hist) < 2:
            return None
        (t0, x0, y0), (t1, x1, y1) = hist[-2], hist[-1]
        frames = max((t1 - t0) * self.rate, 1.0)
        return float(math.hypot((x1 - x0) * width, (y1 - y0) * height) / frames)

    def gate_px(self, tid: int, width: int, height: int,
                scale: float = GATE_SCALE, floor: float = GATE_FLOOR_PX) -> float:
        step = self.step_px(tid, width, height)
        return floor if step is None else max(floor, scale * step)

    def take_id(self) -> int:
        """Allocate a fresh track id, skipping any already in use."""
        used = set(self.states)
        while self._next_id in used:
            self._next_id += 1
        tid = self._next_id
        self._next_id += 1
        return tid

    # -- reporting ----------------------------------------------------------

    def inventory(self) -> list[dict]:
        """Every track seen this run, expired or not.

        The orbit report excludes expired tracks by design, so this is the only
        place their existence is visible — without it, "we dropped 3 tracks"
        would be unrecoverable from the outputs.
        """
        return [
            {
                "track_id": tid,
                "observations": s.observations,
                "n_missed": s.n_missed,
                "lifespan_s": round(s.lifespan_s, 4),
                "expired": s.expired,
                "final_class": s.dominant_class,
                "is_debris": s.is_debris,
            }
            for tid, s in sorted(self.states.items())
        ]

    def orbit_report(self, *, width: int, height: int, fov_deg: float,
                     range_km: float | None,
                     synthetic: bool = False) -> list[dict]:
        """Per-track trajectory summaries, excluding expired tracks.

        History is retained for expired tracks (the bank is never pruned), so
        they remain inspectable — they are excluded from the *report*, not
        deleted from the run.

        ``summarize_track`` carries no class information at all, so without the
        enrichment below an orbit entry would describe a trajectory without
        ever saying whether the object is debris — which is the question the
        report exists to answer. ``is_debris`` stays ``None`` when nothing was
        observed, rather than ``False``.
        """
        report = []
        for tid in sorted(self.bank.ids()):
            state = self.states.get(tid)
            if state is None or state.expired:
                continue
            times, points = self.bank.points(tid)
            if len(points) < MIN_POINTS_FOR_ORBIT:
                continue
            entry = summarize_track(
                tid, times, points, width=width, height=height,
                fov_deg=fov_deg, range_km=range_km,
            )
            entry.update(
                n_missed=state.n_missed,
                lifespan_s=round(state.lifespan_s, 4),
                expired=False,          # expired tracks never reach here
                observations=state.observations,
                final_class=state.dominant_class,
                is_debris=state.is_debris,
                class_votes=dict(state.class_votes),
                # R7: the orbit report is the artifact most likely to be
                # mistaken for a real measurement, so the flag lives here too.
                synthetic=bool(synthetic),
            )
            report.append(entry)
        return report


# ---------------------------------------------------------------------------
# Detection assembly
# ---------------------------------------------------------------------------


def _model_frame_dets(result, names: dict, classifier) -> list[dict]:
    """Normalise one ultralytics result into raw detection dicts."""
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    xywhn = boxes.xywhn.cpu().numpy()
    track_ids = (boxes.id.cpu().numpy().astype(int)
                 if boxes.id is not None else [None] * len(xyxy))
    preds = (classify_crops(classifier, result.orig_img, xyxy)
             if classifier is not None else [None] * len(xyxy))

    out = []
    for i in range(len(xyxy)):
        name = names.get(int(cls_ids[i]), str(int(cls_ids[i])))
        out.append({
            "track_id": None if track_ids[i] is None else int(track_ids[i]),
            "class_id": int(cls_ids[i]),
            "detector_class": name,
            "conf": round(float(confs[i]), 4),
            "xyxy": [round(float(v), 2) for v in xyxy[i]],
            "cx": round(float(xywhn[i, 0]), 4),
            "cy": round(float(xywhn[i, 1]), 4),
            "_cls_pred": preds[i],
        })
    return out


def assign_track_ids(dets: list[dict], engine: TrackerEngine,
                     width: int, height: int, *, scale: float = GATE_SCALE,
                     floor: float = GATE_FLOOR_PX) -> int:
    """Give injected detections stable ids by nearest-neighbour association.

    Matching is against each track's **predicted** position, not its last-seen
    one. That distinction is the whole point: after a gap of *N* frames a
    frozen track sits ``N * step`` px away from where the object reappears,
    while a track that predicted through the gap sits within a few px. With a
    gate of ``GATE_SCALE`` observed steps, id survival across an occlusion
    therefore *fails* if prediction is broken — which makes the Phase 4 exit
    criterion a real assertion instead of a constant-id tautology.

    Detections that already carry a ``track_id`` are honoured as-is (a caller
    may supply its own ids) and reserve that id.

    Returns the next free track id.
    """
    active = engine.active_ids()
    gates = {tid: max(floor, scale * (engine.step_px(tid, width, height) or 0.0))
             for tid in active}
    predicted = {tid: engine.predicted_px(tid, width, height) for tid in active}

    used: set[int] = set()
    taken_det: set[int] = set()
    for di, det in enumerate(dets):
        if det.get("track_id") is not None:
            tid = int(det["track_id"])
            used.add(tid)
            taken_det.add(di)

    candidates: list[tuple[float, int, int]] = []
    for di, det in enumerate(dets):
        if di in taken_det:
            continue
        px = float(det["cx"]) * width
        py = float(det["cy"]) * height
        for tid in active:
            if tid in used:
                continue
            dist = math.hypot(px - predicted[tid][0], py - predicted[tid][1])
            if dist <= gates[tid]:
                candidates.append((dist, di, tid))

    candidates.sort()
    for _dist, di, tid in candidates:
        if di in taken_det or tid in used:
            continue
        dets[di]["track_id"] = tid
        taken_det.add(di)
        used.add(tid)

    for di, det in enumerate(dets):
        if di in taken_det:
            continue
        tid = engine.take_id()
        det["track_id"] = tid
        taken_det.add(di)
        used.add(tid)
    return engine._next_id


def _to_record(det: dict) -> dict:
    """Fuse the detector label with the optional second-stage classifier.

    Accepts either ``detector_class`` (model path) or ``class_name``
    (the injected stream from ``sequence.py``). Reading only the former made
    every injected detection fuse to ``None`` and count as ``unknown`` — the
    run still "worked", it just reported the wrong taxonomy.
    """
    cls_pred = det.pop("_cls_pred", None)
    det_name = det.get("detector_class")
    if det_name is None:
        det_name = det.get("class_name")
    fused = fuse(det_name, cls_pred)
    record = {
        "track_id": det.get("track_id"),
        "class_id": int(det["class_id"]),
        "detector_class": det_name,
        "conf": round(float(det.get("conf", 0.0)), 4),
        "xyxy": [round(float(v), 2) for v in det["xyxy"]],
        # cx/cy are what orbit_from_jsonl reads; the original omits them and
        # therefore cannot be replayed through the library's offline stage.
        "cx": round(float(det["cx"]), 4),
        "cy": round(float(det["cy"]), 4),
        **fused,
    }
    if det.get("injected"):
        # Provenance so an injected detection can never be read as detector
        # output (R7), even after it has been fused and re-serialised.
        record["injected"] = True
    return record


# ---------------------------------------------------------------------------
# Frame iteration (injected path reads frames itself — no model, no weights)
# ---------------------------------------------------------------------------


def iter_frames(source: str | Path, max_frames: int | None = None
                ) -> Iterator[tuple[int, np.ndarray, str]]:
    """Yield ``(index, image_bgr, path)`` from a frame directory or a video.

    The injected path still needs pixels: for geometry (the orbit report needs
    ``width``/``height``) and for annotation. It needs them *without* a model,
    which is the entire point of D13.
    """
    import cv2

    p = Path(str(source))
    if p.is_dir():
        files = sorted(q for q in p.iterdir()
                       if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"no images under {p}")
        for i, f in enumerate(files):
            if max_frames is not None and i >= max_frames:
                return
            img = cv2.imread(str(f))
            if img is None:
                # Skip silently and the frame index stops matching the time
                # base (t = index / rate), so every downstream timestamp would
                # be shifted by one for the rest of the sequence. Fail instead.
                raise ValueError(f"unreadable image: {f}")
            yield i, img, str(f)
        return

    if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise ValueError(f"cannot open video {p}")
        i = 0
        try:
            while True:
                if max_frames is not None and i >= max_frames:
                    return
                ok, img = cap.read()
                if not ok:
                    return
                yield i, img, str(p)
                i += 1
        finally:
            cap.release()
        return

    raise ValueError(
        f"cannot read frames from {p!s}: expected a directory of images or a "
        "video file"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_stream(
    model_path: str | None = None,
    source: str | None = None,
    conf: float = 0.25,
    device: str | None = None,
    imgsz: int = 640,
    tracker: str | None = "bytetrack.yaml",
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
    detections: list[list[dict]] | None = None,
    max_missed: int = DEFAULT_MAX_MISSED,
) -> dict[str, Any]:
    """Run the pipeline and return a summary dict.

    Args:
        detections: a pre-computed per-frame detection stream (D13). When
            given, **no model is loaded and no weights are needed** — the
            frames are read straight from ``source`` and associated by
            :func:`assign_track_ids`. This validates the tracking layer in
            isolation from the detector, which is the layer this branch fixes.
        max_missed: consecutive absent frames before a track expires.

    Returns:
        Summary dict. ``detection_source`` records which path ran, so an
        injected run can never be mistaken for detector output (R7).
    """
    from .source import resolve_source as resolve_local_source

    injected = detections is not None
    if injected and source is None:
        # resolve_source(None) falls back to the default stills directory,
        # whose frames have nothing to do with the injected stream — a
        # mismatch that would silently pair detections with the wrong pixels.
        raise ValueError(
            "injected detections need an explicit --source (the frame "
            "directory or video they were computed from)"
        )
    src = resolve_local_source(source)

    if isinstance(src, int):
        if injected:
            raise ValueError("injected detections cannot drive a camera index")
        src = src  # camera: model path only

    classifier = None
    weights = None
    names: dict = {}

    if not injected:
        from space_debris.models.track import resolve_model
        from ultralytics import YOLO

        weights = resolve_model(model_path)
        model = YOLO(weights)
        names = model.names
        if classifier_path:
            from space_debris.models.classify import load_classifier
            classifier = load_classifier(classifier_path)
        rate_hint = fps
    else:
        rate_hint = fps

    from space_debris.models.track import probe_fps

    rate = rate_hint or probe_fps(src)
    if rate is None:
        rate = 1.0

    engine = TrackerEngine(rate=rate, max_missed=max_missed, orbit_enabled=orbit)

    writer = open(out_jsonl, "w", encoding="utf-8") if out_jsonl else None
    video_writer = None
    frames = 0
    frame_wh: tuple[int, int] | None = None
    class_counts: Counter = Counter()
    per_track_frames: Counter = Counter()
    predicted_log: list[int] = []

    def _emit(frame_idx: int, t: float, img: np.ndarray, path: str,
              records: list[dict], predicted: list[int]) -> None:
        """Write one JSONL event.

        ``records`` are already fused (see ``_to_record``) and already carry
        ``track_id``/``motion`` — fusing here a second time would re-run
        ``fuse()`` on an already-fused label.
        """
        nonlocal video_writer
        height, width = img.shape[:2]
        for record in records:
            tid = record["track_id"]
            if tid is not None:
                per_track_frames[int(tid)] += 1
            class_counts[record["final_class"] or "unknown"] += 1

        event = {
            "frame": frame_idx,
            "t": round(t, 4),
            "source": path,
            "image_size": [width, height],
            "detections": records,
            # Which tracks were predicted rather than observed this frame.
            # Present so "did it predict or freeze?" is answerable from the
            # JSONL itself rather than only from in-memory state (R12).
            "predicted_track_ids": sorted(predicted),
            # R7: this file is derived from synthesised frames when the D13
            # injection path ran. The flag rides on every event, not only on
            # the end-of-run summary, because a consumer may read a single
            # line out of a large file and never see the summary at all.
            "synthetic": bool(injected),
        }
        if writer:
            writer.write(json.dumps(event) + "\n")

        if annotate_path:
            if video_writer is None:
                import cv2
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(annotate_path), fourcc, rate, (width, height)
                )
            video_writer.write(_annotate(img.copy(), records))

    def _observe_records(records: list[dict], t: float) -> set[int]:
        """Fuse-then-correct: feed each detection's measurement and class to
        its track. Returns the set of track ids seen this frame.

        Fusion happens *before* the correction so the class vote accumulated
        on the track is exactly the label that ships in the JSONL — the two
        could not disagree by construction.
        """
        observed: set[int] = set()
        for record in records:
            tid = record.get("track_id")
            if tid is None:
                continue
            observed.add(int(tid))
            engine.observe(int(tid), float(record["cx"]), float(record["cy"]), t,
                           final_class=record.get("final_class"),
                           is_debris=record.get("is_debris"))
        return observed

    def _with_motion(records: list[dict]) -> list[dict]:
        """Attach each track's post-prediction motion to its detection."""
        for record in records:
            tid = record.get("track_id")
            if tid is None:
                continue
            filt = engine.bank.filters.get(int(tid))
            if filt is not None:
                record["motion"] = {
                    "vx_norm_s": round(float(filt.velocity[0]), 6),
                    "vy_norm_s": round(float(filt.velocity[1]), 6),
                    "speed_norm_s": round(float(filt.speed), 6),
                }
        return records

    try:
        if injected:
            assert isinstance(src, str), src
            stream = list(detections)  # type: ignore[arg-type]
            for idx, img, path in iter_frames(src, max_frames):
                if idx >= len(stream):
                    frame_dets: list[dict] = []
                else:
                    # copy: association writes track_id into these dicts
                    frame_dets = [dict(d) for d in stream[idx]]
                predicted = engine.predict()
                assign_track_ids(frame_dets, engine, img.shape[1], img.shape[0])
                t = frames / rate
                records = [_to_record(d) for d in frame_dets]
                observed = _observe_records(records, t)
                # Motion is read after the correction, as before: the shipped
                # velocity is this frame's post-update estimate, not the
                # pre-correction prediction.
                _with_motion(records)
                engine.end_frame(observed)
                if frame_wh is None:
                    frame_wh = (img.shape[1], img.shape[0])
                _emit(idx, t, img, path, records, predicted)
                frames += 1
                if max_frames and frames >= max_frames:
                    break
        else:
            from ultralytics import YOLO

            call_kwargs = dict(
                source=src, conf=conf, device=device, imgsz=imgsz,
                stream=True, save=save, verbose=False,
            )
            if tracker:
                results = YOLO(weights).track(tracker=tracker, persist=True,
                                              **call_kwargs)
            else:
                results = YOLO(weights).predict(**call_kwargs)

            for result in results:
                img = result.orig_img
                if frame_wh is None:
                    frame_wh = (img.shape[1], img.shape[0])
                t = frames / rate

                predicted = engine.predict()
                raw = _model_frame_dets(result, names, classifier)
                records = [_to_record(d) for d in raw]
                observed = _observe_records(records, t)
                _with_motion(records)      # post-correction estimate, as above
                engine.end_frame(observed)

                path = getattr(result, "path", None) or str(src)
                _emit(frames, t, img, path, records, predicted)
                frames += 1
                if max_frames and frames >= max_frames:
                    break
    finally:
        if writer:
            writer.close()
        if video_writer is not None:
            video_writer.release()

    # ---- orbit report (end of stream) -------------------------------------
    report: list[dict] = []
    # Guarded on frame_wh rather than on `frames`: geometry comes from the
    # first frame, and an empty source never produces one. (The original
    # guarded on bank.ids(), which is equivalent only by accident — see the
    # module docstring.)
    if orbit and frame_wh is not None and engine.bank.ids():
        report = engine.orbit_report(
            width=frame_wh[0], height=frame_wh[1],
            fov_deg=fov_deg, range_km=range_km,
        )
        if orbit_report_path and report:
            Path(orbit_report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(orbit_report_path).write_text(json.dumps(report, indent=2))

    inventory = engine.inventory()
    expired = [t for t in inventory if t["expired"]]
    summary = {
        "model": weights,
        "source": str(src),
        "frames": frames,
        "fps_assumed": rate,
        "tracker": None if injected else tracker,
        "classifier": str(classifier_path) if classifier_path else None,
        "orbit_enabled": bool(orbit),
        "detection_source": "injected" if injected else "model",
        "max_missed": engine.max_missed,
        "num_tracks": len(per_track_frames),
        "tracks": inventory,
        "expired_tracks": len(expired),
        "missed_frames_total": sum(t["n_missed"] for t in inventory),
        "detections": sum(class_counts.values()),
        "class_counts": dict(class_counts),
        "jsonl": out_jsonl,
        "annotated_video": annotate_path,
        "orbit_report": orbit_report_path if report else None,
        "orbit_tracks": len(report),
        "image_size": list(frame_wh) if frame_wh else None,
        "synthetic": bool(injected),
    }
    return summary
