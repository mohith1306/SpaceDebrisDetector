"""Synthesise ordered sequences with known ground truth from Drive stills.

Why this module exists
----------------------
The dataset is 88,000 **independent** single-object stills: no temporal
ordering, no multi-object frames (``memory.md`` §Verified Dataset Facts). Fed
to ByteTrack as-is, every image becomes a track of length 1, nothing reaches
``MIN_POINTS_FOR_ORBIT = 3``, and the pipeline reports an empty orbit — a
system that "works" and answers nothing.

So the stills are turned into *sequences* with exact ground truth. This makes
the tracking layer measurable: track continuity, missed-detection propagation
and angular-rate recovery can all be checked against a known answer (D8, R9).

Scope of the claim (R7)
-----------------------
Everything this module produces is ``synthetic: true``. It is a **validation
instrument**, never a measurement of debris motion. A report implying a real
orbit was determined from a ``warpAffine`` translation is a correctness
failure, not a presentation choice.

Ground truth uses the library's own maths
-----------------------------------------
The angular-rate truth is computed with ``fov_vertical_deg`` imported from
``space_debris.utils.orbit`` — the *same* function ``estimate_trajectory``
uses — rather than reimplementing it. ``scripts/make_demo_video.py:73``
hardcodes the small-angle approximation ``fov * h / w`` inline; for a
1920x1080 frame that differs from the library's exact ``arctan`` form by
6.21%, which would silently consume most of the 10% tolerance R9 allows
before any tracking error was measured at all. Importing keeps truth and
estimate in agreement by construction.

(Our Drive stills are 1024x1024, where both formulas coincide — but relying
on that would make the check vacuous for any other aspect ratio.)

Relationship to ``scripts/make_demo_video.py``
---------------------------------------------
Out of bounds (R0), and its geometry is inline inside ``main()`` with no
callable to import — so the stated intent to "import its geometry" is not
achievable without editing it. Instead the translation formula is matched
exactly and pinned by ``test_sequence.py::test_matches_make_demo_video_formula``.

Occlusion is erasure, not translation (D12)
-------------------------------------------
Translating the object past the border can leave it detectable depending on
padding. Erasing its region guarantees a true detection dropout, which is the
only way to test FR-4.1.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

# The library's own vertical-FOV formula. Importing rather than reimplementing
# is what keeps ground truth and estimate_trajectory in agreement (see the
# module docstring).
from space_debris.utils.orbit import fov_vertical_deg

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Ground-truth model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Motion:
    """Ground-truth motion of the object, in pixels per frame at t = 0.

    ``mode`` selects how ``dx``/``dy`` evolve:
        ``linear``    constant velocity (displacement = dx * i)
        ``accel``     constant acceleration (adds 0.5 * ddx * i^2)
        ``orbit_arc`` alias of ``accel`` reserved for curved paths; treated
                      as linear here so it degrades to a checkable case
                      rather than an unvalidated one.

    Units are **pixels per frame**, matching ``make_demo_video.py``.
    """

    mode: str = "linear"
    dx: float = 6.0
    dy: float = -3.0
    ddx: float = 0.0
    ddy: float = 0.0

    def displacement(self, frame: int) -> tuple[float, float]:
        """Cumulative pixel displacement of the whole image at ``frame``."""
        if self.mode == "accel":
            return (
                self.dx * frame + 0.5 * self.ddx * frame * frame,
                self.dy * frame + 0.5 * self.ddy * frame * frame,
            )
        # "linear" and "orbit_arc" (degraded to linear)
        return self.dx * frame, self.dy * frame


@dataclass(frozen=True)
class Occlusion:
    """Frame interval where the object is erased from the image.

    ``[start_frame, end_frame)`` — the object is invisible throughout. The
    ground-truth centres still advance (the object keeps "moving"); only the
    pixels are removed, so a tracker that merely stops seeing it must
    **predict** through the gap rather than freeze.
    """

    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.end_frame <= self.start_frame:
            raise ValueError(
                f"occlusion end_frame ({self.end_frame}) must be after "
                f"start_frame ({self.start_frame})"
            )
        if self.start_frame < 0:
            raise ValueError(f"occlusion start_frame must be >= 0, got {self.start_frame}")

    def covers(self, frame: int) -> bool:
        return self.start_frame <= frame < self.end_frame

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class GroundTruth:
    """Everything needed to score a tracking run against a known answer."""

    fps: float
    fov_deg: float
    width: int
    height: int
    motion: Motion
    frames: int
    # Per-frame normalised centres, index == frame number. Length == frames.
    centres: list[tuple[float, float]]
    # Implied angular rate in deg/s, computed with the library's own formula.
    angular_rate_deg_s: list[float]
    occlusion: Occlusion | None = None
    # Starting normalised centre and bounding box (from the YOLO label).
    start_centre: tuple[float, float] = (0.5, 0.5)
    bbox_norm: tuple[float, float, float, float] | None = None  # cx, cy, w, h
    label_path: str | None = None
    synthetic: bool = True  # R7 — never omit
    source_image: str | None = None
    frame_dir: str | None = None

    @property
    def occluded_frames(self) -> list[int]:
        if self.occlusion is None:
            return []
        return [i for i in range(self.frames) if self.occlusion.covers(i)]

    @property
    def visible_frames(self) -> list[int]:
        return [i for i in range(self.frames) if i not in set(self.occluded_frames)]

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict() already recursed into Motion/Occlusion; keep centres as
        # plain lists of floats so json round-trips cleanly.
        d["centres"] = [[float(x), float(y)] for x, y in self.centres]
        d["angular_rate_deg_s"] = [float(v) for v in self.angular_rate_deg_s]
        d["synthetic"] = True  # R7: refuse to serialise without the flag
        return d

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "GroundTruth":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["motion"] = Motion(**raw["motion"])
        if raw.get("occlusion"):
            raw["occlusion"] = Occlusion(**raw["occlusion"])
        raw["centres"] = [(c[0], c[1]) for c in raw["centres"]]
        return cls(**raw)


# ---------------------------------------------------------------------------
# Label / centre helpers
# ---------------------------------------------------------------------------


def read_yolo_label(path: Path) -> tuple[int, float, float, float, float]:
    """Parse a YOLO label line: ``(class_id, cx, cy, w, h)`` all normalised.

    Raises on malformed input rather than defaulting to the image centre —
    a silently wrong starting centre would put ground truth off by whole
    pixels and the error would look like tracker drift.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"label not found: {path}")
    line = path.read_text(encoding="utf-8").strip()
    if not line:
        raise ValueError(f"label file is empty: {path}")
    parts = line.split()
    if len(parts) < 5:
        raise ValueError(
            f"expected 'class cx cy w h' in {path}, got {len(parts)} field(s): {line!r}"
        )
    cls_id = int(float(parts[0]))
    cx, cy, w, h = (float(v) for v in parts[1:5])
    for name, v in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name}={v} outside [0,1] in {path} — normalised coords required (R3)")
    return cls_id, cx, cy, w, h


def find_label_for(image_path: Path, labels_dir: Path) -> Path | None:
    """Locate ``<stem>.txt`` under ``labels_dir``, else None."""
    candidate = Path(labels_dir) / (Path(image_path).stem + ".txt")
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Centre and rate computation
# ---------------------------------------------------------------------------


def compute_centres(
    start: tuple[float, float],
    frames: int,
    width: int,
    height: int,
    motion: Motion,
) -> list[tuple[float, float]]:
    """Per-frame **normalised** object centres (R3: normalised, not pixels).

    The whole image is translated by ``motion.displacement(i)`` — exactly as
    ``make_demo_video.py`` does — so the object's centre moves with it. The
    object therefore cannot leave the frame by its own motion; only image
    translation moves it, and the centre is recorded whether or not the
    object is visible (occlusion erases pixels, it does not stop motion).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"bad frame size {width}x{height}")
    cx0, cy0 = start
    centres = []
    for i in range(frames):
        tx, ty = motion.displacement(i)
        centres.append((cx0 + tx / width, cy0 + ty / height))
    return centres


def angular_rate_truth(
    motion: Motion,
    fps: float,
    fov_deg: float,
    width: int,
    height: int,
) -> list[float]:
    """Ground-truth angular rate ``[deg/s, deg/s]`` for ``[x, y]``.

    Normalised units/frame -> per second -> degrees, using **the library's**
    ``fov_vertical_deg`` for the y axis. This mirrors
    ``orbit.py:206`` ``omega_deg = [v_norm[0] * fov_deg, v_norm[1] * fov_v]``
    so the truth and the estimate are computed the same way.

    For ``mode="accel"`` this returns the rate at frame 0 (the instantaneous
    initial rate), because a straight-line fit over an accelerating track has
    no single constant answer. Callers scoring an accelerating sequence must
    therefore compare against the fit, not this value — see
    ``fit_residual`` below.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if fov_deg <= 0:
        raise ValueError(f"fov_deg must be > 0, got {fov_deg}")

    v_norm_x = motion.dx / width   # normalised units / frame
    v_norm_y = motion.dy / height
    fov_v = fov_vertical_deg(fov_deg, width, height)

    return [
        float(v_norm_x * fov_deg * fps),
        float(v_norm_y * fov_v * fps),
    ]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _estimate_background(img: np.ndarray) -> tuple[int, int, int]:
    """Median colour of the image border — a robust empty-space estimate.

    Filling the erasure with black would carve a suspicious rectangle into a
    non-black background, and that rectangle is itself detectable. Sampling
    the border makes the erased region resemble surrounding space instead.
    """
    h, w = img.shape[:2]
    band = max(2, min(h, w) // 40)
    border = np.concatenate([
        img[:band, :, :].reshape(-1, 3),
        img[-band:, :, :].reshape(-1, 3),
        img[:, :band, :].reshape(-1, 3),
        img[:, -band:, :].reshape(-1, 3),
    ])
    med = np.median(border, axis=0)
    return tuple(int(round(v)) for v in med)  # type: ignore[return-value]


def erase_region(img: np.ndarray, box_xyxy: tuple[int, int, int, int],
                 fill: tuple[int, int, int]) -> None:
    """Erase a pixel rectangle in place (D12).

    ``box_xyxy`` is ``(x0, y0, x1, y1)`` in pixels, clamped to the frame. The
    rectangle is expanded by a small margin so no anti-aliased edge of the
    object survives for a detector to latch onto.
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box_xyxy
    margin = max(2, min(h, w) // 100)
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w, x1 + margin)
    y1 = min(h, y1 + margin)
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = fill


def synthesise_sequence(
    image_path: Path,
    out_dir: Path,
    frames: int = 30,
    motion: Motion | None = None,
    fps: float = 5.0,
    fov_deg: float = 60.0,
    occlusion: Occlusion | None = None,
    label_path: Path | None = None,
    labels_dir: Path | None = None,
    write_video: bool = False,
) -> tuple[Path, GroundTruth]:
    """Render an ordered sequence plus its ground truth.

    Args:
        image_path: source still from the Drive cache.
        out_dir: destination; frames are written as ``frame_%04d.png``.
        frames: number of frames to emit.
        motion: ground-truth motion; defaults to the library-matching
            ``Motion("linear", 6.0, -3.0)``.
        fps: frames per second, used for the deg/s ground truth.
        fov_deg: horizontal FOV assumed by both truth and estimate.
        occlusion: optional erase-the-object interval (D12).
        label_path: explicit YOLO label for the starting centre/bbox.
        labels_dir: searched for ``<stem>.txt`` when ``label_path`` is None.
        write_video: also emit ``sequence.mp4`` alongside the frames.

    Returns:
        ``(frames_dir, ground_truth)``.

    Raises:
        FileNotFoundError: the source still cannot be read.
        ValueError: frames <= 0, or the occluded object box cannot be placed
            inside the frame — a degenerate setup we refuse rather than
            silently render.
    """
    if frames <= 0:
        raise ValueError(f"frames must be > 0, got {frames}")

    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"cannot read source still: {image_path}")
    height, width = img.shape[:2]

    motion = motion or Motion()
    if occlusion is not None and occlusion.end_frame > frames:
        raise ValueError(
            f"occlusion ends at frame {occlusion.end_frame} but the sequence "
            f"has only {frames} frames — the gap would never close, so track "
            "recovery could not be tested"
        )

    # Starting centre / bbox: explicit label, else sibling label dir, else
    # the image centre. A bbox is required only for occlusion, because that
    # is the one feature that must know where the object *is*.
    resolved_label = Path(label_path) if label_path else (
        find_label_for(image_path, labels_dir) if labels_dir else None
    )
    bbox_norm: tuple[float, float, float, float] | None = None
    start: tuple[float, float] = (0.5, 0.5)
    if resolved_label is not None:
        _, cx, cy, bw, bh = read_yolo_label(resolved_label)
        start = (cx, cy)
        bbox_norm = (cx, cy, bw, bh)
    elif occlusion is not None:
        raise ValueError(
            "occlusion requires a bounding box to erase, but no label was "
            f"found for {image_path.name}. Pass label_path= or labels_dir=."
        )

    centres = compute_centres(start, frames, width, height, motion)
    rate = angular_rate_truth(motion, fps, fov_deg, width, height)

    gt = GroundTruth(
        fps=fps,
        fov_deg=fov_deg,
        width=width,
        height=height,
        motion=motion,
        frames=frames,
        centres=centres,
        angular_rate_deg_s=rate,
        occlusion=occlusion,
        start_centre=start,
        bbox_norm=bbox_norm,
        label_path=str(resolved_label) if resolved_label else None,
        source_image=str(image_path),
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fill = _estimate_background(img)

    # Precompute the object's pixel box at frame 0 so per-frame work is a
    # single translation, then clamping.
    obj_box0 = None
    if occlusion is not None and bbox_norm is not None:
        bcx, bcy, bw, bh = bbox_norm
        x0 = int(round((bcx - bw / 2) * width))
        y0 = int(round((bcy - bh / 2) * height))
        x1 = int(round((bcx + bw / 2) * width))
        y1 = int(round((bcy + bh / 2) * height))
        obj_box0 = (x0, y0, x1, y1)

    writer = None
    if write_video:
        video_path = out_dir / "sequence.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer at {video_path}")

    try:
        for i in range(frames):
            tx, ty = motion.displacement(i)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            # Same call as make_demo_video.py: translate the whole image,
            # constant black border, so the object rides along with it.
            frame = cv2.warpAffine(
                img, M, (width, height),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )

            if occlusion is not None and occlusion.covers(i) and obj_box0 is not None:
                bx0, by0, bx1, by1 = obj_box0
                erase_region(frame, (bx0 + int(tx), by0 + int(ty),
                                      bx1 + int(tx), by1 + int(ty)), fill)

            frame_path = out_dir / f"frame_{i:04d}.png"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"failed to write {frame_path}")
            if writer is not None:
                writer.write(frame)
    finally:
        if writer is not None:
            writer.release()

    gt.frame_dir = str(out_dir)
    return out_dir, gt


# ---------------------------------------------------------------------------
# D13: detections injected at the known centres
# ---------------------------------------------------------------------------


def detections_from_ground_truth(
    gt: GroundTruth,
    *,
    conf: float = 0.90,
    class_id: int = 1,
    visible_only: bool = True,
) -> list[list[dict]]:
    """Emit a per-frame detection stream at the known centres (D13).

    This is the injection path that validates the **tracking layer** without
    any detector: `sequence.py` already knows exactly where the object is, so
    it can hand ByteTrack/Kalman perfect detections and we can score track
    continuity, missed-detection propagation and angular-rate recovery
    against ground truth.

    Args:
        gt: ground truth from :func:`synthesise_sequence`.
        conf: confidence attached to each injected detection.
        class_id: class to claim; 1 == ``"debris"`` (R1 taxonomy).
        visible_only: when True (default), frames the occlusion erases
            produce **no** detection — that is the whole point of erasure
            (D12). Set False only to test the tracker with a dense stream.

    Returns:
        One list per frame; each element shaped like a tracker detection so
        Phase 4 can feed it in without translating shapes.

    Note:
        Injected detections carry ground-truth class and confidence, so this
        measures **association and motion**, never detection quality. Do not
        report it as detector performance (R7).
    """
    occluded = set(gt.occluded_frames) if visible_only else set()
    stream: list[list[dict]] = []
    for i, (cx, cy) in enumerate(gt.centres):
        if i in occluded:
            stream.append([])
            continue
        # Normalised centre -> pixel box, using the label's extent when we
        # have one, else a small default so the box stays sane.
        if gt.bbox_norm is not None:
            _, _, bw, bh = gt.bbox_norm
        else:
            bw = bh = 0.05
        x0 = (cx - bw / 2) * gt.width
        y0 = (cy - bh / 2) * gt.height
        x1 = (cx + bw / 2) * gt.width
        y1 = (cy + bh / 2) * gt.height
        stream.append([{
            "track_id": None,          # assigned by the tracker
            "class_id": class_id,
            "class_name": "debris",
            "conf": conf,
            "xyxy": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
            "cx": round(cx, 4),
            "cy": round(cy, 4),
            "injected": True,          # marks the D13 provenance
        }])
    return stream
