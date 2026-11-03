"""Phase 6 — end-to-end validation of the whole chain (D13 path).

Runs the complete pipeline and writes one machine-readable report:

    Drive folder -> sync -> synthesise a sequence from a real still ->
    inject its ground-truth detections -> associate -> Kalman -> orbit

Everything here derives from a *synthesised* sequence, so the report is
flagged ``synthetic: true`` throughout (R7). It is a validation instrument
that measures the tracking layer against a known answer; it is never a
measurement of real debris motion.

What this module can and cannot claim
-------------------------------------
Without temporal identity ground truth in the dataset there is no MOTA,
IDF1 or identity-switch metric available for real footage. The honest
claims are track-continuity proxies on synthetic motion:

* mean track length and track count against the **degenerate baseline**
  (independent stills fed as if they were a sequence, which is what the
  original pipeline would see);
* angular rate recovered against ground truth (R9, 10% tolerance);
* whether the filter predicted or froze through an occlusion (R11).

Stating that plainly is the point: an honest limitation beats an
unsupported metric.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

# R7: every artifact this module writes carries this flag.
SYNTHETIC = True

# FOV and range are *inputs* to the library's orbit maths, not fitted
# parameters. Sweeping them documents how the reported numbers respond so
# nobody mistakes a range assumption for a measurement.
FOV_SWEEP_DEG = (45.0, 60.0, 90.0)
RANGE_SWEEP_KM = (None, 200.0, 800.0, 5000.0)

MISSING_RANGE_REASON = "range_km_needed_for_physical_speed"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def track_continuity(inventory: list[dict]) -> dict:
    """Summarise a run's track inventory into continuity proxies."""
    if not inventory:
        return {
            "tracks": 0, "mean_track_length": 0.0, "max_track_length": 0,
            "fragmented_tracks": 0, "orbit_eligible": 0,
        }
    lengths = [t["observations"] for t in inventory]
    return {
        "tracks": len(inventory),
        "mean_track_length": round(statistics.mean(lengths), 3),
        "max_track_length": max(lengths),
        # An expired track is one the occlusion forced us to give up on.
        "fragmented_tracks": sum(1 for t in inventory if t["expired"]),
        "orbit_eligible": sum(
            1 for t in inventory if not t["expired"]
            and t["observations"] >= 3
        ),
    }


def compare_continuity(baseline: dict, ours: dict) -> dict:
    """Baseline vs. synthesised-sequence continuity, with the deltas.

    The baseline is the degenerate case: independent stills, one object per
    frame at whatever position it happens to occupy. No temporal identity
    exists, so every frame is a new track of length 1 and nothing reaches
    ``MIN_POINTS_FOR_ORBIT``.
    """
    def delta(key: str) -> float:
        return round(ours[key] - baseline[key], 3)

    return {
        "baseline": baseline,
        "synthetic_sequence": ours,
        "delta_mean_track_length": delta("mean_track_length"),
        "delta_tracks": delta("tracks"),
        "delta_orbit_eligible": delta("orbit_eligible"),
        # The claim this exists to support: longer tracks, fewer of them,
        # and something that finally qualifies for an orbit fit.
        "improved": (
            ours["mean_track_length"] > baseline["mean_track_length"]
            and ours["tracks"] < baseline["tracks"]
            and ours["orbit_eligible"] > baseline["orbit_eligible"]
        ),
    }


# ---------------------------------------------------------------------------
# The validation run
# ---------------------------------------------------------------------------

def run_validation(
    out_dir: str | Path,
    *,
    frames: int = 30,
    fps: float = 5.0,
    occlusion: tuple[int, int] = (10, 20),
    max_missed: int = 30,
    still: str | Path | None = None,
    label: str | Path | None = None,
    no_sync: bool = True,
) -> dict[str, Any]:
    """Execute the full D13 chain and return the validation report.

    Args:
        out_dir: directory for frames, JSONL, orbit reports and the report.
        frames: frames per sequence.
        occlusion: half-open ``[start, end)`` gap, or ``None``-equivalent by
            passing a zero-width interval.
        still: explicit source still; defaults to the first cached Drive still
            (the sync stage of the chain).
        no_sync: resolve the Drive source from the warm cache only, so the
            chain is reproducible offline after the one-time sync.

    Returns:
        The report dict, also written to ``out_dir/validation_report.json``.
        Always carries ``synthetic: true`` (R7).
    """
    from .sequence import (
        GroundTruth,
        Motion,
        Occlusion,
        detections_from_ground_truth,
        synthesise_sequence,
    )
    from .source import resolve_source
    from .stream import run_stream

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- stage 1: Drive -> local still -------------------------------------
    if still is None:
        source = resolve_source(None, no_sync=no_sync)
        if isinstance(source, int):
            raise ValueError("camera index is not a valid validation source")
        candidates = sorted(
            p for p in Path(source).iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not candidates:
            raise ValueError(f"no stills under {source}")
        image = candidates[0]
    else:
        image = Path(still)
        if not image.exists():
            raise FileNotFoundError(f"source still not found: {image}")

    label_path = Path(label) if label else None
    if label_path is None:
        # Convention: .../images/<split>/<img>.jpg -> .../labels/<split>/
        if image.parent.parent.name == "images":
            guess = image.parent.parent.parent / "labels" / image.parent.name
            candidate = guess / (image.stem + ".txt")
            if candidate.exists():
                label_path = candidate

    # -- stage 2: synthesise with a known answer ---------------------------
    seq_dir = out / "sequence"
    gap = Occlusion(*occlusion) if occlusion[1] > occlusion[0] else None
    _, gt = synthesise_sequence(
        image, seq_dir, frames=frames,
        motion=Motion("linear", 6.0, -3.0),
        fps=fps, fov_deg=60.0, occlusion=gap, label_path=label_path,
    )
    gt.save(seq_dir / "ground_truth.json")
    stream = detections_from_ground_truth(gt)

    # -- stage 3: track -> Kalman -> orbit ---------------------------------
    jsonl = out / "stream.jsonl"
    orbit_path = out / "orbit.json"
    summary = run_stream(
        source=str(seq_dir), detections=stream, fps=fps, fov_deg=gt.fov_deg,
        orbit=True, out_jsonl=str(jsonl), orbit_report_path=str(orbit_path),
        max_missed=max_missed,
    )
    orbit = json.loads(orbit_path.read_text("utf-8")) if orbit_path.exists() else []

    # -- stage 4: the degenerate baseline ----------------------------------
    baseline_dir = out / "baseline"
    baseline_summary, baseline_orbit = _run_baseline(
        baseline_dir, frames=frames, fps=fps, fov_deg=gt.fov_deg,
        max_missed=max_missed,
    )

    # -- stage 5: FOV / range sweep ----------------------------------------
    sweep = _sweep(seq_dir, stream, out, fps=fps, max_missed=max_missed)

    # -- stage 6: continuity comparison ------------------------------------
    continuity = compare_continuity(
        track_continuity(baseline_summary["tracks"]),
        track_continuity(summary["tracks"]),
    )

    # -- assemble ----------------------------------------------------------
    rate_truth = list(gt.angular_rate_deg_s)
    rate_got = orbit[0]["trajectory"]["angular_rate_deg_s"] if orbit else None
    rate_err = _rate_error(rate_got, rate_truth) if rate_got else None

    report: dict[str, Any] = {
        # R7 — synthetic, top-level and unmissable.
        "synthetic": SYNTHETIC,
        "claim": (
            "Validation of the tracking layer against a synthesised sequence "
            "derived from a Drive still. NOT a measured debris orbit."
        ),
        "chain": [
            "drive_sync", "synthesise", "inject_ground_truth",
            "associate", "kalman", "orbit",
        ],
        "source_image": str(image),
        "label_path": str(label_path) if label_path else None,
        "frames": frames,
        "fps": fps,
        "occlusion": list(occlusion) if gap else None,
        "ground_truth": {
            "angular_rate_deg_s": rate_truth,
            "synthetic": True,
        },
        "tracking": {
            "detection_source": summary["detection_source"],
            "tracks": summary["tracks"],
            "orbit_tracks": summary["orbit_tracks"],
            "missed_frames_total": summary["missed_frames_total"],
            "expired_tracks": summary["expired_tracks"],
            "image_size": summary["image_size"],
        },
        "orbit": [
            {
                "track_id": e["track_id"],
                "final_class": e.get("final_class"),
                "is_debris": e.get("is_debris"),
                "observations": e.get("observations"),
                "n_missed": e.get("n_missed"),
                "angular_rate_deg_s": e["trajectory"]["angular_rate_deg_s"],
                "angular_rate_error_pct": rate_err,
                "speed_km_s": e["trajectory"]["speed_km_s"],
                "missing": e["trajectory"]["missing"],
                "assumptions": e["trajectory"]["assumptions"],
            }
            for e in orbit
        ],
        "rate_recovery": {
            "truth_deg_s": rate_truth,
            "recovered_deg_s": rate_got,
            "error_pct": rate_err,
            "tolerance_pct": 10.0,
            "within_tolerance": rate_err is not None and rate_err < 10.0,
        },
        "continuity": continuity,
        "sweep": sweep,
        "artifacts": {
            "frames": str(seq_dir),
            "ground_truth": str(seq_dir / "ground_truth.json"),
            "jsonl": str(jsonl),
            "orbit": str(orbit_path),
            "baseline_jsonl": str(baseline_dir / "stream.jsonl"),
        },
        "limitations": [
            "No temporal identity ground truth exists in the dataset, so "
            "MOTA/IDF1/IDF1 are unavailable; claims are track-continuity "
            "proxies on synthetic motion.",
            "speed_km_s is linear in the ASSUMED range — it is not a "
            "measurement of distance.",
            "angular_rate scales linearly with the ASSUMED fov — it is an "
            "input, not a fit.",
            "Injected detections carry class/confidence from ground truth, so "
            "this validates association and motion, not detection quality.",
        ],
    }

    report_path = out / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["artifacts"]["report"] = str(report_path)
    return report


def _rate_error(got: list[float], truth: list[float]) -> float | None:
    """Worst per-axis relative error, in percent."""
    errs = []
    for g, t in zip(got, truth):
        if abs(t) < 1e-12:
            continue
        errs.append(abs(g - t) / abs(t) * 100.0)
    return round(max(errs), 6) if errs else None


def _run_baseline(dir_path: Path, *, frames: int, fps: float,
                  fov_deg: float, max_missed: int) -> tuple[dict, list[dict]]:
    """The degenerate case: independent stills fed as a sequence.

    Each frame is an unrelated image with an object at its own position, so
    no temporal identity exists. This is what the original pipeline would
    report over the raw dataset, and the number the synthesised sequence has
    to beat.
    """
    import random

    import cv2

    from .source import resolve_source
    from .stream import run_stream

    dir_path.mkdir(parents=True, exist_ok=True)
    source = resolve_source(None, no_sync=True)
    if isinstance(source, int):
        raise ValueError("camera index is not a valid baseline source")
    stills = sorted(
        p for p in Path(source).iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:frames]
    if not stills:
        raise ValueError("no stills available for the baseline")

    rng = random.Random(0)          # deterministic, so the baseline is stable
    stream = []
    for i, s in enumerate(stills):
        img = cv2.imread(str(s))
        if img is None:
            raise ValueError(f"unreadable still: {s}")
        cv2.imwrite(str(dir_path / f"frame_{i:04d}.png"), img)
        # Object where it actually sits in that still when a label exists,
        # else scattered — either way it does not track across frames.
        cx, cy = rng.random(), rng.random()
        lab = _sibling_label(s)
        if lab is not None:
            try:
                from .sequence import read_yolo_label
                _, cx, cy, _, _ = read_yolo_label(lab)
            except (ValueError, OSError):
                pass
        stream.append([{
            "track_id": None, "class_id": 1, "class_name": "debris",
            "conf": 0.9, "cx": cx, "cy": cy, "xyxy": [0, 0, 10, 10],
        }])

    jsonl = dir_path / "stream.jsonl"
    orbit_path = dir_path / "orbit.json"
    summary = run_stream(
        source=str(dir_path), detections=stream, fps=fps, fov_deg=fov_deg,
        orbit=True, out_jsonl=str(jsonl), orbit_report_path=str(orbit_path),
        max_missed=max_missed,
    )
    orbit = json.loads(orbit_path.read_text("utf-8")) if orbit_path.exists() else []
    return summary, orbit


def _sibling_label(image: Path) -> Path | None:
    if image.parent.parent.name != "images":
        return None
    guess = image.parent.parent.parent / "labels" / image.parent.name
    candidate = guess / (image.stem + ".txt")
    return candidate if candidate.exists() else None


def _sweep(seq_dir: Path, stream: list, out: Path, *, fps: float,
           max_missed: int) -> dict:
    """Record how the reported numbers respond to FOV and range assumptions.

    Both are inputs to ``summarize_track``, not fitted quantities:

    * ``angular_rate_deg_s`` is linear in ``fov_deg`` (rate/fov constant);
    * ``speed_km_s`` is linear in ``range_km``.

    Swept and pinned so neither can be mistaken for a measurement.
    """
    from .stream import run_stream

    rows = []
    for fov in FOV_SWEEP_DEG:
        for rng in RANGE_SWEEP_KM:
            path = out / f"sweep_fov{fov}_range{rng}.json"
            run_stream(
                source=str(seq_dir), detections=stream, fps=fps, fov_deg=fov,
                range_km=rng, orbit=True, orbit_report_path=str(path),
                max_missed=max_missed,
            )
            entries = json.loads(path.read_text("utf-8"))
            if not entries:
                continue
            t = entries[0]["trajectory"]
            rows.append({
                "fov_deg": fov,
                "range_km": rng,
                "angular_rate_deg_s": t["angular_rate_deg_s"],
                "speed_km_s": t["speed_km_s"],
                "consistent_with_circular_orbit": t.get(
                    "consistent_with_circular_orbit"),
                "missing": t["missing"],
            })

    no_range = [r for r in rows if r["range_km"] is None]
    with_range = [r for r in rows if r["range_km"] is not None]
    # Speed scales with BOTH range and FOV, so linearity in range must be
    # checked within one FOV — pooling FOVs mixes two independent factors and
    # would report a false negative.
    per_fov = {}
    for r in with_range:
        per_fov.setdefault(r["fov_deg"], []).append(
            (r["range_km"], r["speed_km_s"])
        )
    return {
        "rows": rows,
        # Acceptance criterion: without a range, speed must be null and the
        # reason recorded — never a silently invented number.
        "all_missing_range_reason": all(
            MISSING_RANGE_REASON in r["missing"] for r in no_range
        ),
        "all_speed_null_without_range": all(
            r["speed_km_s"] is None for r in no_range
        ),
        "speed_linear_in_range": all(
            _is_linear(pts) for pts in per_fov.values()
        ) and bool(per_fov),
        "rate_linear_in_fov": _is_linear(
            [(r["fov_deg"], r["angular_rate_deg_s"][0]) for r in rows
             if r["range_km"] is None]
        ),
        "note": (
            "speed_km_s is linear in the assumed range and angular rate is "
            "linear in the assumed FOV; consistent_with_circular_orbit "
            "therefore checks the assumption, not the observation."
        ),
    }


def _is_linear(points: list[tuple[float, float]]) -> bool:
    """True when y/x is constant across the non-zero points."""
    ratios = [y / x for x, y in points if x and y is not None]
    if len(ratios) < 2:
        return True
    lo, hi = min(ratios), max(ratios)
    return hi - lo <= 1e-6 * max(1.0, abs(hi))


# ---------------------------------------------------------------------------
# R7 audit
# ---------------------------------------------------------------------------

def audit_synthetic_flags(report: dict) -> list[str]:
    """Return a list of R7 violations in a validation report.

    Empty means every synthetic-derived artifact is flagged and nothing
    claims to be a measurement.
    """
    problems: list[str] = []
    if report.get("synthetic") is not True:
        problems.append("report top-level synthetic flag is not true")
    if report.get("ground_truth", {}).get("synthetic") is not True:
        problems.append("ground_truth.synthetic is not true")
    claim = report.get("claim", "")
    if "NOT a measured" not in claim and "not a measured" not in claim:
        problems.append("claim does not state that this is not a measurement")
    for entry in report.get("orbit", []):
        if entry.get("synthetic") is False:
            problems.append(
                f"orbit entry {entry.get('track_id')} claims synthetic=false"
            )
    if not report.get("limitations"):
        problems.append("report carries no limitations")
    return problems
