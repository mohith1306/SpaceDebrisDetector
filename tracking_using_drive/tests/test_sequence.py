"""Phase 3 tests: sequence synthesis and its ground truth.

Covers the rules that make validation meaningful:
  R9  synthesised motion is the oracle; angular rate within 10% of truth
  R11 occlusion must be tested explicitly: gap present, motion predicted
  R12 assert on the JSONL / ground-truth structure, not stdout
  R7  synthetic output must never lose its ``synthetic`` flag
  D12 occlusion is erasure, not translation
  D13 detection injection at known centres

The network is never touched: a still is drawn from the local Drive cache
when present, else a synthetic 1024x1024 image is generated in tmp, so this
suite runs on a fresh clone before any sync.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tracking_using_drive import sequence as seq
from space_debris.utils.orbit import estimate_trajectory, fov_vertical_deg

_CACHE = Path(__file__).resolve().parents[1] / "_data" / "gdrive_cache"
_VAL_IMG = _CACHE / "yolo_dataset" / "images" / "val"
_VAL_LAB = _CACHE / "yolo_dataset" / "labels" / "val"
_HAS_CACHE = _VAL_IMG.is_dir() and any(_VAL_IMG.glob("*.jpg"))


def _cache_still() -> tuple[Path, Path | None]:
    """One cached still and its label, or a generated stand-in."""
    if _HAS_CACHE:
        img = sorted(_VAL_IMG.glob("*.jpg"))[0]
        lab = _VAL_LAB / (img.stem + ".txt")
        return img, (lab if lab.exists() else None)
    pytest.skip("no Drive cache present and generator not yet invoked")


@pytest.fixture()
def still(request) -> tuple[Path, Path | None]:
    """A source still: cached if available, else generated in tmp."""
    if _HAS_CACHE:
        return _cache_still()
    # Fresh clone with no cache: make a deterministic stand-in so the
    # structural tests still run rather than skipping wholesale.
    import cv2

    tmp = Path(request.config.cache.makedir("stills"))
    img_path = tmp / "generated.png"
    if not img_path.exists():
        rng = np.random.default_rng(42)
        cv2.imwrite(str(img_path), rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    return img_path, None


# ---------------------------------------------------------------------------
# Motion model
# ---------------------------------------------------------------------------

def test_linear_displacement_is_proportional_to_frame():
    m = seq.Motion("linear", dx=6.0, dy=-3.0)
    assert m.displacement(0) == (0.0, -0.0)
    assert m.displacement(1) == (6.0, -3.0)
    assert m.displacement(10) == (60.0, -30.0)


def test_accel_displacement_follows_quadratic():
    m = seq.Motion("accel", dx=6.0, dy=-3.0, ddx=2.0, ddy=0.5)
    # p(i) = d*i + 0.5*a*i^2
    assert m.displacement(0) == (0.0, -0.0)
    assert m.displacement(4)[0] == pytest.approx(6 * 4 + 0.5 * 2 * 16)
    assert m.displacement(4)[1] == pytest.approx(-3 * 4 + 0.5 * 0.5 * 16)


def test_orbit_arc_degrades_to_linear_rather_than_unvalidated():
    """'orbit_arc' must not silently take an untested code path."""
    arc = seq.Motion("orbit_arc", dx=6.0, dy=-3.0)
    lin = seq.Motion("linear", dx=6.0, dy=-3.0)
    for i in range(12):
        assert arc.displacement(i) == lin.displacement(i)


# ---------------------------------------------------------------------------
# Occlusion (R11, D12)
# ---------------------------------------------------------------------------

def test_occlusion_interval_is_half_open():
    o = seq.Occlusion(10, 20)
    assert o.covers(10) is True      # start included
    assert o.covers(19) is True      # end excluded
    assert o.covers(20) is False
    assert o.covers(9) is False
    assert o.n_frames == 10


def test_occlusion_rejects_inverted_or_negative_ranges():
    with pytest.raises(ValueError, match="after"):
        seq.Occlusion(20, 10)
    with pytest.raises(ValueError, match="end_frame"):
        seq.Occlusion(15, 15)
    with pytest.raises(ValueError, match=">= 0"):
        seq.Occlusion(-1, 5)


# ---------------------------------------------------------------------------
# Centres and ground-truth rate (R3, R9)
# ---------------------------------------------------------------------------

def test_centres_are_normalised_not_pixels():
    """R3: orbit.py assumes cx/cy in [0,1]."""
    centres = seq.compute_centres((0.5, 0.5), 40, 1024, 1024,
                                  seq.Motion("linear", 6.0, -3.0))
    for cx, cy in centres:
        assert -1.0 <= cx <= 2.0, cx
        assert -1.0 <= cy <= 2.0, cy


def test_centre_step_matches_displacement_over_size():
    w, h = 1024, 1024
    m = seq.Motion("linear", 6.0, -3.0)
    centres = seq.compute_centres((0.4, 0.4), 10, w, h, m)
    dx_norm = (centres[1][0] - centres[0][0])
    assert dx_norm == pytest.approx(m.dx / w)
    assert (centres[1][1] - centres[0][1]) == pytest.approx(m.dy / h)


def test_angular_rate_truth_matches_library_estimate_within_10pct():
    """R9 acceptance criterion #5: < 10% error.

    The truth must be recovered by the *library's* estimator. This is a real
    check on formula agreement, not a tautology: it would fail if the truth
    were computed with a different vertical-FOV formula than orbit.py uses.
    """
    for fps in (5.0, 10.0, 30.0):
        for motion in (seq.Motion("linear", 6.0, -3.0),
                       seq.Motion("linear", -8.0, 4.0),
                       seq.Motion("linear", 0.0, -5.0)):
            w = h = 1024
            truth = seq.angular_rate_truth(motion, fps, 60.0, w, h)
            centres = seq.compute_centres((0.4, 0.4), 30, w, h, motion)
            times = np.arange(30) / fps
            est = estimate_trajectory(times, np.asarray(centres, dtype=float),
                                      width=w, height=h, fov_deg=60.0)
            got = est["angular_rate_deg_s"]
            for t, g in zip(truth, got):
                if abs(t) < 1e-9:
                    assert abs(g) < 1e-6, (t, g)
                    continue
                err = abs(g - t) / abs(t) * 100
                assert err < 10.0, f"{err:.3f}% exceeds R9 (fps={fps}, motion={motion})"


def test_vertical_fov_uses_exact_formula_not_small_angle():
    """The small-angle approximation is 6.21% off at 16:9 — enough to eat
    most of R9's budget before any tracking error is measured."""
    exact = fov_vertical_deg(60.0, 1920, 1080)
    small_angle = 60.0 * 1080 / 1920
    assert exact != pytest.approx(small_angle, rel=1e-6)
    assert abs(exact - small_angle) / exact * 100 == pytest.approx(6.21, abs=0.01)
    # Our truth must follow the library, i.e. the exact value.
    m = seq.Motion("linear", 0.0, -10.0)
    rate = seq.angular_rate_truth(m, 5.0, 60.0, 1920, 1080)
    assert rate[1] == pytest.approx(
        (m.dy / 1080) * exact * 5.0, rel=1e-9
    )


def test_truth_rejects_nonpositive_fps_or_fov():
    with pytest.raises(ValueError, match="fps"):
        seq.angular_rate_truth(seq.Motion(), 0.0, 60.0, 1024, 1024)
    with pytest.raises(ValueError, match="fov_deg"):
        seq.angular_rate_truth(seq.Motion(), 5.0, -1.0, 1024, 1024)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_read_yolo_label_parses_normalised_bbox(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.4052734375 0.37646484375 0.810546875 0.7529296875\n")
    cls, cx, cy, w, h = seq.read_yolo_label(p)
    assert cls == 0
    assert cx == pytest.approx(0.4052734375)
    assert w == pytest.approx(0.810546875)


def test_read_yolo_label_fails_loudly_on_bad_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        seq.read_yolo_label(tmp_path / "missing.txt")

    empty = tmp_path / "empty.txt"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        seq.read_yolo_label(empty)

    short = tmp_path / "short.txt"
    short.write_text("0 0.5 0.5\n")
    with pytest.raises(ValueError, match="expected"):
        seq.read_yolo_label(short)

    # R3: pixel centres must be rejected, not silently accepted.
    oob = tmp_path / "oob.txt"
    oob.write_text("0 512 300 64 64\n")
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        seq.read_yolo_label(oob)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_synthesise_writes_frames_and_ground_truth(still, tmp_path):
    img, lab = still
    out = tmp_path / "seq"
    _, gt = seq.synthesise_sequence(
        img, out, frames=8, motion=seq.Motion("linear", 6.0, -3.0),
        fps=5.0, fov_deg=60.0, label_path=lab,
    )
    frames = sorted(out.glob("frame_*.png"))
    assert len(frames) == 8
    assert [f.name for f in frames] == [f"frame_{i:04d}.png" for i in range(8)]
    assert gt.synthetic is True                       # R7
    assert gt.frames == 8 and len(gt.centres) == 8
    assert gt.angular_rate_deg_s                      # non-empty


def test_synthesise_refuses_zero_frames(still, tmp_path):
    img, _ = still
    with pytest.raises(ValueError, match="frames"):
        seq.synthesise_sequence(img, tmp_path / "x", frames=0)


def test_synthesise_refuses_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        seq.synthesise_sequence(tmp_path / "nope.jpg", tmp_path / "x", frames=3)


def test_occlusion_beyond_sequence_is_refused(still, tmp_path):
    """A gap that never closes means recovery could never be tested."""
    img, lab = still
    with pytest.raises(ValueError, match="never close"):
        seq.synthesise_sequence(
            img, tmp_path / "x", frames=10,
            occlusion=seq.Occlusion(5, 20), label_path=lab,
        )


def test_occlusion_without_label_is_refused(still, tmp_path):
    """Erasing needs to know where the object is."""
    img, _ = still
    with pytest.raises(ValueError, match="bounding box"):
        seq.synthesise_sequence(
            img, tmp_path / "x", frames=10,
            occlusion=seq.Occlusion(2, 5), label_path=None, labels_dir=None,
        )


def test_erasure_flattens_the_object_region(still, tmp_path):
    """D12: the object must genuinely disappear, not shift off-canvas."""
    cv2 = pytest.importorskip("cv2")
    img, lab = still
    if lab is None:
        pytest.skip("no label available, so occlusion cannot be constructed")
    out = tmp_path / "seq"
    occl = seq.Occlusion(4, 8)
    _, gt = seq.synthesise_sequence(
        img, out, frames=12, motion=seq.Motion("linear", 6.0, -3.0),
        fps=5.0, occlusion=occl, label_path=lab,
    )
    cx, cy, bw, bh = gt.bbox_norm

    def roi_std(i: int) -> float:
        f = cv2.imread(str(out / f"frame_{i:04d}.png"))
        tx, ty = gt.motion.displacement(i)
        x0 = max(0, int((cx - bw / 2) * gt.width) + int(tx))
        y0 = max(0, int((cy - bh / 2) * gt.height) + int(ty))
        x1 = min(gt.width, int((cx + bw / 2) * gt.width) + int(tx))
        y1 = min(gt.height, int((cy + bh / 2) * gt.height) + int(ty))
        return float(np.std(f[y0:y1, x0:x1]))

    visible_std = roi_std(2)
    occluded_std = roi_std(6)
    assert visible_std > 1.0, "object region should have structure while visible"
    # Erased region is filled with the border median -> near-zero variance.
    assert occluded_std < visible_std * 0.35, (
        f"erasure incomplete: occluded std {occluded_std:.2f} vs visible {visible_std:.2f}"
    )


def test_motion_continues_unchanged_through_the_gap(still, tmp_path):
    """R11: centres must keep advancing during occlusion."""
    img, lab = still
    if lab is None:
        pytest.skip("no label available")
    out = tmp_path / "seq"
    o = seq.Occlusion(5, 9)
    _, gt = seq.synthesise_sequence(
        img, out, frames=14, motion=seq.Motion("linear", 6.0, -3.0),
        fps=5.0, occlusion=o, label_path=lab,
    )
    assert gt.occluded_frames == [5, 6, 7, 8]
    # Identical step entering and leaving the gap.
    step_in = (gt.centres[5][0] - gt.centres[4][0], gt.centres[5][1] - gt.centres[4][1])
    step_out = (gt.centres[9][0] - gt.centres[8][0], gt.centres[9][1] - gt.centres[8][1])
    assert step_in[0] == pytest.approx(step_out[0], abs=1e-12)
    assert step_in[1] == pytest.approx(step_out[1], abs=1e-12)
    # And it is genuinely non-zero, or "prediction" would be vacuous.
    assert abs(step_in[0]) > 0 or abs(step_in[1]) > 0


# ---------------------------------------------------------------------------
# Ground-truth serialisation (R7, R12)
# ---------------------------------------------------------------------------

def test_ground_truth_round_trips_through_json(still, tmp_path):
    img, lab = still
    out = tmp_path / "seq"
    _, gt = seq.synthesise_sequence(
        img, out, frames=6, motion=seq.Motion("linear", 6.0, -3.0),
        fps=5.0, occlusion=seq.Occlusion(2, 4), label_path=lab,
    )
    p = gt.save(tmp_path / "gt.json")
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["synthetic"] is True, "R7: synthetic flag must be on the JSON"
    back = seq.GroundTruth.load(p)
    assert back.centres == gt.centres
    assert back.angular_rate_deg_s == gt.angular_rate_deg_s
    assert back.occlusion == gt.occlusion
    assert back.synthetic is True


def test_synthetic_flag_cannot_be_serialised_away():
    """R7: even if constructed with False, save() forces True."""
    gt = seq.GroundTruth(
        fps=5.0, fov_deg=60.0, width=100, height=100,
        motion=seq.Motion(), frames=2, centres=[(0.5, 0.5), (0.6, 0.5)],
        angular_rate_deg_s=[1.0, 2.0], synthetic=False,
    )
    assert gt.to_dict()["synthetic"] is True


# ---------------------------------------------------------------------------
# D13: injected detections
# ---------------------------------------------------------------------------

def test_injected_stream_suppresses_only_occluded_frames(still, tmp_path):
    img, lab = still
    if lab is None:
        pytest.skip("no label available")
    out = tmp_path / "seq"
    _, gt = seq.synthesise_sequence(
        img, out, frames=16, motion=seq.Motion("linear", 6.0, -3.0),
        fps=5.0, occlusion=seq.Occlusion(6, 10), label_path=lab,
    )
    stream = seq.detections_from_ground_truth(gt)
    assert len(stream) == 16
    for i in range(16):
        if 6 <= i < 10:
            assert stream[i] == [], f"frame {i} is occluded and must be empty"
        else:
            assert len(stream[i]) == 1, f"frame {i} must carry one detection"


def test_injected_detections_carry_provenance_and_normalised_centres(still, tmp_path):
    img, lab = still
    out = tmp_path / "seq"
    _, gt = seq.synthesise_sequence(img, out, frames=4, fps=5.0, label_path=lab)
    det = seq.detections_from_ground_truth(gt)[0][0]
    assert det["injected"] is True
    assert det["track_id"] is None, "the tracker assigns ids, not us"
    assert det["class_id"] == 1          # "debris", R1 taxonomy
    assert det["class_name"] == "debris"
    assert 0.0 <= det["cx"] <= 1.0 and 0.0 <= det["cy"] <= 1.0   # R3
    assert math.isclose(det["cx"], gt.centres[0][0], abs_tol=1e-4)


def test_dense_stream_option_overrides_suppression(still, tmp_path):
    img, lab = still
    out = tmp_path / "seq"
    _, gt = seq.synthesise_sequence(
        img, out, frames=10, fps=5.0,
        occlusion=seq.Occlusion(3, 6) if lab else None, label_path=lab,
    )
    dense = seq.detections_from_ground_truth(gt, visible_only=False)
    assert all(len(f) == 1 for f in dense)


def test_injection_does_not_claim_detection_accuracy(still, tmp_path):
    """Injected detections are perfect by construction — they must be
    distinguishable so results cannot be misread as detector metrics (R7)."""
    img, lab = still
    out = tmp_path / "seq"
    _, gt = seq.synthesise_sequence(img, out, frames=3, fps=5.0, label_path=lab)
    dets = seq.detections_from_ground_truth(gt, conf=0.9)
    assert all(d["injected"] for f in dets for d in f)


# ---------------------------------------------------------------------------
# Pin: agreement with scripts/make_demo_video.py (out of bounds, R0)
# ---------------------------------------------------------------------------

def _demo_video_formula(dx, dy, fov, fps, w, h):
    """The formula inlined at scripts/make_demo_video.py:63 and :72-73.

    Reproduced here literally (the script is out of bounds and offers no
    callable) so that a change to either side is caught rather than silently
    shifting the ground truth the pipeline is scored against.
    """
    centres = [[0.5 + dx * i / w, 0.5 + dy * i / h] for i in range(30)]
    rate = [dx / w * fov * fps, dy / h * (fov * h / w) * fps]
    return centres, rate


def test_matches_make_demo_video_formula():
    """Our geometry must equal the reference script's — except the y-axis
    rate at non-square aspect, where we deliberately follow the library's
    exact ``arctan`` form instead of the script's small-angle approximation.
    """
    dx, dy, fov, fps, w, h = 6.0, -3.0, 60.0, 5.0, 1024, 1024
    motion = seq.Motion("linear", dx, dy)
    ref_centres, ref_rate = _demo_video_formula(dx, dy, fov, fps, w, h)

    centres = seq.compute_centres((0.5, 0.5), 30, w, h, motion)
    for (cx, cy), (rx, ry) in zip(centres, ref_centres):
        assert cx == pytest.approx(rx, abs=1e-12)
        assert cy == pytest.approx(ry, abs=1e-12)

    rate = seq.angular_rate_truth(motion, fps, fov, w, h)
    assert rate[0] == pytest.approx(ref_rate[0], rel=1e-12)   # x is identical everywhere
    # Square frame: the two vertical-FOV forms coincide, so y agrees too.
    assert rate[1] == pytest.approx(ref_rate[1], rel=1e-9)


def test_deliberately_departs_from_demo_video_at_16_9():
    """The one intended divergence, pinned so it cannot be 'fixed' by accident.

    ``make_demo_video.py:73`` uses ``fov * h / w`` for the vertical FOV. At
    1920x1080 that is 6.21% off the library's exact form — enough to consume
    most of R9's 10% budget before any tracking error is measured. We follow
    the library, because ``estimate_trajectory`` is what we are scored against.
    """
    dx, dy, fov, fps, w, h = 6.0, -3.0, 60.0, 5.0, 1920, 1080
    motion = seq.Motion("linear", dx, dy)
    _, ref_rate = _demo_video_formula(dx, dy, fov, fps, w, h)
    rate = seq.angular_rate_truth(motion, fps, fov, w, h)

    assert rate[0] == pytest.approx(ref_rate[0], rel=1e-12)   # x unaffected
    rel = abs(rate[1] - ref_rate[1]) / abs(rate[1]) * 100
    assert rel == pytest.approx(6.21, abs=0.01)
    # Ours must track the library, which is the side that scores us.
    assert rate[1] == pytest.approx(
        (dy / h) * fov_vertical_deg(fov, w, h) * fps, rel=1e-12
    )


# ---------------------------------------------------------------------------
# CLI surface (`synth`)
# ---------------------------------------------------------------------------

def _synth_cli(tmp_path, *argv):
    from tracking_using_drive.cli import main
    return main(["synth", "--out", str(tmp_path / "out"), "--json", *argv])


def test_cli_synth_emits_structured_json(still, tmp_path):
    """R12: assert on the JSON payload, not stdout."""
    img, _ = still
    assert _synth_cli(tmp_path, str(img), "--frames", "6") == 0
    payload = json.loads((tmp_path / "out" / "ground_truth.json").read_text("utf-8"))
    assert payload["synthetic"] is True            # R7
    assert payload["frames"] == 6
    assert len(payload["angular_rate_deg_s"]) == 2
    assert sorted(p.name for p in (tmp_path / "out").glob("frame_*.png")) == [
        f"frame_{i:04d}.png" for i in range(6)
    ]


def test_cli_synth_writes_video_on_request(still, tmp_path):
    img, _ = still
    assert _synth_cli(tmp_path, str(img), "--frames", "4", "--video") == 0
    assert (tmp_path / "out" / "sequence.mp4").exists()


def test_cli_synth_reports_occluded_frames(still, tmp_path):
    img, lab = still
    if lab is None:
        pytest.skip("no label available")
    assert _synth_cli(tmp_path, str(img), "--frames", "14",
                      "--occlude-start", "4", "--occlude-end", "9") == 0
    gt = seq.GroundTruth.load(tmp_path / "out" / "ground_truth.json")
    assert gt.occluded_frames == [4, 5, 6, 7, 8]
    assert len(gt.visible_frames) == 9


def test_cli_synth_rejects_gap_beyond_sequence(still, tmp_path):
    img, _ = still
    assert _synth_cli(tmp_path, str(img), "--frames", "5",
                      "--occlude-start", "2", "--occlude-end", "9") == 2


def test_cli_synth_rejects_missing_image(tmp_path):
    assert _synth_cli(tmp_path, str(tmp_path / "nope.jpg"), "--frames", "3") == 2


def test_cli_synth_auto_discovers_sibling_label_dir(still, tmp_path):
    """.../images/<split>/x.jpg should find .../labels/<split>/x.txt."""
    img, lab = still
    if lab is None:
        pytest.skip("no label available")
    # Place image + label in the canonical layout so the convention is
    # exercised regardless of where the fixture's still came from.
    layout = tmp_path / "ds" / "images" / "val"
    layout.mkdir(parents=True)
    copied = layout / img.name
    copied.write_bytes(img.read_bytes())
    labels = tmp_path / "ds" / "labels" / "val"
    labels.mkdir(parents=True)
    (labels / (img.stem + ".txt")).write_bytes(lab.read_bytes())
    assert _synth_cli(tmp_path, str(copied), "--frames", "3") == 0
    gt = seq.GroundTruth.load(tmp_path / "out" / "ground_truth.json")
    assert gt.label_path is not None and gt.label_path.endswith(".txt")


def test_cli_synth_does_not_guess_a_label_dir_that_is_absent(still, tmp_path):
    """Discovery must be a convention, not a hope: a missing sibling dir
    leaves label_path None rather than inventing a bbox."""
    img, _ = still
    layout = tmp_path / "lonely" / "images" / "val"
    layout.mkdir(parents=True)
    copied = layout / img.name
    copied.write_bytes(img.read_bytes())
    assert _synth_cli(tmp_path, str(copied), "--frames", "3") == 0
    gt = seq.GroundTruth.load(tmp_path / "out" / "ground_truth.json")
    assert gt.label_path is None
    # JSON round-trip yields lists; compare as floats.
    assert tuple(gt.start_centre) == pytest.approx((0.5, 0.5)), "no label means the image centre"


# ---------------------------------------------------------------------------
# Batch mode (D8: an error bar needs a spread, not one sample)
# ---------------------------------------------------------------------------

def test_cli_synth_batch_writes_index_and_subdirs(tmp_path):
    """D8: one sequence gives a single number with no spread."""
    assert _synth_cli(tmp_path, "--count", "4", "--frames", "5") == 0
    out = tmp_path / "out"
    index = json.loads((out / "index.json").read_text("utf-8"))
    assert index["synthetic"] is True                    # R7
    assert index["count"] == 4
    assert len(index["sequences"]) == 4
    for i, entry in enumerate(index["sequences"]):
        sub = out / f"seq_{i:03d}"
        assert sub.is_dir(), f"missing {sub}"
        assert (sub / "ground_truth.json").exists()
        assert len(list(sub.glob("frame_*.png"))) == 5
        # Each sequence has its own ground truth, not a shared one.
        gt = seq.GroundTruth.load(Path(entry["ground_truth"]))
        assert gt.frames == 5 and gt.synthetic is True


def test_cli_synth_batch_refuses_more_than_available(tmp_path, monkeypatch):
    """Failing early beats rendering 500 sequences then dying."""
    monkeypatch.setattr(
        "tracking_using_drive.source.is_still", lambda p: str(p).endswith(".jpg")
    )
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        (src / f"s{i}.jpg").write_bytes(b"")
    from tracking_using_drive.cli import main
    assert main(["synth", str(src), "--out", str(tmp_path / "o"),
                 "--count", "9", "--json"]) == 2


def test_cli_synth_batch_refuses_pinned_label(tmp_path):
    """One --label cannot describe N different stills; refuse rather than
    silently stamp the same bbox onto every sequence."""
    assert _synth_cli(tmp_path, "--count", "3", "--label", "x.txt") == 2


def test_cli_synth_batch_json_is_structured(tmp_path):
    """R12: the batch payload, not stdout, is what assertions read."""
    from tracking_using_drive.cli import main
    out = tmp_path / "out"
    assert main(["synth", "--out", str(out), "--count", "3",
                 "--frames", "4", "--json"]) == 0
    payload = json.loads((out / "index.json").read_text("utf-8"))
    assert payload["count"] == 3
    assert all(len(e["angular_rate_deg_s"]) == 2 for e in payload["sequences"])
