"""Phase 4 tests: tracking correctness across occlusion.

Covers the rules this phase exists to satisfy:
  R2  persist=True on every tracking call (tested against a fake model)
  R7  injected runs are flagged synthetic and never look like detector output
  R9  angular rate within 10% of ground truth, end to end
  R11 occlusion tested explicitly: the gap is present and the filter predicts
  R12 assert on the JSONL structure, not stdout

The suite needs no weights and no network. A still is drawn from the local
Drive cache when present, else a generated image stands in, so a fresh clone
runs it before any sync.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tracking_using_drive import sequence as seq
from tracking_using_drive.stream import (
    DEFAULT_MAX_MISSED,
    GATE_FLOOR_PX,
    GATE_SCALE,
    MIN_POINTS_FOR_ORBIT,
    TrackerEngine,
    assign_track_ids,
    iter_frames,
    run_stream,
)

_CACHE = Path(__file__).resolve().parents[1] / "_data" / "gdrive_cache"
_VAL_IMG = _CACHE / "yolo_dataset" / "images" / "val"
_VAL_LAB = _CACHE / "yolo_dataset" / "labels" / "val"
_HAS_CACHE = _VAL_IMG.is_dir() and any(_VAL_IMG.glob("*.jpg"))

FPS = 5.0
FOV = 60.0
GAP = (10, 20)
FRAMES = 30


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def occluded_run(tmp_path_factory):
    """Render the Phase 3 occlusion fixture once and share it.

    Module-scoped because rendering 30 PNGs per test would dominate the
    suite's runtime, and the fixture is immutable for the whole module.
    """
    import cv2

    if _HAS_CACHE:
        img = sorted(_VAL_IMG.glob("*.jpg"))[0]
        lab = _VAL_LAB / (img.stem + ".txt")
        label = lab if lab.exists() else None
    else:
        rng = np.random.default_rng(42)
        img = tmp_path_factory.mktemp("still") / "generated.png"
        cv2.imwrite(str(img), rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
        label = None

    out = tmp_path_factory.mktemp("occluded")
    occl = seq.Occlusion(*GAP) if label is not None else None
    _, gt = seq.synthesise_sequence(
        img, out, frames=FRAMES,
        motion=seq.Motion("linear", 6.0, -3.0),
        fps=FPS, fov_deg=FOV, occlusion=occl, label_path=label,
    )
    gt.save(out / "ground_truth.json")
    stream = seq.detections_from_ground_truth(gt)
    return {"dir": out, "gt": gt, "stream": stream, "has_label": label is not None}


@pytest.fixture(scope="module")
def run(occluded_run, tmp_path_factory):
    """Execute the injected pipeline once over the occlusion fixture."""
    root = tmp_path_factory.mktemp("run4")
    jsonl = root / "run.jsonl"
    report = root / "orbit.json"
    summary = run_stream(
        source=str(occluded_run["dir"]),
        detections=occluded_run["stream"],
        fps=FPS, fov_deg=FOV, orbit=True,
        out_jsonl=str(jsonl),
        orbit_report_path=str(report),
        max_missed=30,
    )
    records = [json.loads(line) for line in jsonl.read_text("utf-8").splitlines()
               if line.strip()]
    orbit = json.loads(report.read_text("utf-8")) if report.exists() else []
    return {"summary": summary, "records": records, "orbit": orbit,
            "jsonl": jsonl, "report": report, "gt": occluded_run["gt"]}


def _gap_records(records):
    return [r for r in records if GAP[0] <= r["frame"] < GAP[1]]


# ---------------------------------------------------------------------------
# Exit criterion 1: the id survives the gap
# ---------------------------------------------------------------------------


def test_id_survives_the_gap(run):
    ids = {d["track_id"] for r in run["records"] for d in r["detections"]}
    assert len(ids) == 1, f"track fragmented: ids {sorted(ids)}"
    assert None not in ids


def test_gap_frames_are_empty_but_predict_the_track(run):
    """R11: detections stop, prediction does not."""
    gap = _gap_records(run["records"])
    assert len(gap) == GAP[1] - GAP[0], "the gap must be present and exact"
    for rec in gap:
        assert rec["detections"] == [], f"frame {rec['frame']} should see nothing"
        assert rec["predicted_track_ids"] == [1], (
            f"frame {rec['frame']} must advance track 1 — a frozen filter is "
            "the defect this phase fixes"
        )


def test_track_is_counted_once_in_the_orbit_report(run):
    """Not twice: fragmentation would show as two short entries."""
    ids = [entry["track_id"] for entry in run["orbit"]]
    assert len(ids) == 1, f"expected one orbit entry, got {ids}"
    assert run["summary"]["orbit_tracks"] == 1


def test_summary_counts_the_gap_as_missed(run):
    summary = run["summary"]
    assert summary["missed_frames_total"] == GAP[1] - GAP[0]
    assert summary["expired_tracks"] == 0
    (track,) = summary["tracks"]
    assert track["n_missed"] == 10
    assert track["observations"] == FRAMES - 10
    assert track["expired"] is False


# ---------------------------------------------------------------------------
# Exit criterion 2: it predicted, it did not freeze
# ---------------------------------------------------------------------------


def test_motion_is_continuous_across_the_gap(run):
    """The velocity estimate must not collapse while the object is hidden."""
    before = [d for r in run["records"] if r["frame"] == GAP[0] - 1
              for d in r["detections"]][0]["motion"]
    after = [d for r in run["records"] if r["frame"] == GAP[1]
             for d in r["detections"]][0]["motion"]
    assert before["speed_norm_s"] > 0, "filter was frozen before the gap"
    assert after["speed_norm_s"] > 0, "filter froze during the gap"
    # Same object, same motion: speed must not drift materially.
    rel = abs(after["speed_norm_s"] - before["speed_norm_s"]) / before["speed_norm_s"]
    assert rel < 0.10, f"speed jumped {rel:.1%} across the gap"


def test_filter_state_advances_through_every_gap_frame(occluded_run):
    """Direct check on the bank: position moves each frame it is unobserved."""
    engine = TrackerEngine(rate=FPS, max_missed=30)
    positions = []
    for i, frame in enumerate(occluded_run["stream"]):
        engine.predict()
        dets = [dict(d) for d in frame]
        assign_track_ids(dets, engine, 1024, 1024)
        for d in dets:
            engine.observe(int(d["track_id"]), float(d["cx"]), float(d["cy"]),
                           i / FPS)
        engine.end_frame({int(d["track_id"]) for d in dets
                          if d.get("track_id") is not None})
        pos = engine.bank.filters[1].position
        positions.append((float(pos[0]), float(pos[1])))

    gap_pos = positions[GAP[0]:GAP[1]]
    steps = [math.hypot(gap_pos[i + 1][0] - gap_pos[i][0],
                        gap_pos[i + 1][1] - gap_pos[i][1])
             for i in range(len(gap_pos) - 1)]
    assert all(s > 0 for s in steps), "the filter froze inside the gap"
    # Constant-velocity motion -> every step the same size.
    assert max(steps) - min(steps) < 1e-9, f"uneven prediction: {steps}"


def test_naive_miss_plus_update_would_corrupt_velocity():
    """Guard the design decision itself.

    ``update()`` predicts with ``dt`` before correcting, and ``miss()`` does
    not append to history. Pairing them across a gap therefore advances the
    filter once per missed frame *and* once more for the whole gap. This test
    pins the measurement so nobody "simplifies" the engine back into it.
    """
    from space_debris.utils.orbit import TrajectoryBank

    truth = 0.06

    def velocity(naive: bool) -> float:
        bank = TrajectoryBank()
        for f in range(10):
            bank.update(1, 0.4 + truth * f, 0.4, float(f))
        for _ in range(10):
            bank.miss(1, dt=1.0)
        bank.update(1, 0.4 + truth * 20, 0.4, 20.0, **({} if naive else {"dt": 0.0}))
        return float(bank.filters[1].velocity[0])

    naive_err = abs(velocity(True) - truth) / truth
    ours_err = abs(velocity(False) - truth) / truth
    assert naive_err > 0.50, f"expected the naive form to be badly wrong, got {naive_err:.1%}"
    assert ours_err < naive_err, "predict-then-correct must beat the naive pairing"


# ---------------------------------------------------------------------------
# Non-vacuity: the assertion above must be able to fail
# ---------------------------------------------------------------------------


def _ids_with_frozen_prediction(occluded_run):
    """Replay the fixture with prediction switched off."""
    engine = TrackerEngine(rate=FPS, max_missed=30)
    seen = []
    for i, frame in enumerate(occluded_run["stream"]):
        for tid in list(engine.bank.ids()):
            state = engine.states.get(tid)
            if state is None or state.expired:
                continue
            engine.bank.filters[tid].x[2:] = 0.0   # freeze: no advance, no velocity
        dets = [dict(d) for d in frame]
        assign_track_ids(dets, engine, 1024, 1024)
        for d in dets:
            engine.observe(int(d["track_id"]), float(d["cx"]), float(d["cy"]),
                           i / FPS)
        engine.end_frame({int(d["track_id"]) for d in dets
                          if d.get("track_id") is not None})
        if dets:
            seen.append((i, dets[0]["track_id"]))
    return seen


def test_id_survival_would_fail_without_prediction(occluded_run):
    """Prove the exit criterion is not a tautology.

    A constant-id injection would "preserve" the id no matter what the filter
    did. With nearest-neighbour association against *predicted* positions, a
    frozen track sits ~67 px from the re-appearing object — outside the gate —
    so the id must be lost. If this test ever passes, the gate has grown loose
    enough to make criterion 1 meaningless.
    """
    if not occluded_run["has_label"]:
        pytest.skip("no label, so no occlusion fixture")

    before = _ids_with_frozen_prediction(occluded_run)
    pre_gap = [tid for i, tid in before if i == GAP[0] - 1]
    post_gap = [tid for i, tid in before if i == GAP[1]]
    assert pre_gap and post_gap
    assert pre_gap[0] != post_gap[0], (
        "the id survived even with prediction frozen — the association gate "
        "is too loose for criterion 1 to mean anything"
    )


def test_gate_is_sized_from_observed_motion(occluded_run):
    """Gate ~4 observed steps, so a 10-frame gap is comfortably outside it."""
    engine = TrackerEngine(rate=FPS, max_missed=30)
    frame = occluded_run["stream"][0]
    dets = [dict(d) for d in frame]
    assign_track_ids(dets, engine, 1024, 1024)
    engine.observe(int(dets[0]["track_id"]), float(dets[0]["cx"]),
                   float(dets[0]["cy"]), 0.0)
    # A single observation has no motion evidence -> the floor applies.
    assert engine.gate_px(1, 1024, 1024) == GATE_FLOOR_PX

    engine.observe(1, float(dets[0]["cx"]) + 6 / 1024,
                   float(dets[0]["cy"]) - 3 / 1024, 1 / FPS)
    gate = engine.gate_px(1, 1024, 1024)
    step = engine.step_px(1, 1024, 1024)
    assert gate == max(GATE_FLOOR_PX, GATE_SCALE * step)
    # 10 frames of travel must exceed the gate, or the id could never be lost.
    travel = step * (GAP[1] - GAP[0])
    assert travel > gate, f"gap travel {travel:.1f}px inside gate {gate:.1f}px"


# ---------------------------------------------------------------------------
# R9: angular rate recovered end to end
# ---------------------------------------------------------------------------


def test_angular_rate_recovered_within_10pct(run):
    """R9 acceptance criterion #5, now measured through the tracker."""
    assert len(run["orbit"]) == 1
    got = run["orbit"][0]["trajectory"]["angular_rate_deg_s"]
    want = run["gt"].angular_rate_deg_s

    for est, truth in zip(got, want):
        if abs(truth) < 1e-9:
            assert abs(est) < 1e-6
            continue
        err = abs(est - truth) / abs(truth) * 100
        assert err < 10.0, f"{err:.3f}% exceeds R9 tolerance"


def test_orbit_report_states_its_assumptions(run):
    """R7/R8: an orbit claim must declare what it assumed and what is missing."""
    traj = run["orbit"][0]["trajectory"]
    assert traj["assumptions"]["fov_deg_horizontal"] == FOV
    assert "range_km_needed_for_physical_speed" in traj["missing"]
    assert traj["speed_km_s"] is None, "no range was supplied, so no speed"


# ---------------------------------------------------------------------------
# R7: injected output must never pass as detector output
# ---------------------------------------------------------------------------


def test_injected_run_is_flagged_and_loads_no_weights(run):
    summary = run["summary"]
    assert summary["detection_source"] == "injected"
    assert summary["model"] is None, "injection must not load weights"
    assert summary["synthetic"] is True, "R7"
    assert summary["tracker"] is None


def test_every_detection_carries_provenance(run):
    dets = [d for r in run["records"] for d in r["detections"]]
    assert dets, "fixture produced no detections"
    for d in dets:
        assert d["class_id"] == 1
        assert d["final_class"] == "debris"
        assert d["detector_class"] == "debris", (
            "injected detections must fuse to their real class, not 'unknown'"
        )
        assert d["injected"] is True, "R7: provenance must survive fusion"
        assert 0.0 <= d["cx"] <= 1.0 and 0.0 <= d["cy"] <= 1.0   # R3


def test_records_expose_predicted_ids_for_offline_scoring(run):
    """R12: 'did it predict?' must be answerable from the file alone."""
    for rec in run["records"]:
        assert "predicted_track_ids" in rec
        assert isinstance(rec["predicted_track_ids"], list)


def test_records_carry_normalised_centres(run):
    """``orbit_from_jsonl`` reads ``cx``/``cy``; the original never wrote them."""
    for rec in run["records"]:
        for det in rec["detections"]:
            assert "cx" in det and "cy" in det
            assert 0.0 <= det["cx"] <= 1.0


# ---------------------------------------------------------------------------
# Geometry: first frame, not last
# ---------------------------------------------------------------------------


def test_geometry_comes_from_the_first_frame(tmp_path):
    """The original took width/height from the last frame, which mis-fits a
    mixed-size source."""
    import cv2

    src = tmp_path / "mixed"
    src.mkdir()
    sizes = [(100, 80), (300, 240), (300, 240)]
    for i, (w, h) in enumerate(sizes):
        cv2.imwrite(str(src / f"f{i:03d}.png"), np.full((h, w, 3), 40, np.uint8))

    stream = [[{"track_id": None, "class_id": 1, "class_name": "debris",
                "conf": 0.9, "cx": 0.5, "cy": 0.5,
                "xyxy": [10, 10, 50, 50]}] for _ in sizes]

    summary = run_stream(source=str(src), detections=stream, fps=FPS,
                         orbit=False, max_missed=30)
    assert summary["image_size"] == [100, 80], (
        f"expected the FIRST frame's geometry, got {summary['image_size']}"
    )


def test_empty_run_is_reported_not_crashed(tmp_path):
    """A run that processes no frames must not crash building the report.

    ``max_frames=0`` is the reachable zero-frame path: ``resolve_source``
    already refuses an empty directory up front (Phase 2), so a genuinely
    empty source never gets this far.
    """
    import cv2

    src = tmp_path / "some"
    src.mkdir()
    cv2.imwrite(str(src / "a.png"), np.zeros((8, 8, 3), np.uint8))

    summary = run_stream(source=str(src), detections=[[]], fps=FPS,
                         orbit=True, max_frames=0)
    assert summary["frames"] == 0
    assert summary["image_size"] is None
    assert summary["orbit_tracks"] == 0
    assert summary["tracks"] == []
    assert summary["detections"] == 0


# ---------------------------------------------------------------------------
# Expiry (max_missed)
# ---------------------------------------------------------------------------


def test_track_expires_after_max_missed_frames():
    engine = TrackerEngine(rate=FPS, max_missed=3)
    engine.observe(1, 0.5, 0.5, 0.0)

    for frame in range(1, 6):
        engine.predict()
        engine.end_frame(set())
        state = engine.states[1]
        if frame <= 3:
            assert state.expired is False, f"expired too early at frame {frame}"
        else:
            assert state.expired is True, f"never expired by frame {frame}"

    # Expired tracks stop advancing and are not resurrected.
    before = engine.bank.filters[1].position.copy()
    engine.predict()
    after = engine.bank.filters[1].position
    assert np.allclose(before, after), "an expired track must be frozen"
    engine.observe(1, 0.9, 0.9, 10.0)
    assert engine.states[1].expired is True
    assert engine.states[1].observations == 1, "expired ids must not be revived"


def test_default_max_missed_is_unchanged_from_plan():
    assert DEFAULT_MAX_MISSED == 30


def test_engine_rejects_nonsense_parameters():
    with pytest.raises(ValueError, match="rate"):
        TrackerEngine(rate=0.0)
    with pytest.raises(ValueError, match="rate"):
        TrackerEngine(rate=-1.0)
    with pytest.raises(ValueError, match="max_missed"):
        TrackerEngine(rate=FPS, max_missed=-1)


def test_expired_track_is_excluded_from_orbit_report(tmp_path):
    """Excluded from the report, but history retained (still inspectable)."""
    engine = TrackerEngine(rate=FPS, max_missed=2, orbit_enabled=True)
    for i in range(4):
        engine.observe(1, 0.4 + 0.01 * i, 0.4, i / FPS)
    for _ in range(4):
        engine.predict()
        engine.end_frame(set())
    assert engine.states[1].expired is True
    assert engine.bank.get_history(1), "history must survive expiry"
    # Counting stops at expiry: it took max_missed + 1 absent frames to
    # exceed the threshold, and the 4th absent frame adds nothing.
    assert engine.states[1].n_missed == 3
    assert engine.states[1].consecutive_misses == 3

    report = engine.orbit_report(width=1024, height=1024, fov_deg=FOV,
                                 range_km=None)
    assert report == [], "expired tracks must not reach the orbit report"
    # ...but the inventory still shows it, or the drop would be invisible.
    (entry,) = engine.inventory()
    assert entry["expired"] is True
    assert entry["n_missed"] == 3


def test_short_track_is_withheld_from_orbit_report():
    engine = TrackerEngine(rate=FPS, max_missed=30)
    for i in range(MIN_POINTS_FOR_ORBIT - 1):
        engine.observe(1, 0.5 + 0.01 * i, 0.5, i / FPS)
    report = engine.orbit_report(width=1024, height=1024, fov_deg=FOV,
                                 range_km=None)
    assert report == [], f"{MIN_POINTS_FOR_ORBIT - 1} points is below the minimum"
    assert engine.inventory()[0]["observations"] == MIN_POINTS_FOR_ORBIT - 1


def test_multiple_tracks_are_reported_separately():
    engine = TrackerEngine(rate=FPS, max_missed=30)
    for i in range(5):
        engine.predict()
        engine.observe(1, 0.3 + 0.01 * i, 0.4, i / FPS)
        engine.observe(2, 0.7, 0.6 + 0.01 * i, i / FPS)
        engine.end_frame({1, 2})
    report = engine.orbit_report(width=1024, height=1024, fov_deg=FOV,
                                 range_km=None)
    assert [e["track_id"] for e in report] == [1, 2]
    assert all(e["observations"] == 5 for e in report)


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------


def test_association_matches_nearest_predicted_position():
    engine = TrackerEngine(rate=FPS, max_missed=30)
    # Establish a track moving +6px/frame in x (normalised 6/1024).
    step = 6 / 1024
    for i in range(3):
        engine.predict()
        engine.observe(1, 0.4 + step * i, 0.5, i / FPS)
        engine.end_frame({1})

    engine.predict()
    # Object reappears 6px from the *predicted* position, not the last-seen one.
    dets = [{"track_id": None, "class_id": 1, "class_name": "debris", "conf": 0.9,
             "cx": 0.4 + step * 3, "cy": 0.5, "xyxy": [0, 0, 10, 10]}]
    assign_track_ids(dets, engine, 1024, 1024)
    assert dets[0]["track_id"] == 1, "nearest prediction should keep the id"


def test_association_allocates_a_new_id_when_nothing_matches():
    engine = TrackerEngine(rate=FPS, max_missed=30)
    engine.observe(1, 0.1, 0.1, 0.0)
    dets = [{"track_id": None, "class_id": 1, "class_name": "debris", "conf": 0.9,
             "cx": 0.9, "cy": 0.9, "xyxy": [0, 0, 10, 10]}]
    assign_track_ids(dets, engine, 1024, 1024)
    assert dets[0]["track_id"] == 2, "a far-away object is a different track"


def test_association_honours_caller_supplied_ids():
    engine = TrackerEngine(rate=FPS, max_missed=30)
    dets = [{"track_id": 7, "class_id": 1, "class_name": "debris", "conf": 0.9,
             "cx": 0.5, "cy": 0.5, "xyxy": [0, 0, 10, 10]}]
    assign_track_ids(dets, engine, 1024, 1024)
    assert dets[0]["track_id"] == 7, "a supplied id must not be reassigned"


def test_association_does_not_mutate_the_caller_stream(occluded_run):
    """The injected stream is shared; association writes into its copies."""
    engine = TrackerEngine(rate=FPS, max_missed=30)
    original = occluded_run["stream"][0]
    assert all(d["track_id"] is None for d in original), (
        "the source stream must stay untouched for reuse"
    )
    copies = [dict(d) for d in original]
    assign_track_ids(copies, engine, 1024, 1024)
    assert all(d["track_id"] is None for d in original)


# ---------------------------------------------------------------------------
# Injection contract
# ---------------------------------------------------------------------------


def test_injection_requires_an_explicit_source():
    with pytest.raises(ValueError, match="explicit --source"):
        run_stream(detections=[[]], fps=FPS)


def test_injection_refuses_a_camera_index():
    with pytest.raises(ValueError, match="camera index"):
        run_stream(source="0", detections=[[]], fps=FPS)


def test_missing_source_directory_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="cannot read frames"):
        list(iter_frames(tmp_path / "nope"))


def test_directory_with_no_images_fails_loudly(tmp_path):
    (tmp_path / "d").mkdir()
    with pytest.raises(ValueError, match="no images"):
        list(iter_frames(tmp_path / "d"))


def test_unreadable_image_fails_rather_than_skipping(tmp_path):
    """Skipping would desynchronise the frame index from the time base."""
    import cv2

    src = tmp_path / "bad"
    src.mkdir()
    cv2.imwrite(str(src / "a.png"), np.zeros((8, 8, 3), np.uint8))
    (src / "b.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="unreadable"):
        list(iter_frames(src))


def test_max_frames_truncates_iteration(tmp_path):
    import cv2

    src = tmp_path / "many"
    src.mkdir()
    for i in range(10):
        cv2.imwrite(str(src / f"{i:03d}.png"), np.zeros((8, 8, 3), np.uint8))
    assert len(list(iter_frames(src, max_frames=4))) == 4
    assert len(list(iter_frames(src))) == 10


# ---------------------------------------------------------------------------
# R2: persist=True on every tracking call
# ---------------------------------------------------------------------------


class _FakeArray:
    """Mimics a torch tensor closely enough for the box-access pattern."""

    def __init__(self, values):
        self._values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _FakeBoxes:
    def __init__(self, n):
        self.xyxy = _FakeArray([[10, 10, 50, 50]] * n)
        self.conf = _FakeArray([0.9] * n)
        self.cls = _FakeArray([1] * n)
        self.id = _FakeArray([1] * n)
        self.xywhn = _FakeArray([[0.5, 0.5, 0.1, 0.1]] * n)

    def __len__(self):
        return len(self.xyxy.numpy())


class _FakeResult:
    def __init__(self, n=1):
        self.orig_img = np.zeros((64, 64, 3), np.uint8)
        self.boxes = _FakeBoxes(n)
        self.path = "fake.png"


class _FakeYOLO:
    """Records every kwargs dict handed to track()/predict()."""
    calls: list[dict] = []

    def __init__(self, weights=None):
        self.names = {0: "cheops", 1: "debris"}

    def track(self, **kwargs):
        _FakeYOLO.calls.append(kwargs)
        return iter([_FakeResult() for _ in range(3)])

    def predict(self, **kwargs):
        _FakeYOLO.calls.append(kwargs)
        return iter([_FakeResult() for _ in range(3)])


def test_persist_is_true_on_every_tracking_call(tmp_path, monkeypatch):
    """R2: without persist=True every frame's identity is destroyed."""
    import ultralytics

    dummy = tmp_path / "weights.pt"
    dummy.write_bytes(b"x")

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    _FakeYOLO.calls.clear()

    summary = run_stream(model_path=str(dummy), source=str(tmp_path),
                         fps=FPS, orbit=False, max_missed=30)
    assert _FakeYOLO.calls, "model.track()/predict() was never called"
    for call in _FakeYOLO.calls:
        assert call.get("persist") is True, (
            f"R2 violated: persist={call.get('persist')!r} on a tracking call"
        )
    assert summary["detection_source"] == "model"
    assert summary["frames"] == 3


def test_model_path_without_tracker_does_not_claim_persist(tmp_path, monkeypatch):
    """predict() has no tracker to persist; R2 constrains tracking calls."""
    import ultralytics

    dummy = tmp_path / "weights.pt"
    dummy.write_bytes(b"x")
    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    _FakeYOLO.calls.clear()

    run_stream(model_path=str(dummy), source=str(tmp_path), tracker=None,
               fps=FPS, orbit=False)
    assert _FakeYOLO.calls
    for call in _FakeYOLO.calls:
        assert "tracker" not in call or call["tracker"] is None


# ---------------------------------------------------------------------------
# Summary shape
# ---------------------------------------------------------------------------


def test_summary_records_the_detection_path_and_geometry(run):
    summary = run["summary"]
    assert summary["frames"] == FRAMES
    assert summary["fps_assumed"] == FPS
    assert summary["image_size"] == [1024, 1024]
    assert summary["detections"] == FRAMES - (GAP[1] - GAP[0])
    assert summary["num_tracks"] == 1
    assert summary["max_missed"] == 30


def test_jsonl_has_one_record_per_frame(run):
    records = run["records"]
    assert len(records) == FRAMES
    assert [r["frame"] for r in records] == list(range(FRAMES))
    assert records[0]["t"] == pytest.approx(0.0)
    assert records[-1]["t"] == pytest.approx((FRAMES - 1) / FPS)


def test_class_counts_match_detections(run):
    summary = run["summary"]
    assert sum(summary["class_counts"].values()) == summary["detections"]
    assert summary["class_counts"].get("debris") == summary["detections"]


# ---------------------------------------------------------------------------
# CLI surface (`stream`, `orbit`)
# ---------------------------------------------------------------------------


def _run_cli(*argv):
    from tracking_using_drive.cli import main
    return main(list(argv))


def test_cli_stream_from_ground_truth_is_structured(tmp_path, occluded_run, capsys):
    """R12: assert on the JSON payload, not stdout."""
    out = tmp_path / "run.jsonl"
    report = tmp_path / "orbit.json"
    rc = _run_cli("stream",
                  "--ground-truth", str(occluded_run["dir"] / "ground_truth.json"),
                  "--out", str(out), "--orbit-report", str(report), "--json")
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["detection_source"] == "injected"
    assert payload["synthetic"] is True, "R7"
    assert payload["model"] is None, "the D13 path must not need weights"
    assert payload["frames"] == FRAMES
    assert payload["num_tracks"] == 1
    assert payload["orbit_tracks"] == 1
    assert payload["missed_frames_total"] == 10
    assert out.exists() and report.exists()


def test_cli_stream_takes_fps_and_fov_from_ground_truth(tmp_path, occluded_run, capsys):
    """A mismatched fps would make R9 measure an argument error, not tracking."""
    gt = occluded_run["dir"] / "ground_truth.json"
    assert _run_cli("stream", "--ground-truth", str(gt), "--fps", "999",
                    "--fov", "1", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["fps_assumed"] == occluded_run["gt"].fps, (
        "explicit --fps must lose to the ground truth's own value"
    )


def test_cli_orbit_replays_the_stream_jsonl(tmp_path, occluded_run, capsys):
    out = tmp_path / "run.jsonl"
    assert _run_cli("stream", "--ground-truth",
                    str(occluded_run["dir"] / "ground_truth.json"),
                    "--out", str(out), "--json") == 0
    capsys.readouterr()

    rc = _run_cli("orbit", str(out), "--fov", "60", "--json")
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tracks"] == 1
    assert payload["frames"] == FRAMES
    assert payload["image_size"] == [1024, 1024]


def test_cli_stream_without_weights_fails_actionably(tmp_path, capsys, monkeypatch):
    """The weights blocker is real; the message must say what to do about it."""
    import tracking_using_drive.cli as cli_mod

    # Point runs/ at an empty dir so resolve_model finds nothing.
    monkeypatch.setattr("space_debris.config.RUNS_DIR", tmp_path / "runs", raising=False)
    src = tmp_path / "frames"
    src.mkdir()
    import cv2
    cv2.imwrite(str(src / "a.png"), np.zeros((8, 8, 3), np.uint8))

    rc = _run_cli("stream", str(src), "--json")
    assert rc == 2
    err = capsys.readouterr().err
    assert "ERROR" in err
    assert "--model" in err or "weights" in err, (
        f"the message must point at the fix, got: {err}"
    )


def test_cli_stream_honours_max_missed(tmp_path, occluded_run, capsys):
    """A tighter budget expires the track and SPLITS the object in two.

    The gap is 10 frames, so ``max_missed=3`` cannot bridge it: track 1
    (frames 0-9) expires, and when the object returns at frame 20 it is a
    *new* track 2 with 10 fresh observations. Track 2 is neither expired nor
    too short, so it still reaches the orbit report — the split shows up as
    ``tracks == 2`` with ``orbit_tracks == 1``, not as an empty report.
    """
    gt = occluded_run["dir"] / "ground_truth.json"
    rc = _run_cli("stream", "--ground-truth", str(gt), "--max-missed", "3",
                  "--json")
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["max_missed"] == 3

    # The gap splits one object into two tracks.
    assert payload["num_tracks"] == 2, (
        "a 10-frame gap with max_missed=3 must fragment the track"
    )
    assert payload["expired_tracks"] == 1, "track 1 must expire in the gap"
    first, second = payload["tracks"]
    assert (first["track_id"], first["observations"], first["expired"]) == (1, 10, True)
    assert (second["track_id"], second["observations"], second["expired"]) == (2, 10, False)
    # Only the surviving fragment is reportable: the expired one is excluded,
    # but the new one has 10 obs (>= MIN_POINTS_FOR_ORBIT), so one entry ships.
    assert payload["orbit_tracks"] == 1, (
        "only the non-expired fragment may reach the orbit report"
    )


def test_cli_stream_default_max_missed_bridges_the_same_gap(tmp_path, occluded_run, capsys):
    """The contrasting case: default 30 bridges 10 frames, so no split.

    Without this, the assertion above would prove only that a small number is
    smaller than a gap — not that the *default* configuration preserves the
    id. Together the pair pins the boundary the parameter controls.
    """
    gt = occluded_run["dir"] / "ground_truth.json"
    assert _run_cli("stream", "--ground-truth", str(gt), "--max-missed", "30",
                    "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["max_missed"] == 30
    assert payload["num_tracks"] == 1, "30 frames must bridge a 10-frame gap"
    assert payload["expired_tracks"] == 0
    assert payload["tracks"][0]["n_missed"] == 10
    assert payload["orbit_tracks"] == 1
