"""Phase 6 tests — end-to-end validation and honest reporting.

Covers the rules this phase exists to satisfy:
  R7  every synthetic artifact is flagged and never presented as measured
  R9  angular rate within 10% of ground truth, through the whole chain
  R11 occlusion tested explicitly, at chain level
  R12 assert on the JSONL / report, not stdout

The suite runs the full D13 chain (Drive still -> synthesise -> inject ->
associate -> Kalman -> orbit) without weights or network.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tracking_using_drive.validate import (
    MISSING_RANGE_REASON,
    audit_synthetic_flags,
    compare_continuity,
    run_validation,
    track_continuity,
)

_CACHE = Path(__file__).resolve().parents[1] / "_data" / "gdrive_cache"
_VAL_IMG = _CACHE / "yolo_dataset" / "images" / "val"
_HAS_CACHE = _VAL_IMG.is_dir() and any(_VAL_IMG.glob("*.jpg"))

FPS = 5.0
FRAMES = 30
GAP = (10, 20)


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    """Run the whole chain once and share the report across the module.

    Rendering 30 frames plus a 12-point sweep per test would dominate the
    suite; the report is immutable, so module scope is safe.
    """
    if not _HAS_CACHE:
        pytest.skip("no cached Drive stills — run `sync` first")
    out = tmp_path_factory.mktemp("validate")
    return run_validation(out, frames=FRAMES, fps=FPS, occlusion=GAP)


# ---------------------------------------------------------------------------
# The chain itself
# ---------------------------------------------------------------------------


def test_full_chain_runs_end_to_end(report):
    """D13: Drive -> synthesise -> inject -> track -> Kalman -> orbit."""
    assert report["chain"] == [
        "drive_sync", "synthesise", "inject_ground_truth",
        "associate", "kalman", "orbit",
    ]
    assert report["frames"] == FRAMES
    assert report["tracking"]["detection_source"] == "injected"
    assert report["tracking"]["image_size"] == [1024, 1024]
    # Every stage left an artifact behind.
    for key in ("frames", "ground_truth", "jsonl", "orbit",
                "baseline_jsonl", "report"):
        assert report["artifacts"][key], f"missing artifact: {key}"
        assert Path(report["artifacts"][key]).exists(), (
            f"artifact {key} not written: {report['artifacts'][key]}"
        )


def test_jsonl_has_one_record_per_frame(report):
    """R12: assertions read the file, not a summary dict."""
    records = [
        json.loads(line)
        for line in Path(report["artifacts"]["jsonl"])
        .read_text("utf-8").splitlines() if line.strip()
    ]
    assert len(records) == FRAMES
    assert [r["frame"] for r in records] == list(range(FRAMES))
    # The occlusion is present and exact.
    gap = [r for r in records if GAP[0] <= r["frame"] < GAP[1]]
    assert len(gap) == GAP[1] - GAP[0]
    assert all(r["detections"] == [] for r in gap)
    assert all(r["predicted_track_ids"] == [1] for r in gap)


def test_orbit_entry_states_assumptions_and_missing(report):
    """Every orbit entry declares what it assumed and what it lacks."""
    (entry,) = report["orbit"]
    assert entry["missing"] == [MISSING_RANGE_REASON]
    assert entry["assumptions"]["range_km"] is None
    assert entry["assumptions"]["fov_deg_horizontal"] == 60.0
    assert entry["speed_km_s"] is None, "no range in, no speed out"


def test_orbit_entry_carries_class_and_debris_flag(report):
    """An orbit entry must say whether the track it describes is debris."""
    (entry,) = report["orbit"]
    assert entry["final_class"] == "debris"
    assert entry["is_debris"] is True
    assert entry["observations"] == FRAMES - (GAP[1] - GAP[0])
    assert entry["n_missed"] == GAP[1] - GAP[0]


# ---------------------------------------------------------------------------
# R9 — rate recovery through the whole chain
# ---------------------------------------------------------------------------


def test_rate_recovered_within_10pct(report):
    """R9 / acceptance criterion #5."""
    rate = report["rate_recovery"]
    assert rate["within_tolerance"] is True, rate
    assert rate["error_pct"] < 10.0
    # Not merely "under 10%" — it should be near-exact, since injected
    # detections sit on the ground-truth centres.
    assert rate["error_pct"] < 1.0, (
        f"expected sub-1% recovery on injected detections, got "
        f"{rate['error_pct']}%"
    )
    for got, want in zip(rate["recovered_deg_s"], rate["truth_deg_s"]):
        assert abs(got - want) / abs(want) < 0.01


# ---------------------------------------------------------------------------
# Track continuity vs the degenerate baseline
# ---------------------------------------------------------------------------


def test_continuity_improves_on_the_degenerate_baseline(report):
    """The Phase 6 exit criterion: longer tracks, fewer of them."""
    c = report["continuity"]
    base, ours = c["baseline"], c["synthetic_sequence"]

    # Baseline is genuinely degenerate: independent stills -> length-1 tracks.
    assert base["mean_track_length"] == 1.0, base
    assert base["tracks"] == FRAMES, "every frame should be its own track"
    assert base["orbit_eligible"] == 0, (
        "nothing in the baseline reaches MIN_POINTS_FOR_ORBIT"
    )

    # Ours is one track across the whole sequence, occlusion included.
    assert ours["tracks"] == 1, ours
    assert ours["mean_track_length"] == FRAMES - (GAP[1] - GAP[0])
    assert ours["orbit_eligible"] == 1

    assert c["improved"] is True
    assert c["delta_mean_track_length"] > 0
    assert c["delta_tracks"] < 0


def test_track_continuity_helper_shapes():
    """Unit-check the metric itself, independent of a run."""
    empty = track_continuity([])
    assert empty["tracks"] == 0 and empty["mean_track_length"] == 0.0

    inv = [
        {"observations": 10, "expired": False},
        {"observations": 4, "expired": False},
        {"observations": 2, "expired": True},
    ]
    got = track_continuity(inv)
    assert got["tracks"] == 3
    assert got["mean_track_length"] == pytest.approx(5.333, abs=1e-3)
    assert got["max_track_length"] == 10
    assert got["fragmented_tracks"] == 1
    assert got["orbit_eligible"] == 2      # the expired one does not qualify


def test_compare_continuity_is_honest_about_the_direction():
    """A regression must not be reported as an improvement."""
    base = track_continuity([{"observations": 1, "expired": False}] * 10)
    worse = track_continuity([{"observations": 1, "expired": False}] * 10)
    assert compare_continuity(base, worse)["improved"] is False

    better = track_continuity([{"observations": 8, "expired": False}] * 2)
    c = compare_continuity(base, better)
    assert c["improved"] is True
    assert c["delta_mean_track_length"] == pytest.approx(7.0)
    assert c["delta_tracks"] == -8


# ---------------------------------------------------------------------------
# The FOV / range sweep
# ---------------------------------------------------------------------------


def test_sweep_has_all_points(report):
    rows = report["sweep"]["rows"]
    assert len(rows) == 12, "3 FOVs x 4 ranges"
    assert {r["fov_deg"] for r in rows} == {45.0, 60.0, 90.0}
    assert any(r["range_km"] is None for r in rows)


def test_no_range_means_no_speed_and_a_recorded_reason(report):
    """Never invent a number: null speed plus the reason, everywhere."""
    sweep = report["sweep"]
    assert sweep["all_speed_null_without_range"] is True
    assert sweep["all_missing_range_reason"] is True
    for row in sweep["rows"]:
        if row["range_km"] is None:
            assert row["speed_km_s"] is None
            assert MISSING_RANGE_REASON in row["missing"]
        else:
            assert isinstance(row["speed_km_s"], (int, float))


def test_speed_is_linear_in_the_assumed_range(report):
    """speed_km_s scales with the ASSUMED range — it is not a measurement."""
    sweep = report["sweep"]
    assert sweep["speed_linear_in_range"] is True, (
        "speed must scale linearly with range; anything else means the "
        "reported value is not a pure function of the assumption"
    )
    # And it must actually vary — a constant would also be "linear".
    at_200 = [r for r in sweep["rows"]
              if r["range_km"] == 200.0 and r["fov_deg"] == 60.0]
    at_800 = [r for r in sweep["rows"]
              if r["range_km"] == 800.0 and r["fov_deg"] == 60.0]
    assert at_200 and at_800
    assert at_800[0]["speed_km_s"] > at_200[0]["speed_km_s"] * 3.5, (
        "4x the range should give 4x the speed"
    )


def test_rate_is_linear_in_the_assumed_fov(report):
    """angular_rate scales with the ASSUMED fov — an input, not a fit."""
    assert report["sweep"]["rate_linear_in_fov"] is True
    rows = [r for r in report["sweep"]["rows"] if r["range_km"] is None]
    by_fov = {r["fov_deg"]: r["angular_rate_deg_s"][0] for r in rows}
    assert by_fov[90.0] > by_fov[60.0] > by_fov[45.0]
    # 90/60 = 1.5 -> the x-rate should scale by the same factor.
    assert by_fov[90.0] / by_fov[60.0] == pytest.approx(1.5, rel=1e-6)


def test_sweep_document_states_the_caveat(report):
    note = report["sweep"]["note"]
    assert "assumed" in note
    assert "not the observation" in note or "not a measurement" in note


# ---------------------------------------------------------------------------
# R7 — synthetic output is never presented as measured
# ---------------------------------------------------------------------------


def test_report_is_flagged_synthetic_end_to_end(report):
    """R7: the flag is on the report, the ground truth, and every claim."""
    assert report["synthetic"] is True
    assert report["ground_truth"]["synthetic"] is True
    assert audit_synthetic_flags(report) == []


def test_report_states_its_limitations(report):
    """An honest limitation beats an unsupported metric."""
    text = " ".join(report["limitations"]).lower()
    assert "mota" in text or "idf1" in text, (
        "the missing-metrics limitation must be stated, not implied"
    )
    assert "range" in text, "the range assumption must be stated"
    assert "detection quality" in text, (
        "injected detections must not be read as detector performance"
    )


def test_claim_says_it_is_not_a_measurement(report):
    claim = report["claim"].lower()
    assert "not a measured" in claim or "not a measurement" in claim


def test_audit_flags_a_report_that_claims_to_be_real():
    """Non-vacuity: the audit must be able to fail."""
    good = {
        "synthetic": True,
        "ground_truth": {"synthetic": True},
        "claim": "Validation run. NOT a measured debris orbit.",
        "orbit": [{"track_id": 1, "synthetic": True}],
        "limitations": ["range assumed"],
    }
    assert audit_synthetic_flags(good) == []

    bad = dict(good, synthetic=False)
    assert audit_synthetic_flags(bad), "top-level flag must be checked"

    bad = dict(good, claim="Measured debris orbit from real footage.")
    assert any("not a measurement" in p for p in audit_synthetic_flags(bad))

    bad = dict(good, orbit=[{"track_id": 1, "synthetic": False}])
    assert any("synthetic=false" in p for p in audit_synthetic_flags(bad))

    bad = dict(good, limitations=[])
    assert any("limitations" in p for p in audit_synthetic_flags(bad))

    bad = dict(good, ground_truth={"synthetic": False})
    assert any("ground_truth" in p for p in audit_synthetic_flags(bad))


# ---------------------------------------------------------------------------
# Annotated output — debris must be visually distinct
# ---------------------------------------------------------------------------


def test_debris_is_visually_distinct_in_the_annotation(tmp_path, report):
    """The annotation colour must separate debris from everything else."""
    import cv2
    import numpy as np

    from tracking_using_drive.stream import _annotate

    frame = np.full((120, 160, 3), 30, np.uint8)
    det = {
        "track_id": 1, "xyxy": [20, 20, 80, 80], "conf": 0.9,
        "final_class": "debris", "is_debris": True,
    }
    other = dict(det, track_id=2, final_class="cheops", is_debris=False,
                 xyxy=[90, 20, 140, 80])

    out = _annotate(frame.copy(), [det, other])

    # Sample the rectangle BORDER (thickness 2 at y=20), not the interior —
    # the interior is untouched background and would compare equal for both.
    debris_edge = out[20:22, 22:78]
    other_edge = out[20:22, 92:138]

    # Debris is drawn in red-dominant BGR; non-debris in green-dominant.
    assert int(debris_edge[..., 2].mean()) > int(debris_edge[..., 1].mean()), (
        "debris must be red-dominant"
    )
    assert int(other_edge[..., 1].mean()) > int(other_edge[..., 2].mean()), (
        "non-debris must be green-dominant"
    )
    # And they must differ from each other, not just from the background.
    # Compare per-channel means: the boxes have different widths, so the
    # raw arrays cannot be broadcast against each other.
    assert not np.allclose(debris_edge.mean(axis=(0, 1)),
                           other_edge.mean(axis=(0, 1))), (
        "debris and non-debris must be drawn in different colours"
    )


def test_annotation_uses_the_fused_debris_flag(report):
    """The flag driving the colour is the fused one, not the raw class."""
    records = [
        json.loads(line)["detections"]
        for line in Path(report["artifacts"]["jsonl"])
        .read_text("utf-8").splitlines() if line.strip()
    ]
    dets = [d for frame in records for d in frame]
    assert dets, "fixture produced no detections"
    assert all(d["is_debris"] is True for d in dets)
    assert all(d["final_class"] == "debris" for d in dets)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_validate_reports_pass_structured(tmp_path, capsys):
    """R12: assert on the payload, and exit code gates the criteria."""
    from tracking_using_drive.cli import main

    rc = main(["validate", "--out", str(tmp_path / "v"),
               "--no-sync", "--frames", "12",
               "--occlude-start", "4", "--occlude-end", "8", "--json"])
    assert rc == 0, "all acceptance checks must pass"
    payload = json.loads(capsys.readouterr().out)
    assert payload["synthetic"] is True, "R7"
    assert payload["passed"] is True
    assert len(payload["checks"]) == 7
    assert all(c["ok"] for c in payload["checks"]), payload["checks"]
    assert Path(payload["artifacts"]["report"]).exists()


def test_cli_validate_writes_the_report_to_disk(tmp_path, capsys):
    from tracking_using_drive.cli import main

    out = tmp_path / "v"
    assert main(["validate", "--out", str(out), "--no-sync",
                 "--frames", "8", "--occlude-start", "3",
                 "--occlude-end", "6", "--json"]) == 0
    capsys.readouterr()
    path = out / "validation_report.json"
    assert path.exists()
    on_disk = json.loads(path.read_text("utf-8"))
    assert on_disk["synthetic"] is True, "R7 must survive serialisation"
    assert on_disk["chain"][-1] == "orbit"


def test_cli_validate_human_output_says_synthetic(tmp_path, capsys):
    """A reader of the terminal output alone must not mistake it for real."""
    from tracking_using_drive.cli import main

    out = tmp_path / "v"
    assert main(["validate", "--out", str(out), "--no-sync",
                 "--frames", "8", "--occlude-start", "3",
                 "--occlude-end", "6"]) == 0
    text = capsys.readouterr().out
    assert "SYNTHETIC" in text
    assert "never a measured" in text or "not a measured" in text
    assert "R7" in text


def test_cli_validate_refuses_a_gap_that_never_closes(tmp_path, capsys):
    """A gap past the end of the sequence would test nothing; refuse it."""
    from tracking_using_drive.cli import main

    rc = main(["validate", "--out", str(tmp_path / "v"), "--no-sync",
               "--frames", "8", "--occlude-start", "10",
               "--occlude-end", "20", "--json"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "occlusion" in err.lower()
    assert "never close" in err
