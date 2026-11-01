# Phases — `tracking_using_drive`

Seven phases, ordered so each leaves the branch in a working state.

**Constraint carried into every phase:** nothing outside
`tracking_using_drive/` is modified (`rules.md` R0). Verify with
`git status --porcelain | grep -v "tracking_using_drive/"` before each commit.

---

## Phase 0 — Baseline and Data Verification

**Goal:** Confirm the verified dataset facts hold locally, and capture a
measurable starting point.

**Work**
- [ ] Set `CAPSTONE_DIR` bootstrap and confirm `import tracking_using_drive.config` works.
- [ ] Re-verify the Drive facts against a local sample:
  - [ ] `labels/train.csv` → 66,000 rows, 66,000 unique filenames, max 1 annot/image.
  - [ ] `labels/val.csv` → 22,000 rows, zero filename overlap with train.
  - [ ] All `.png`.
- [ ] Record the baseline on the **untouched** `space_debris` pipeline:
  ```bash
  python -m space_debris.main stream --source yolo_dataset/images/val \
    --max-frames 200 --out runs/baseline.jsonl
  ```
  Note frames, `num_tracks`, track-length distribution.
- [ ] Demonstrate the degenerate case on purpose: run `stream` over a batch of
  **unrelated** stills and record that every track is length 1 and the orbit
  report is empty. This is the evidence that §2 of `requirements.md` is real
  and not theoretical.
- [ ] Run `scripts/make_demo_video.py` and record the angular-rate error.

**Exit criterion**
Numbers written to `memory.md` §Current State. The degenerate case is
reproduced and recorded. **Zero code changes.**

**Why first:** "tracking is broken on stills" must be demonstrated, not
asserted, before a synthesis phase can be justified.

---

## Phase 1 — Package Skeleton and Bootstrap

**Goal:** A runnable, isolated package that imports the library correctly.

**Work**
- [ ] `tracking_using_drive/__init__.py`
- [ ] `config.py` — `CAPSTONE_DIR` bootstrap before importing
  `space_debris.config`; scoped `GDRIVE_CACHE_DIR`.
- [ ] `cli.py` / `main.py` — argument parsing, dispatch, `--help`.
- [ ] Confirm `space_debris` imports resolve from inside the package.
- [ ] Confirm no directories are created outside `tracking_using_drive/_data/`.

**Exit criterion**
```bash
python -m tracking_using_drive.main --help
git status --porcelain | grep -v "tracking_using_drive/"   # empty
```
`_data/` is created, nothing else. Import takes no longer than importing
`space_debris.config` directly.

**Risk:** the bootstrap must be the *first* import in every entrypoint. A
stray `from space_debris...` at the top of another module silently captures the
wrong `BASE_DIR`. Add a test asserting the resolved path is under
`tracking_using_drive/`.

---

## Phase 2 — Drive Access

**Goal:** Reach the Drive dataset with correct credentials and selective sync.

**Work**
- [ ] `drive.py` — construct `GDriveSync` with explicit `cache_dir` and
  `credentials_file`, defeating `gdrive_sync.py:15` and `:18`.
- [ ] Deterministic credential path, `GDRIVE_CREDENTIALS` override.
- [ ] Actionable error when credentials are absent.
- [ ] Selective sync allowlist: `labels/train.csv`, `labels/val.csv`,
  `yolo_dataset/images/val/` — **not** the whole 88k-image folder.
- [ ] `source.py` — `resolve_source()` with `gdrive://` support.
- [ ] Record `folder_id` and `cache_hit` for the output metadata.

**Exit criterion**
Cold cache: syncs the allowlisted slice and resolves to a local path.
Warm cache with `--no-sync`: **zero** network calls.
`grep -rE "E:/Capstone|Path\(.credentials" tracking_using_drive/` → empty.

**Risk:** PyDrive2 OAuth opens a browser on first run. Validate the warm path
immediately after one successful auth, or the credential bug will hide.

---

## Phase 3 — Sequence Synthesis

**Goal:** Turn a Drive still into an ordered sequence with known ground truth.
This is the branch's core addition and the only way the dataset can support
tracking at all.

**Work**
- [ ] `sequence.py` — `Motion`, `Occlusion`, `GroundTruth` dataclasses.
- [ ] `synthesise_sequence()` — constant, accelerating, curved motion.
- [ ] **Occlusion by erasure**, not off-canvas translation. Guarantee a true
  detection dropout.
- [ ] Emit `ground_truth.json` with per-frame centres, pixel shift, FPS, and
  implied angular rate.
- [ ] `synthetic: true` on every artifact (R7).
- [ ] `scripts/make_occlusion_video.py` — the occlusion test fixture.
- [ ] `scripts/validate_angular_rate.py` — estimated vs. ground truth, < 10%.
- [ ] Batch mode over multiple Drive stills.

**Exit criterion**
A sequence synthesised from a real Drive still is tracked end-to-end and the
recovered angular rate is within 10% of `ground_truth.json`. The occlusion
fixture produces a genuine detection gap (verified by inspecting the frames).

**Risk:** if the object is small relative to the translation step, it may
alias between frames and ByteTrack will fail to associate even with correct
synthesis. Tune `dx`/`dy` against object size before blaming the tracker.

---

## Phase 4 — Tracking Correctness

**Goal:** Tracks survive occlusion. The branch's second core contribution.

**Work**
- [ ] `stream.py` — reimplement `run_stream` importing `TrajectoryBank`,
  `summarize_track`, `classify_crops`, `fuse`, `resolve_model`, `probe_fps`.
- [ ] Missed-detection propagation: `bank.miss(tid, dt=1.0/rate)` for every
  active, unobserved, unexpired track.
- [ ] `missed` / `expired` counters; `--max-missed` (default 30).
- [ ] `frame_wh` captured from the **first** frame; orbit block guarded on
  `frames > 0`. (Fixes the `stream.py:181` unbound-variable bug.)
- [ ] `n_missed`, `lifespan_s`, `expired` in the per-track report.
- [ ] Expired tracks excluded from the orbit report but history retained.

**Exit criterion**
On the occlusion fixture from Phase 3:
- track ID preserved across the gap;
- during the gap `motion` is non-zero and consistent with pre-gap velocity
  (the filter **predicted**, it did not freeze);
- the track appears **once** in the orbit report, not twice.

Plus: no regression on the Phase 3 angular-rate check.

**Risk:** ByteTrack also predicts internally and may recover the ID on its own,
disagreeing with our timing about when to resume. If IDs still fragment, raise
`max_missed` above ByteTrack's cooldown — do not remove the Kalman prediction.

---

## Phase 5 — End-to-End Validation

**Goal:** Prove the full chain, and document honestly what it can claim.

**Work**
- [ ] Full run: Drive → sync → synthesise → detect → track → Kalman → orbit.
- [ ] Sweep `--fov` and `--range-km`; record how `speed_km_s` and
  `consistent_with_circular_orbit` respond.
- [ ] Confirm that with no `--range-km` every track reports
  `missing: ["range_km_needed_for_physical_speed"]`.
- [ ] Track-continuity metrics vs. Phase 0 baseline: mean track length, track
  count, fragmentation rate.
- [ ] Verify `is_debris` populates and debris tracks are visually distinct in
  the annotated output.
- [ ] **Confirm every synthetic-derived artifact carries `synthetic: true` and
  that no report describes it as a measured orbit** (R7).
- [ ] Update `PREPROCESSING_AND_TRACKING.md` §13/§14 — as a *proposed* upstream
  change recorded in `memory.md`, since the file itself is out of bounds.

**Exit criterion**
Measurable improvement in mean track length and fewer tracks for the same frame
range. Every orbit entry states assumptions and `missing`. No synthetic result
is presented as a measurement.

**Risk:** without temporal identity ground truth there is no MOTA/IDF1. The
available claims are track-continuity proxies on **synthetic** motion. State
that plainly — an honest limitation beats an unsupported metric.

---

## Phase 6 — Hardening and Closeout

**Goal:** Safe to merge, legible to the next person.

**Work**
- [ ] Full pre-commit checklist (`rules.md` Git Workflow).
- [ ] Confirm no `gdrive_cache/`, `credentials.json`, `*.pt`, `runs/`, or
  synthesised videos are staged.
- [ ] Re-run Phase 0, 3, 4 validations on final code.
- [ ] Tests: `test_source.py`, `test_sequence.py`, `test_missed_detections.py`.
- [ ] Close out `requirements.md` §8 — Q7 and Q8 answered or explicitly
  "accepted as unknown".
- [ ] Prune stale entries from `memory.md`; write the final state.
- [ ] PR against `main`, evidence attached phase by phase.

**Exit criterion**
All 8 acceptance criteria in `requirements.md` §7 demonstrated or explicitly
waived with a reason. `git status` clean outside the folder. `memory.md`
reflects reality.

---

## Dependency Graph

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6
              │            │           │            │           │
              │            │           └─── synthesises the sequences Phase 4 tests on
              │            └─── Drive stills are the Phase 3 input
              └─── bootstrap everything else depends on
```

Strictly sequential — each phase's output is the next phase's input. The one
parallelism available is Phase 6's test scaffolding, which can be written from
Phase 1 onward.

---

## Effort Estimate

| Phase | Effort | Notes |
|-------|--------|-------|
| 0 — Baseline | Small | Running commands, recording numbers. |
| 1 — Skeleton | Small | Bootstrap + CLI wiring. |
| 2 — Drive access | Medium | OAuth validation, selective sync. |
| 3 — Sequence synthesis | **Largest** | New capability, ground truth correctness, occlusion-by-erasure. |
| 4 — Tracking correctness | Medium-Large | New state machine, real risk of ByteTrack interaction. |
| 5 — Validation | Medium | FOV/range sweep, honest reporting. |
| 6 — Closeout | Small | Mechanical. |

---

## Scope Gate — Resolved

**Q6** (`requirements.md` §8) was the branch's gate:

> Is "complete the tracking of debris" satisfied by validating the pipeline on
> **synthesised** sequences derived from the Drive stills? Or is real
> observation video expected from another source?

**Answered 2026-11-01 — Option A confirmed.** The deliverable is a validated
tracking pipeline with quantified error against known ground truth, derived
from synthesised sequences over Drive stills.

Two consequences worth carrying into Phases 3–5:

1. **Success is a measured error bar, not an orbit.** Phase 5 reports angular-
   rate recovery accuracy and track-continuity metrics. It does not report
   determined orbits, and must not be written up as if it did.
2. **The 22k-image `val.csv` is the natural synthesis pool** (Q8). Drawing
   stills from a held-out split keeps the synthesised sequences independent of
   whatever the detector was trained on.

No gate remains. Phases proceed sequentially from Phase 0.
