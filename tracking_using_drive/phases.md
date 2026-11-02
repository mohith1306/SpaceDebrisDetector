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

## Phase 2 — Drive Access and Local Staging

**Goal:** Reach the Drive dataset with correct credentials, selective sync,
and the local layout `scripts/jarvis.sh upload` expects (D12/G12).

**Status: core complete 2026-11-01. JarvisLabs staging deferred** — it needs
an R0 decision (see the last work item) and only Phase 5 consumes it.

**Work**
- [x] `drive.py` — anonymous-first access (D15). No OAuth required: the shared
  folder is public, so listing and download both work credential-free. This
  removed the `credentials.json` blocker from Phases 2–4 entirely.
- [x] `GDriveSync` retained as the OAuth path for private folders, constructed
  with explicit `cache_dir` and `credentials_file`, defeating
  `gdrive_sync.py:15` and `:18`.
- [x] Actionable error when the anonymous path hits a private folder, naming
  `--oauth` as the remedy.
- [x] Selective sync allowlist: `labels/train.csv`, `labels/val.csv`,
  `yolo_dataset/images/val/`, `yolo_dataset/labels/val/` — **not** the whole
  88k-image folder (D11). Measured: **1,086 files / ~164 MB**, against ~14 GB
  for a whole-folder sync.
- [x] Warm path via a manifest — **zero** network calls on a second run (R10).
- [x] `source.py` — `resolve_source()` with `gdrive://` support, plus
  `is_still()` so a single image can be routed to `sequence.py` instead of
  being tracked as a length-1 track.
- [x] Record `folder_id` and `cache_hit` via `SyncReport.as_metadata()`.
- [ ] **Local staging layout** — emit `train/` and `labels/train.csv` at the
  repo root, because `jarvis.sh upload` pushes exactly those paths and cannot
  be edited (R0). **Deferred: needs an R0 decision.** Both `train/` and
  `labels/` are gitignored upstream, so creating them is not a *source* change
  — but R0's wording ("no file outside it may be added") is broad enough to
  warrant explicit confirmation. Options and the OneDrive caveat are in
  `memory.md` Next Actions.

**Exit criterion**
Cold cache: syncs the allowlisted slice and resolves to a local path.
Warm cache with `--no-sync`: **zero** network calls.
`python -m pytest tracking_using_drive/tests -q -k hardcoded` → passes.
(A raw grep for the drive-path literal is unsatisfiable — the guard and the
docstrings quote it deliberately. FR-3.1 constrains *code*, so the check is
AST-based.)
`train/` and `labels/train.csv` exist where `jarvis.sh upload` expects them.

**Risk:** PyDrive2 OAuth opens a browser on first run. Validate the warm path
immediately after one successful auth, or the credential bug will hide.

---

## Phase 3 — Sequence Synthesis

**Goal:** Turn a Drive still into an ordered sequence with known ground truth.
This is the branch's core addition and the only way the dataset can support
tracking at all.

**Status: complete (2026-11-01).** Core delivered in Phase 3; the end-to-end
exit criterion was measured in Phase 4 (0.0013% rate error).

**Work**
- [x] `sequence.py` — `Motion`, `Occlusion`, `GroundTruth` dataclasses.
- [x] `synthesise_sequence()` — linear and accelerating motion.
  (`orbit_arc` is accepted but **deliberately degrades to linear**, pinned by
  `test_orbit_arc_degrades_to_linear_rather_than_unvalidated` — a curved path
  we do not actually render must not take an untested branch.)
- [x] **Occlusion by erasure**, not off-canvas translation. Verified: the
  erased ROI's pixel std is exactly **0.0** against a visible std of 65.34
  (D12, `test_erasure_flattens_the_object_region`).
- [x] Emit `ground_truth.json` with per-frame centres, pixel shift, FPS, and
  implied angular rate.
- [x] `synthetic: true` on every artifact (R7) — forced in `to_dict()`, so it
  cannot be serialised away even if constructed with `False`.
- [x] **Occlusion fixture** — delivered as the `synth --occlude-start/--occlude-end`
  flags rather than `scripts/make_occlusion_video.py`. That path is out of
  bounds (R0); the fixture belongs in this package.
- [x] **Angular-rate validation** — delivered as
  `test_angular_rate_truth_matches_library_estimate_within_10pct` rather than
  `scripts/validate_angular_rate.py`, for the same R0 reason.
- [x] Batch mode — `synth --count N` renders N sequences into `seq_NNN/` with an
  `index.json`. D8 says success is a *measured error bar*; one sequence yields
  one number with no spread, which cannot separate a real 8% from a lucky 8%.
- [x] **Tracked end-to-end recovery** — closed by Phase 4. Measured through
  `stream.py`: recovered `[1.75779, -0.87886]` vs truth
  `[1.75781, -0.87891]` = **0.0013%**, well inside R9's 10%.

**Measured so far**
- R9 formula agreement: worst error **0.0001%** against `estimate_trajectory`
  (tolerance 10%). Truth and estimate use the *same* `fov_vertical_deg`, so
  they agree by construction — see the caveat below.
- Motion continues unchanged through the gap: step into and out of the occlusion
  both `(0.005859, -0.00293)`.
- D13 stream over a 10-frame gap: `111111111100000000001111111111`.

**Caveat we deliberately did not dodge:** on 1024×1024 stills the library's
exact `arctan` vertical-FOV formula and `make_demo_video.py:73`'s small-angle
approximation *coincide* (60.0 = 60.0), which would make the R9 check vacuous.
At 1920×1080 they differ by **6.21%** — enough to consume most of the 10%
budget before any tracking error is measured. We import the library's formula
and pin both behaviours:
`test_vertical_fov_uses_exact_formula_not_small_angle`,
`test_matches_make_demo_video_formula`,
`test_deliberately_departs_from_demo_video_at_16_9`.

**Exit criterion**
A sequence synthesised from a real Drive still is tracked end-to-end and the
recovered angular rate is within 10% of `ground_truth.json`. The occlusion
fixture produces a genuine detection gap (verified by inspecting the frames).

*The occlusion half is met (std 0.0, gap visible in the frames). The
"tracked end-to-end" half is Phase 4's exit criterion — this phase stops at
the ground truth the tracker will be scored against.*

**Risk:** if the object is small relative to the translation step, it may
alias between frames and ByteTrack will fail to associate even with correct
synthesis. Tune `dx`/`dy` against object size before blaming the tracker.

*Now exercised: Phase 4 drove `stream.py` over this fixture end-to-end and
recovered the rate to 0.0013% (see Phase 4).*

---

## Phase 4 — Tracking Correctness

**Goal:** Tracks survive occlusion. The branch's second core contribution.

**Unblocked without weights (D13).** Because `sequence.py` knows the exact
object centre for every frame it renders, it can emit a detection stream
directly at those centres. That validates ByteTrack → Kalman → orbit
independently of any detector, and isolates precisely the layer this branch
fixes. Real weights (Phase 5) are needed only for the *full* end-to-end path.

**Status: core complete (2026-11-01). All exit criteria met on the Phase 3
occlusion fixture — measured below.**

**Work**
- [x] `stream.py` — reimplemented `run_stream` importing `TrajectoryBank`,
  `summarize_track`, `classify_crops`, `fuse`, `probe_fps`.
- [x] **D13 injection path** — `run_stream(..., detections=...)` accepts a
  pre-computed stream in place of `model.track()`. **No model is loaded and no
  weights are needed**; frames are read straight from disk and associated by
  `assign_track_ids()`.
- [x] Missed-detection propagation — `engine.predict()` advances *every* active
  track one frame each frame. **Not** the naive `bank.miss()` + `update()`
  pairing: see the measured table in `stream.py` §Design, where the naive form
  lands **-58.77%** on velocity at re-acquisition versus **+13.30%** for
  predict-then-correct.
- [x] `n_missed` / `expired` counters; `max_missed` (default 30).
- [x] `frame_wh` captured from the **first** frame; orbit report guarded on
  `frame_wh is not None`. *(Originally described as fixing an unbound-variable
  crash at `stream.py:181` — **not reachable**, verified 2026-11-01: the report
  block is guarded by `bank.ids()`, which is only truthy if the loop ran and
  bound `width`. The real defect is that geometry came from the **last** frame,
  which silently mis-fits mixed-size sources. See `stream.py` §Why.)*
- [x] `n_missed`, `lifespan_s`, `expired` in the per-track report, plus a full
  `tracks` inventory in the summary (otherwise expired tracks would be
  invisible in every output).
- [x] Expired tracks excluded from the orbit report but history retained — the
  bank is never pruned, so they stay inspectable.
- [x] Records now carry `cx`/`cy`. The original omitted them while
  `orbit_from_jsonl` (`orbit.py:294`) reads them unconditionally, so the
  original's own JSONL could not be replayed through the library's offline
  orbit stage.

**Measured on the Phase 3 occlusion fixture (30 frames, gap 10–19, injected)**

| exit criterion | result |
|---|---|
| track ID preserved across the gap | **1** distinct id, before and after |
| filter predicted rather than froze | `predicted_track_ids == [1]` on **all 10** gap frames; `motion` resumes at 0.032763 vs 0.032616 pre-gap |
| appears **once** in the orbit report | `orbit_tracks = 1`, ids `[1]` |
| `n_missed` recorded | **10** (exactly the gap) |
| angular rate vs ground truth | `[1.75779, -0.87886]` vs `[1.75781, -0.87891]` → **0.0013%** (R9: 10%) |
| no regression on Phase 3 | R9 still 0.0013%, well inside tolerance |

**The non-vacuity check** — id survival is only meaningful if it *fails* when
prediction fails. It does:

| | distinct ids | id after gap | verdict |
|---|---|---|---|
| prediction working | `[1]` | 1 | preserved |
| prediction frozen | `[1, 2]` | 2 | **splits into two fragments** |

Without prediction the re-appearing object sits 67 px from where the frozen
track waits, outside the gate (4 observed steps ≈ 27 px), gets a fresh id, and
the track fragments into two length-10 pieces — precisely the failure mode
this phase exists to prevent. So the exit criterion is a real assertion, not a
constant-id tautology.

**Exit criterion — MET (2026-11-01), measured on the Phase 3 fixture**
On the occlusion fixture from Phase 3, using injected detections:
- track ID preserved across the gap: **1** distinct id, before and after;
- during the gap `motion` is non-zero and consistent with pre-gap velocity
  (the filter **predicted**, it did not freeze): `predicted_track_ids == [1]`
  on all 10 gap frames, `motion` resumes at 0.032763 vs 0.032616 pre-gap;
- the track appears **once** in the orbit report, not twice: `orbit_tracks = 1`.

Plus: no regression on the Phase 3 angular-rate check — recovered
`[1.75779, -0.87886]` vs truth `[1.75781, -0.87891]` = **0.0013%** (R9: 10%).

**The `max_missed` boundary, measured both sides.** The default of 30 bridges
the 10-frame gap; a tight budget of 3 does not, and the failure mode is a
*split*, not an empty report:

| `max_missed` | tracks | expired | orbit entries | what happened |
|---|---|---|---|---|
| 30 (default) | 1 | 0 | 1 | id 1 survived, `n_missed=10` |
| 3 | **2** | 1 | 1 | track 1 expired in the gap; the object returned as **track 2** with 10 fresh obs, which is neither expired nor too short — so it still ships one orbit entry |

The second row is why `test_cli_stream_honours_max_missed` asserts
`orbit_tracks == 1`, not `0`: an expired track is excluded, but the
re-acquired fragment is a legitimately new track. The split is exactly the
failure mode the default avoids, and both sides are pinned so the parameter's
effect cannot drift unnoticed.

**Risk:** ByteTrack also predicts internally and may recover the ID on its own,
disagreeing with our timing about when to resume. If IDs still fragment, raise
`max_missed` above ByteTrack's cooldown — do not remove the Kalman prediction.
*Not yet exercised: no real detector has run in this phase (D13 injects
centres), so ByteTrack's own association has not been pitted against ours.
Open until Phase 5 or 6.*

**Note:** injected detections carry `class_id`/`conf` from ground truth, so
this phase validates *association and motion*, not detection quality. Do not
report it as detector performance.

---

## Phase 5 — Detector Training on JarvisLabs

**Goal:** Produce a real `best.pt` so the full detection→tracking path can be
exercised. D14.

**Prerequisite:** Phase 2 must have staged `train/` and `labels/train.csv`
locally — `scripts/jarvis.sh upload` pushes exactly those paths and cannot be
edited (R0/G12).

**Work**
- [ ] Confirm JarvisLabs auth: `jl status`. **Needs a token — see Q9.**
- [ ] `jl create L4 --storage 120` (raw ~14GB + processed ~17GB +
  yolo_dataset ~17GB + checkpoints).
- [ ] `jarvis.sh upload <id>` — pushes `space_debris/`, `train_jarvis.py`,
  `requirements-gpu.txt`, `train/`, `labels/`.
- [ ] `jarvis.sh train <id> --epochs 30 --batch 16` → `train_jarvis.py` →
  `train_with_recovery()` (P1/P2 head, crash recovery, per-epoch logging).
- [ ] Fetch `best.pt` back to `tracking_using_drive/_data/weights/`.
- [ ] `jarvis.sh pause <id>` to stop billing.
- [ ] Record the ultralytics version on the instance vs. local **8.4.197**
  (G11) — a mismatch can explain result drift.

**Exit criterion**
`best.pt` present locally, `jarvis.sh pause` run, and the training session ID
plus final metrics recorded in `memory.md`.

**Risk:** this is the only phase needing external credentials, GPU budget, and
roughly an hour. It is also the only phase that can be deferred indefinitely —
Phases 3, 4 and 6 all work via D13 without it.

---

## Phase 6 — End-to-End Validation

**Goal:** Prove the full chain, and document honestly what it can claim.

**Work**
- [ ] **Via D13:** full chain Drive → sync → synthesise → injected detections →
  track → Kalman → orbit.
- [ ] **Via D14 (if `best.pt` available):** repeat with the real detector so
  detection→tracking integration is covered. Report the two separately — they
  are different claims.
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

## Phase 7 — Hardening and Closeout

**Goal:** Safe to merge, legible to the next person.

**Work**
- [ ] Full pre-commit checklist (`rules.md` Git Workflow).
- [ ] Confirm no `gdrive_cache/`, `credentials.json`, `*.pt`, `runs/`, or
  synthesised videos are staged.
- [ ] Re-run Phase 0, 3, 4 validations on final code.
- [ ] Tests: `test_source.py`, `test_sequence.py`, `test_missed_detections.py`.
- [ ] Close out `requirements.md` §8 — Q7, Q8, Q9 answered or explicitly
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
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 6 ──► Phase 7
              │            │           │            │           ▲
              │            │           │            │           │
              │            │           │            └─ D13 ─────┘
              │            │           │        (injected detections:
              │            │           │         no weights required)
              │            │           │                              │
              │            │           └── synthesises the sequences │
              │            │               Phase 4 tests on          │
              │            │                                         │
              │            └── Drive stills are the Phase 3 input;   │
              │                staged layout feeds Phase 5           │
              │                                                      │
              └── bootstrap everything else depends on               │
                                                                 │
              Phase 5 (JarvisLabs training) ──── best.pt ────────────┘
                  OPTIONAL — only needed for the full
                  detection→tracking path in Phase 6
```

**Phase 5 is the only optional phase.** Phases 3, 4, 6 and 7 all complete via
D13 without any detector. Phase 5 exists to additionally cover
detection→tracking integration, and depends on external credentials, GPU
budget, and Phase 2's local staging.

Everything else is strictly sequential.

---

## Effort Estimate

| Phase | Effort | Notes |
|-------|--------|-------|
| 0 — Baseline | Small | Environment + data verification. **Mostly done.** |
| 1 — Skeleton | Small | Bootstrap + CLI wiring. |
| 2 — Drive access and staging | Medium | OAuth validation, selective sync, JarvisLabs layout. |
| 3 — Sequence synthesis | **Largest** | New capability, ground-truth correctness, occlusion-by-erasure. |
| 4 — Tracking correctness | Medium-Large | New state machine + D13 injection path. |
| 5 — JarvisLabs training | **External** | Needs token, GPU time, ~1hr. Deferrable. |
| 6 — Validation | Medium | FOV/range sweep, honest reporting. |
| 7 — Closeout | Small | Mechanical. |

---

## Scope Gate — Resolved

**Q6** (`requirements.md` §8) was the branch's gate:

> Is "complete the tracking of debris" satisfied by validating the pipeline on
> **synthesised** sequences derived from the Drive stills? Or is real
> observation video expected from another source?

**Answered 2026-11-01 — Option A confirmed.** The deliverable is a validated
tracking pipeline with quantified error against known ground truth, derived
from synthesised sequences over Drive stills.

Consequences carried into Phases 3–7:

1. **Success is a measured error bar, not an orbit.** Phase 6 reports angular-
   rate recovery accuracy and track-continuity metrics. It does not report
   determined orbits, and must not be written up as if it did.
2. **The 22k-image `val.csv` is the natural synthesis pool** (Q8). Drawing
   stills from a held-out split keeps the synthesised sequences independent of
   whatever the detector was trained on.
3. **Two validation claims, reported separately (D13 vs. D14).** Tracking-layer
   correctness is provable with injected detections and no weights. Detector
   accuracy needs the Phase 5 JarvisLabs run. Merging them into one number
   would misrepresent both.

No gate remains. Phases proceed sequentially from Phase 0.
