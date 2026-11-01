# Requirements — `tracking_using_drive`

**Goal:** Complete the tracking of debris using the dataset stored in Google Drive.

**Data source:** `Capstone` folder —
`https://drive.google.com/drive/u/0/folders/1vxqeBHOglilc1PP_ZSNA1fCx_jfTTdUY`
(folder ID `1vxqeBHOglilc1PP_ZSNA1fCx_jfTTdUY`, publicly readable).

> **ISOLATION CONSTRAINT (hard):** All work on this branch happens inside
> `tracking_using_drive/`. No file outside that folder may be modified, added,
> or deleted. The `space_debris/` package is a **read-only library** — import
> from it, never write to it. See `rules.md` R0.

---

## 1. Verified Dataset Facts

Established by direct inspection of the Drive folder on 2026-11-01. These are
measurements, not assumptions.

### 1.1 Folder structure

```
Capstone/                          (ID 1vxqeBHOglilc1PP_ZSNA1fCx_jfTTdUY)
├── train/                         raw images
├── processed/                     preprocessed images
├── labels/
│   ├── train.csv                  66,000 annotations
│   ├── val.csv                    22,000 annotations
│   └── sample_submission.csv      (Kaggle-style; unused)
├── labels_yolo_2k/                YOLO labels, 2k subset
├── yolo_dataset/
│   ├── images/
│   └── labels/
├── space_debris.yaml
├── train.zip
├── CAPSTONE.ipynb
└── CAPSTONE (4)_backup.ipynb
```

### 1.2 Annotation statistics

| File | Rows | Unique images | Max annots/image | Classes |
|------|------|---------------|------------------|---------|
| `labels/train.csv` | 66,000 | 66,000 | **1** | 11 × 6,000 |
| `labels/val.csv` | 22,000 | 22,000 | **1** | 11 × 2000 |

- Schema: `filename,class,bbox` with `bbox` as a literal list
  `"[x_min, y_min, x_max, y_max]"`.
- All images are `.png`.
- **Zero filename overlap** between `train.csv` and `val.csv`. No leakage.
- Class balance is exact.

### 1.3 The fact that governs this entire branch

> **The dataset contains 88,000 independent, single-object still images. There
> are no video sequences, no temporal ordering, and no multi-object frames.**

Every downstream requirement follows from this. It resolves three of the five
open questions at once and creates one hard constraint (§2).

---

## 2. The Data Mismatch — the central problem

The goal is *tracking*. Tracking requires the same object to appear across an
ordered sequence of frames. This dataset provides none of that:

| Property needed for tracking | Present in Drive data? |
|------------------------------|------------------------|
| Temporally ordered frames | **No** — images are unrelated stills |
| Same object across frames | **No** — one object, one frame |
| Multiple objects per frame | **No** — exactly 1 annotation per image |
| Known frame rate | **No** — no video |
| Camera motion model | **No** |

Consequence: running `stream` over these images hands ByteTrack 88,000
unrelated frames. Each frame yields a detection with no motion correspondence,
so ByteTrack assigns a **new track ID to every image**. The "tracks" produced
are length 1, fall below `MIN_POINTS_FOR_ORBIT = 3`, and are discarded from the
orbit report. The pipeline runs without error and produces nothing.

The repository already knows this. `scripts/make_demo_video.py:5-6`:

> *"The dataset is a set of unrelated still images, so tracking/orbit cannot be
> validated against it."*

That script is the existing workaround: take **one** still, translate it by a
known pixel shift per frame, and emit a synthetic video plus ground-truth
motion. This branch must generalise that idea rather than pretend the stills
are a sequence.

### 2.1 What "complete the tracking of debris" can therefore mean

Three candidate interpretations, stated explicitly so the scope is chosen
rather than assumed:

| Option | Description | Produces real debris orbits? |
|--------|-------------|------------------------------|
| **A — Synthetic sequence validation** | Generalise `make_demo_video.py`: synthesise sequences from Drive stills with known ground-truth motion, then run the full track→Kalman→orbit pipeline and verify recovery. | No — trajectories are synthetic by construction |
| **B — Per-image track assembly** | Treat each still as an independent single-observation track and report detection-level results without cross-frame association. | No — no temporal data exists |
| **C — Real sequences required** | Obtain actual telescope/observation video. Cannot be done from this Drive data. | Yes, but out of reach here |

**CONFIRMED SCOPE (2026-11-01): Option A.** The deliverable is a validated,
end-to-end tracking pipeline whose correctness is demonstrated on synthesised
sequences derived from the Drive stills, with Option B as the honest degenerate
case. Any claim of "real debris orbit determined from this data" is unsupported
and must not be made.

This is the single most important scope decision on the branch. Logged as
**D8** in `memory.md`, confirmed by the project owner. Consequence: real
orbital determination is structurally out of reach here — it would require
different data, not different code. The branch succeeds by producing a pipeline
with **quantified error against known ground truth**, not by producing orbits.

---

## 3. Problem Statement — the Drive/tracking disconnect

Even setting the mismatch aside, the tracker cannot reach the Drive data:

- `track`, `stream`, `orbit` **do not auto-sync**. Only `preprocess`,
  `convert_labels`, `split_dataset`, `train`, `train_resilient` call
  `ensure_data_synced()` (`main.py:161-164`).
- `resolve_source()` (`track.py:41-48`) defaults to local
  `yolo_dataset/images/val` and raises if absent.
- `E:/Capstone/gdrive_cache` is hardcoded at `gdrive_sync.py:15` and `:133`.
- `credentials.json` resolves against `Path.cwd()` (`gdrive_sync.py:18`), so
  the working directory decides whether OAuth re-triggers.

Under the isolation constraint **none of these can be fixed in place**. The
branch must supply corrected implementations inside `tracking_using_drive/`
that import the untouched originals where they are correct.

---

## 4. Functional Requirements

### FR-1 — Drive as a first-class source
| ID | Requirement |
|----|-------------|
| FR-1.1 | The pipeline MUST accept a Drive folder ID or name and resolve it to a local path. |
| FR-1.2 | Default source MUST be the Drive folder, with a local-path override. |
| FR-1.3 | Drive-sourced frames MUST use the identical inference path as local frames. |
| FR-1.4 | The resolved path AND the Drive folder ID MUST both be recorded in the output. |

### FR-2 — Synchronisation
| ID | Requirement |
|----|-------------|
| FR-2.1 | Sync MUST run before tracking when local data is absent. |
| FR-2.2 | Sync MUST be incremental — size-matched files skipped. |
| FR-2.3 | A warm cache MUST make zero network calls (`--no-sync`). |
| FR-2.4 | Partial sync MUST warn per file and not abort silently. |
| FR-2.5 | Selective sync MUST be possible — the full folder holds 88k images across `train/`, `processed/`, and `yolo_dataset/`, and a whole-folder download is wasteful for a tracking run. |

### FR-3 — Path and credential handling
| ID | Requirement |
|----|-------------|
| FR-3.1 | No `E:/Capstone` literal may appear anywhere in this branch's code. |
| FR-3.2 | Credentials MUST resolve deterministically, overridable by `GDRIVE_CREDENTIALS`. |
| FR-3.3 | Absent credentials MUST raise an error naming the expected path. |
| FR-3.4 | Because `space_debris.config` cannot be edited, the branch MUST set `CAPSTONE_DIR` before importing it, so its paths land where this branch controls them. |

### FR-4 — Complete the debris track
| ID | Requirement |
|----|-------------|
| FR-4.1 | Missed detections MUST propagate: call `TrajectoryBank.miss(track_id, dt)` for active tracks absent from the current frame. |
| FR-4.2 | Tracks MUST expire after `--max-missed` consecutive misses (default 30). |
| FR-4.3 | Track IDs MUST not be reused within a run. |
| FR-4.4 | The orbit report MUST use the **first** frame's dimensions, not the last (`stream.py:181` currently uses the last, and is unbound on zero frames). |
| FR-4.5 | Every reported track MUST carry `n_points`, `n_missed`, `lifespan_s`, `expired`. |

### FR-5 — Sequence synthesis (from §2)
| ID | Requirement |
|----|-------------|
| FR-5.1 | The branch MUST provide a synthesiser that turns a Drive still into an ordered sequence with known ground-truth motion. |
| FR-5.2 | Ground truth MUST include per-frame centres, pixel shift per frame, FPS, and the implied angular rate. |
| FR-5.3 | The synthesiser MUST generalise `scripts/make_demo_video.py` — supporting varying shift, duration, FPS, and optional occlusion gaps. |
| FR-5.4 | An occlusion mode MUST exist (object leaves and re-enters frame) so FR-4 is testable. |
| FR-5.5 | Any output derived from synthesised sequences MUST be labelled synthetic in the report metadata. Nothing synthetic may be presented as a measured debris orbit. |

### FR-6 — Reproducible output
| ID | Requirement |
|----|-------------|
| FR-6.1 | Per-frame JSONL remains the primary output. |
| FR-6.2 | A per-track summary JSON MUST be written every run. |
| FR-6.3 | The run MUST record weights, tracker, conf, FPS, FOV, range, source, Drive folder ID, and `synthetic: true/false`. |
| FR-6.4 | Identical inputs MUST yield identical track IDs. |

---

## 5. Non-Functional Requirements

| ID | Requirement | Rationale |
|----|-------------|-----------|
| NFR-1 | Warm cache → zero network calls. | Drive latency must not sit in the frame loop. |
| NFR-2 | Frames streamed, never accumulated. | Bounded memory. |
| NFR-3 | Throughput must not regress vs. `stream.py`. | Synthesis is a startup cost. |
| NFR-4 | Correct on Windows, macOS, Linux. | `Path` throughout; no string joining. |
| NFR-5 | Local-disk workflow must keep working. | Backwards compatibility. |
| NFR-6 | **Zero modifications outside `tracking_using_drive/`.** | Isolation constraint. Verified by `git status` before every commit. |
| NFR-7 | No new heavy dependencies. | `pydrive2` already present in both requirements files. |

---

## 6. Out of Scope

- Uploading results back to Drive.
- Per-frame streaming from the Drive API. Sync-then-process is the model.
- **Real orbital determination.** The data cannot support it: no camera
  attitude, no astrometry, no range, no TLE. Synthesised sequences are worse
  still — their "motion" is a `warpAffine`, not physics.
- Quantitative tracking benchmark (MOTA/MOTP/IDF1). Requires temporal identity
  ground truth that does not exist here.
- Any change to `space_debris/`, `scripts/`, `train_jarvis.py`, or either
  `requirements*.txt`.

---

## 7. Acceptance Criteria

1. A fresh clone resolves the Drive folder by ID and produces a JSONL with
   non-null `track_id` values.
2. A second identical run performs **zero** downloads.
3. `grep -rE "E:/Capstone|Path\(.credentials" tracking_using_drive/` returns
   nothing.
4. `git status` outside `tracking_using_drive/` is clean after every commit.
5. A synthesised sequence with known `dx`/`dy` yields an angular rate within
   **10%** of ground truth.
6. An occlusion-mode sequence yields **one** track with the ID preserved across
   the gap, and non-zero predicted motion during the gap.
7. Every orbit report entry states its assumptions and its `missing` list.
   Nothing with `synthetic: true` is described as a measured orbit.
8. The untouched local default path still works with no arguments and no
   network.

---

## 8. Question Status

| # | Question | Status | Answer |
|---|----------|--------|--------|
| Q1 | Is the Drive folder flat or full layout? | **RESOLVED** | Full layout — `train/`, `labels/`, `processed/`, `labels_yolo_2k/`, `yolo_dataset/`. |
| Q2 | Are Drive frames raw or preprocessed? | **RESOLVED** | Both. `train/` (raw) and `processed/` coexist. |
| Q3 | Video file or stills? | **RESOLVED** | **Stills only.** No video anywhere in the folder. Drives §2. |
| Q4 | Observer-to-object range? | **OPEN** | Unknowable from data. Keep `--range-km` optional; leave physical speed `null` when absent. |
| Q5 | Multi-object images? | **RESOLVED** | **None.** 88,000 images, exactly 1 annotation each. `convert_labels.py:111` is therefore correct, not a bug. |

### New questions

| # | Question | Blocks |
|---|----------|--------|
| Q7 | Should `yolo_dataset/` be synced directly (it is already split into `images/` + `labels/`), or rebuilt from `train/` + `labels/train.csv`? | Sync scope and runtime. |
| Q8 | Should `val.csv` (22k images, currently unused by the codebase) be used as the pool from which synthesis stills are drawn? | Evaluation design. |
