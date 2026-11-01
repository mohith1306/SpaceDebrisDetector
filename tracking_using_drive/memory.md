# Memory — `tracking_using_drive`

Persistent working context. Source of truth for *where things stand*,
*what was decided and why*, and *what would bite the next person*. Update at
the end of every work session.

---

## Current State

| Item | Value |
|------|-------|
| Branch | `tracking_using_drive` (tracks `origin/tracking_using_drive`) |
| Created | 2026-11-01 |
| Active phase | **Phase 0 — Baseline and Data Verification** (not started) |
| Code changes made | **None.** Planning docs only. |
| Baseline recorded | No |
| Blocking question | **RESOLVED 2026-11-01** — Option A confirmed. Scope is locked. |

### Goal
Complete the tracking of debris using the data stored in Google Drive.

### Drive source
`Capstone` folder — ID **`1vxqeBHOglilc1PP_ZSNA1fCx_jfTTdUY`**
`https://drive.google.com/drive/u/0/folders/1vxqeBHOglilc1PP_ZSNA1fCx_jfTTdUY`
Publicly readable; listing obtained via
`https://drive.google.com/embeddedfolderview?id=<ID>#list`.

### Isolation constraint
**All work happens inside `tracking_using_drive/`.** No file outside it may be
modified, added, or deleted. `space_debris/` is a read-only library. Enforced
by `rules.md` R0 and the pre-commit check
`git status --porcelain | grep -v "tracking_using_drive/"`.

---

## Verified Dataset Facts

Measured 2026-11-01 by downloading `labels/train.csv` (3.16 MB) and
`labels/val.csv` (1.05 MB) and counting. Not assumptions.

### Drive folder structure
```
Capstone/  (1vxqeBHOglilc1PP_ZSNA1fCx_jfTTdUY)
├── train/                  raw images
├── processed/              preprocessed images
├── labels/
│   ├── train.csv           66,000 rows
│   ├── val.csv             22,000 rows
│   └── sample_submission.csv
├── labels_yolo_2k/         YOLO labels, 2k subset
├── yolo_dataset/{images,labels}/
├── space_debris.yaml
├── train.zip
├── CAPSTONE.ipynb
└── CAPSTONE (4)_backup.ipynb
```

### Annotation statistics
| File | Rows | Unique images | Max annots/img | Classes |
|------|------|---------------|----------------|---------|
| `train.csv` | 66,000 | 66,000 | **1** | 11 × 6,000 |
| `val.csv` | 22,000 | 22,000 | **1** | 11 × 2,000 |

- Schema `filename,class,bbox`, bbox as `"[x_min, y_min, x_max, y_max]"`.
- All `.png`. **Zero filename overlap** between train and val — no leakage.
- Balance is exact across all 11 classes.

### THE GOVERNING FACT
> **88,000 independent, single-object still images. No video sequences, no
> temporal ordering, no multi-object frames.**

ByteTrack associates across frames of the same video. Fed unrelated stills it
assigns a **new track ID per image**. Every track is length 1, nothing reaches
`MIN_POINTS_FOR_ORBIT = 3`, and the orbit report comes back **empty** — with no
error raised. The repository already knew this; `scripts/make_demo_video.py:5-6`
says so in its docstring.

---

## Decisions

| # | Date | Decision | Rationale |
|---|------|----------|-----------|
| D1 | 2026-11-01 | Drive is a **delivery** mechanism, not a per-frame transport. `resolve_source()` syncs then returns a local path. | Network cost O(dataset) instead of O(frames); downstream untouched. |
| D2 | 2026-11-01 | Reuse `GDriveSync`; change only how it is *called*. | Its incremental skip (`gdrive_sync.py:67-81`) is what makes a second run free. A rewrite risks it. |
| D3 | 2026-11-01 | Credentials resolve deterministically, overridable via `GDRIVE_CREDENTIALS`. | `Path("credentials.json")` is CWD-relative and scatters secrets. |
| D4 | 2026-11-01 | Missed detections use `TrajectoryBank.miss()` with `dt = 1/rate`, expiring after `--max-missed` (30). | `miss()` exists and is correct. Raw `dt=1.0` over-predicts at any other rate. |
| D5 | 2026-11-01 | Do not rely on ByteTrack's internal Kalman. | Unexposed in the JSONL; orbit stage needs the centres anyway. Two disagreeing motion models is worse than one. |
| D6 | 2026-11-01 | Expire on a generous threshold, not first miss. | Single-frame dropouts are common; ByteTrack recovers. |
| D7 | 2026-11-01 | No new Drive client dependency. | `pydrive2` already in both requirements files. |
| **D8** | **2026-11-01** | **"Complete the tracking" = validating the pipeline on synthesised sequences derived from Drive stills, with per-image detection as the honest degenerate case. CONFIRMED by the project owner.** | The Drive data has no sequences. Real debris orbits cannot be determined from it. Synthesis is the only path producing a defensible, measurable result — success is a pipeline with quantified error against known ground truth, not a set of orbits. |
| D9 | 2026-11-01 | `space_debris/` is read-only; broken pieces are reimplemented inside `tracking_using_drive/`, correct pieces imported. | Isolation constraint. `stream.py`, `main.py`, `gdrive_sync.py` defaults, and `resolve_source` cannot be patched in place. |
| D10 | 2026-11-01 | Set `CAPSTONE_DIR` **before** importing `space_debris.config`, scoping its paths under `tracking_using_drive/_data/`. | `config.py:19-20` reads the env at import and `:117-118` creates dirs at import. Cannot edit either; must control via environment. |
| D11 | 2026-11-01 | Selective sync via allowlist, not whole-folder. | Folder holds `train/`, `processed/`, `yolo_dataset/` — three overlapping copies totalling 88k images. A tracking run needs a slice. |
| **D12** | **2026-11-01** | **Occlusion is implemented by erasing the object region, not translating it off-canvas.** | Off-canvas translation can leave the object detectable depending on padding. Erasure guarantees a true dropout, which is the only way to test D4/FR-4.1. |

---

## Codebase Facts Worth Not Forgetting

### Verified NOT a bug
**`convert_labels.py:111-113`** writes one label file per annotation *row*,
which looks like a multi-object overwrite defect. **Verified 2026-11-01: it is
correct.** All 88,000 images have exactly one annotation, so the per-row write
never overwrites anything. Do not "fix" it. (`rules.md` R8.)

### The occlusion defect
- `TrajectoryBank.miss()` is defined at `orbit.py:115-118` and is **never
  called anywhere in the codebase**.
- The stream loop (`stream.py:118-144`) only calls `bank.update()` for tracks
  present in the current frame.
- Consequence: an occluded track freezes. On reappearance ByteTrack may assign
  a new ID → fragmentation → both fragments below `MIN_POINTS_FOR_ORBIT = 3` →
  both discarded. Occlusion silently deletes results.
- Listed as "Partially implemented" in `PREPROCESSING_AND_TRACKING.md:804`.

### Latent bugs (unfixable in place — reimplement instead)
- **`stream.py:181`** — `width`/`height` are loop-local but used after the
  loop. Zero frames → `UnboundLocalError`. Variable resolution → last frame's
  size applied to every track.
- **`gdrive_sync.py:15`, `:133`** — `E:/Capstone/gdrive_cache` literal.
- **`gdrive_sync.py:18`** — `Path("credentials.json")` is CWD-relative.
- **`main.py:161-164`** — auto-sync gate excludes `track`/`stream`/`orbit`.
- **`track.py:41-48`** — `resolve_source` cannot accept `gdrive://`.
- **`train.py:9`** — lists `"yolo11"` in `VALID_ARCHITECTURES`, but
  `main.py:68-70` never offers it. Unreachable from the CLI.
- **`yolov8_p1p2.yaml:56`** — `Detect` takes `[21,24,27,30]` while the header
  comment implies P1–P4. Comment and code disagree.

### Correct — do not touch
- `ConstantVelocityFilter` (`orbit.py:26-91`) — numpy-only, no scipy/filterpy
  by design.
- `estimate_trajectory`'s `missing` list (`orbit.py:171`) — reports what could
  **not** be determined. Preserve.
- Fusion thresholds `min_cls_conf=0.45`, `agreement_boost=0.10`
  (`classify.py:193-194`) — tuned; changing invalidates prior runs.
- `make_confusion_matrix.py:33-54` raising `max_time_img` — deliberate;
  Ultralytics silently drops detections when it trips.

### The 11-class taxonomy (`config.py:43-55`)
```
cheops, debris, double_star, earth_observation_sat_1, lisa_pathfinder,
proba_2, proba_3_csc, proba_3_ocs, smart_1, soho, xmm_newton
```
`DEBRIS_CLASSES = ["debris"]`.

---

## Gotchas

| # | Gotcha | Why it matters |
|---|--------|----------------|
| G1 | `CAPSTONE_DIR` must be set **before** `space_debris.config` is first imported anywhere in the process. | It is read once at import (`config.py:19-20`). A stray `from space_debris...` elsewhere captures the wrong `BASE_DIR` and paths silently escape the R0 boundary. |
| G2 | `config.py:117-118` creates four directories at import. | Unavoidable side effect. D10 scopes it under `_data/`. |
| G3 | `persist=True` required on every `model.track()`. | No error without it; all cross-frame identity destroyed. |
| G4 | `torch.backends.mkldnn.enabled = False` (`train_resilient.py:12`). | oneDNN SIGFPEs on virtualised CPUs (JarvisLabs). |
| G5 | `requirements-gpu.txt` must not pin `torch`/`torchvision`. | JarvisLabs template supplies CUDA builds. **Note: editing it is now out of bounds anyway.** |
| G6 | Split seed is 42 (`split_dataset.py:35`). | Changing invalidates prior comparisons. Prefer Drive's shipped `train.csv`/`val.csv` — already zero-overlap. |
| G7 | `preprocess.py` caps at 2000 images by default. | Workspace holds ~66k. |
| G8 | Track/Stream pass source **directly** to Ultralytics — no preprocessing. | The data pipeline applies grayscale + CLAHE. Parity is R5. |
| G9 | `orbit.py` assumes `cx`/`cy` in `[0,1]`. | The `abs(cx) > 1.5` check (`orbit.py:295`) is a legacy fallback, not a licence for pixel centres. |
| G10 | Drive folder holds **three overlapping copies** of the images (`train/`, `processed/`, `yolo_dataset/`). | Whole-folder sync is wasteful and slow. D11. |

---

## Open Questions

| # | Question | Blocks | Status |
|---|----------|--------|--------|
| Q4 | Observer-to-object range? | Physical speed stays `null` | OPEN — accepted as unknowable from data |
| Q7 | Sync `yolo_dataset/` directly (already split) or rebuild from `train/` + `train.csv`? | Sync scope, runtime | OPEN |
| Q8 | Use `val.csv` (22k images) as the pool for synthesis stills? | Evaluation design | OPEN |

Resolved: **Q1** (full layout), **Q2** (both raw and processed present),
**Q3** (stills only — no video anywhere), **Q5** (no multi-object images),
**Q6** (**Option A confirmed 2026-11-01** — synthetic-sequence validation).

---

## Proposed Upstream Changes (NOT applied — out of bounds)

Recorded here because `space_debris/` cannot be modified on this branch. If
the isolation constraint is ever lifted, these are worth upstreaming:

1. `gdrive_sync.py` — remove `E:/Capstone` defaults; derive credential path.
2. `main.py` — add `track`/`stream` to the `ensure_data_synced()` gate.
3. `stream.py` — call `bank.miss()` for absent tracks; capture first-frame dims.
4. `track.py` — `resolve_source` accepts a `gdrive://` scheme.
5. `train.py:9` — remove unreachable `"yolo11"` from `VALID_ARCHITECTURES`, or
   expose it in `main.py`.
6. `yolov8_p1p2.yaml` — reconcile the header comment with the 4-scale Detect.

---

## Session Log

### 2026-11-01 — Branch setup, planning, data reconnaissance
- Cloned `mohith1306/SpaceDebrisDetector` into `Capstone/`.
- Created and pushed branch `tracking_using_drive`.
- Read the full codebase: `config.py`, `main.py`, all of `data/`, all of
  `models/`, `utils/orbit.py`, `utils/visualize.py`, both requirements files,
  `train_jarvis.py`, `scripts/*`, both model YAMLs, and
  `PREPROCESSING_AND_TRACKING.md`.
- Created the planning folder: `requirements.md`, `architecture.md`,
  `rules.md`, `phases.md`, `design.md`, `memory.md`.
- **Inspected the live Drive folder.** Listed `Capstone/` and subfolders via
  the embedded-folderview endpoint (public folder, no auth needed).
- **Downloaded and analysed both CSVs.** Established the 88k single-object
  stills fact, the exact class balance, and the zero train/val overlap.
  Resolved Q1, Q2, Q3, Q5. Raised Q6, Q7, Q8.
- Recorded the isolation constraint and reworked all six docs for the shadow-
  package architecture (D9, D10) and the sequence-synthesis stage (D8, D12).
- **Q6 confirmed with the project owner: Option A — validate on synthetic
  sequences.** Scope locked; D8 upgraded from assumption to decision. No gate
  remains before Phase 3.
- **No runtime code changed.**

---

## Next Actions

1. **Start Phase 0.** Reproduce the degenerate case (stream over unrelated
   stills → empty orbit report) so the need for synthesis is documented as
   evidence, not assertion.
2. **Phase 1** — package skeleton and the `CAPSTONE_DIR` bootstrap, with a
   test that the resolved paths stay inside the R0 boundary.
3. **Phase 2** — Drive access: deterministic credentials, selective sync
   allowlist, `gdrive://` resolution.
4. Commit the planning folder before any code, so the reasoning is in version
   control ahead of the first implementation commit.
5. Resolve Q7 (which Drive subfolder to sync) during Phase 2 design — it
   affects runtime but not architecture.

The scope is locked (D8). No further gate remains before Phase 3.
