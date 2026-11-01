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
| Active phase | **Phase 2 — complete.** Anonymous Drive sync, allowlist, `gdrive://` resolution, CLI. |
| Code changes made | **Yes, all inside `tracking_using_drive/`** (`drive.py`, `source.py`, `cli.py`, `tests/test_drive.py`). |
| Isolation verified | `git status` clean outside the folder; 14/14 library paths scoped under `_data/` |
| Baseline recorded | **Not yet — unblocked via D13** (injected detections, no weights needed) |
| Blocking question | **None.** Scope locked (D8); weights blocker bypassed (D13); Phase 2 needs no credentials (D15). Only Q9 (JarvisLabs token) is outstanding, and it gates the optional Phase 5 alone. |

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

## Environment (audited 2026-11-01)

Local machine, `C:\Users\naras\OneDrive\Desktop\Capstone\SpaceDebrisDetector`:

| Item | Value | Consequence |
|------|-------|-------------|
| Python | 3.12.2 | OK |
| torch | **2.15.0+cpu** (upgraded from 2.7.1 by ultralytics) | CPU only locally. No CUDA, no MPS. |
| ultralytics | **8.4.197** ✅ installed | Repo pins `8.4.161` — see G11. |
| pydrive2 | **installed** ✅ | Drive sync now possible. |
| cv2 / numpy / pandas | 4.11.0 / 2.0.1 / 2.2.2 | Present. |
| Data locally | **none** | No `train/`, `labels/`, `processed/`, `yolo_dataset/`. |
| Weights locally | **none** | No `.pt` anywhere in the repo. |
| **Training platform** | **JarvisLabs (GPU)** | Local torch is CPU-only by design. See D14. |

Git identity had to be set locally (`user.name = "naras"`) — `user.email` was
already configured. Change it if a different name is wanted.

A pre-existing unrelated pip warning: `wsproto 1.3.2 requires h11<1,>=0.16.0,
but you have h11 0.14.0`. Does not affect this work.

### The weights blocker

**No trained detector weights are reachable.** Verified:

- Not in the repo — no `.pt` anywhere (gitignored upstream).
- Not in the shared Drive folder — the `Capstone` listing has 10 top-level
  items (`train/`, `processed/`, `labels/`, `labels_yolo_2k/`,
  `yolo_dataset/`, `space_debris.yaml`, `train.zip`, two notebooks,
  `.ipynb_checkpoints/`) and **no `runs/` or `model_checkpoints/` folder**.
- The notebook `CAPSTONE.ipynb` shows they were trained in **Google Colab**
  and written to the author's **private MyDrive**:
  ```
  /content/drive/MyDrive/Capstone/runs/detect/weights/best.pt
  /content/drive/MyDrive/Capstone/runs/detect/weights/last.pt
  /content/drive/MyDrive/Capstone/model_checkpoints
  ```
  `MyDrive/Capstone` is the author's own drive, **not** the shared folder.
  They were never uploaded to the share.

Consequence: nothing in Phases 0, 3, 4, 5 that requires *detections* can run
until weights are obtained or the tracking layer is validated without them.
See **D13** for the proposed workaround.

### Phase 0 task status

| Task | Status |
|------|--------|
| Environment audit | ✅ Done |
| Install `ultralytics` + `pydrive2` | ✅ Done (8.4.197, pydrive2) |
| Verify Drive CSV facts locally | ✅ Done (66k/22k, 1 annot each, zero overlap) |
| Confirm R0 isolation | ✅ Done — `git status` clean outside the folder |
| Baseline `stream` run | ⏳ Needs weights (D14 path) or injected detections (D13) |
| Reproduce degenerate case | ⏳ Same — doable via D13 without any weights |
| `make_demo_video.py` run | ⏳ Needs one synced image |

---

## Phase 1 Record (complete 2026-11-01)

### Delivered
| File | Role |
|------|------|
| `__init__.py` | Cheap bootstrap — puts repo root on `sys.path` and sets `CAPSTONE_DIR`. No torch, no library import. |
| `config.py` | Heavy import, path guard, `verify_scoped_paths()`, `data_presence()`. |
| `cli.py` | argparse entry point; `status` and `verify` implemented, roadmap in `--help`. |
| `main.py` | `python -m tracking_using_drive.main <cmd>` |
| `tests/test_config.py` | 11 tests incl. two subprocess tests of the bootstrap. |
| `.gitignore` | Guards `_data/`, `credentials.json`, weights, runs. |

### The two-layer bootstrap
1. `__init__.py` sets `CAPSTONE_DIR` **cheaply** on package import — before
   anything can import the library. This is what makes the scoping robust no
   matter how the package is entered.
2. `config.py` adds `_guard_pre_existing_import()`, which raises
   `RuntimeError` if `space_debris.config` was already imported with a
   `BASE_DIR` outside the package. Once imported, that value is fixed for the
   life of the process and re-importing cannot move it.

Both layers are exercised by subprocess tests, so a regression fails the suite
rather than silently scattering directories.

### G2 reproduced and confirmed
Importing `space_debris.config` with an unscoped `CAPSTONE_DIR` created
`processed/`, `labels_yolo/`, `model_checkpoints/` and `runs/` **in the
repository root**. Observed directly during Phase 0 verification, then cleaned
up. The bootstrap now routes all four into `tracking_using_drive/_data/`.

### New gotchas found while building it
- **G14** — `space_debris.config.verify_paths()` prints U+2713/U+2717. The
  Windows console default is cp1252, which cannot encode them, so calling it
  raises `UnicodeEncodeError`. We do not call it; `data_presence()` checks the
  paths directly and returns a dict. `cli.py` also reconfigures stdio to UTF-8
  with `errors="replace"` so an unencodable glyph cannot kill the process.
- **G15** — `space_debris.config` prints `Using device: cpu` on import. A
  library side effect we cannot remove (R0). Harmless; appears above our
  banner. Suppressing it would risk hiding real warnings, so it stays.
- **G16** — running pytest from the repo root creates `.pytest_cache/` there.
  Gitignored by the upstream `.gitignore`, so no R0 violation, but remove it
  rather than let it accumulate. Run with `-p no:cacheprovider` to avoid.

### Test results
```
11 passed in 10.33s
```
Covers: no path escapes the package, `CAPSTONE_DIR` points inside it,
`BASE_DIR` matches the env var, class taxonomy order (R1 guard), `DEBRIS_CLASSES`,
`class_id` hit/miss, bootstrap scoping in a fresh interpreter, the
unscoped-import guard firing, and `cli verify` returning 0.

---

## Phase 2 Record (complete 2026-11-01)

### Delivered
| File | Role |
|------|------|
| `drive.py` | Anonymous-first Drive access, selective allowlist sync, manifest, `DriveError`. |
| `source.py` | `resolve_source()` with `gdrive://` support; `is_still()` / `is_video()`. |
| `cli.py` | Added `sync` and `source` subcommands. |
| `tests/test_drive.py` | 30 tests (2 network-marked, opt-in via `CAPSTONE_NETWORK_TESTS=1`). |

### D15 — anonymous access removes the credential blocker
The shared folder is **public**. Listing (`embeddedfolderview?id=`) and
download (`uc?export=download&id=`) both work with no OAuth. Verified live.
Consequence: Phases 2–4 carry **no credential dependency at all**. OAuth is
retained only for private folders and for enumeration past the 5,500 listing
cap (G17). This is a stronger position than the original plan, which assumed
`credentials.json` was a hard prerequisite.

### Measured facts (live, 2026-11-01)
| Fact | Value |
|------|-------|
| Cold sync of the allowlist | **1,086 files, 171,915,492 bytes (~164 MB)** |
| Warm sync | **0 network calls**, 1,086 cache hits |
| `network_calls` accounting | exact: 4 on a 2-file cold sync (2 listings + 2 downloads) |
| `yolo_dataset/images/val` | **542** stills, all stems in `train.csv` |
| `train/` listing | capped at **5,500** (indices 1–19,075 of 110,000) — G17 |
| `train.zip` | **2,456 bytes** — broken placeholder, ignore it — G19 |
| CSV vs disk extension | CSV says `.png`, disk says `.jpg`; matched on **stem** — G18 |
| `val.csv` images | **none exist** in the shared folder — G21 |

### Q7 answered for the JarvisLabs path
`jarvis.sh upload` pushes local `train/` + `labels/`, and `train_jarvis.py`
sets `CAPSTONE_DIR=ROOT` then rebuilds `yolo_dataset/` **remotely**. So the
training path needs the **raw** layout, not `yolo_dataset/`. Rebuilding from
`train/` + `train.csv` is correct; syncing `yolo_dataset/` for training is
not. (For Phases 3–4, `yolo_dataset/images/val` *is* the right source — 542
stems, all resolvable against `train.csv` for bboxes.)

### Q8 answered — and it overturns an assumption
`val.csv` cannot be a synthesis pool. **All 22,000 of its images are absent**
from the shared folder; every image present (in `train/`, `processed/`, and
`yolo_dataset/`) has a stem in `train.csv`, none in `val.csv`. `val.csv` is
annotation-only. Synthesis must draw from `train.csv` stems.

### Two real bugs caught by the tests
1. **Warm path never triggered.** The manifest is keyed per *file*
   (`.../val/img000013.jpg`), but `_warm_check` looked up the allowlist entry
   verbatim (`.../val`) — a directory prefix that is never itself a key. Every
   warm run silently fell through to a full re-resolve. Fixed with
   `_expand_entry()`.
2. **Corruption was blessed permanently.** `_needs_fetch` re-downloaded only
   when a file was *absent or zero-length*. A size mismatch passed straight
   through, and `_write_manifest` then recorded the corrupted size — making
   the damage permanent and invisible. Fixed to compare against the manifest's
   recorded byte count. Caught by appending one byte to `train.csv` and
   asserting exactly one re-download.

### A rule conflict, resolved
`rules.md` demanded `grep -rE "E:/Capstone" tracking_using_drive/` be empty.
That is **unsatisfiable**: the guard test must contain the literal to check
for it, and `drive.py` quotes it in a docstring to explain the defect. FR-3.1
says the literal may not appear in *code*. Reconciled: the check is now
AST-based (`test_no_hardcoded_drive_path`), ignoring docstrings. Updated in
`rules.md`, `phases.md`, and `requirements.md` §8.

### Test results
```
30 passed in 15.60s        # incl. 2 live-network tests
```
Offline: listing parser against recorded HTML, HTML-interstitial detection,
`_expand_entry`, allowlist smallness, no hardcoded path in code, camera-index
round-trip, still/video classification, empty-dir failure, warm cache zero
calls, `--no-sync` on empty cache, unknown path, report metadata.
Network (opt-in): live folder listing, live CSV download and header check.

---

## JarvisLabs Training Flow (D14)

The repo is already provisioned for GPU training off-box. What the branch must
add is the **Drive → local → JarvisLabs** handoff, which does not exist yet.

### Existing tooling
| Asset | Role |
|-------|------|
| `train_jarvis.py` | Entrypoint: rebuild the YOLO dataset if needed, then `train_with_recovery()`. Sets `CAPSTONE_DIR` to the project root. |
| `scripts/jarvis.sh` | Instance lifecycle: `status`, `create [GPU]`, `upload <id>`, `train <id>`, `logs`, `pause`, `destroy`. |
| `requirements-gpu.txt` | GPU deps. Deliberately unpins torch. |
| `models/train_resilient.py` | Crash recovery, per-epoch logging, weight-carry across failures. |

### Current flow
```bash
jl setup --token YOUR_TOKEN --agents universal --yes   # one-time auth
./scripts/jarvis.sh create L4        # ~120GB: raw + processed + yolo_dataset + ckpts
./scripts/jarvis.sh upload <id>      # pushes space_debris/, train_jarvis.py,
                                     #   requirements-gpu.txt, train/, labels/
./scripts/jarvis.sh train <id> --epochs 30 --batch 16
```

### The gap this branch must close
`jarvis.sh upload` pushes **local** `train/` and `labels/`. Those do not exist
locally — they live in Drive. So today the flow cannot start.

**Required addition:** a `tracking_using_drive`-owned step that syncs
`labels/train.csv`, `labels/val.csv` (and the images) from Drive into the
layout `jarvis.sh` expects, **before** `upload`. This is Phase 2's deliverable
and is why Phase 2 precedes any training.

### What the branch does NOT do
- It does not edit `jarvis.sh`, `train_jarvis.py`, or `train_resilient.py`
  (R0). It orchestrates around them.
- It does not retrain the detector as part of tracking validation. Training is
  a one-time upstream step whose output (`best.pt`) the branch consumes.

### Two paths to detections, used together
| Path | Unblocks | Needs |
|------|----------|-------|
| **D13 — injected detections** | Tracking-layer validation now (Phases 3–4) | Nothing |
| **D14 — JarvisLabs training** | Full end-to-end validation with a real detector (Phase 5) | JarvisLabs token, synced data, GPU time |

D13 is not a fallback for D14 — it validates a *different layer*. Tracking
correctness (missed-detection propagation, track continuity, Kalman, orbit) is
provable without any detector. Detection accuracy is a separate claim that
needs the JarvisLabs run.

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
| **D13** | **2026-11-01** | **Validate the tracking layer in isolation by injecting synthesised detections at known centres, decoupling it from the detector.** `sequence.py` already knows the exact object position per frame, so it can emit a detection stream directly. | No weights are reachable from the share. This unblocks Phases 3–5 immediately, and isolates precisely the layer this branch fixes — missed-detection propagation and track continuity. Complements D14 rather than replacing it: real weights still matter for the full detection→tracking path. |
| **D14** | **2026-11-01** | **Training runs on JarvisLabs GPU, not locally. Local torch stays CPU-only.** | Stated by the project owner. The repo is already provisioned for this: `train_jarvis.py` (rebuild pipeline + train), `scripts/jarvis.sh` (instance lifecycle), `requirements-gpu.txt` (no torch pin — the pytorch template supplies CUDA builds). Local CPU is used only for inference and validation. |
| **D15** | **2026-11-01** | **Phase 2 Drive access is anonymous-first; OAuth is an optional fallback, not a prerequisite.** | Verified directly: the shared `Capstone` folder is public, so listing (`embeddedfolderview`) and download (`uc?export=download`) both succeed with no credentials. This removes the `credentials.json` blocker from Phases 2–4 entirely. `GDriveSync`/OAuth is retained only for private folders and for enumeration beyond the public listing cap (G17). |

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
| G6 | Split seed is 42 (`split_dataset.py:35`). | Changing invalidates prior comparisons. Prefer Drive's shipped `train.csv`/`val.csv` — already zero-overlap. |
| G7 | `preprocess.py` caps at 2000 images by default. | Workspace holds ~66k. |
| G8 | Track/Stream pass source **directly** to Ultralytics — no preprocessing. | The data pipeline applies grayscale + CLAHE. Parity is R5. |
| G9 | `orbit.py` assumes `cx`/`cy` in `[0,1]`. | The `abs(cx) > 1.5` check (`orbit.py:295`) is a legacy fallback, not a licence for pixel centres. |
| G10 | Drive folder holds **three overlapping copies** of the images (`train/`, `processed/`, `yolo_dataset/`). | Whole-folder sync is wasteful and slow. D11. |
| G11 | Installed ultralytics is **8.4.197**; `requirements.txt` pins **8.4.161**. | Tracker internals and default YAMLs can differ across versions. Record the version in every run's metadata, and pin back to 8.4.161 if a result looks version-dependent. Cannot edit `requirements.txt` (R0). |
| G12 | **`scripts/jarvis.sh upload` pushes local `train/` and `labels/`** to the instance. | Drive sync (Phase 2) is a hard prerequisite for JarvisLabs training — the data must be local before it can be uploaded. |
| G13 | `requirements-gpu.txt` must **not** pin `torch`/`torchvision`. | The JarvisLabs pytorch template supplies CUDA builds; pinning overrides them with CPU wheels. Out of bounds to edit anyway (R0). |
| G14 | `space_debris.config.verify_paths()` prints U+2713/U+2717 (`✓`/`✗`). | The Windows console default is **cp1252**, which cannot encode those glyphs — calling it raises `UnicodeEncodeError`. Use `data_presence()` from `tracking_using_drive.config`, which returns a dict and prints nothing. |
| G15 | `space_debris.config` prints `Using device: cpu` on import. | An unavoidable library side effect (R0). It appears above our banner. Suppressed stdout could hide real warnings, so it is left visible. |
| G16 | Running pytest from the repo root creates `.pytest_cache/` there. | Gitignored by the upstream `.gitignore`, so it is not an R0 violation — but remove it rather than let it accumulate. Run with `-p no:cacheprovider` to avoid it entirely. |
| G17 | The public Drive listing endpoint (`embeddedfolderview`) **caps at 5,500 entries per folder** and ignores `page=`. | `train/` holds far more (CSV references indices up to 110,000) but only the first 5,500 are enumerable anonymously. A full 66k anonymous sync is therefore **impossible** — it needs OAuth. A small allowlisted slice is unaffected. |
| G18 | `train.csv` names **`.png`**; the Drive folders hold **`.jpg`**. | Not a bug: `convert_labels.py:73` and `:80` join on `Path(csv_name).stem` and probe both extensions. Matching is by stem, so the mismatch is tolerated by design. Never "fix" it. |
| G19 | `train.zip` in the shared folder is **2,456 bytes** — an empty/corrupt placeholder. | Cannot be used as a data source; `jarvis.sh` does not reference it either. Ignore it. |
| G20 | `yolo_dataset/images/val` is only **542 images**, and all its stems are in **train.csv**, not `val.csv`. | The Drive `yolo_dataset` was built by `split_dataset.py` (seed 42, 90/10) from the 2,000 preprocessed images — a split of `train/`, unrelated to `val.csv`'s 22k. Never assume `images/val` corresponds to `val.csv`. |
| G21 | **All 22,000 `val.csv` images are absent from the shared folder.** `train/`, `processed/`, and `yolo_dataset/` contain only `train.csv` stems (verified: 0 `val.csv` stems anywhere). | `val.csv` is annotation-only. It cannot be used as a synthesis pool (Q8) — there are no images behind it. Synthesis must draw from `train.csv` stems. |

---

## Open Questions

| # | Question | Blocks | Status |
|---|----------|--------|--------|
| Q4 | Observer-to-object range? | Physical speed stays `null` | OPEN — accepted as unknowable from data |
| Q7 | Sync `yolo_dataset/` directly (already split) or rebuild from `train/` + `train.csv`? | Sync scope, runtime | OPEN |
| Q8 | Use `val.csv` (22k images) as the pool for synthesis stills? | Evaluation design | OPEN |
| Q9 | JarvisLabs token available? GPU budget? | **Phase 5 only** — Phases 3/4/6/7 work via D13 without it | OPEN |

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
- **Committed the planning set** as `d1ddc75` (7 files, 1,661 insertions),
  plus `tracking_using_drive/.gitignore` guarding `_data/`, `credentials.json`,
  and weights. Set local `user.name = "naras"` (was unset).
- **Phase 0 environment audit.** Python 3.12.2, torch **CPU-only**, cv2/numpy/
  pandas present, **`ultralytics` and `pydrive2` missing**. No data and no
  weights locally.
- **Traced the weights.** `CAPSTONE.ipynb` shows models were trained in Google
  Colab and written to the author's **private** `MyDrive/Capstone/runs/`, which
  is not the shared folder. **No weights are reachable.** Raised **D13**
  (proposed) to validate the tracking layer via injected detections instead.
- **Installed `ultralytics` 8.4.197 and `pydrive2`** (approved). torch bumped
  2.7.1+cpu → 2.15.0+cpu as a transitive upgrade. Local remains CPU-only.
- **D14 recorded: training runs on JarvisLabs GPU, not locally.** The repo is
  already provisioned (`train_jarvis.py`, `scripts/jarvis.sh`,
  `requirements-gpu.txt`). Added **Phase 5** to `phases.md` for the
  Drive→local→JarvisLabs handoff, which does not exist today — `jarvis.sh
  upload` pushes local `train/` and `labels/`, and those only live in Drive.
  Phase 2 now owns that staging.
- **Reframed D13 vs. D14 as complementary, not either/or.** D13 validates the
  tracking layer with zero detector dependency; D14 produces `best.pt` for the
  full detection→tracking path. Their results must be reported separately.
- Added G11 (ultralytics 8.4.197 vs. pinned 8.4.161), G12 (jarvis.sh needs
  local data), G13 (never pin torch in requirements-gpu.txt).
- **No runtime code changed.**
- **Pushed** `d1ddc75`, `d3f4604`, `4494973` to `origin/tracking_using_drive`
  on request.
### 2026-11-01 — Phase 1: package skeleton and bootstrap
- **Wrote the Phase 1 code**, all inside `tracking_using_drive/`:
  `__init__.py`, `config.py`, `cli.py`, `main.py`, `tests/test_config.py`,
  plus the `.gitignore` from earlier.
- **G2 reproduced during verification.** Importing `space_debris.config` with
  an unscoped `CAPSTONE_DIR` created `processed/`, `labels_yolo/`,
  `model_checkpoints/` and `runs/` **in the repository root**. Removed all four
  (empty) afterward. This is the exact hazard the bootstrap exists to prevent,
  and it is now confirmed empirically rather than by reading.
- **Implemented the two-layer bootstrap.** `__init__.py` primes
  `CAPSTONE_DIR` cheaply (no torch, no library import) on *any* package entry;
  `config.py` adds `_guard_pre_existing_import()` which raises `RuntimeError`
  if `space_debris.config` was already imported out of scope. Both layers are
  covered by fresh-interpreter subprocess tests.
- **Found G14:** `space_debris.config.verify_paths()` prints U+2713/U+2717,
  which the Windows cp1252 console default cannot encode — calling it crashes
  with `UnicodeEncodeError`. Replaced with `data_presence()`, which returns a
  dict instead of printing. `cli.py` also reconfigures stdio to UTF-8 with
  `errors="replace"` so no unencodable glyph can kill the process.
- **Found G15:** the library prints `Using device: cpu` on import — a side
  effect we cannot remove (R0). Left visible rather than suppressed, since
  suppressing stdout could hide real warnings.
- **Found G16:** pytest from the repo root drops `.pytest_cache/` there.
  Gitignored upstream (no R0 violation) but removed after the run.
- **11/11 tests pass** in 10.33s.
- `python -m tracking_using_drive.main status` reports all 14 library paths
  scoped under `tracking_using_drive/_data/`. `verify` exits 0.

### 2026-11-01 — Phase 2: Drive access and local staging
- **Pushed Phase 1** (`11b1405`) to `origin/tracking_using_drive`.
- **Read all four Phase 2 dependencies**: `gdrive_sync.py`, `jarvis.sh`,
  `train_jarvis.py`, root `.gitignore`, `space_debris/config.py`,
  `rules.md`, `design.md`.
- **D15 established: the shared folder is public.** Listing and download both
  succeed anonymously. Probed with five scripts before writing any production
  code — this finding removed the `credentials.json` blocker from Phases 2–4.
- **Reconciled four data contradictions** (all recorded as G17–G21): the
  5,500 listing cap; `.png` in CSV vs `.jpg` on disk (matched on stem, not a
  bug); the broken 2,456-byte `train.zip`; and the absence of all 22,000
  `val.csv` images.
- **Wrote `drive.py`** — anonymous-first sync, allowlist, manifest, warm path,
  transport-level network counter, `DriveError`.
- **Wrote `source.py`** — `resolve_source()` with `gdrive://`, plus
  `is_still()`/`is_video()` for the still-vs-sequence routing decision.
- **Extended `cli.py`** with `sync` and `source`.
- **Two real bugs caught by tests and fixed:**
  1. warm path never triggered — manifest keyed per file, looked up per
     directory prefix;
  2. size-mismatch corruption was permanently blessed by the next manifest
     write.
- **Resolved an unsatisfiable rule** (raw grep for the drive-path literal)
  by moving the check to AST and updating `rules.md`, `phases.md`,
  `requirements.md` §8.
- **30/30 tests pass**, including 2 live-network tests (opt-in via
  `CAPSTONE_NETWORK_TESTS=1`).
- Cold sync measured at **1,086 files / ~164 MB**; warm sync at **0 network
  calls**. Both recorded above.
- **Phase 2's JarvisLabs staging half is deliberately deferred** — it needs an
  R0 decision (see Next Actions 3), and only Phase 5 consumes it.

---

## Next Actions

1. **Phase 3 — sequence synthesis.** `sequence.py`: `Motion`, `Occlusion`,
   `GroundTruth`, occlusion-by-erasure (D12). Unblocked and weight-free; the
   542 cached stills are ready as input.
2. **Phase 4 — tracking correctness.** `stream.py` reimplemented with the
   missed-detection propagation fix and the **D13 injection path**, which
   needs no weights.
3. **Decide the JarvisLabs staging layout** (only matters for Phase 5). Two
   options, both R0-compatible since `train/` and `labels/` are gitignored
   upstream:
   - (a) `stage` writes real dirs at the repo root where `jarvis.sh upload`
     expects them, with `unstage` to remove them.
   - (b) `stage` keeps the bytes in `_data/` and creates repo-root
     **junctions**. Cleaner R0 story, but the repo lives inside **OneDrive**,
     and junctions under OneDrive are unreliable — likely not worth it.
   Leaning (a) with an explicit warning print, but this touches R0 (the
   controlling rule), so confirm before implementing.
4. **Ask about Q9** (JarvisLabs token) only when Phase 5 is actually reached —
   it is the sole optional phase.

The scope is locked (D8) and no gate remains. Phases 3–4 and 6–7 need no
external credentials, no GPU budget, and — since D15 — no Drive credentials
either.
