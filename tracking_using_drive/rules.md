# Rules — `tracking_using_drive`

Binding constraints for work on this branch.

---

## R0 — Isolation: nothing outside this folder changes

**This is the controlling rule.** All work on the `tracking_using_drive`
branch happens inside `tracking_using_drive/`. No file outside it may be
modified, added, or deleted — not `space_debris/`, not `scripts/`, not
`train_jarvis.py`, not `requirements*.txt`, not `.gitignore`.

`space_debris/` is a **read-only library**. Import from it; never write to it.

### What this means in practice

| Wanted change | Cannot do | Must do instead |
|---------------|-----------|-----------------|
| Add `gdrive://` to `resolve_source()` | Edit `track.py:41-48` | Write `tracking_using_drive/source.py` with its own resolver |
| Fix the `stream.py:181` unbound `width` | Edit `stream.py` | Reimplement `run_stream` in `tracking_using_drive/stream.py` |
| Call `bank.miss()` for absent tracks | Edit `stream.py:118-144` | Own stream loop |
| Fix `E:/Capstone` in `gdrive_sync.py:15,133` | Edit `gdrive_sync.py` | Own Drive module |
| Auto-sync before `track` | Edit `main.py:161-164` | Own CLI that calls `ensure_data_synced()` itself |

### The pattern

```
tracking_using_drive/          ← we own this, read-write
    ↓ imports (read-only)
space_debris/                  ← we do not own this, read-only
```

Import the correct pieces, reimplement only what is broken:

```python
# CORRECT — reuse what works
from space_debris.utils.orbit import TrajectoryBank, estimate_trajectory, summarize_track
from space_debris.models.classify import classify_crops, fuse, load_classifier

# CORRECT — own implementation of what is broken
from .stream import run_stream          # not space_debris.models.stream
from .source import resolve_source      # not space_debris.models.track.resolve_source
```

### Verification
```bash
git status --porcelain | grep -v "tracking_using_drive/"   # must be empty
```
Run before every commit. A single stray file outside the folder is a rule
violation, not a convenience.

### One permitted side effect
Importing `space_debris.config` creates `processed/`, `labels_yolo/`,
`model_checkpoints/`, and `runs/` in the project root (`config.py:117-118`).
That is an import side effect, not a source edit, and is unavoidable without
editing the file. Mitigate it by setting `CAPSTONE_DIR` to a
`tracking_using_drive/`-scoped cache **before** the import, so the directories
land under our control (see `design.md` §3).

---

## R1 — Never reorder `CLASS_NAMES`

`config.py:43-55` fixes the order of the 11 classes. `convert_labels.py` writes
these ids into YOLO label files and `get_yaml_config()` writes the same order
into `space_debris.yaml`. Reordering raises no error — it silently relabels the
training set against existing checkpoints.

Verified class order and balance (2026-11-01):
```
cheops(6000) debris(6000) double_star(6000) earth_observation_sat_1(6000)
lisa_pathfinder(6000) proba_2(6000) proba_3_csc(6000) proba_3_ocs(6000)
smart_1(6000) soho(6000) xmm_newton(6000)          # train; val is 2000 each
```
Append-only. Never insert, never reorder.

---

## R2 — `persist=True` on every tracking call

`track.py:139`, `stream.py:85`. Dropping it makes every frame an independent
detection set and destroys all cross-frame identity. No error is raised.

---

## R3 — Track with normalised centres

`orbit.py` assumes `cx`/`cy` in `[0, 1]`. The `abs(cx) > 1.5` tolerance at
`orbit.py:295` is a legacy fallback for old pixel-centre records, not a licence
to write pixel centres.

---

## R4 — Split seed is 42

`split_dataset.py:35`. Changing it invalidates every comparison against a prior
run. Note the Drive data already ships a `train.csv`/`val.csv` split with zero
overlap — prefer it over re-splitting.

---

## R5 — Preprocessing must be identical everywhere

`track.py:132-143` and `stream.py:80-87` pass the source **directly** to
Ultralytics with no preprocessing, while the data pipeline applies grayscale +
CLAHE. If preprocessing joins the tracking path it must join **both**, and must
reuse `preprocess_image()` — never a second inline copy.

---

## R6 — `mkldnn` stays disabled on CPU paths

`train_resilient.py:12`. oneDNN SIGFPEs on virtualised CPUs (JarvisLabs VMs).

---

## R7 — Synthetic output must never be presented as measured

This branch generates sequences from stills. Any artifact derived from them
carries `synthetic: true` in its metadata. It is a **validation instrument**,
not a measurement of debris motion. A report that implies a real orbit was
determined from a `warpAffine` translation is a correctness failure, not a
presentation choice.

---

## R8 — Never "fix" `convert_labels.py`

`convert_labels.py:111-113` writes one label file per annotation *row*, which
looks like a multi-object overwrite bug. **Verified 2026-11-01: it is not a
bug.** All 88,000 images have exactly one annotation. The per-row write is
correct for this dataset.

It would become a bug only if a multi-object dataset were ever introduced.
Leave it alone.

---

## Code Style

| Rule | Detail |
|------|--------|
| `pathlib.Path` | No `os.path` string joining. |
| No hardcoded absolute paths | `E:/Capstone` must never reappear. |
| Relative imports inside our package | `from .config import ...` within `tracking_using_drive/`. |
| Absolute import for the library | `from space_debris.utils.orbit import ...` — `space_debris` is a sibling package, not a parent. |
| JSONL records stay flat-ish | One object per line; `detections` the only nested list. |
| Round at the boundary | `round(..., 2)` for boxes, `round(..., 4)` for normalised coords and confidence. Never round inside the Kalman filter. |
| Docstrings state assumptions | `orbit.py` is the model. |
| Fail loud on unknown classes | `config.class_id()` raises `KeyError` with the sorted expectation list. |

---

## Git Workflow

```
main ─────────────────────────────────────────►
  └── tracking_using_drive ──────────────────►  (this branch)
```

- Branch `tracking_using_drive`, tracks `origin/tracking_using_drive`.
- Commit prefix by area: `drive:`, `source:`, `stream:`, `track:`, `synth:`,
  `docs:`.
- Imperative mood: `drive: resolve credentials path deterministically`.
- Never force-push.
- Never commit `gdrive_cache/`, `credentials.json`, `*.pt`, `runs/`,
  `processed/`, or synthesised videos.

### Pre-commit checklist
```
1. git status --porcelain | grep -v "tracking_using_drive/"   → empty
2. No hardcoded absolute path in CODE: run the AST guard —
   python -m pytest tracking_using_drive/tests/test_drive.py -q -k hardcoded
   (A raw grep for the literal is unsatisfiable: the guard and the module
   docstrings quote it deliberately. FR-3.1 constrains code, not prose.)
3. python -c "import tracking_using_drive"
4. python tracking_using_drive/main.py --help
5. No gdrive_cache/, credentials.json, *.pt, or runs/ staged
```

---

## Testing Rules

### R9 — Synthetic video is the oracle
The dataset has no sequences, so synthesised motion is the **only** available
ground truth. Any change to tracking, Kalman, or orbit code MUST be validated
against it, and the angular rate must land within 10% of ground truth.

### R10 — Test cold cache and warm cache
Both must pass. Most Drive bugs live in the warm path, where stale
`credentials.json` or a mismatched file size causes a silent re-download or a
silent skip.

### R11 — Test occlusion explicitly
Construct a sequence where the object exits and re-enters. Assert the track ID
is preserved and that missed frames show **predicted**, not frozen, motion.
This is the only way to verify FR-4.1.

### R12 — Assert on the JSONL, not stdout
The summary dict is convenient but lossy.

---

## Performance Rules

| Rule | Detail |
|------|--------|
| Never buffer all frames | `stream=True` yields lazily. Never `list(results)`. |
| Sync is a startup cost | No network inside the frame loop. |
| Kalman is per-track O(1) | Miss-propagation is O(active tracks) per frame. |
| Cache image sizes | `convert_labels.py` memoises to `.image_sizes.json`. |
| Selective sync | Do not download all 88k images for a 300-frame validation run. |
| Do not raise NMS limits casually | `make_confusion_matrix.py:33-54` does so deliberately — Ultralytics silently drops detections when it trips. |

---

## Dependency Rules

- No second Drive client. `pydrive2` is already in both requirements files.
- Do not edit `requirements*.txt`. If a new dependency is genuinely required,
  record it in `memory.md` as a blocked action needing upstream approval.
- No `filterpy`/`scipy` for the Kalman filter. `orbit.py` is numpy-only by
  design.

---

## What Not To Do

- Do not restructure `space_debris/` "for cleanliness". It is out of bounds.
- Do not present synthetic trajectories as measured debris orbits (R7).
- Do not add a full orbital determination module. The data cannot support it.
- Do not make Drive a per-frame transport.
- Do not silently swallow exceptions in the sync path. Per-file warnings at
  `gdrive_sync.py:98-99` are the right granularity; a blanket `except: pass`
  at folder level is not.
- Do not download the whole `Capstone` folder for a tracking run. `train/`,
  `processed/`, and `yolo_dataset/` are three copies of overlapping data.
