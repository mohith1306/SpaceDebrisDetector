# 3-Channel Physics-Motivated Preprocessing — Experiment Branch

Branch: `physics-motivated-preprocessing`

This branch contains a **controlled experiment** for the capstone project
*Detection and Tracking of Space Debris*: it answers whether a new image
preprocessing technique — the **3-Channel Physics-Motivated Pipeline** —
improves space-debris detection over the existing (baseline) preprocessing,
while keeping **everything else identical** (model, split, training config,
annotations, evaluation thresholds).

> **No improvement is claimed anywhere in this repository until actual
> experimental results demonstrate it.** Baseline code and outputs
> (`space_debris/`, `runs/detect/`) are untouched.

---

## 1. The preprocessing techniques

### Baseline (existing, unchanged)

```
RGB → grayscale → Gaussian 3×3 → CLAHE(1.5, 8×8) → Gaussian 3×3
    → min-max normalize → copy the SAME grayscale into all 3 channels
```

The network effectively sees **one** duplicated channel; a single bright
outlier can also dominate the min-max stretch.

### New: 3-Channel Physics-Motivated Pipeline

`preprocessing/physics_pipeline.py` produces **three engineered feature
maps** (not RGB colour channels), each pre-solving one physical imaging
problem:

| Channel | Operation | Why |
|---|---|---|
| **Ch1 — percentile-normalized intensity** | clip to 1st/99th percentiles, rescale 0–255 (degenerate range handled) | Robust global appearance: stars/hot pixels no longer squash the dynamic range (fixes min-max sensitivity) |
| **Ch2 — CLAHE local contrast** | Gaussian denoise **3×3, σ=0** (documented) → `CLAHE(clipLimit=1.5, tileGridSize=(8,8))` | Faint debris differs from its local background by only a few DN — adaptive local contrast reveals it |
| **Ch3 — background residual** | `GaussianBlur(σx=25)` of Ch1 → subtract → `+128` shift → clip 0–255 → `GaussianBlur 3×3` | Removes large-scale sky gradients/vignetting; compact targets glow on a near-black field |

Final image: `cv2.merge([ch1, ch2, ch3])` → `uint8 (H, W, 3)`.

Saved as JPEG **quality 95, 4:4:4 subsampling** (PNG lossless) so the
independent channel content is never averaged away by chroma subsampling.

---

## 2. Controlled-experiment guarantees

| Factor | Baseline | This experiment |
|---|---|---|
| Model | **p1p2-n** — custom `yolov8_p1p2.yaml` (P1/P2 heads), scale `n`, from scratch | **identical** (`create_custom_model` imported from baseline) |
| Training | 30 epochs, imgsz 640, batch 16, workers 4, patience 30, save every epoch | **identical** (+ crash recovery, same as `train_resilient.py`) |
| Annotations | 11-class YOLO `.txt` from the Drive dataset | **byte-identical copies** (SHA-verified by `verify_dataset.py`) |
| Classes | `space_debris.config.CLASS_NAMES` (fixed order) | **identical** (imported, not re-typed) |
| Eval thresholds | conf 0.25, IoU 0.5, imgsz 640 | **identical** (from `configs/physics_config.yaml`) |
| **HSV augmentation** | ultralytics defaults | **disabled** (`hsv_h=hsv_s=hsv_v=0`) — channels are engineered features, not RGB colours; geometric (box-aligned) augmentations retained |
| Inputs | duplicated-grayscale images | **3-channel physics pipeline** |
| Outputs | `runs/detect/` | `runs/physics_detect/` (baseline never touched) |

**Split decision (recorded):** the source Drive dataset is pooled and
re-split **80/20 with seed 42** (`configs/physics_config.yaml → split`).
For a strictly like-for-like comparison, the baseline should be retrained on
this same split; otherwise note the split difference in the report.

---

## 3. Installation

```bash
# baseline requirements (ultralytics, opencv, pydrive2, ...)
pip install -r requirements.txt          # local/CPU
pip install -r requirements-gpu.txt      # JarvisLabs (torch via template)

# Google Drive access (sync + automatic upload) - already in requirements
# OAuth client secrets: place credentials.json in the repo root
# (gitignored; same file the baseline gdrive_sync flow uses)
```

Python ≥ 3.10. No new third-party dependencies are introduced by this branch.

---

## 4. Google Drive dataset configuration

* **Dataset folder (Capstone root):**
  `https://drive.google.com/drive/u/0/folders/1vxqeBHOglilc1PP_ZSNA1fCx_jfTTdUY`
* Expected layout (auto-detected):

  ```
  Capstone/
  └── yolo_dataset/
      ├── images/train/*.jpg   images/val/*.jpg
      ├── labels/train/*.txt   labels/val/*.txt     (YOLO format)
      └── space_debris.yaml                        (optional)
  ```

* Processed output is written **locally** to `processed_dataset/` and then
  **automatically uploaded** to a new Drive folder
  `processed_dataset/` (inside the Capstone folder when write access allows,
  otherwise My Drive root). Uploads are resumable via
  `processed_dataset/.upload_manifest.json` — reruns never duplicate files.
* Original images/annotations are **never modified or overwritten**.

---

## 5. Running the experiment (JarvisLabs recommended)

All commands from the repository root.

### 5.1 Preprocessing (≈66k images)

```bash
# Dataset already on the machine (or uploaded):
python scripts/run_preprocessing.py --source /path/to/Capstone --workers 8

# Pull from Google Drive first (first run opens OAuth):
python scripts/run_preprocessing.py --sync --workers 8

# Quick trial (50 images, no upload):
python scripts/run_preprocessing.py --source /path/to/Capstone --limit 50 --no-upload

# Resume an interrupted Drive upload only:
python scripts/run_preprocessing.py --upload-only --out processed_dataset
```

Behaviour: batch/multiprocess (spawn) processing, tqdm progress,
checkpoint-skip of already-processed images, train and val partitions
processed **separately**, original labels copied byte-for-byte, processing
log + `processed_dataset/processing_summary.json` written
(processed / cached / corrupt / failed counts).

Via JarvisLabs helpers:

```bash
./scripts/jarvis_physics.sh create L4
./scripts/jarvis_physics.sh upload <id>
./scripts/jarvis_physics.sh preprocess <id> --sync      # build processed_dataset/
./scripts/jarvis_physics.sh train <id> --epochs 30 --batch 16
```

### 5.2 Verification (run before training)

```bash
python scripts/verify_dataset.py --processed processed_dataset --source /path/to/Capstone
```

Checks: YOLO structure, 1:1 image↔label pairing, dataset YAML, label format
and `[0,1]` ranges, **byte-identity of labels vs source**, 3-channel uint8
images with source dimensions, and **channel independence** (exact-equality
fractions + Pearson correlations per channel pair).

It also renders **20 six-panel comparisons**
(original / grayscale / Ch1 / Ch2 / Ch3 / final) into
`logs/visualizations/` plus `channel_statistics.json`.

### 5.3 Training (separate, explicit step — never starts automatically)

```bash
python scripts/train_physics_model.py                     # full run (30 epochs)
python scripts/train_physics_model.py --preflight-only    # dataset + forward-pass check only
python scripts/train_physics_model.py --resume            # continue after interruption
```

* Preflight proves the model accepts the 3-channel input before any epoch runs.
* Outputs: `runs/physics_detect/weights/{best,last}.pt`,
  `train_config.json`, `train_summary.json` (duration + peak GPU memory),
  logs in `logs/physics_training/`.
* Baseline entrypoint remains `python -m space_debris.main train_resilient`
  (writes `runs/detect/`, untouched by this branch).

### 5.4 Evaluation & comparison

```bash
# physics run
python scripts/evaluate_model.py --tag physics

# baseline run (same thresholds; run it on the baseline model/dataset)
python scripts/evaluate_model.py --model runs/detect/weights/best.pt \
    --data yolo_dataset/space_debris.yaml --tag baseline

# side-by-side table + bar charts
python scripts/compare_experiments.py \
    --baseline logs/metrics_baseline.json --new logs/metrics_physics.json
```

Recorded metrics: **precision, recall, F1, mAP@50, mAP@50–95, false
positives, false negatives, training time, peak GPU memory** (+ per-class
breakdown). `compare_experiments.py` prints raw deltas and explicitly makes
no quality claims.

---

## 6. Repository structure (this branch)

```
preprocessing/
├── physics_pipeline.py      # 3-channel pipeline (Ch1/Ch2/Ch3 + save)
├── dataset_processor.py     # locate → pool → 80/20 split → process → labels → yaml
├── drive_upload.py          # automatic, resumable Drive upload (background thread)
├── visualization.py         # 6-panel samples + channel statistics/correlations
├── validation.py            # dataset/annotation/channel verification
└── experiment_config.py     # YAML config loader

configs/
├── physics_config.yaml      # single source of truth (split, model, training, eval)
└── physics_dataset.yaml     # template (authoritative copy generated into processed_dataset/)

scripts/
├── run_preprocessing.py     # 5.1 preprocessing CLI (+ automatic Drive upload)
├── verify_dataset.py        # 5.2 verification + visualizations
├── train_physics_model.py   # 5.3 training (p1p2-n, HSV off)
├── evaluate_model.py        # 5.4 metrics JSON
├── compare_experiments.py   # 5.4 baseline-vs-physics table + plots
└── jarvis_physics.sh        # JarvisLabs instance helpers

train_physics_jarvis.py      # JarvisLabs entrypoint (preprocess if needed → train)
logs/                        # summaries, metrics, visualizations (gitignored)
processed_dataset/           # generated: images/{train,val}, labels/{train,val},
                             # space_debris.yaml, processing_summary.json (gitignored)

# untouched baseline
space_debris/ ...            # original pipeline (preprocess/train/track/stream)
```

---

## 7. Known limitations

1. **Split differs from the original baseline run** (80/20 seed 42 pooling vs
   the earlier 90/10 build). Retrain the baseline on the same split for a
   strictly controlled claim.
2. **HSV disabled is a deliberate deviation** required by the engineered
   channels; it is the *only* augmentation change and is documented in
   `runs/physics_detect/train_config.json`.
3. **Drive upload of ~66k files is network-bound** (hours on typical
   uplinks). It is automatic but resumable; use `--no-upload` to skip and
   `--upload-only` later.
4. **JPEG q95/4:4:4 is not bit-lossless** — Ch3 residuals are low-amplitude;
   PNG would be exact but ~2–3× larger. Verification thresholds account for this.
5. **Correlation ≠ duplication**: channels are derived from the same source,
   so moderate Pearson correlation is expected. The hard check is
   byte-identity (exact-equality fraction), which must be 0.
6. **No claim of improvement yet** — run the experiment, evaluate both tags,
   then interpret `logs/comparison_*.png`.
