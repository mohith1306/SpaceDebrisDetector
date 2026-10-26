# Space Debris Detection: Preprocessing and Tracking

This document consolidates the current preprocessing pipeline, tracking
implementation, possible preprocessing improvements, novelty direction, and
remaining work discussed for the space-debris detection project.

## 1. Current End-to-End Pipeline

The project currently follows this flow:

```text
Google Drive or local dataset
        ↓
Image preprocessing
        ↓
CSV bounding-box labels converted to YOLO format
        ↓
Deterministic train/validation split
        ↓
YOLO object detector
        ↓
Optional second-stage crop classifier
        ↓
ByteTrack or BoT-SORT object tracking
        ↓
Kalman motion filtering
        ↓
Trajectory and orbit estimation
        ↓
JSONL events, annotated video, and orbit report
```

The command-line entry point is [`space_debris/main.py`](space_debris/main.py).

## 2. Current Image Preprocessing

The main preprocessing implementation is in
[`space_debris/data/preprocess.py`](space_debris/data/preprocess.py).

### 2.1 Existing sequence

The current image preprocessing sequence is:

```text
RGB image
    ↓
grayscale conversion
    ↓
3 × 3 Gaussian blur
    ↓
CLAHE local contrast enhancement
    ↓
3 × 3 Gaussian blur
    ↓
min-max normalization to 0–255
    ↓
grayscale image converted back to RGB
```

### 2.2 Processing steps

#### RGB to grayscale

The image is converted to one intensity channel. This is appropriate for
space-object imagery where brightness, contrast, shape, and edges are more
important than color.

#### Gaussian denoising

A `3 × 3` Gaussian filter reduces sensor noise, isolated bright pixels, and
compression artifacts while attempting to preserve small targets.

#### CLAHE

CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied with:

```python
clipLimit=1.5
tileGridSize=(8, 8)
```

It enhances local contrast in images with uneven illumination or faint
objects. The clip limit reduces excessive amplification of noise.

#### Second Gaussian blur

The second `3 × 3` blur smooths artifacts that may be introduced by CLAHE.

#### Min-max normalization

Pixel values are mapped to the `0–255` range. This reduces brightness
variation between images, although it can be sensitive to extreme outlier
pixels.

#### RGB restoration

The processed grayscale image is converted back to three channels so that it
can be consumed by standard YOLO models. At present, all three channels
contain the same grayscale information.

### 2.3 Dataset behavior

`preprocess_dataset()`:

- processes up to 2,000 images by default;
- skips images already present in the processed directory;
- supports `--force` to reprocess cached images;
- writes output to `processed/`;
- reports failed images.

Example:

```bash
.venv/bin/python -m space_debris.main preprocess --num 66000
```

The default limit should be checked when running a full experiment because the
workspace currently contains approximately 66,000 processed images.

## 3. Label and Dataset Preparation

### 3.1 CSV to YOLO conversion

[`space_debris/data/convert_labels.py`](space_debris/data/convert_labels.py)
converts CSV annotations into YOLO label files.

For each annotation, it:

1. reads the image and bounding box;
2. parses the bounding-box coordinates;
3. clamps coordinates to the image boundaries;
4. rejects invalid or zero-area boxes;
5. converts pixel coordinates into normalized YOLO coordinates;
6. writes:

```text
class_id x_center y_center width height
```

The configured taxonomy contains 11 classes and is defined in
[`space_debris/config.py`](space_debris/config.py).

The current configuration uses multi-class labels:

```python
LEGACY_COLLAPSE_TO_SINGLE_CLASS = False
```

The legacy single-class behavior remains available for reproducing old
checkpoints.

### 3.2 Train/validation split

[`space_debris/data/split_dataset.py`](space_debris/data/split_dataset.py)
creates the YOLO directory structure:

```text
yolo_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── space_debris.yaml
```

The split is shuffled using seed `42`, making it reproducible.

The CLI currently defaults to a 10% validation ratio:

```bash
.venv/bin/python -m space_debris.main split_dataset --val-ratio 0.1
```

The function documentation describes an 80/20 split, but the CLI default
produces a 90/10 train/validation split. This should be made explicit when
reporting experimental results.

## 4. Detector and Second-Stage Classifier

The first stage is a YOLO object detector implemented through
[`space_debris/models/train.py`](space_debris/models/train.py) and
[`space_debris/models/train_resilient.py`](space_debris/models/train_resilient.py).

The detector predicts:

- object bounding boxes;
- detector confidence;
- one of the configured object classes.

Supported training options include:

- standard YOLO architecture;
- custom P1/P2 architecture;
- YOLO11 through the resilient training path;
- CPU, CUDA, and Apple MPS devices;
- checkpoint saving and resuming;
- epoch logging and retry-based recovery.

The optional second-stage classifier is implemented in
[`space_debris/models/classify.py`](space_debris/models/classify.py).

Its purpose is:

```text
detector: where is the object?
classifier: what is the object?
```

The crop workflow is:

```bash
.venv/bin/python -m space_debris.main build_crops
.venv/bin/python -m space_debris.main train_classifier
```

The classifier prediction is fused with the detector prediction. If the
classifier disagrees and has confidence of at least `0.45`, the classifier
overrides the detector. Otherwise, the detector label is retained. The final
record includes both predictions, their agreement, the source of the final
decision, and whether the final class is debris.

## 5. Current Tracking Implementation

The primary tracking implementation is in
[`space_debris/models/track.py`](space_debris/models/track.py).

### 5.1 Detector plus tracker

The detector first generates frame-level detections. Those detections are
passed to Ultralytics tracking:

```python
results = model.track(
    source=src,
    conf=conf,
    device=device,
    imgsz=imgsz,
    tracker=tracker,
    persist=True,
    stream=True,
    save=save,
    verbose=False,
)
```

The default tracker is:

```python
DEFAULT_TRACKER = "bytetrack.yaml"
```

The implementation supports:

- ByteTrack by default;
- BoT-SORT through another tracker YAML;
- video files;
- image directories;
- camera indexes;
- configurable confidence;
- configurable image size;
- configurable FPS;
- maximum frame limits;
- JSONL output;
- optional annotated results.

Example:

```bash
.venv/bin/python -m space_debris.main track \
  --model runs/detect/weights/best.pt \
  --source scene.mp4 \
  --tracker bytetrack.yaml \
  --out runs/track.jsonl
```

`persist=True` allows the tracker to maintain object identity across frames.

### 5.2 Track IDs and per-frame records

The `_track_record()` function converts each result into a JSON-compatible
record. Each detection contains:

- `track_id`;
- `class_id`;
- `class_name`;
- confidence;
- pixel bounding box (`xyxy`);
- normalized center coordinates (`cx`, `cy`).

An example detection has this shape:

```json
{
  "track_id": 4,
  "class_id": 1,
  "class_name": "debris",
  "conf": 0.82,
  "xyxy": [120.4, 80.2, 165.8, 130.1],
  "cx": 0.31,
  "cy": 0.24
}
```

The frame-level record also includes:

- frame number;
- time;
- source path;
- image size;
- all detections;
- IDs present in that frame.

The tracker returns a summary containing:

- number of processed frames;
- tracker configuration;
- number of unique tracks;
- number of observations per track;
- JSONL output path.

## 6. Full Streaming Tracking Pipeline

The end-to-end implementation is in
[`space_debris/models/stream.py`](space_debris/models/stream.py).

It runs:

```text
YOLO detection
    ↓
ByteTrack or BoT-SORT association
    ↓
optional crop classification
    ↓
detector/classifier fusion
    ↓
Kalman motion estimation
    ↓
trajectory and orbit reporting
```

Example:

```bash
.venv/bin/python -m space_debris.main stream \
  --model runs/detect/weights/best.pt \
  --source scene.mp4 \
  --classifier model_checkpoints/classifier/weights/best.pt \
  --out runs/stream.jsonl \
  --annotate runs/stream.mp4 \
  --orbit-report runs/orbit.json
```

Tracking can be disabled with:

```bash
--no-tracker
```

When tracking is disabled, detections are still produced, but persistent
object identity and track-level trajectory estimation are not reliable.

The annotated output displays the track ID, final class, and confidence.
Debris and non-debris objects use different annotation colors.

## 7. Kalman Motion Filtering

The motion-filtering layer is in
[`space_debris/utils/orbit.py`](space_debris/utils/orbit.py).

### 7.1 Constant-velocity filter

`ConstantVelocityFilter` uses the state:

```text
[cx, cy, vx, vy]
```

where:

- `cx`, `cy` are normalized image coordinates;
- `vx`, `vy` are normalized image velocities.

The filter:

1. predicts the next position;
2. receives the detector measurement;
3. corrects the prediction;
4. estimates smoothed velocity.

### 7.2 Trajectory bank

`TrajectoryBank` maintains:

```text
one Kalman filter per track ID
one observation history per track ID
```

For every detected object with a track ID, the stream pipeline updates the
corresponding trajectory.

The output can include:

```json
"motion": {
  "vx_norm_s": 0.0012,
  "vy_norm_s": -0.0004,
  "speed_norm_s": 0.0013
}
```

The current tracking hierarchy is therefore:

```text
ByteTrack or BoT-SORT:
    associates detections and assigns identities

Kalman filter:
    smooths positions and estimates motion for each identity
```

## 8. Trajectory and Orbit Estimation

`estimate_trajectory()` calculates:

- track duration;
- pixel velocity;
- pixel speed;
- angular velocity in degrees per second;
- angular velocity in radians per second;
- line-fit residual;
- physical speed if range is supplied;
- circular-orbit speed;
- approximate circular-orbit altitude;
- circular-orbit consistency.

The stream reports tracks only after they have at least three observations.

For an image sequence, an FPS value should be supplied:

```bash
.venv/bin/python -m space_debris.main stream \
  --source yolo_dataset/images/val \
  --fps 30 \
  --out runs/stream.jsonl
```

For physical speed and orbit checks, a range is also required:

```bash
--range-km 700
```

Without a range, the system reports angular movement but correctly leaves
physical speed and orbit checks unavailable.

The current orbit calculation is an image-motion-based estimate. It is not a
complete orbital determination system because it does not yet use:

- camera calibration;
- camera attitude;
- observer latitude and longitude;
- accurate object range measurements;
- synchronized astronomical timestamps;
- astrometric coordinates;
- radar or TLE/ephemeris data;
- full six-element orbital fitting.

## 9. Offline Orbit Processing

The tracking JSONL can be processed after tracking has finished:

```bash
.venv/bin/python -m space_debris.main orbit \
  --jsonl runs/track.jsonl \
  --fov 60 \
  --range-km 700 \
  --out runs/orbit.json
```

This makes it possible to test different FOV and range assumptions without
rerunning detection and tracking.

## 10. Preprocessing Improvements and Novelty Direction

The strongest practical improvement is to replace the current three identical
grayscale channels with three complementary, physics-motivated channels.

### 10.1 Recommended three-channel representation

Use:

```text
Channel 1: percentile-normalized grayscale image
Channel 2: CLAHE-enhanced image
Channel 3: background-subtracted residual
```

Conceptually:

```python
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

low, high = np.percentile(gray, (1, 99))
raw = np.clip(gray, low, high)
raw = ((raw - low) / max(high - low, 1) * 255).astype(np.uint8)

clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
local_contrast = clahe.apply(raw)

background = cv2.GaussianBlur(raw, (0, 0), sigmaX=15)
residual = cv2.subtract(raw, background)

processed = cv2.merge([
    raw,
    local_contrast,
    residual,
])
```

The channels represent:

| Channel | Information |
|---|---|
| Raw normalized image | Original brightness and object appearance |
| CLAHE image | Local contrast and faint structure |
| Background residual | Compact targets after large-scale background suppression |

This is more useful than copying one grayscale image into all three channels.

### 10.2 Background subtraction

Background subtraction can suppress sky gradients and uneven illumination:

```python
background = cv2.GaussianBlur(gray, (0, 0), sigmaX=15)
residual = cv2.subtract(gray, background)
```

Morphological white top-hat filtering is another option for small bright
objects:

```python
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (15, 15),
)
opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
top_hat = cv2.subtract(gray, opened)
```

The kernel size must be tested because an unsuitable value can remove larger
objects or create halos.

### 10.3 Robust percentile normalization

The current min-max normalization can be dominated by a single bright outlier.
Percentile clipping is more robust:

```python
low, high = np.percentile(image, (1, 99))
clipped = np.clip(image, low, high)
normalized = ((clipped - low) / max(high - low, 1) * 255).astype(np.uint8)
```

The percentile values should be tuned experimentally.

### 10.4 Logarithmic intensity transform

A logarithmic transform can make faint objects more visible:

```python
image_float = image.astype(np.float32) / 255.0
log_image = np.log1p(5.0 * image_float)
log_image = cv2.normalize(
    log_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX,
).astype(np.uint8)
```

This compresses very bright regions and expands darker intensities.

### 10.5 Multi-scale residual preprocessing

A stronger novelty experiment is to calculate residuals at multiple spatial
scales:

```text
fine residual   = gray - medium-scale background
coarse residual = gray - large-scale background
local contrast  = CLAHE(gray)
```

These can be supplied as three channels:

```python
processed = cv2.merge([
    normalize(local_contrast),
    normalize(fine_residual),
    normalize(coarse_residual),
])
```

A possible research contribution is:

> A multi-scale background-residual preprocessing pipeline for small
> space-object detection under non-uniform illumination.

### 10.6 Adaptive quality-aware preprocessing

Instead of using one fixed transformation for every image, calculate image
quality features such as:

- mean brightness;
- standard deviation;
- contrast;
- saturated-pixel percentage;
- estimated noise;
- background gradient.

Then select or blend preprocessing operations:

```text
low contrast       → stronger CLAHE
high noise         → denoising plus moderate CLAHE
strong gradient    → background subtraction
overexposure       → percentile clipping
normal image       → light enhancement
```

This can support a novelty claim such as:

> Adaptive Quality-Aware Preprocessing for Robust Space-Debris Detection.

### 10.7 Learnable preprocessing

A more advanced alternative is a trainable preprocessing module before YOLO,
such as learnable filters, residual enhancement, attention, or trainable
gamma correction.

This is more novel but requires deeper integration with the current Ultralytics
training pipeline. It should be considered only after fixed preprocessing
experiments have been evaluated.

## 11. Recommended Ablation Study

Novelty should be supported by measurable improvement rather than by adding
many filters without evidence.

Use the same model, data split, image size, epochs, batch size, confidence
threshold, and random seed for every experiment.

| Experiment | Preprocessing |
|---|---|
| A | Raw images |
| B | Current grayscale plus CLAHE pipeline |
| C | Percentile normalization plus CLAHE |
| D | Background subtraction only |
| E | Raw plus CLAHE plus residual three-channel input |
| F | Multi-scale residual input |
| G | Adaptive quality-aware preprocessing |

Measure detector performance using:

- mAP50;
- mAP50-95;
- precision;
- recall;
- F1 score;
- small-object recall;
- false positives;
- inference time.

Measure tracking performance using:

- average track length;
- number of lost tracks;
- track fragmentation;
- identity switches;
- consecutive-frame recall;
- MOTA;
- MOTP;
- IDF1.

For this project, small-object recall and track continuity may be more
informative than overall mAP alone.

## 12. Important Implementation Warnings

### Do not stack filters without evidence

Applying Gaussian blur, median blur, bilateral filtering, CLAHE, sharpening,
thresholding, and morphology together may remove faint targets or create
artificial structures.

### Avoid hard thresholding before YOLO

Binary thresholding can discard low-contrast objects. It is generally safer to
provide a residual or enhanced channel and allow YOLO to learn the useful
decision boundary.

### Preserve original information

At least one channel should remain close to the original image. If every
channel is aggressively enhanced, useful brightness and shape information may
be lost.

### Keep preprocessing consistent

The same preprocessing must be applied to:

- training images;
- validation images;
- test images;
- live stream frames;
- tracking frames;
- classifier crops.

At present, the regular inference and tracking paths pass their source
directly to Ultralytics. If a new preprocessing method is adopted, it must be
integrated consistently into those paths.

### Avoid classifier split leakage

The crop classifier split is currently performed at the annotation-row level.
If multiple annotations come from the same source image, related crops can
potentially appear in both train and validation. A source-image-level split is
safer.

## 13. Current Tracking Limitations and Next Steps

### Missed detections are only partially handled

`TrajectoryBank.miss()` can predict a track forward without a measurement, but
the streaming loop does not yet call it for every absent track on every frame.

Current behavior is effectively:

```text
detection present → Kalman update
detection missing  → no explicit stream-level prediction
```

Full occlusion handling should identify tracks not observed in the current
frame, call `miss()` with the correct time step, and expire tracks after a
configurable number of missed frames.

### No quantitative tracking benchmark

The current code reports track counts and lengths, but does not calculate
standard tracking metrics such as MOTA, MOTP, IDF1, identity switches, or
fragmentation.

### No temporal identity ground truth

The current annotations are frame-level bounding boxes. Without ground-truth
object identities across frames, tracker identity quality cannot be measured
reliably.

### FPS assumptions affect motion

For videos, FPS can often be read automatically. For image directories, supply
`--fps`; otherwise the fallback rate affects velocity and trajectory estimates.

### Orbit estimation remains approximate

The current code estimates image motion and optionally converts it using a
user-supplied range. It is not yet a full orbit determination system.

## 14. Current Completion Status

| Area | Status |
|---|---|
| Google Drive synchronization | Implemented |
| Image preprocessing | Implemented |
| CSV to YOLO label conversion | Implemented |
| Bounding-box validation | Implemented |
| Reproducible dataset split | Implemented |
| YOLO detector training | Implemented |
| P1/P2 architecture | Implemented |
| Training recovery and logging | Implemented |
| Detector inference | Implemented |
| ByteTrack integration | Implemented |
| BoT-SORT option | Supported |
| Persistent track IDs | Implemented |
| Per-frame JSONL output | Implemented |
| Track-length summary | Implemented |
| Kalman position smoothing | Implemented |
| Velocity estimation | Implemented |
| Annotated stream output | Implemented |
| Second-stage classifier | Implemented |
| Detector/classifier fusion | Implemented |
| Angular trajectory estimation | Implemented |
| Physical speed estimation | Implemented when range is supplied |
| Circular-orbit plausibility check | Implemented |
| Quantitative tracking benchmark | Not implemented |
| Temporal identity ground truth | Not implemented |
| Full missed-detection propagation | Partially implemented |
| Full orbital determination | Not implemented |

## 15. Suggested Research Contribution

A practical and defensible novelty statement is:

> We propose a quality-aware, multi-channel preprocessing method for
> space-object detection that combines percentile-normalized intensity,
> locally enhanced contrast, and multi-scale background residuals to improve
> the visibility of faint debris under uneven illumination.

This statement should be supported by an ablation study showing improvements
in faint-object recall, false-positive control, and track continuity.

The recommended implementation order is:

1. establish a raw-image baseline;
2. measure the current preprocessing baseline;
3. implement raw + CLAHE + background-residual channels;
4. compare multiple background scales;
5. evaluate small-object detection and track continuity;
6. add adaptive preprocessing only if it provides measurable benefit;
7. integrate the selected preprocessing consistently into inference and
   tracking.
