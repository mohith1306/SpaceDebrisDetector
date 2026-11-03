# Design — `tracking_using_drive`

Detailed design. Each section states the problem, the chosen approach, the
rejected alternative, and the concrete signature.

**Governing constraint:** nothing outside `tracking_using_drive/` is modified
(`rules.md` R0). Every "fix" below is therefore an *adaptation*, not a patch.

---

## 1. Sequence Synthesis — the core addition

### Problem
The Drive data is 88,000 independent single-object stills (`requirements.md`
§1.2). There is no temporal ordering, so ByteTrack cannot associate anything:
every image becomes a track of length 1, and nothing reaches
`MIN_POINTS_FOR_ORBIT = 3`. The pipeline runs and produces an empty orbit
report.

### Decision
Generalise `scripts/make_demo_video.py` into a first-class synthesiser that
turns any Drive still into an ordered sequence with known ground truth.

```python
# tracking_using_drive/sequence.py

@dataclass(frozen=True)
class Motion:
    """Ground-truth motion model for a synthetic sequence."""
    mode: str                      # "linear" | "accel" | "orbit_arc"
    dx: float                      # px/frame at t=0 (x)
    dy: float                      # px/frame at t=0 (y)
    ddx: float = 0.0               # px/frame^2 (accel mode)
    ddy: float = 0.0

@dataclass(frozen=True)
class Occlusion:
    """Optional gap where the object leaves the frame and returns."""
    start_frame: int
    end_frame: int                 # exclusive; object invisible in [start, end)

@dataclass(frozen=True)
class GroundTruth:
    fps: float
    fov_deg: float
    width: int
    height: int
    motion: Motion
    occlusion: Occlusion | None
    centres: list[tuple[float, float]]     # per-frame normalised centres
    angular_rate_deg_s: list[float]        # implied, for validation
    synthetic: bool = True                 # R7

def synthesise_sequence(
    image_path: Path,
    out_dir: Path,
    frames: int = 30,
    motion: Motion = Motion("linear", 6.0, -3.0),
    fps: float = 5.0,
    fov_deg: float = 60.0,
    occlusion: Occlusion | None = None,
    label_path: Path | None = None,
) -> tuple[Path, GroundTruth]:
    """Render frames to out_dir/frame_%04d.png plus ground_truth.json.

    Object centre comes from label_path if given (normalised), else image centre.
    During an occlusion gap the object is erased, not merely moved off-canvas,
    so the detector genuinely loses it.
    """
```

**Occlusion is implemented by erasing the object region**, not by translating
it beyond the border. Translating off-canvas leaves the object detectable in
some frames depending on padding; erasing guarantees a true detection dropout
and is the only reliable way to test FR-4.1. The erase is filled with the
**median colour of the image border** rather than black, because a black
rectangle is itself a detectable shape against a non-black background.

### Decision: ground truth imports the library's vertical-FOV formula
`angular_rate_truth()` calls `fov_vertical_deg` from
`space_debris.utils.orbit` — the same function `estimate_trajectory` uses —
rather than reimplementing it.

This matters because `scripts/make_demo_video.py:73` hardcodes the small-angle
approximation `fov * h / w` inline. The two agree only when `w == h`:

| size | exact (`orbit.py`) | small-angle | rel diff |
|------|-------|-------|-------|
| 1920×1080 | 35.9834 | 33.7500 | **6.21%** |
| 1024×1024 | 60.0000 | 60.0000 | 0.00% |

Our Drive stills are 1024×1024, where the formulas coincide — so a validation
run on those alone would never detect the discrepancy, making the R9 check
vacuous for any other aspect ratio. 6.21% would silently consume most of the
10% tolerance before any tracking error was measured. Importing keeps truth
and estimate in agreement by construction; the divergence is pinned explicitly
in `test_deliberately_departs_from_demo_video_at_16_9`.

### Rejected: treating stills as a sequence
Feeding 88k unrelated images to `stream` and calling the result tracking would
produce length-1 tracks and an empty orbit report — a pipeline that "works" and
answers nothing. Worse, it would invite the report to describe those IDs as
tracked debris.

### Rejected: using `make_demo_video.py` as-is
It is single-image, single fixed shift, no occlusion, no batch, and writes to
`/tmp` by default. Adequate for a one-off check, not for a validation suite.
`sequence.py` imports its geometry so the two cannot drift apart, but owns its
API.

### Validation
`test_sequence.py::test_angular_rate_truth_matches_library_estimate_within_10pct`
recovers the ground-truth rate with `estimate_trajectory` and asserts < 10%
error — acceptance criterion #5. Measured worst case: **0.0001%**.

It is a test rather than the originally planned `scripts/validate_angular_rate.py`
because R0 puts `scripts/` out of bounds; a test also runs on every commit
instead of on demand.

The check is only meaningful alongside the FOV decision above: truth and
estimate must use the *same* vertical-FOV formula, or the number being
asserted measures a formula mismatch rather than tracking accuracy.

---

## 2. Missed-Detection Propagation

### Problem
`TrajectoryBank.miss()` (`orbit.py:115-118`) is defined and **never called
anywhere in the codebase**. The stream loop only updates tracks present in the
current frame.

Effect: an occluded object freezes in the Kalman state. When it reappears,
ByteTrack may assign a new ID → fragmentation → both fragments fall below
`MIN_POINTS_FOR_ORBIT = 3` → both discarded. Occlusion silently destroys
results.

Because `stream.py` is out of bounds, we reimplement `run_stream` in
`tracking_using_drive/stream.py`, importing the pieces that work:

```python
from space_debris.utils.orbit import TrajectoryBank, summarize_track
from space_debris.models.classify import classify_crops, fuse, load_classifier
from space_debris.models.track import resolve_model, probe_fps, DEFAULT_TRACKER
```

### The loop

```python
missed: dict[int, int] = defaultdict(int)
expired: set[int] = set()
frame_wh: tuple[int, int] | None = None          # fixes stream.py:181

for result in results:
    ...
    if frame_wh is None:
        frame_wh = (w, h)                        # first frame, not last

    detections = [...]                            # as before, incl. fuse()
    observed = {d["track_id"] for d in detections if d["track_id"] is not None}

    for tid in bank.ids():
        if tid in expired:
            continue
        if tid in observed:
            missed[tid] = 0
        else:
            bank.miss(tid, dt=1.0 / rate)         # ← FR-4.1
            missed[tid] += 1
            if missed[tid] > max_missed:
                expired.add(tid)

    ...write record...
```

### Why `dt = 1.0/rate` and not `1.0`
`rate` is the resolved FPS. A raw `1.0` is correct only at 1 FPS and would
systematically over-predict displacement otherwise.

### Why `max_missed = 30`
≈1s at 30 FPS. Generous enough to survive real occlusion. ByteTrack has its
own cooldown; if IDs still fragment the fix is to raise this above ByteTrack's
cooldown, **not** to remove the Kalman prediction.

### Rejected: relying on ByteTrack's internal Kalman
It does handle short occlusions, but its predictions are not exposed in the
JSONL, and the orbit stage needs smoothed normalised centres regardless. Two
disagreeing motion models is worse than one.

### Rejected: expiring on first miss
Single-frame dropouts are common and ByteTrack recovers from them. Aggressive
expiry would fragment healthy tracks.

---

## 3. Drive Source Resolution

### Problem
`resolve_source()` (`track.py:41-48`) accepts only local paths and camera
indices. It is out of bounds.

### Decision
Own resolver in `tracking_using_drive/source.py`.

```python
GDRIVE_PREFIX = "gdrive://"

def resolve_source(source: str | None) -> str | int:
    """Resolve a source to something Ultralytics can consume.

    None        -> TRACK_SOURCE env, else configured Drive folder, else local val
    "0"         -> camera index 0
    "gdrive://x"-> sync folder x, return its local path
    otherwise   -> local path or file
    """
    if source is None:
        source = os.environ.get("TRACK_SOURCE") or DEFAULT_SOURCE
    if source is None:
        return str(YOLO_BASE / "images" / "val")
    if source.startswith(GDRIVE_PREFIX):
        from .drive import sync_folder
        return str(sync_folder(source[len(GDRIVE_PREFIX):]))
    return int(source) if source.isdigit() else source
```

Prefix check, not a URL parser. Sync returns a **local path**, so everything
downstream stays path-based and the rest of the pipeline is untouched.

### Rejected: passing a Drive ID to Ultralytics directly
Ultralytics accepts URLs, but that means one HTTP GET per frame — rate limits,
no caching, no guaranteed frame ordering. Sync-then-process makes the network
cost O(dataset) instead of O(frames).

### Rejected: subclassing `GDriveSync`
The class is correct; its defaults and call sites are not. Rewriting it risks
the incremental-skip behaviour at `gdrive_sync.py:67-81`, which is the one
property making a second run free. We pass `cache_dir` and
`credentials_file` explicitly instead.

---

## 4. Drive Wrapper

```python
# tracking_using_drive/drive.py
from space_debris.data.gdrive_sync import GDriveSync

CREDENTIALS = Path(os.environ.get(
    "GDRIVE_CREDENTIALS", GDRIVE_CACHE_DIR / "credentials.json"))

# Subpaths worth syncing for a tracking run — NOT the whole 88k-image folder.
TRACKING_ALLOWLIST = (
    "labels/train.csv",
    "labels/val.csv",
    "yolo_dataset/images/val/",
)

def sync_folder(folder_id: str, allowlist: Sequence[str] = TRACKING_ALLOWLIST) -> Path:
    """Sync only the subpaths a tracking run needs. Returns the local root."""
    sync = GDriveSync(
        cache_dir=str(GDRIVE_CACHE_DIR),        # defeats gdrive_sync.py:15
        credentials_file=str(CREDENTIALS),      # defeats gdrive_sync.py:18
    )
    ...
```

**Selective sync is an explicit requirement** (`FR-2.5`): the folder holds
`train/`, `processed/`, and `yolo_dataset/` — three overlapping copies. A
tracking run needs a small slice, and downloading all of it would be both slow
and pointless.

---

## 5. Import Bootstrap

```python
# tracking_using_drive/config.py  — must be imported first by every entrypoint
import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent
os.environ.setdefault("CAPSTONE_DIR", str(_PKG / "_data"))   # BEFORE the import

from space_debris.config import *            # noqa
from space_debris import config as _sc

GDRIVE_CACHE_DIR = _PKG / "_data" / "gdrive_cache"
```

**Why:** `space_debris/config.py:19-20` reads `CAPSTONE_DIR` at import, and
`config.py:117-118` creates four directories at import. We cannot edit either,
so we set the environment variable first and let its paths land inside
`tracking_using_drive/_data/` — inside the R0 boundary.

**Ordering is load-bearing.** `CAPSTONE_DIR` must be set before
`space_debris.config` is first imported *anywhere in the process*. This module
must be first in the import list of every entrypoint.

### Rejected: importing `space_debris.config` lazily per-function
The variable is read once at module import. Lazy import in a later function
still respects it, but the first import anywhere wins — so a stray
`from space_debris...` at the top of another module would silently capture the
wrong `BASE_DIR`. Centralising the bootstrap removes that hazard.

---

## 6. Output Schema

### Per-frame JSONL
```json
{
  "frame": 17, "t": 3.4,
  "source": "_data/sequences/demo/frame_0017.png",
  "image_size": [1920, 1080],
  "synthetic": true,
  "source_meta": {"provider": "gdrive", "folder_id": "1vxqeBHOglilc1PP_ZSNA1fCx_jfTTdUY",
                   "cache_hit": true},
  "detections": [
    {"track_id": 4, "class_id": 1, "class_name": "debris", "conf": 0.82,
     "xyxy": [120.4, 80.2, 165.8, 130.1], "cx": 0.31, "cy": 0.24,
     "final_class": "debris", "is_debris": true, "source": "agree",
     "motion": {"vx_norm_s": 0.0012, "vy_norm_s": -0.0004, "speed_norm_s": 0.0013}}
  ]
}
```

### Per-track summary
```json
{"run": {"model": "...", "tracker": "bytetrack.yaml", "conf": 0.25,
          "fps_assumed": 5.0, "fov_deg": 60.0, "range_km": null,
          "synthetic": true, "frames": 300, "folder_id": "1vxqeBHO..."},
 "tracks": [{"track_id": 4, "n_points": 42, "n_missed": 3, "lifespan_s": 8.4,
             "expired": false, "is_debris": true,
             "trajectory": {"angular_rate_deg_s": [1.02, -0.51], "speed_km_s": null,
                             "missing": ["range_km_needed_for_physical_speed"]}}]}
```

`missing` comes from `estimate_trajectory` (`orbit.py:171`) — the mechanism for
reporting what could **not** be determined. With no `--range-km`,
`speed_km_s` stays `null`. Correct, not a gap to fill with a guess.

---

## 7. Constants and Justification

| Constant | Value | Source |
|----------|-------|--------|
| `MIN_POINTS_FOR_ORBIT` | 3 | `stream.py:29`. Line fit needs 2; 3 gives a residual. |
| `max_missed` | 30 | New. ≈1s at 30 FPS. |
| `min_cls_conf` | 0.45 | `classify.py:193`. |
| `agreement_boost` | 0.10 | `classify.py:194`. |
| `clipLimit` | 1.5 | `preprocess.py:15`. |
| `tileGridSize` | (8, 8) | `preprocess.py:15`. |
| Split seed | 42 | `split_dataset.py:35`. |
| `MU_EARTH` | 398600.4418 km³/s² | `orbit.py:22`. |
| `R_EARTH` | 6371.0 km | `orbit.py:23`. |
| Orbit consistency band | 10% | `orbit.py:223`. |
| `FOV_DEFAULT` | 60° | Placeholder — must be set per instrument. |
| Angular-rate tolerance | 10% | Acceptance criterion #5. |

---

## 8. Failure Modes

| Failure | Handling |
|---------|----------|
| Folder not found | Catch in `resolve_source`; name it, suggest alternatives. |
| Credentials missing | Error naming the exact expected path + `GDRIVE_CREDENTIALS`. |
| Partial download | Per-file warning, continue (`gdrive_sync.py:98-99` granularity). |
| Zero frames | Guard orbit block on `frame_wh is not None`. |
| Source is a single still | Route through `sequence.py` — do not track it as-is. |
| `--range-km` omitted | `speed_km_s = null`, `missing` populated. |
| Empty resolved dir | Fail early with the path, before loading the model. |
| Synthetic output escaping into a report | `synthetic: true` written on every record; report writer refuses to omit it (R7). |

---

## 9. What Deliberately Does Not Change

- **The JSONL record shape** — `orbit_from_jsonl` (`orbit.py:242`) parses it.
- **`ConstantVelocityFilter`** — numpy-only and correct.
- **The fusion thresholds** — tuned; changing them invalidates prior runs.
- **The class taxonomy** — R1.
- **`GDriveSync` internals** — we pass parameters, we do not edit the class.
- **`convert_labels.py`** — verified correct for this dataset (R8).
