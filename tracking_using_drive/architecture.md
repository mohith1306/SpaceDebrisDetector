# Architecture — `tracking_using_drive`

## 0. The Isolation Constraint Shapes Everything

Work happens **only** inside `tracking_using_drive/`. The `space_debris/`
package is read-only. This produces a *shadow package* architecture: we own a
parallel module tree that imports the correct pieces of the library and
reimplements only what is broken.

```
┌─────────────────────────────────────────────────────────────┐
│  tracking_using_drive/          WE OWN THIS (read-write)    │
│  ─────────────────────────                                  │
│  cli.py          orchestration + argument parsing           │
│  config.py       scoped paths, CAPSTONE_DIR bootstrap       │
│  drive.py        Drive sync (fixed credentials/defaults)    │
│  source.py       source resolution incl. gdrive://          │
│  sequence.py     still → ordered synthetic sequence         │
│  stream.py       detect → track → classify → Kalman → orbit │
│  report.py       per-track summary JSON                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ imports, never writes
┌──────────────────────────▼──────────────────────────────────┐
│  space_debris/              READ-ONLY LIBRARY               │
│  ─────────────────────────                                  │
│  config.py       class taxonomy, DEVICE, ensure_data_synced │
│  utils/orbit.py  ConstantVelocityFilter, TrajectoryBank,    │
│                  estimate_trajectory, summarize_track       │
│  models/classify.py  classify_crops, fuse, load_classifier  │
│  models/track.py     resolve_model, _track_record, probe_fps│
│  data/preprocess.py  preprocess_image                       │
│  data/gdrive_sync.py GDriveSync (imported, defaults ignored)│
└─────────────────────────────────────────────────────────────┘
```

### Reuse vs. reimplement

| Module | Action | Reason |
|--------|--------|--------|
| `utils/orbit.py` | **Import as-is** | Correct and numpy-only. Nothing to fix. |
| `models/classify.py` | **Import as-is** | Fusion rule is tuned; changing it invalidates prior runs. |
| `models/track.py` → `resolve_model`, `_track_record`, `probe_fps` | **Import as-is** | Correct. |
| `models/track.py` → `resolve_source` | **Reimplement** | Cannot accept `gdrive://`. |
| `models/stream.py` → `run_stream` | **Reimplement** | Never calls `bank.miss()`; uses last-frame dims; unbound on zero frames. |
| `data/gdrive_sync.py` → `GDriveSync` | **Import class, ignore defaults** | Logic correct; `E:/Capstone` and CWD-relative credentials are wrong. |
| `main.py` | **Reimplement** | Auto-sync gate excludes `track`/`stream`. |
| `config.py` | **Import, then override** | Cannot edit. We set `CAPSTONE_DIR` first so its paths land under our control. |

---

## 1. Data Flow

```
DRIVE (folder 1vxqeBHOglilc1PP_ZSNA1fCx_jfTTdUY)
   │  selective sync → gdrive_cache/
   ▼
SOURCE RESOLUTION  source.resolve_source()
   │  gdrive://<id> | local path | camera index
   ▼
┌──────────────────────────────────────────┐
│  IF source is a single still  →          │
│  sequence.py synthesises an ordered      │◄─── THE BRANCH'S CORE ADDITION
│  sequence with known ground truth        │     (dataset has no sequences)
└──────────────────────────────────────────┘
   │  ordered frames + gt JSON
   ▼
DETECTION   YOLO(P1/P2 head) → boxes + class + conf
   ▼
ASSOCIATION ByteTrack (persist=True) → stable track_id
   ▼
CLASSIFY    crop → yolo11n-cls → fuse()      [optional]
   ▼
MOTION      TrajectoryBank → ConstantVelocityFilter per id
            + MISS PROPAGATION for absent tracks   ◄── FR-4.1
   ▼
REPORT      per-frame JSONL │ annotated MP4 │ per-track orbit JSON
                              (tagged synthetic:true)
```

---

## 2. Why the sequence stage exists

The dataset is **88,000 independent single-object stills** (verified — see
`requirements.md` §1.2). ByteTrack associates across *frames of the same
video*. Feed it unrelated stills and every image becomes a new track of length
1; nothing reaches `MIN_POINTS_FOR_ORBIT = 3`; the orbit report is empty.

`scripts/make_demo_video.py` already demonstrates the workaround for a single
image. `sequence.py` generalises it:

| Capability | `make_demo_video.py` (existing) | `sequence.py` (this branch) |
|------------|--------------------------------|-----------------------------|
| Source | one `--image` | any Drive still, or a batch |
| Motion | fixed `--dx`/`--dy` | constant, accelerating, or curved |
| Occlusion | none | **gap injection** — object leaves and re-enters |
| Output | one mp4 + one gt json | sequence catalogue + per-sequence gt |
| Batch | no | yes |
| Purpose | ad-hoc check | first-class validation instrument |

Occlusion support is not a nicety — it is the only way to test FR-4.1
(missed-detection propagation), because the Drive data contains no natural
occlusion.

---

## 3. Package Layout

```
tracking_using_drive/
├── requirements.md      ┐
├── architecture.md      │
├── rules.md             ├── planning context (never imported by runtime)
├── phases.md            │
├── design.md            │
├── memory.md            ┘
├── __init__.py
├── config.py            scoped paths, CAPSTONE_DIR bootstrap
├── drive.py             DriveSync wrapper: fixed defaults, selective sync
├── source.py            resolve_source() with gdrive:// support
├── sequence.py          still → synthetic sequence + ground truth
├── stream.py            run_stream() with missed-detection propagation
├── report.py            per-track summary JSON writer
├── cli.py               argument parsing + dispatch
├── main.py              entrypoint (python -m tracking_using_drive.main)
├── scripts/
│   ├── make_occlusion_video.py
│   └── validate_angular_rate.py
└── tests/
    ├── test_source.py
    ├── test_sequence.py
    └── test_missed_detections.py
```

Runtime entrypoint:
```bash
python -m tracking_using_drive.main track --source gdrive://1vxqeBHOglilc1PP_ZSNA1fCx_jfTTdUY
python -m tracking_using_drive.main synth --image <still> --occlusion --out runs/seq
python -m tracking_using_drive.main stream --source runs/seq/demo.mp4 --orbit-report runs/orbit.json
```

---

## 4. Import Bootstrap — the `CAPSTONE_DIR` trick

`space_debris/config.py:19-20`:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = Path(os.environ.get("CAPSTONE_DIR", PROJECT_ROOT))
```

It also creates four directories at import time (`config.py:117-118`). Since we
cannot edit the file, we control it through the environment **before** the
import:

```python
# tracking_using_drive/config.py
import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent
# Scope every space_debris path under our own folder BEFORE importing it.
os.environ.setdefault("CAPSTONE_DIR", str(_PKG / "_data"))

from space_debris.config import *          # noqa: F401,F403
from space_debris import config as _sc

# Repoint at our scoped cache
GDRIVE_CACHE_DIR = _PKG / "_data" / "gdrive_cache"
```

**Ordering is load-bearing.** `CAPSTONE_DIR` must be set before
`space_debris.config` is first imported anywhere in the process. Put this
module first in every entrypoint.

The side effect is directories created under `tracking_using_drive/_data/` —
which is inside the allowed boundary, satisfying R0.

---

## 5. Drive Integration

Three defects, all fixable only in our own `drive.py`:

| Defect | Location (unmodifiable) | Our fix |
|--------|------------------------|---------|
| `E:/Capstone/gdrive_cache` literal | `gdrive_sync.py:15`, `:133` | Pass `cache_dir=GDRIVE_CACHE_DIR` explicitly at every construction |
| `Path("credentials.json")` is CWD-relative | `gdrive_sync.py:18` | Pass `credentials_file` explicitly, derived from cache dir, overridable by `GDRIVE_CREDENTIALS` |
| No selective sync — full folder is 88k images | `gdrive_sync.py:83-104` | Wrap `sync_folder` with an allowlist (`labels/train.csv`, `yolo_dataset/**`) |

### Resolution

```
resolve_source("0")                         → camera index 0
resolve_source("scene.mp4")                 → local video
resolve_source("yolo_dataset/images/val")   → local directory
resolve_source("gdrive://<id>")             → sync → local path
resolve_source(None)                        → configured Drive folder
```

Prefix check, not a URL parser. Sync returns a local path so everything
downstream stays path-based. **Drive is a delivery mechanism, never a
per-frame transport.**

---

## 6. State Ownership

| State | Lifetime | Owner |
|-------|----------|-------|
| ByteTrack internal state | one run | Ultralytics, via `persist=True` |
| `TrajectoryBank.filters` / `.history` | one run | our `stream.py` |
| `missed` / `expired` counters | one run | our `stream.py` ← **new** |
| Drive cache | cross-run | filesystem |
| `credentials.json` | cross-run | filesystem |
| Ground-truth JSON | cross-run | `sequence.py` output |

Nothing is stateful across runs except the cache and credentials. A tracking
run must never require a previous tracking run.

---

## 7. Output Schema

### Per-frame JSONL
```json
{
  "frame": 17,
  "t": 3.4,
  "source": "_data/sequences/demo/frame_0017.png",
  "image_size": [1920, 1080],
  "synthetic": true,
  "source_meta": {"provider": "gdrive", "folder_id": "1vxqeBHOglilc1PP_ZSNA1fCx_jfTTdUY", "cache_hit": true},
  "detections": [
    {
      "track_id": 4,
      "class_id": 1,
      "class_name": "debris",
      "conf": 0.82,
      "xyxy": [120.4, 80.2, 165.8, 130.1],
      "cx": 0.31, "cy": 0.24,
      "final_class": "debris",
      "is_debris": true,
      "motion": {"vx_norm_s": 0.0012, "vy_norm_s": -0.0004, "speed_norm_s": 0.0013}
    }
  ]
}
```

`synthetic` is mandatory (R7). `source_meta.folder_id` makes a run
reproducible without re-resolving the folder.

### Per-track summary
```json
{
  "run": {"model": "...", "tracker": "bytetrack.yaml", "conf": 0.25,
           "fps_assumed": 5.0, "fov_deg": 60.0, "range_km": null,
           "synthetic": true, "frames": 300},
  "tracks": [
    {"track_id": 4, "n_points": 42, "n_missed": 3, "lifespan_s": 8.4,
     "expired": false, "is_debris": true,
     "trajectory": {"angular_rate_deg_s": [1.02, -0.51], "speed_km_s": null,
                     "missing": ["range_km_needed_for_physical_speed"]}}
  ]
}
```

`missing` is inherited from `estimate_trajectory` (`orbit.py:171`) — the
mechanism by which the system reports what it could **not** determine. Preserve
it. With `range_km` absent, `speed_km_s` stays `null`; that is correct, not a
gap to be filled with a guess.

---

## 8. The Fusion Rule (unchanged, `classify.py:192-231`)

| Condition | `source` | Final class |
|-----------|----------|-------------|
| No classifier | `detector_only` | detector |
| No detection | `classifier_only` | classifier |
| Agreement | `agree` | detector (== classifier) |
| Disagree, `cls_conf ≥ 0.45` | `classifier_overrode` | **classifier** |
| Disagree, `cls_conf < 0.45` | `detector_kept` | detector |

Confidence = `cls_conf` on agreement, else `max(cls_conf - 0.10, 0.0)`.
`is_debris = final_class in ["debris"]`.

---

## 9. Failure Modes

| Failure | Handling |
|---------|----------|
| Folder not found | Catch in `resolve_source`; name the folder and suggest alternatives. |
| Credentials missing | Error naming the exact expected path and the env var. |
| Partial download | Per-file warning, continue. |
| Zero frames processed | Guard orbit block on `frame_wh is not None`. |
| Source has no sequences | Route through `sequence.py` rather than failing or silently tracking stills. |
| `--range-km` omitted | `speed_km_s = null`, `missing` populated. Correct behaviour. |
| Empty resolved directory | Fail early with the resolved path, before the model loads. |
