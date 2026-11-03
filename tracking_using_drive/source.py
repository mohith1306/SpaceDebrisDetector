"""Source resolution — adds ``gdrive://`` support (design.md §3).

Why this module exists
----------------------
``space_debris.models.track.resolve_source`` (``track.py:41-48``) accepts only
local paths and camera indices. It is out of bounds (R0), so we own the
resolver here.

Design decisions
----------------
* **Prefix check, not a URL parser.** ``gdrive://<path>`` is a routing
  convention, not a network location. A URL parser would invite query params,
  credentials and escaping rules that serve no purpose and create failure
  modes.
* **Sync returns a local path.** Everything downstream stays path-based, so
  the rest of the pipeline is untouched. Ultralytics never sees a Drive URL —
  which is what keeps network cost O(dataset) instead of O(frames).
* **A single still is refused, not silently tracked.** The dataset is 88k
  unrelated stills (``memory.md`` §Verified Dataset Facts). Tracking one as-is
  yields a length-1 track and an empty orbit report — a pipeline that "works"
  and answers nothing. ``is_still()`` exists so callers can route stills
  through ``sequence.py`` instead (design.md §8).
"""
from __future__ import annotations

import os
from pathlib import Path

from .config import YOLO_BASE
from .drive import DEFAULT_FOLDER_ID

GDRIVE_PREFIX = "gdrive://"

# What a bare ``gdrive://`` resolves to: the validation slice synced by
# ``drive.TRACKING_ALLOWLIST``. Chosen over the whole folder because a
# tracking run needs a small slice, and ``yolo_dataset/images/val`` is the
# only folder whose contents are fully enumerable anonymously (G17).
DEFAULT_GDRIVE_PATH = "yolo_dataset/images/val"

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})


def is_still(path: str | Path) -> bool:
    """True when ``path`` is a single image rather than a sequence.

    A still must be synthesised before tracking (D8/R9). Detecting it here is
    what lets ``stream`` fail early with an actionable message instead of
    producing an empty orbit report.
    """
    p = Path(str(path))
    return p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES


def is_video(path: str | Path) -> bool:
    """True when ``path`` is a video container Ultralytics can iterate."""
    p = Path(str(path))
    return p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES


def resolve_gdrive(
    rel_path: str | None = None,
    *,
    no_sync: bool = False,
    folder_id: str | None = None,
) -> Path:
    """Sync a Drive subpath and return its **local** directory.

    Parameters
    ----------
    rel_path
        Path relative to the shared folder, e.g. ``yolo_dataset/images/val``.
        ``None`` or ``""`` selects ``DEFAULT_GDRIVE_PATH``.
    no_sync
        Resolve from cache only; raises if the cache cannot satisfy it.
    folder_id
        Override the shared folder id.

    Returns
    -------
    pathlib.Path
        A local directory. Never a URL — see the module docstring.
    """
    from .drive import DriveError, sync_allowlist

    rel = (rel_path or DEFAULT_GDRIVE_PATH).strip().strip("/")
    if not rel:
        rel = DEFAULT_GDRIVE_PATH

    fid = folder_id or os.environ.get("CAPSTONE_GDRIVE_FOLDER") or DEFAULT_FOLDER_ID

    # sync_allowlist takes an allowlist of one, so the returned report's
    # `files` are exactly the files under `rel`.
    try:
        report = sync_allowlist(
            (rel,), folder_id=fid, no_sync=no_sync,
        )
    except DriveError as exc:
        raise DriveError(
            f"Could not resolve gdrive://{rel}\n{exc}"
        ) from exc

    if not report.files:
        raise DriveError(
            f"gdrive://{rel} resolved to zero files under folder {fid}. "
            "Either the path is empty or the listing was truncated (G17). "
            "Check the folder id and the path spelling."
        )

    # The cache root holds the allowlist mirrored by relative path, so the
    # resolved directory is root/rel.
    resolved = report.root / rel
    if not resolved.exists():
        # A single-file allowlist lands as root/<file>, not root/<file>/.
        candidate = report.root / report.files[0]
        if candidate.exists():
            return candidate.parent if candidate.is_file() else candidate
        raise DriveError(
            f"Sync reported {len(report.files)} file(s) for gdrive://{rel} "
            f"but {resolved} does not exist. Cache and manifest disagree."
        )
    return resolved


def resolve_source(
    source: str | None = None,
    *,
    no_sync: bool = False,
) -> str | int:
    """Resolve a source to something Ultralytics can consume.

    ==================  ==================================================
    ``None``            ``TRACK_SOURCE`` env, else the configured Drive
                        folder, else the local YOLO val split
    ``"0"``             camera index 0
    ``"gdrive://x"``    sync subpath ``x``, return its local path
    otherwise           local path or file
    ==================  ==================================================

    Raises ``ValueError`` when a resolved local path is an empty directory —
    failing early with the path beats loading a model and then discovering
    there is nothing to run it on (design.md §8).
    """
    if source is None:
        source = os.environ.get("TRACK_SOURCE")
        if source is None:
            # Default to the local val split when it exists, else Drive.
            local_val = YOLO_BASE / "images" / "val"
            if local_val.exists() and any(local_val.iterdir()):
                return str(local_val)
            return str(resolve_gdrive(no_sync=no_sync))

    if source.startswith(GDRIVE_PREFIX):
        return str(resolve_gdrive(source[len(GDRIVE_PREFIX):], no_sync=no_sync))

    if source.isdigit():
        return int(source)

    # A local path: fail early if it is an empty directory, which is the
    # failure mode that otherwise surfaces as "no results" after a long load.
    p = Path(source)
    if p.is_dir() and not any(p.iterdir()):
        raise ValueError(
            f"source directory is empty: {p}\n"
            "Populate it, or pass a gdrive:// path to sync a slice."
        )
    return source
