"""Scoped configuration. Import this before any other ``space_debris`` import.

Why the ordering matters
------------------------
``space_debris/config.py`` does two things at import time that we cannot edit
around (``rules.md`` R0):

* ``:19-20``  ``BASE_DIR = Path(os.environ.get("CAPSTONE_DIR", PROJECT_ROOT))``
* ``:117-118`` ``mkdir`` four directories derived from ``BASE_DIR``

So the very first import of ``space_debris.config`` in a process permanently
decides where every library path points, and creates directories as a side
effect. If it happens before this module runs, the paths land in the
repository root — outside this package — and re-importing cannot fix it,
because Python caches modules.

``tracking_using_drive/__init__.py`` therefore sets ``CAPSTONE_DIR`` cheaply on
package import, and this module adds a guard that fails loudly if the library
was somehow already imported out of scope.

Evidence this is not theoretical: importing ``space_debris.config`` with an
unscoped ``CAPSTONE_DIR`` creates ``processed/``, ``labels_yolo/``,
``model_checkpoints/`` and ``runs/`` in the repository root. That was observed
directly during Phase 0. (``memory.md`` G2)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Importing the package first guarantees CAPSTONE_DIR is primed.
from . import PACKAGE_DIR, REPO_ROOT, __version__  # noqa: F401

DATA_ROOT = PACKAGE_DIR / "_data"
GDRIVE_CACHE_DIR = DATA_ROOT / "gdrive_cache"
WEIGHTS_DIR = DATA_ROOT / "weights"
SEQUENCE_DIR = DATA_ROOT / "sequences"

# setdefault: an explicit operator override is respected, and `verify` below
# will then report the resulting escape rather than silently ignoring it.
os.environ.setdefault("CAPSTONE_DIR", str(DATA_ROOT))


def _guard_pre_existing_import() -> None:
    """Fail loudly if space_debris.config was imported before this module.

    Once imported, its BASE_DIR is fixed for the life of the process and no
    later assignment can move it. Detecting that here beats letting every
    downstream path silently point at the repository root.
    """
    prior = sys.modules.get("space_debris.config")
    if prior is None:
        return
    prior_base = Path(getattr(prior, "BASE_DIR", "") or "").resolve()
    if not prior_base.is_relative_to(PACKAGE_DIR.resolve()):
        raise RuntimeError(
            "space_debris.config was imported before tracking_using_drive.config, "
            "so its paths are scoped to:\n"
            f"    {prior_base}\n"
            "instead of inside this package. Those directories have already been "
            "created and re-importing cannot move them (Python caches modules).\n"
            "Fix: import tracking_using_drive.config — or merely "
            "`import tracking_using_drive` — before any space_debris import. "
            "See memory.md D10/G1."
        )


_guard_pre_existing_import()

# --- the library import (triggers its mkdir side effect, now scoped) -------
from space_debris import config as _lib  # noqa: E402

# Re-exports our code uses. Note the library's own modules read
# space_debris.config directly, so these are conveniences for us only — they
# are not a second source of truth.
BASE_DIR = _lib.BASE_DIR
DATA_DIR = _lib.DATA_DIR
PROCESSED_DIR = _lib.PROCESSED_DIR
LABELS_DIR = _lib.LABELS_DIR
YOLO_LABEL_DIR = _lib.YOLO_LABEL_DIR
YOLO_BASE = _lib.YOLO_BASE
CHECKPOINT_DIR = _lib.CHECKPOINT_DIR
RUNS_DIR = _lib.RUNS_DIR
IMAGE_DIR = _lib.IMAGE_DIR
LABEL_CSV = _lib.LABEL_CSV

DEVICE = _lib.DEVICE
CLASS_NAMES = _lib.CLASS_NAMES
NUM_CLASSES = _lib.NUM_CLASSES
DEBRIS_CLASSES = _lib.DEBRIS_CLASSES
class_id = _lib.class_id
get_yaml_config = _lib.get_yaml_config
verify_paths = _lib.verify_paths
ensure_data_synced = _lib.ensure_data_synced


def scoped_paths() -> dict[str, Path]:
    """Every library path this package is responsible for keeping inside itself."""
    return {
        "BASE_DIR": Path(_lib.BASE_DIR),
        "DATA_DIR": Path(_lib.DATA_DIR),
        "GDRIVE_CACHE_DIR": Path(_lib.GDRIVE_CACHE_DIR),
        "PROCESSED_DIR": Path(_lib.PROCESSED_DIR),
        "LABELS_DIR": Path(_lib.LABELS_DIR),
        "YOLO_LABEL_DIR": Path(_lib.YOLO_LABEL_DIR),
        "YOLO_BASE": Path(_lib.YOLO_BASE),
        "CHECKPOINT_DIR": Path(_lib.CHECKPOINT_DIR),
        "RUNS_DIR": Path(_lib.RUNS_DIR),
        "IMAGE_DIR": Path(_lib.IMAGE_DIR),
        "LABEL_CSV": Path(_lib.LABEL_CSV),
        "DATA_ROOT": DATA_ROOT,
        "WEIGHTS_DIR": WEIGHTS_DIR,
        "SEQUENCE_DIR": SEQUENCE_DIR,
    }


def verify_scoped_paths() -> dict:
    """Assert no library path escapes this package (rules.md R0).

    Returns a dict with ``ok``, ``escapes`` and ``paths``. ``escapes`` is empty
    only when every path resolves inside ``tracking_using_drive/``.
    """
    boundary = PACKAGE_DIR.resolve()
    paths = scoped_paths()
    escapes = []
    for name, path in paths.items():
        resolved = path.expanduser().resolve()
        if not resolved.is_relative_to(boundary):
            escapes.append({"name": name, "path": str(resolved)})
    return {
        "ok": not escapes,
        "boundary": str(boundary),
        "escapes": escapes,
        "paths": {k: str(v) for k, v in paths.items()},
    }


def ensure_local_dirs() -> None:
    """Create this package's own directories. Idempotent."""
    for path in (DATA_ROOT, GDRIVE_CACHE_DIR, WEIGHTS_DIR, SEQUENCE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def data_presence() -> dict:
    """Whether the library's expected inputs exist locally. Does not print.

    ``space_debris.config.verify_paths()`` is the library's equivalent, but it
    prints tick/cross glyphs and returns only a bool. Those glyphs are not
    encodable in the Windows cp1252 console default, and the bool is useless
    for JSON output — so we check the paths directly instead.
    """
    def _info(path: Path) -> dict:
        p = Path(path)
        entry: dict = {"path": str(p), "exists": p.exists()}
        if p.is_dir():
            try:
                entry["entries"] = sum(1 for x in p.iterdir() if x.is_file())
            except OSError:
                entry["entries"] = None
        return entry

    return {"IMAGE_DIR": _info(IMAGE_DIR), "LABEL_CSV": _info(LABEL_CSV)}


__all__ = [
    "PACKAGE_DIR", "REPO_ROOT", "DATA_ROOT", "GDRIVE_CACHE_DIR",
    "WEIGHTS_DIR", "SEQUENCE_DIR", "BASE_DIR", "DATA_DIR", "PROCESSED_DIR",
    "LABELS_DIR", "YOLO_LABEL_DIR", "YOLO_BASE", "CHECKPOINT_DIR", "RUNS_DIR",
    "IMAGE_DIR", "LABEL_CSV", "DEVICE", "CLASS_NAMES", "NUM_CLASSES",
    "DEBRIS_CLASSES", "class_id", "get_yaml_config", "verify_paths",
    "ensure_data_synced", "scoped_paths", "verify_scoped_paths",
    "ensure_local_dirs", "data_presence",
]
