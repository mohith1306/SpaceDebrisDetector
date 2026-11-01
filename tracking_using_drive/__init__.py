"""Tracking debris using the Google Drive-hosted dataset.

All work on this branch happens inside this package. ``space_debris/`` is a
read-only library: import from it, never write to it (see ``rules.md`` R0).

Bootstrapping
-------------
This module performs two *cheap* operations on import (no torch, no library
imports) so that the scope is primed no matter how the package is entered:

1. puts the repository root on ``sys.path`` so ``space_debris`` resolves;
2. sets ``CAPSTONE_DIR`` to ``tracking_using_drive/_data``.

(2) is load-bearing. ``space_debris/config.py:19-20`` reads ``CAPSTONE_DIR``
and ``:117-118`` creates four directories, both *at import time*. Setting the
variable here — before anything can import the library — is what keeps those
paths inside this package. See ``config.py`` for the full explanation and
``memory.md`` D10/G1/G2.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.1.0"

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

# 1. Make `import space_debris` work regardless of the caller's CWD.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 2. Prime CAPSTONE_DIR before any space_debris import can happen.
#    `setdefault` respects an explicit operator override.
os.environ.setdefault("CAPSTONE_DIR", str(PACKAGE_DIR / "_data"))

__all__ = ["PACKAGE_DIR", "REPO_ROOT", "__version__"]
