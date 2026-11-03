"""Entrypoint: ``python -m tracking_using_drive.main <command>``.

Thin wrapper over :func:`tracking_using_drive.cli.main`. Importing the package
first is what primes ``CAPSTONE_DIR`` before the library is touched — see
``__init__.py``.

Also runnable directly as ``python tracking_using_drive/main.py --help``
(`rules.md` pre-commit checklist step 4). That form executes the file with no
package context, so ``from .cli import main`` raises ``ImportError:
attempted relative import with no known parent package``. The bootstrap below
resolves the package absolutely instead, so both invocation styles behave
identically.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # Direct execution: make the repo root importable, then import absolutely.
    # The package's own parent is one level up from this file.
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from tracking_using_drive.cli import main
else:
    from .cli import main

if __name__ == "__main__":
    sys.exit(main())
