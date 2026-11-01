"""Entrypoint: ``python -m tracking_using_drive.main <command>``.

Thin wrapper over :func:`tracking_using_drive.cli.main`. Importing the package
first is what primes ``CAPSTONE_DIR`` before the library is touched — see
``__init__.py``.
"""
from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
