"""Phase 1 tests: the R0 isolation invariant and the import bootstrap.

These are the guard rails for the whole branch. If the bootstrap regresses,
every library path silently escapes into the repository root and starts
creating directories there — with no error raised anywhere.

Run with::

    python -m pytest tracking_using_drive/tests -q
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tracking_using_drive import PACKAGE_DIR, REPO_ROOT
from tracking_using_drive import config


# ---------------------------------------------------------------------------
# R0: nothing escapes the package
# ---------------------------------------------------------------------------

def test_no_library_path_escapes_the_package():
    result = config.verify_scoped_paths()
    assert result["ok"], (
        "library paths escaped tracking_using_drive/: "
        f"{result['escapes']}\nexpected boundary {result['boundary']}"
    )


def test_every_scoped_path_is_inside_the_package():
    boundary = PACKAGE_DIR.resolve()
    for name, path in config.scoped_paths().items():
        assert path.expanduser().resolve().is_relative_to(boundary), (
            f"{name} = {path} is outside {boundary}"
        )


def test_capstone_dir_env_points_into_the_package():
    env = os.environ.get("CAPSTONE_DIR")
    assert env is not None, "CAPSTONE_DIR was never primed"
    assert Path(env).expanduser().resolve().is_relative_to(PACKAGE_DIR.resolve()), (
        f"CAPSTONE_DIR={env} is outside the package"
    )


def test_library_base_dir_matches_capstone_dir():
    import space_debris.config as lib

    assert Path(lib.BASE_DIR).resolve() == Path(os.environ["CAPSTONE_DIR"]).resolve()


# ---------------------------------------------------------------------------
# R1: the class taxonomy must never be reordered
# ---------------------------------------------------------------------------

EXPECTED_CLASS_NAMES = [
    "cheops",
    "debris",
    "double_star",
    "earth_observation_sat_1",
    "lisa_pathfinder",
    "proba_2",
    "proba_3_csc",
    "proba_3_ocs",
    "smart_1",
    "soho",
    "xmm_newton",
]


def test_class_taxonomy_order_is_unchanged():
    assert config.CLASS_NAMES == EXPECTED_CLASS_NAMES


def test_debris_classes_is_debris_only():
    assert config.DEBRIS_CLASSES == ["debris"]


def test_class_id_maps_debris():
    assert config.class_id("debris") == 1


def test_class_id_rejects_unknown():
    with pytest.raises(KeyError):
        config.class_id("satellite_that_does_not_exist")


# ---------------------------------------------------------------------------
# The bootstrap, exercised in a fresh interpreter
# ---------------------------------------------------------------------------

def _run(code: str, *, drop: tuple[str, ...] = (), env_extra: dict[str, str] | None = None):
    """Run ``code`` in a fresh interpreter at the repo root."""
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_bootstrap_primes_the_library_in_a_fresh_process():
    """Importing the package first must scope space_debris's paths.

    CAPSTONE_DIR is dropped from the environment so the package's own priming
    is the only thing that can decide BASE_DIR.
    """
    code = (
        "from pathlib import Path\n"
        "import tracking_using_drive\n"
        "import space_debris.config as lib\n"
        "from tracking_using_drive import PACKAGE_DIR\n"
        "b = Path(lib.BASE_DIR).resolve()\n"
        "assert b.is_relative_to(PACKAGE_DIR.resolve()), f'escaped: {b}'\n"
        "print(b)\n"
    )
    proc = _run(code, drop=("CAPSTONE_DIR",))
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert str(PACKAGE_DIR) in proc.stdout


def test_guard_rejects_an_unscoped_pre_existing_import():
    """If space_debris.config lands outside the package, we must fail loudly."""
    with tempfile.TemporaryDirectory() as tmp:
        # Scope the library to a temp dir first, so its mkdir side effect lands
        # in the temp dir rather than the repo root. Then importing our config
        # must raise, because BASE_DIR is already fixed out of bounds.
        code = (
            "import os\n"
            f"os.environ['CAPSTONE_DIR'] = {str(tmp)!r}\n"
            "import space_debris.config\n"          # scoped to tmp -> out of bounds
            "import tracking_using_drive.config\n"  # must raise RuntimeError
        )
        proc = _run(code)
        assert proc.returncode != 0, (
            "guard failed to fire: importing space_debris.config out of scope "
            "must raise RuntimeError"
        )
        assert "imported before" in (proc.stderr or ""), proc.stderr


def test_cli_verify_passes():
    from tracking_using_drive.cli import main

    assert main(["verify"]) == 0
