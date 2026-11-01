"""Phase 2 tests: Drive access, selective sync, and source resolution.

Two categories, deliberately separated:

* **Offline tests** — pure functions and cache behaviour. These always run.
* **Network tests** — marked and skipped unless ``CAPSTONE_NETWORK_TESTS=1``.
  They hit the live public Drive folder, so they are opt-in: a CI box or an
  offline machine must not fail because Google is unreachable.

The warm-path assertions only run when a populated cache is already present,
which is the normal state after one ``sync``.
"""
from __future__ import annotations

import inspect
import os
import tempfile
from pathlib import Path

import pytest

from tracking_using_drive import drive, source

# Opt-in: real requests to the shared Drive folder.
_network = pytest.mark.skipif(
    os.environ.get("CAPSTONE_NETWORK_TESTS") != "1",
    reason="set CAPSTONE_NETWORK_TESTS=1 to hit the live Drive folder",
)

_CACHE = drive.GDRIVE_CACHE_DIR
_HAS_CACHE = (
    drive.load_manifest(_CACHE) is not None
    and (_CACHE / "labels" / "train.csv").exists()
)


# ---------------------------------------------------------------------------
# Pure functions — no I/O
# ---------------------------------------------------------------------------

def test_listing_parser_reads_the_observed_structure():
    """Parse the exact HTML shape embeddedfolderview emitted on 2026-11-01.

    If Google changes the markup this fails loudly, which is the point: a
    silent parse failure would look like a successfully synced empty folder.
    """
    html = (
        '<div class="flip-entry" id="entry-1ABCdefGHIjklMNOpqrSTUvwx12" '
        'tabindex="0" role="link"><div class="flip-entry-info">'
        '<a href="https://drive.google.com/drive/folders/1ABC">'
        '<div class="flip-entry-icon"><div aria-label="Folder" '
        'class="drive-sprite-folder-grid-shared-icon"></div></div>'
        '<div class="flip-entry-title">yolo_dataset</div></a></div></div>'
        '<div class="flip-entry" id="entry-1XYZ1234567890abcdefghij" '
        'tabindex="0" role="link"><div class="flip-entry-info">'
        '<div class="flip-entry-title">train.csv</div></div></div>'
    )
    entries = drive._parse_listing(html)
    assert len(entries) == 2
    assert entries[0].name == "yolo_dataset"
    assert entries[0].is_folder is True
    assert entries[0].file_id == "1ABCdefGHIjklMNOpqrSTUvwx12"
    assert entries[1].name == "train.csv"
    assert entries[1].is_folder is False


def test_listing_parser_unescapes_names():
    html = (
        '<div class="flip-entry" id="entry-1ABCdefGHIjklMNOpqrSTUvwx12">'
        '<div class="flip-entry-title">a &amp; b</div></div>'
    )
    assert drive._parse_listing(html)[0].name == "a & b"


def test_html_detection_catches_interstitials():
    """uc?export=download serves HTML for large files; writing it would
    silently corrupt the cache."""
    assert drive._looks_like_html(b"<!DOCTYPE html><html>")
    assert drive._looks_like_html(b"   \n<html lang=en>")
    assert not drive._looks_like_html(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    assert not drive._looks_like_html(b"filename,class,bbox\n")


def test_expand_entry_covers_files_and_directory_prefixes():
    """The manifest is keyed per file; an allowlist entry may be a directory.

    This is the exact bug the warm path originally had: a directory entry
    looked up verbatim found nothing, so every warm run fell through to a
    full re-sync.
    """
    recorded = {
        "labels/train.csv": {"bytes": 10},
        "yolo_dataset/images/val/img1.jpg": {"bytes": 20},
        "yolo_dataset/images/val/img2.jpg": {"bytes": 30},
        "yolo_dataset/labels/val/img1.txt": {"bytes": 5},
    }
    assert drive._expand_entry(recorded, "labels/train.csv") == ["labels/train.csv"]
    assert drive._expand_entry(recorded, "yolo_dataset/images/val") == [
        "yolo_dataset/images/val/img1.jpg",
        "yolo_dataset/images/val/img2.jpg",
    ]
    assert drive._expand_entry(recorded, "nothing/here") == []


def test_allowlist_is_small_not_the_whole_folder():
    """FR-2.5 / G10: the folder holds three overlapping image copies."""
    assert "train" not in drive.TRACKING_ALLOWLIST
    assert "processed" not in drive.TRACKING_ALLOWLIST
    assert "labels/train.csv" in drive.TRACKING_ALLOWLIST
    assert "yolo_dataset/images/val" in drive.TRACKING_ALLOWLIST


def _code_string_constants(module) -> set[str]:
    """Every string constant in ``module``'s AST, excluding docstrings.

    Docstrings legitimately quote the offending path to explain the defect —
    you cannot document why a default is wrong without naming it. FR-3.1
    constrains *code*, so the check must ignore documentation.
    """
    import ast

    tree = ast.parse(inspect.getsource(module))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
    # Drop module/class/function docstrings: they are the first statement of
    # their enclosing body.
    docstrings = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if not isinstance(first, ast.Expr):
            continue
        val = getattr(first, "value", None)
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            docstrings.add(val.value)
    return found - docstrings


def test_no_hardcoded_drive_path():
    """FR-3.1: the author's absolute drive path must never be used in code.

    Deliberately AST-based rather than a raw text grep. A text grep is
    unsatisfiable here: the guard itself must contain the literal to check
    for it, and the module docstrings legitimately quote it to explain why
    the default is wrong. FR-3.1 constrains *code*, so only code constants
    are in scope.
    """
    assert "E:/Capstone" not in _code_string_constants(drive), (
        "FR-3.1 violated: the hardcoded default appears as a code constant"
    )


# ---------------------------------------------------------------------------
# Source resolution — offline
# ---------------------------------------------------------------------------

def test_camera_index_round_trips():
    assert source.resolve_source("0") == 0
    assert source.resolve_source("2") == 2


def test_still_and_video_classification(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0")
    vid = tmp_path / "a.mp4"
    vid.write_bytes(b"\x00\x00\x00")
    assert source.is_still(img) is True
    assert source.is_video(vid) is True
    assert source.is_still(tmp_path) is False
    assert source.is_still(tmp_path / "missing.jpg") is False


def test_local_path_passthrough(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    assert source.resolve_source(str(tmp_path)) == str(tmp_path)


def test_empty_directory_fails_early(tmp_path):
    """design.md §8: fail before loading a model, not after."""
    with pytest.raises(ValueError, match="empty"):
        source.resolve_source(str(tmp_path))


def test_prefix_constant_is_not_a_url_parse():
    assert source.GDRIVE_PREFIX == "gdrive://"


# ---------------------------------------------------------------------------
# Cache behaviour — offline, but needs a populated cache for the warm path
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_CACHE, reason="no populated Drive cache present")
def test_warm_cache_makes_zero_network_calls():
    """R10: the warm path is where most Drive bugs hide."""
    before = drive.network_calls()
    report = drive.sync_allowlist(no_sync=True)
    assert report.network_calls == 0
    assert drive.network_calls() == before
    assert report.from_cache is True
    assert len(report.files) > 0


@pytest.mark.skipif(not _HAS_CACHE, reason="no populated Drive cache present")
def test_gdrive_resolution_returns_local_directory():
    resolved = source.resolve_gdrive("yolo_dataset/images/val", no_sync=True)
    assert resolved.is_dir()
    assert sum(1 for x in resolved.iterdir() if x.is_file()) == 542


@pytest.mark.skipif(not _HAS_CACHE, reason="no populated Drive cache present")
def test_manifest_records_folder_id():
    manifest = drive.load_manifest(_CACHE)
    assert manifest["folder_id"] == drive.DEFAULT_FOLDER_ID


def test_no_sync_on_empty_cache_raises(tmp_path):
    """--no-sync must never quietly reach for the network."""
    with pytest.raises(drive.DriveError, match="no-sync"):
        drive.sync_allowlist(no_sync=True, cache_dir=tmp_path)


def test_unknown_allowlist_path_raises(tmp_path):
    with pytest.raises(drive.DriveError, match="not found"):
        drive.sync_allowlist(("definitely/not/a/path",), cache_dir=tmp_path)


def test_report_metadata_carries_folder_and_cache_flag():
    meta = drive.SyncReport(
        root=Path("/x"), folder_id="abc", files=["a/b.csv"],
    ).as_metadata()
    assert meta["folder_id"] == "abc"
    assert meta["cache_hit"] is False
    assert meta["n_files"] == 1
    assert "provider" in meta


# ---------------------------------------------------------------------------
# Network — opt-in only
# ---------------------------------------------------------------------------

@_network
def test_live_list_and_download(tmp_path):
    entries, truncated = drive.list_folder(drive.DEFAULT_FOLDER_ID)
    assert len(entries) > 0
    names = {e.name for e in entries}
    assert {"train", "labels", "yolo_dataset"} <= names
    # G17: observed cap, and train/ is known to exceed it.
    assert drive.PUBLIC_LISTING_CAP == 5500
    assert isinstance(truncated, bool)


@_network
def test_live_csv_download_is_valid(tmp_path):
    dest = tmp_path / "train.csv"
    size = drive.download_file(
        "1DypZMDRZxnSjbJoWN0_5CcQzMh0b_jp8", dest
    )
    assert size > 0
    assert dest.stat().st_size == size
    head = dest.read_text(encoding="utf-8").splitlines()[0]
    assert head == "filename,class,bbox"
