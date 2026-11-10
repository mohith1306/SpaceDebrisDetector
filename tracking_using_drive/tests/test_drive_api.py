"""Phase 5 tests — Drive API listing/sync for folders the public listing caps.

Covers the G17 problem: the anonymous endpoint stops at 5,500 children per
folder and ignores paging, while ``train/`` holds 66,000 flat images. The API
path must paginate without a cap, fail loudly on quota/auth problems, skip
already-present files so reruns are incremental, and collect per-file
failures rather than aborting 66k downloads over one error.

Network tests are gated: the live listing test needs ``GDRIVE_API_KEY`` (or
``CAPSTONE_DRIVE_API_KEY``) plus ``CAPSTONE_NETWORK_TESTS=1``.
"""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

from tracking_using_drive import drive_api
from tracking_using_drive.drive import DriveError

FOLDER_MIME = "application/vnd.google-apps.folder"
FILE_MIME = "image/png"

TEST_KEY = "test-key-not-a-secret"


class FakeAPI:
    """Serves paginated ``files.list`` responses from an in-memory tree."""

    def __init__(self, folders: dict[str, list[dict]], page_size: int = 2):
        self.folders = folders
        self.page_size = page_size
        self.calls = 0

    def __call__(self, url: str, **kwargs) -> bytes:
        self.calls += 1
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        folder_id = query["q"][0].split("'")[1]
        token = query.get("pageToken", [None])[0]
        items = self.folders.get(folder_id, [])
        start = int(token) if token else 0
        chunk = items[start:start + self.page_size]
        body: dict = {"files": chunk}
        nxt = start + self.page_size
        if nxt < len(items):
            body["nextPageToken"] = str(nxt)
        return json.dumps(body).encode()


def _file(name: str, mime: str = FILE_MIME) -> dict:
    return {"id": f"id-{name}", "name": name, "mimeType": mime}


@pytest.fixture(autouse=True)
def _isolated_key(monkeypatch):
    """No ambient key: tests must pass one explicitly or expect the error."""
    monkeypatch.delenv(drive_api.API_KEY_ENV, raising=False)


# ---------------------------------------------------------------------------
# Listing: pagination, folders, failure modes
# ---------------------------------------------------------------------------


def test_pagination_collects_every_page(monkeypatch):
    """The whole point: more than 5,500 children come back in full."""
    files = [_file(f"img_{i:03d}.png") for i in range(7)]  # 4 pages @ size 2
    fake = FakeAPI({"ROOT": files}, page_size=2)
    monkeypatch.setattr(drive_api, "_api_get", fake)

    entries, calls = drive_api.list_folder_api("ROOT", TEST_KEY)

    assert len(entries) == 7, "pagination must not drop entries"
    assert [e.name for e in entries] == [f["name"] for f in files]
    assert calls == 4, "one API call per page"
    assert fake.calls == 4


def test_folder_mime_marks_is_folder(monkeypatch):
    fake = FakeAPI({"ROOT": [_file("train", FOLDER_MIME), _file("a.png")]})
    monkeypatch.setattr(drive_api, "_api_get", fake)

    entries, _ = drive_api.list_folder_api("ROOT", TEST_KEY)
    by_name = {e.name: e for e in entries}

    assert by_name["train"].is_folder is True
    assert by_name["a.png"].is_folder is False


def test_empty_folder_returns_empty_list(monkeypatch):
    monkeypatch.setattr(drive_api, "_api_get", FakeAPI({"ROOT": []}))
    entries, calls = drive_api.list_folder_api("ROOT", TEST_KEY)
    assert entries == [] and calls == 1


def test_repeated_pagetoken_refuses_to_loop(monkeypatch):
    def sticky(url: str, **kwargs) -> bytes:
        return json.dumps(
            {"files": [_file("a.png")], "nextPageToken": "same"}
        ).encode()

    monkeypatch.setattr(drive_api, "_api_get", sticky)
    with pytest.raises(DriveError, match="pageToken"):
        drive_api.list_folder_api("ROOT", TEST_KEY)


def test_http_error_surfaces_the_api_message(monkeypatch):
    """A 403 must say *why* (key invalid / API not enabled), not just 403."""
    body = io.BytesIO(json.dumps(
        {"error": {"message": "API key not valid. Please pass a valid API key."}}
    ).encode())

    def denied(url: str, **kwargs):
        raise urllib.error.HTTPError(
            url, 403, "Forbidden", {}, body
        )

    monkeypatch.setattr(drive_api, "_api_get", denied)
    with pytest.raises(DriveError) as excinfo:
        drive_api.list_folder_api("ROOT", "bad-key")
    assert "API key not valid" in str(excinfo.value)
    assert "403" in str(excinfo.value)


def test_missing_key_is_a_clear_error():
    with pytest.raises(DriveError, match=drive_api.API_KEY_ENV):
        drive_api.list_folder_api("ROOT", None)


def test_explicit_key_wins_over_missing_env():
    assert drive_api._require_key(f" {TEST_KEY} ") == TEST_KEY


# ---------------------------------------------------------------------------
# Path resolution and recursion
# ---------------------------------------------------------------------------


def test_resolve_path_api_walks_nested_paths(monkeypatch):
    tree = {
        "ROOT": [
            _file("train", FOLDER_MIME),
            _file("labels", FOLDER_MIME),
        ],
        "id-labels": [_file("train.csv")],
    }
    monkeypatch.setattr(drive_api, "_api_get", FakeAPI(tree, page_size=50))

    entry, calls = drive_api.resolve_path_api("ROOT", "labels/train.csv", TEST_KEY)
    assert entry is not None and entry.name == "train.csv"
    assert calls == 2

    entry, _ = drive_api.resolve_path_api("ROOT", "train", TEST_KEY)
    assert entry is not None and entry.is_folder

    entry, _ = drive_api.resolve_path_api("ROOT", "nope", TEST_KEY)
    assert entry is None


def test_resolve_path_api_reuses_the_ancestor_cache(monkeypatch):
    tree = {"ROOT": [_file("a", FOLDER_MIME), _file("b", FOLDER_MIME)]}
    fake = FakeAPI(tree, page_size=50)
    monkeypatch.setattr(drive_api, "_api_get", fake)

    cache: dict = {}
    drive_api.resolve_path_api("ROOT", "a", TEST_KEY, _cache=cache)
    drive_api.resolve_path_api("ROOT", "b", TEST_KEY, _cache=cache)

    assert fake.calls == 1, "the shared ancestor must be listed once"


def test_recursive_api_returns_files_with_prefixes(monkeypatch):
    tree = {
        "ROOT": [_file("train", FOLDER_MIME)],
        "id-train": [
            _file("a.png"),
            _file("sub", FOLDER_MIME),
        ],
        "id-sub": [_file("b.png")],
    }
    monkeypatch.setattr(drive_api, "_api_get", FakeAPI(tree, page_size=50))

    files, calls = drive_api.list_folder_recursive_api("id-train", TEST_KEY)
    assert [f.name for f in files] == ["a.png", "sub/b.png"]
    assert calls == 2, "one listing per folder visited"


# ---------------------------------------------------------------------------
# Sync: skip-existing, error collection, modes
# ---------------------------------------------------------------------------


def _sync_tree() -> dict[str, list[dict]]:
    return {
        "ROOT": [_file("train", FOLDER_MIME)],
        "id-train": [_file("a.png"), _file("b.png"), _file("c.png")],
    }


def test_sync_skips_existing_and_counts_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(
        drive_api, "_api_get", FakeAPI(_sync_tree(), page_size=50)
    )
    fetched: list[str] = []

    def fake_download(file_id: str, dest: Path) -> int:
        fetched.append(dest.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"xx")
        return 2

    monkeypatch.setattr(drive_api, "download_file", fake_download)

    # Pre-existing file: presence means complete (atomic .part writes).
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "a.png").write_bytes(b"old")

    report = drive_api.sync_paths(
        ["train"], folder_id="ROOT", dest_root=tmp_path,
        api_key=TEST_KEY, workers=2,
    )

    assert report["n_files"] == 3
    assert report["skipped"] == 1
    assert report["downloaded"] == 2
    assert report["bytes"] == 4
    assert report["errors"] == 0
    assert sorted(fetched) == ["b.png", "c.png"]
    assert report["truncated_folders"] == [], "the API path never truncates"
    assert report["provider"] == "gdrive-api"


def test_sync_collects_failures_without_aborting(monkeypatch, tmp_path):
    monkeypatch.setattr(
        drive_api, "_api_get", FakeAPI(_sync_tree(), page_size=50)
    )

    def flaky(file_id: str, dest: Path) -> int:
        if dest.name == "b.png":
            raise DriveError("rate limited after 4 attempts")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return 1

    monkeypatch.setattr(drive_api, "download_file", flaky)

    report = drive_api.sync_paths(
        ["train"], folder_id="ROOT", dest_root=tmp_path,
        api_key=TEST_KEY, workers=1,
    )

    assert report["errors"] == 1, "one failure must not sink the other 65,999"
    assert report["downloaded"] == 2
    assert any("b.png" in s for s in report["error_samples"])


def test_sync_rejects_an_unknown_download_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(
        drive_api, "_api_get", FakeAPI(_sync_tree(), page_size=50)
    )
    with pytest.raises(DriveError, match="download mode"):
        drive_api.sync_paths(
            ["train"], folder_id="ROOT", dest_root=tmp_path,
            api_key=TEST_KEY, download_mode="carrier-pigeon",
        )


def test_sync_missing_path_names_it(monkeypatch, tmp_path):
    monkeypatch.setattr(
        drive_api, "_api_get", FakeAPI({"ROOT": []}, page_size=50)
    )
    with pytest.raises(DriveError, match="not found"):
        drive_api.sync_paths(
            ["train"], folder_id="ROOT", dest_root=tmp_path, api_key=TEST_KEY
        )


def test_api_download_mode_uses_alt_media(monkeypatch, tmp_path):
    """--download-mode api must go through alt=media with the key attached."""
    seen: list[str] = []

    def fake_http_get(url: str, **kwargs) -> bytes:
        seen.append(url)
        return b"png-bytes"

    monkeypatch.setattr(drive_api, "_api_get", FakeAPI(_sync_tree(), page_size=50))
    monkeypatch.setattr(drive_api, "_http_get", fake_http_get)

    report = drive_api.sync_paths(
        ["train"], folder_id="ROOT", dest_root=tmp_path,
        api_key=TEST_KEY, workers=1, download_mode="api",
    )

    assert report["errors"] == 0 and report["downloaded"] == 3
    assert all("alt=media" in u for u in seen)
    assert all("key=" + TEST_KEY in u for u in seen)
    assert (tmp_path / "train" / "a.png").read_bytes() == b"png-bytes"


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_sync_api_emits_json_and_exits_zero(monkeypatch, tmp_path, capsys):
    from tracking_using_drive.cli import main

    def fake_sync(paths, **kwargs):
        return {
            "provider": "gdrive-api", "folder_id": "F", "paths": list(paths),
            "download_mode": "anonymous", "n_files": 66000,
            "downloaded": 66000, "skipped": 0, "bytes": 1,
            "api_list_calls": 67, "errors": 0, "error_samples": [],
            "truncated_folders": [],
        }

    monkeypatch.setattr(drive_api, "sync_paths", fake_sync)
    rc = main(["sync-api", "train", "--json", "--api-key", TEST_KEY])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "gdrive-api"
    assert payload["n_files"] == 66000
    assert payload["errors"] == 0


def test_cli_sync_api_exits_nonzero_when_files_failed(monkeypatch, capsys):
    from tracking_using_drive.cli import main

    def fake_sync(paths, **kwargs):
        return {
            "provider": "gdrive-api", "folder_id": "F", "paths": list(paths),
            "download_mode": "anonymous", "n_files": 3,
            "downloaded": 2, "skipped": 0, "bytes": 2,
            "api_list_calls": 1, "errors": 1,
            "error_samples": ["b.png: rate limited"],
            "truncated_folders": [],
        }

    monkeypatch.setattr(drive_api, "sync_paths", fake_sync)
    rc = main(["sync-api", "train", "--json", "--api-key", TEST_KEY])
    assert rc == 1, "a partial sync must not present as success"
    out = capsys.readouterr().out
    assert json.loads(out)["errors"] == 1


def test_cli_sync_api_without_key_exits_two(capsys):
    from tracking_using_drive.cli import main

    rc = main(["sync-api", "train", "--json"])
    assert rc == 2
    assert drive_api.API_KEY_ENV in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Live (gated): the actual 66k folder
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (__import__("os").environ.get("CAPSTONE_NETWORK_TESTS")
         and (__import__("os").environ.get("GDRIVE_API_KEY")
              or __import__("os").environ.get("CAPSTONE_DRIVE_API_KEY"))),
    reason="needs CAPSTONE_NETWORK_TESTS=1 and a Drive API key",
)
def test_live_api_lists_the_full_train_folder(monkeypatch):
    """The property this module exists for: train/ comes back complete."""
    import os

    key = os.environ.get("GDRIVE_API_KEY") or os.environ["CAPSTONE_DRIVE_API_KEY"]
    from tracking_using_drive.drive import DEFAULT_FOLDER_ID

    entry, calls = drive_api.resolve_path_api(
        DEFAULT_FOLDER_ID, "train", key
    )
    assert entry is not None and entry.is_folder
    files, total_calls = drive_api.list_folder_recursive_api(
        entry.file_id, key
    )
    assert len(files) >= 60_000, (
        f"expected the full 66k train/ listing, got {len(files)} "
        f"in {total_calls} API calls"
    )
    assert calls >= 1
