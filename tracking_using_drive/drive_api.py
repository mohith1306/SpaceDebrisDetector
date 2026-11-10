"""Drive API listing — for folders the public listing truncates (G17).

Why this module exists
----------------------
The anonymous ``embeddedfolderview`` endpoint behind :mod:`.drive` caps at
``PUBLIC_LISTING_CAP`` (5,500) children per folder and ignores paging (G17).
``train/`` holds **66,000 flat images**, so an anonymous sync of it can never
see more than the first 5,500 — the report says ``truncated`` honestly, but
the data is still incomplete. Phase 5 (full-scale training on JarvisLabs)
needs the whole folder.

Design decisions
----------------
* **Listing uses the official Drive API v3** (``files.list`` with
  ``pageToken``), which paginates without a cap. Downloads stay on the
  **anonymous endpoint by default** — they are public-file fetches and do not
  consume API quota; only the ~67 listing queries do.
* **The API key is never a code constant** (AST guard): ``--api-key`` flag
  or ``GDRIVE_API_KEY`` env var.
* **Skip-if-exists, not manifest.** ``download_file`` writes via a ``.part``
  temp and renames, so an existing file at the destination is a complete
  file. That makes a rerun after partial failure incremental for free — the
  honest fix for 66k downloads is simply to run it again.
* **Per-file failures are collected, not fatal.** With 66k requests some
  will fail after retries; aborting the other 99.9% to raise one exception
  helps nobody. The report counts them and the CLI exits non-zero so the
  operator knows to rerun.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence

from .drive import (
    DEFAULT_FOLDER_ID,
    DriveEntry,
    DriveError,
    _http_get,
    download_file,
)

# Drive API v3 files.list. The key rides along as a query parameter, which is
# the supported auth method for API keys (no OAuth flow needed for public
# folders).
API_LIST_URL = "https://www.googleapis.com/drive/v3/files"
API_MEDIA_URL = "https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"

FOLDER_MIME = "application/vnd.google-apps.folder"
PAGE_SIZE = 1000          # API maximum per page
MAX_PAGES = 10_000        # runaway guard: 10M files would be a bug, not data

API_KEY_ENV = "GDRIVE_API_KEY"

# Test seam: every listing goes through here, so tests paginate without a key.
_api_get: Callable[[str], bytes] = _http_get


# --- listing -----------------------------------------------------------------


def _require_key(api_key: str | None) -> str:
    import os

    key = api_key or os.environ.get(API_KEY_ENV) or ""
    if not key.strip():
        raise DriveError(
            "No Drive API key. Pass --api-key or set "
            f"{API_KEY_ENV}. Create one at console.cloud.google.com "
            "(APIs & Services → Credentials → API key) with the Google "
            "Drive API enabled."
        )
    return key.strip()


def _list_page(
    folder_id: str, api_key: str, page_token: str | None
) -> dict[str, Any]:
    """One ``files.list`` page as a parsed dict."""
    params: dict[str, str] = {
        "q": f"'{folder_id}' in parents and trashed=false",
        "pageSize": str(PAGE_SIZE),
        "fields": "nextPageToken,files(id,name,mimeType)",
        "key": api_key,
    }
    if page_token:
        params["pageToken"] = page_token
    url = f"{API_LIST_URL}?{urllib.parse.urlencode(params)}"
    try:
        payload = _api_get(url, timeout=60.0, retries=4)
    except urllib.error.HTTPError as exc:
        # Surface the API's own message — a 403 says "key invalid" or
        # "Drive API not enabled", which is exactly what the operator needs.
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8", "replace"))[
                "error"
            ].get("message", "")
        except Exception:  # noqa: BLE001 — best-effort detail only
            pass
        raise DriveError(
            f"Drive API listing failed (HTTP {exc.code}) for folder "
            f"{folder_id}: {detail or exc.reason}. Check the API key and "
            "that the Drive API is enabled for the project."
        ) from exc
    return json.loads(payload)


def list_folder_api(folder_id: str, api_key: str) -> tuple[list[DriveEntry], int]:
    """Every direct child of ``folder_id``, paginated. No 5,500 cap.

    Returns ``(entries, api_calls)``. ``api_calls`` is the number of
    ``files.list`` requests this listing consumed, recorded so the report can
    state its quota use honestly.
    """
    key = _require_key(api_key)
    entries: list[DriveEntry] = []
    token: str | None = None
    seen: set[str] = set()
    calls = 0
    for _ in range(MAX_PAGES):
        page = _list_page(folder_id, key, token)
        calls += 1
        for f in page.get("files") or []:
            if not f.get("id") or not f.get("name"):
                continue
            entries.append(
                DriveEntry(
                    name=f["name"],
                    file_id=f["id"],
                    is_folder=f.get("mimeType") == FOLDER_MIME,
                )
            )
        token = page.get("nextPageToken")
        if not token:
            return entries, calls
        if token in seen:
            raise DriveError(
                "Drive API returned a repeated pageToken — refusing to loop."
            )
        seen.add(token)
    raise DriveError(f"Listing exceeded {MAX_PAGES} pages — refusing to loop.")


def resolve_path_api(
    root_id: str, rel_path: str, api_key: str,
    *, _cache: dict[str, list[DriveEntry]] | None = None,
) -> tuple[DriveEntry | None, int]:
    """Walk ``rel_path`` from ``root_id`` using API listings.

    Same contract as :func:`drive.resolve_path` (minus the truncation list —
    the API does not truncate). ``_cache`` lets several paths share the
    listings of their common ancestors, which matters when syncing
    ``train`` and ``labels`` in one call.
    """
    cache = _cache if _cache is not None else {}
    calls = 0

    def _children(fid: str) -> list[DriveEntry]:
        nonlocal calls
        if fid not in cache:
            cache[fid], n = list_folder_api(fid, api_key)
            calls += n
        return cache[fid]

    parts = [p for p in rel_path.strip("/").split("/") if p]
    if not parts:
        return DriveEntry("<root>", root_id, True), calls

    current_id = root_id
    for depth, part in enumerate(parts):
        match = next(
            (c for c in _children(current_id) if c.name == part), None
        )
        if match is None:
            return None, calls
        if depth == len(parts) - 1:
            return match, calls
        if not match.is_folder:
            return None, calls
        current_id = match.file_id
    return None, calls  # pragma: no cover — parts is non-empty here


def list_folder_recursive_api(
    folder_id: str, api_key: str, prefix: str = "",
) -> tuple[list[DriveEntry], int]:
    """All files under ``folder_id``, with ``prefix`` prepended to each name.

    Returns ``(file_entries, api_calls)`` — folders themselves are not
    returned, only the files beneath them, mirroring
    :func:`drive.list_folder_recursive`.
    """
    children, calls = list_folder_api(folder_id, api_key)
    files: list[DriveEntry] = []
    for entry in children:
        rel = f"{prefix}{entry.name}"
        if entry.is_folder:
            sub_files, sub_calls = list_folder_recursive_api(
                entry.file_id, api_key, prefix=rel + "/"
            )
            files.extend(sub_files)
            calls += sub_calls
        else:
            files.append(
                DriveEntry(name=rel, file_id=entry.file_id, is_folder=False)
            )
    return files, calls


# --- download ----------------------------------------------------------------


def _download_via_api(file_id: str, dest: Path, api_key: str) -> int:
    """Download through ``alt=media`` — consumes API quota, but bypasses the
    anonymous endpoint's per-IP abuse limits (the fallback when a datacenter
    IP gets 429'd repeatedly)."""
    url = API_MEDIA_URL.format(file_id=file_id) + "&key=" + urllib.parse.quote(
        api_key
    )
    payload = _http_get(url, timeout=120.0, retries=4)
    from .drive import _looks_like_html

    if _looks_like_html(payload):
        raise DriveError(
            f"API download of {file_id} returned HTML, not file content "
            f"({len(payload)} bytes)."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(payload)
    tmp.replace(dest)
    return len(payload)


# --- sync --------------------------------------------------------------------


def sync_paths(
    paths: Sequence[str],
    *,
    folder_id: str = DEFAULT_FOLDER_ID,
    dest_root: Path,
    api_key: str | None = None,
    workers: int = 4,
    download_mode: str = "anonymous",
) -> dict[str, Any]:
    """Sync ``paths`` (folders or files) from Drive using API listings.

    ``dest_root`` receives the same layout the anonymous ``sync`` produces
    (``train/x.png`` → ``dest_root/train/x.png``), so the rest of the
    pipeline — ``jarvis.sh``, ``train_jarvis.py`` — needs no changes.

    Already-present files are skipped (atomic ``.part`` writes mean presence
    == completeness), which makes reruns after partial failures incremental.

    Returns a report dict; ``errors`` lists per-file failures that a rerun
    should fill in.
    """
    key = _require_key(api_key)
    if download_mode not in ("anonymous", "api"):
        raise DriveError(
            f"unknown download mode {download_mode!r} "
            "(expected 'anonymous' or 'api')"
        )
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    listed: list[tuple[Path, DriveEntry]] = []
    list_calls = 0
    ancestor_cache: dict[str, list[DriveEntry]] = {}
    unresolved: list[str] = []

    for raw in paths:
        rel = raw.strip("/")
        if not rel:
            continue
        entry, calls = resolve_path_api(
            folder_id, rel, key, _cache=ancestor_cache
        )
        list_calls += calls
        if entry is None:
            unresolved.append(rel)
            continue
        base = dest_root / rel
        if entry.is_folder:
            files, sub_calls = list_folder_recursive_api(
                entry.file_id, key, prefix=""
            )
            list_calls += sub_calls
            listed.extend((base / f.name, f) for f in files)
        else:
            listed.append((base, entry))

    if unresolved:
        raise DriveError(
            f"path(s) not found in Drive folder {folder_id}: {unresolved}"
        )

    to_fetch = [(dest, e) for dest, e in listed if not dest.exists()]
    n_skipped = len(listed) - len(to_fetch)

    def _fetch(dest: Path, entry: DriveEntry) -> tuple[Path, int]:
        if download_mode == "api":
            n = _download_via_api(entry.file_id, dest, key)
        else:
            n = download_file(entry.file_id, dest)
        return dest, n

    errors: list[str] = []
    downloaded = 0
    byte_count = 0
    if to_fetch:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(_fetch, dest, entry): dest
                for dest, entry in to_fetch
            }
            for fut in as_completed(futures):
                dest = futures[fut]
                try:
                    _, n = fut.result()
                    downloaded += 1
                    byte_count += n
                except (DriveError, OSError) as exc:
                    errors.append(f"{dest.name}: {exc}")

    return {
        "provider": "gdrive-api",
        "folder_id": folder_id,
        "paths": [p.strip("/") for p in paths if p and p.strip("/")],
        "download_mode": download_mode,
        "n_files": len(listed),
        "downloaded": downloaded,
        "skipped": n_skipped,
        "bytes": byte_count,
        "api_list_calls": list_calls,
        "errors": len(errors),
        "error_samples": errors[:10],
        # The API path exists precisely because this can never happen here;
        # recorded (always empty) so reports from both sync paths share shape.
        "truncated_folders": [],
    }
