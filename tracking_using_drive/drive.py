"""Google Drive access — anonymous first (D15).

Why this module exists
----------------------
``space_debris/data/gdrive_sync.py`` cannot be used as-is (``rules.md`` R0):

* ``:15`` defaults ``cache_dir`` to a hardcoded absolute path on the original
  author's machine (the ``E`` drive). It exists nowhere else, so the default
  is unusable for anyone else.
* ``:18`` defaults ``credentials_file`` to ``credentials.json``, resolved
  relative to the **current working directory**, so the same run finds or
  misses credentials depending on where it was launched from.
* ``:106`` ``sync_capstone_dataset()`` recurses the **entire** folder. The
  shared ``Capstone`` folder holds three overlapping copies of the images
  (``train/``, ``processed/``, ``yolo_dataset/``) — see ``memory.md`` G10 — so
  a whole-folder sync downloads ~88k images to do work that needs ~542.

The class itself is sound. Only its defaults and its call site are wrong, so
we pass explicit parameters rather than reimplement it.

The anonymous path (D15)
------------------------
The shared folder is **public**, so no OAuth is required to read it:

* listing  — ``embeddedfolderview?id=<folder>``
* download — ``uc?export=download&id=<file_id>``

Both were verified working with no credentials on 2026-11-01. That is why
Phases 2–4 carry no credential dependency at all. OAuth remains available as
an explicit opt-in for private folders and for enumeration beyond the public
listing cap (``memory.md`` G17).

Assumptions this module makes
-----------------------------
* The folder is public. If it is ever made private, the anonymous path raises
  with an actionable message rather than returning an empty result.
* The public listing caps at ~5,500 entries per folder and ignores paging
  (G17). We record that fact instead of pretending a truncated listing is
  complete.
* ``embeddedfolderview`` is an unversioned HTML endpoint, not a stable API. A
  structural change must fail loudly — a silent empty listing would look like
  a successful sync of nothing.
"""
from __future__ import annotations

import concurrent.futures
import html as _html
import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .config import GDRIVE_CACHE_DIR

# --- constants ---------------------------------------------------------------

DEFAULT_FOLDER_ID = "1vxqeBHOglilc1PP_ZSNA1fCx_jfTTdUY"

_PUBLIC_HOST = "https://drive.google.com"
_LIST_URL = _PUBLIC_HOST + "/embeddedfolderview?id={folder_id}"
_DOWNLOAD_URL = _PUBLIC_HOST + "/uc?export=download&id={file_id}"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# Observed 2026-11-01: the endpoint returns at most 5,500 entries and ignores
# any paging attempt. See G17. Recording the cap is what lets us report a
# truncated listing as truncated instead of silently complete.
PUBLIC_LISTING_CAP = 5500

# The synthesis slice — deliberately NOT the whole folder (D11, G10).
#   labels/train.csv        66k rows of pixel bboxes: the ground truth source
#   labels/val.csv          22k rows, annotation-only; no images back it (G21)
#   yolo_dataset/images/val 542 stills — the synthesis pool
#   yolo_dataset/labels/val 542 YOLO labels — independent cross-check on the
#                           bbox conversion, and cheap
# Total is ~160 MB, against ~14 GB for a whole-folder sync.
TRACKING_ALLOWLIST = (
    "labels/train.csv",
    "labels/val.csv",
    "yolo_dataset/images/val",
    "yolo_dataset/labels/val",
)

MANIFEST_NAME = "sync_manifest.json"

_RETRYABLE_HTTP = frozenset({429, 500, 502, 503, 504})

# Counts every outbound HTTP request, including listings. Tracking this at the
# transport layer rather than in the download loop is what makes the number
# honest: on a cold sync most calls are listings, and counting only downloads
# would under-report badly and make the "zero network calls" warm-path claim
# unverifiable.
_NETWORK_LOCK = threading.Lock()
_NETWORK_CALLS = 0


def _reset_network_counter() -> None:
    global _NETWORK_CALLS
    with _NETWORK_LOCK:
        _NETWORK_CALLS = 0


def network_calls() -> int:
    with _NETWORK_LOCK:
        return _NETWORK_CALLS


class DriveError(RuntimeError):
    """A Drive operation failed in a way that needs a human decision."""


# --- data --------------------------------------------------------------------


@dataclass(frozen=True)
class DriveEntry:
    """One child of a Drive folder, as reported by the public listing."""

    name: str
    file_id: str
    is_folder: bool


@dataclass
class SyncReport:
    """Outcome of a sync. Carries the metadata later phases write out (D11)."""

    root: Path
    folder_id: str
    files: list[str] = field(default_factory=list)  # repo-relative paths
    downloaded: int = 0
    cache_hits: int = 0
    bytes_downloaded: int = 0
    network_calls: int = 0
    truncated_folders: list[str] = field(default_factory=list)
    from_cache: bool = False

    def as_metadata(self) -> dict:
        """The slice of this report that belongs in an output record."""
        return {
            "provider": "gdrive-anonymous",
            "folder_id": self.folder_id,
            "cache_hit": self.from_cache,
            "n_files": len(self.files),
            "truncated_folders": list(self.truncated_folders),
        }


# --- low-level transport -----------------------------------------------------


def _http_get(url: str, *, timeout: float = 60.0, retries: int = 3) -> bytes:
    """GET with bounded exponential backoff on transient failures.

    Non-retryable HTTP errors propagate immediately; retrying a 404 or 403
    just delays the honest failure.
    """
    last_exc: BaseException | None = None
    for attempt in range(retries):
        try:
            global _NETWORK_CALLS
            with _NETWORK_LOCK:
                _NETWORK_CALLS += 1
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRYABLE_HTTP:
                raise
            last_exc = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
        if attempt < retries - 1:
            time.sleep(1.5 ** attempt)
    raise DriveError(f"GET failed after {retries} attempts: {url}") from last_exc


def _looks_like_html(payload: bytes) -> bool:
    """True if a response is an HTML page where data was expected.

    ``uc?export=download`` serves an interstitial for large files and an error
    page for inaccessible ones. Writing either into a ``.csv`` would corrupt
    the cache *and* look like a successful sync, so we detect it explicitly.
    """
    head = payload[:512].lstrip().lower()
    return head.startswith((b"<!doctype", b"<html"))


# --- listing -----------------------------------------------------------------


_ENTRY_SPLIT = 'class="flip-entry" id="entry-'
_ID_RE = re.compile(r'^([0-9A-Za-z_-]{10,})"')
_TITLE_RE = re.compile(r'class="flip-entry-title">([^<]*)</div>')
_FOLDER_HINT = 'aria-label="Folder"'


def _parse_listing(page: str) -> list[DriveEntry]:
    """Parse ``embeddedfolderview`` HTML into entries.

    Pure function so it can be tested offline against a recorded fragment.
    The markup is an unversioned endpoint, not a stable API — keeping the
    parser separate is what lets a structural change fail here, loudly,
    rather than surfacing as an empty sync much later.
    """
    entries: list[DriveEntry] = []
    for chunk in page.split(_ENTRY_SPLIT)[1:]:
        m_id = _ID_RE.match(chunk)
        if not m_id:
            continue
        m_title = _TITLE_RE.search(chunk)
        if not m_title:
            continue
        entries.append(DriveEntry(
            name=_html.unescape(m_title.group(1)),
            file_id=m_id.group(1),
            is_folder=_FOLDER_HINT in chunk,
        ))
    return entries


def list_folder(folder_id: str) -> tuple[list[DriveEntry], bool]:
    """List the direct children of a public Drive folder.

    Returns ``(entries, truncated)``. ``truncated`` is True when the listing
    hit ``PUBLIC_LISTING_CAP``, meaning the folder holds more children than
    the anonymous endpoint will reveal (G17).

    Raises :class:`DriveError` rather than returning an empty list when the
    page has no recognisable entries — an HTML-structure change must not
    masquerade as an empty folder.
    """
    payload = _http_get(_LIST_URL.format(folder_id=folder_id))
    page = payload.decode("utf-8", "replace")

    entries = _parse_listing(page)

    if not entries:
        # Distinguish "private/changed endpoint" from "genuinely empty".
        hint = ""
        if "You need access" in page or "request access" in page.lower():
            hint = (" The folder appears to require authorisation — it may "
                    "have been made private. Re-run with --oauth.")
        raise DriveError(
            f"Could not parse any entries from folder {folder_id}.{hint} "
            "The embeddedfolderview HTML structure may have changed; this is "
            "deliberately an error rather than a silently empty listing."
        )

    truncated = len(entries) >= PUBLIC_LISTING_CAP
    return entries, truncated


def resolve_path(
    root_folder_id: str,
    rel_path: str,
    *,
    _listing_cache: dict[str, tuple[list[DriveEntry], bool]] | None = None,
    _counter: list[int] | None = None,
) -> tuple[DriveEntry | None, list[str]]:
    """Walk ``rel_path`` (e.g. ``yolo_dataset/images/val``) from the root.

    Returns ``(entry, truncated_ancestors)``. ``entry`` is None when the path
    does not resolve. ``truncated_ancestors`` names any ancestor folder whose
    listing hit the cap, which is how a partial sync announces itself.
    """
    cache = _listing_cache if _listing_cache is not None else {}
    counter = _counter if _counter is not None else [0]
    truncated: list[str] = []

    def _list(fid: str) -> list[DriveEntry]:
        counter[0] += 1
        if fid not in cache:
            cache[fid] = list_folder(fid)
        return cache[fid][0]

    current_id = root_folder_id
    current_name = "<root>"
    parts = [p for p in rel_path.strip("/").split("/") if p]

    for depth, part in enumerate(parts):
        children = _list(current_id)
        if cache[current_id][1] and depth > 0:
            truncated.append(current_name)
        match = next((c for c in children if c.name == part), None)
        if match is None:
            return None, truncated
        if depth == len(parts) - 1:
            return match, truncated
        if not match.is_folder:
            return None, truncated
        current_id, current_name = match.file_id, match.name

    # rel_path was empty: the root itself.
    return DriveEntry(name="<root>", file_id=root_folder_id, is_folder=True), truncated


def list_folder_recursive(folder_id: str, prefix: str = "") -> tuple[list[DriveEntry], list[str]]:
    """All files under ``folder_id``, with ``prefix`` prepended to each name.

    Returns ``(files, truncated_paths)``. Folders that hit the listing cap are
    recorded so the caller can report the sync as partial.
    """
    entries, truncated = list_folder(folder_id)
    files: list[DriveEntry] = []
    cut: list[str] = []
    if truncated:
        cut.append(prefix.rstrip("/") or "<root>")
    for entry in entries:
        rel = f"{prefix}{entry.name}"
        if entry.is_folder:
            sub_files, sub_cut = list_folder_recursive(entry.file_id, prefix=rel + "/")
            files.extend(sub_files)
            cut.extend(sub_cut)
        else:
            files.append(DriveEntry(name=rel, file_id=entry.file_id, is_folder=False))
    return files, cut


# --- download ----------------------------------------------------------------


def download_file(
    file_id: str,
    dest: Path,
    *,
    timeout: float = 120.0,
) -> int:
    """Download one file to ``dest``. Returns the byte count.

    Refuses to write an HTML payload into ``dest`` — that is the interstitial
    or an error page, and silently caching it would poison every later run.
    """
    payload = _http_get(
        _DOWNLOAD_URL.format(file_id=file_id), timeout=timeout, retries=4
    )
    if _looks_like_html(payload):
        raise DriveError(
            f"Download of {file_id} returned an HTML page, not file content "
            f"({len(payload)} bytes). The file may be too large for the "
            "anonymous endpoint, or access may have been revoked."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(payload)
    tmp.replace(dest)  # never leave a half-written file at the final path
    return len(payload)


# --- manifest / warm path ----------------------------------------------------


def _manifest_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / MANIFEST_NAME


def load_manifest(cache_dir: Path) -> dict | None:
    path = _manifest_path(cache_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None  # a corrupt manifest is treated as no manifest


def save_manifest(cache_dir: Path, manifest: dict) -> None:
    path = _manifest_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _expand_entry(recorded: dict, rel: str) -> list[str]:
    """Manifest keys an allowlist entry covers.

    An entry may be a single file (``labels/train.csv``) or a directory prefix
    (``yolo_dataset/images/val``), while the manifest always records one key
    per downloaded file. So a directory entry expands to every key beneath it.
    Returns [] when the entry covers nothing, i.e. it is absent.
    """
    if rel in recorded:
        return [rel]
    prefix = rel + "/"
    return sorted(k for k in recorded if k.startswith(prefix))


def _warm_check(cache_dir: Path, rel_paths: Sequence[str]) -> SyncReport | None:
    """Return a report if every allowlisted path is already present locally.

    This is the ``--no-sync`` guarantee: zero network calls when the cache is
    complete. Any missing or size-mismatched file returns None, and the caller
    falls through to a real sync.
    """
    manifest = load_manifest(cache_dir)
    if not manifest:
        return None
    recorded = manifest.get("files", {})
    if manifest.get("folder_id") != DEFAULT_FOLDER_ID:
        return None

    resolved: list[str] = []
    for rel in rel_paths:
        keys = _expand_entry(recorded, rel)
        if not keys:
            return None
        for key in keys:
            local = Path(cache_dir) / key
            if not local.exists():
                return None
            if local.stat().st_size != recorded[key].get("bytes"):
                return None
            resolved.append(key)

    return SyncReport(
        root=Path(cache_dir),
        folder_id=manifest["folder_id"],
        files=resolved,
        cache_hits=len(resolved),
        network_calls=0,
        truncated_folders=list(manifest.get("truncated_folders", [])),
        from_cache=True,
    )


# --- the public entry point --------------------------------------------------


def sync_allowlist(
    allowlist: Sequence[str] = TRACKING_ALLOWLIST,
    *,
    folder_id: str = DEFAULT_FOLDER_ID,
    cache_dir: Path | None = None,
    no_sync: bool = False,
    workers: int = 4,
) -> SyncReport:
    """Sync only ``allowlist`` subpaths of a public Drive folder.

    Parameters
    ----------
    allowlist
        Paths relative to ``folder_id``. A trailing name that resolves to a
        folder downloads every file beneath it.
    folder_id
        Drive folder to read. Defaults to the shared ``Capstone`` folder.
    cache_dir
        Destination root. Defaults to the package's ``_data/gdrive_cache``,
        which keeps every byte inside the R0 boundary — defeating the
        ``E:/Capstone`` default at ``gdrive_sync.py:15``.
    no_sync
        Resolve from the warm cache only. Raises :class:`DriveError` if the
        cache cannot satisfy the allowlist, rather than reaching for the
        network. This is the FR-2.4 / R10 guarantee.
    workers
        Concurrent downloads. 4 is deliberately modest: the anonymous endpoint
        rate-limits, and a large fan-out trades throughput for 429s.

    Returns
    -------
    SyncReport
        Includes ``files`` (relative paths), download counts, and
        ``truncated_folders`` — populated whenever a listing hit the public
        cap (G17), so a partial sync never presents as complete.
    """
    root = Path(cache_dir) if cache_dir is not None else GDRIVE_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    rels = [p.strip("/") for p in allowlist if p and p.strip("/")]

    if no_sync:
        report = _warm_check(root, rels)
        if report is None:
            missing = _missing_from_cache(root, rels)
            raise DriveError(
                "--no-sync requested but the cache cannot satisfy the "
                f"allowlist.\n  cache dir: {root}\n  missing or stale: "
                f"{missing or rels}\nRun once without --no-sync to populate "
                "the cache."
            )
        return report

    # Warm path: if everything is already present and the manifest agrees,
    # make zero network calls. R10 says the warm path must be tested; this is
    # the property that makes a second run free.
    warm = _warm_check(root, rels)
    if warm is not None:
        return warm

    _reset_network_counter()

    # Resolve every allowlist entry to concrete files.
    listing_cache: dict[str, tuple[list[DriveEntry], bool]] = {}
    counter = [0]
    truncated: list[str] = []
    targets: list[tuple[str, DriveEntry]] = []  # (relative path, entry)

    for rel in rels:
        entry, cut = resolve_path(
            folder_id, rel,
            _listing_cache=listing_cache, _counter=counter,
        )
        for c in cut:
            if c not in truncated:
                truncated.append(c)
        if entry is None:
            raise DriveError(
                f"Allowlist path {rel!r} not found in Drive folder "
                f"{folder_id}. Check the spelling against the folder listing."
            )
        if entry.is_folder:
            files, sub_cut = list_folder_recursive(entry.file_id, prefix=rel + "/")
            for c in sub_cut:
                if c not in truncated:
                    truncated.append(c)
            targets.extend((f.name, f) for f in files)
        else:
            targets.append((rel, entry))

    report = SyncReport(
        root=root,
        folder_id=folder_id,
        files=[t[0] for t in targets],
        truncated_folders=truncated,
    )

    # The manifest records the byte size of every file as it was downloaded.
    # Re-fetch when the local file is absent, empty, or a *different size* than
    # recorded — existence alone is not enough, or a truncated file would be
    # blessed forever by the next _write_manifest. (Caught by test: appending
    # one byte to train.csv must trigger exactly one re-download.)
    prior = load_manifest(root) or {}
    prior_files: dict[str, dict] = prior.get("files", {})

    def _needs_fetch(rel: str) -> bool:
        local = root / rel
        if not local.exists():
            return True
        size = local.stat().st_size
        if size == 0:
            return True
        recorded = prior_files.get(rel, {}).get("bytes")
        return recorded is not None and size != recorded

    to_fetch = [(rel, e) for rel, e in targets if _needs_fetch(rel)]
    report.cache_hits = len(targets) - len(to_fetch)

    if to_fetch:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(download_file, e.file_id, root / rel): rel
                for rel, e in to_fetch
            }
            for fut in concurrent.futures.as_completed(futures):
                rel = futures[fut]
                try:
                    size = fut.result()
                except DriveError:
                    # Per-file granularity, per rules.md "What Not To Do":
                    # warn and continue. A single unreadable file must not
                    # abort a 500-file sync, but it must not vanish either.
                    print(f"  WARNING: failed to download {rel}")
                    continue
                report.downloaded += 1
                report.bytes_downloaded += size

    # Re-verify: only record files that actually landed, so the manifest can
    # never claim a file that is absent on the next run.
    landed = [rel for rel in report.files if (root / rel).exists()]
    report.files = landed

    # Listings and downloads both count. Read from the transport counter so
    # the figure reflects every outbound request, not just the download loop.
    report.network_calls = network_calls()

    _write_manifest(root, report)
    return report


def _missing_from_cache(root: Path, rels: Sequence[str]) -> list[str]:
    """Which allowlist paths the cache cannot currently satisfy."""
    missing: list[str] = []
    for rel in rels:
        local = root / rel
        if local.is_dir():
            if not any(local.iterdir()):
                missing.append(rel)
        elif not local.exists():
            missing.append(rel)
    return missing


def _write_manifest(root: Path, report: SyncReport) -> None:
    """Persist what was synced so a later run can take the warm path."""
    files: dict[str, dict] = {}
    for rel in report.files:
        local = root / rel
        if local.exists():
            files[rel] = {"bytes": local.stat().st_size}
    save_manifest(root, {
        "folder_id": report.folder_id,
        "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
        "truncated_folders": report.truncated_folders,
    })
