"""Automatic Google Drive upload of the processed_dataset/ tree.

Design notes
------------
* Reuses the project's existing pydrive2 authentication
  (`space_debris.data.gdrive_sync.GDriveSync` - same OAuth flow and
  credentials.json the baseline pipeline already uses).
* Uploads run on a SINGLE background thread so the pydrive2/GoogleAuth
  objects are never shared across threads while the CPU-bound preprocessing
  runs in parallel (network overlaps with compute).
* Fully resumable: a manifest maps every local relative path to its remote
  file id + size. Reruns skip identical files, update changed ones in place
  (never creating Drive duplicates) and continue where a crash left off.
* On the very first run (or after manifest loss) the remote folder is listed
  once to seed the manifest, so already-uploaded files are not pushed again.
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Callable

FOLDER_MIME = "application/vnd.google-apps.folder"
MANIFEST_FLUSH_EVERY = 25
PROGRESS_EVERY = 500


class DriveUploader:
    """Background uploader for a local directory tree -> Google Drive."""

    def __init__(
        self,
        local_root: str | Path,
        folder_name: str = "processed_dataset",
        parent_folder_name: str = "Capstone",
        manifest_name: str = ".upload_manifest.json",
        credentials_file: str = "credentials.json",
        log: Callable[[str], None] = print,
    ) -> None:
        self.local_root = Path(local_root).resolve()
        self.folder_name = folder_name
        self.parent_folder_name = parent_folder_name
        self.manifest_path = self.local_root / manifest_name
        self.credentials_file = credentials_file
        self.log = log

        self._queue: "queue.Queue[Path | None]" = queue.Queue(maxsize=1024)
        self._thread: threading.Thread | None = None
        self._drive = None
        self._folder_cache: dict[str, str] = {}  # remote dir relpath -> folder id
        self._manifest: dict[str, dict] = {}
        self._dirty = 0
        self.stats = {
            "uploaded_new": 0,
            "updated": 0,
            "skipped_identical": 0,
            "failed": 0,
            "errors": [],
        }
        self._remote_root_id: str | None = None

    # -- auth / folder resolution ------------------------------------------
    def _authenticate_and_resolve(self) -> None:
        from space_debris.data.gdrive_sync import GDriveSync  # reuse baseline auth

        # Cache lives OUTSIDE the uploaded tree (parent dir, e.g. repo root's
        # gdrive_cache/ which is already gitignored).
        sync = GDriveSync(
            cache_dir=str(self.local_root.parent / "gdrive_cache"),
            credentials_file=self.credentials_file,
        )
        self._drive = sync.drive

        # Prefer placing processed_dataset/ inside the Capstone folder.
        parent_id = None
        try:
            parent = sync.find_shared_with_me_folder(self.parent_folder_name)
            if parent is None:
                parent = sync.find_folder(self.parent_folder_name)
            if parent is not None:
                parent_id = parent["id"]
                self.log(
                    f"[upload] parent folder: {self.parent_folder_name} "
                    f"({parent_id})"
                )
        except Exception as exc:  # noqa: BLE001 - fall back to My Drive root
            self.log(f"[upload] parent folder lookup failed ({exc}); using My Drive root")

        try:
            self._remote_root_id = self._find_or_create_folder(
                self.folder_name, parent_id
            )
        except Exception as exc:  # noqa: BLE001 - e.g. no write access to shared folder
            self.log(
                f"[upload] cannot create '{self.folder_name}' under parent "
                f"({exc}); falling back to My Drive root"
            )
            self._remote_root_id = self._find_or_create_folder(
                self.folder_name, None
            )
        self._folder_cache[""] = self._remote_root_id
        self.log(f"[upload] remote folder id: {self._remote_root_id}")

    def _find_or_create_folder(self, name: str, parent_id: str | None) -> str:
        drive = self._drive
        query = (
            f"title='{name}' and mimeType='{FOLDER_MIME}' and trashed=false"
        )
        if parent_id:
            query += f" and '{parent_id}' in parents"
        else:
            query += " and 'root' in parents"
        found = drive.ListFile({"q": query}).GetList()
        if found:
            return found[0]["id"]

        meta = {"title": name, "mimeType": FOLDER_MIME}
        if parent_id:
            meta["parents"] = [{"id": parent_id}]
        folder = drive.CreateFile(meta)
        folder.Upload()
        return folder["id"]

    def _ensure_remote_dir(self, rel_dir: str) -> str:
        """Create (and cache) the remote folder chain for a relative dir."""
        if rel_dir in self._folder_cache:
            return self._folder_cache[rel_dir]
        parent_rel = str(Path(rel_dir).parent) if "/" in rel_dir or "\\" in rel_dir else ""
        if parent_rel == ".":
            parent_rel = ""
        parent_id = self._ensure_remote_dir(parent_rel)
        name = Path(rel_dir).name
        folder = self._find_or_create_folder(name, parent_id)
        self._folder_cache[rel_dir] = folder
        return folder

    # -- manifest -----------------------------------------------------------
    def _load_manifest(self) -> None:
        if self.manifest_path.exists():
            try:
                self._manifest = json.loads(
                    self.manifest_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                self.log("[upload] manifest unreadable -> rebuilding from Drive")
                self._manifest = {}
        else:
            self._manifest = {}

    def _flush_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self._manifest, indent=1), encoding="utf-8"
        )
        self._dirty = 0

    def _seed_manifest_from_drive(self) -> None:
        """First run / lost manifest: list remote tree so we never duplicate."""
        if self.manifest_path.exists() or not self._remote_root_id:
            return
        self.log("[upload] seeding manifest by listing the remote folder...")
        count = 0

        def walk(folder_id: str, prefix: str) -> None:
            nonlocal count
            page = self._drive.ListFile(
                {"q": f"'{folder_id}' in parents and trashed=false"}
            )
            for item in page.GetList():
                remote_path = f"{prefix}{item['title']}"
                if item["mimeType"] == FOLDER_MIME:
                    walk(item["id"], remote_path + "/")
                else:
                    try:
                        size = int(item.get("fileSize", 0))
                    except (TypeError, ValueError):
                        size = 0
                    self._manifest[remote_path] = {"id": item["id"], "size": size}
                    count += 1

        try:
            walk(self._remote_root_id, "")
        except Exception as exc:  # noqa: BLE001 - seeding is best-effort
            self.log(f"[upload] manifest seeding failed ({exc}); continuing empty")
            self._manifest = {}
            return
        if count:
            self._flush_manifest()
        self.log(f"[upload] manifest seeded with {count} remote file(s)")

    # -- upload one file ----------------------------------------------------
    def _upload_file(self, local: Path) -> None:
        rel = local.relative_to(self.local_root).as_posix()
        local_size = local.stat().st_size
        entry = self._manifest.get(rel)

        if entry and entry.get("size") == local_size:
            self.stats["skipped_identical"] += 1
            return

        parent_rel = str(Path(rel).parent)
        if parent_rel == ".":
            parent_rel = ""
        parent_id = self._ensure_remote_dir(parent_rel)

        try:
            if entry and entry.get("id"):
                # Update existing remote file in place (no Drive duplicates).
                remote = self._drive.CreateFile({"id": entry["id"]})
            else:
                remote = self._drive.CreateFile(
                    {"title": local.name, "parents": [{"id": parent_id}]}
                )
            remote.SetContentFile(str(local))
            remote.Upload()
            self._manifest[rel] = {"id": remote["id"], "size": local_size}
            if entry:
                self.stats["updated"] += 1
            else:
                self.stats["uploaded_new"] += 1
        except Exception as exc:  # noqa: BLE001 - keep the run alive
            self.stats["failed"] += 1
            if len(self.stats["errors"]) < 100:
                self.stats["errors"].append({"file": rel, "reason": str(exc)})

        self._dirty += 1
        done = (
            self.stats["uploaded_new"]
            + self.stats["updated"]
            + self.stats["skipped_identical"]
        )
        if self._dirty >= MANIFEST_FLUSH_EVERY:
            self._flush_manifest()
        if done % PROGRESS_EVERY == 0 and done:
            self.log(
                f"[upload] progress: {done} files "
                f"(new={self.stats['uploaded_new']} "
                f"updated={self.stats['updated']} "
                f"skipped={self.stats['skipped_identical']} "
                f"failed={self.stats['failed']})"
            )

    # -- worker thread ------------------------------------------------------
    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._upload_file(Path(item))
            except Exception as exc:  # noqa: BLE001
                self.stats["failed"] += 1
                if len(self.stats["errors"]) < 100:
                    self.stats["errors"].append(
                        {"file": str(item), "reason": str(exc)}
                    )
        if self._dirty:
            self._flush_manifest()

    # -- public API ---------------------------------------------------------
    def start(self) -> None:
        """Authenticate, resolve folders, seed manifest, start the thread."""
        self._authenticate_and_resolve()
        self._load_manifest()
        self._seed_manifest_from_drive()
        self._thread = threading.Thread(
            target=self._worker, name="drive-uploader", daemon=True
        )
        self._thread.start()
        self.log("[upload] background uploader started")

    def put(self, path: Path) -> None:
        """Enqueue a local file for upload (blocking when the queue is full)."""
        if self._thread is None:
            return
        if path and Path(path).exists():
            self._queue.put(Path(path))

    def finish(self) -> dict:
        """Drain the queue, stop the thread, final manifest flush."""
        if self._thread is None:
            return dict(self.stats)
        self._queue.put(None)
        self._thread.join()
        self._thread = None
        self.log(
            f"[upload] finished: new={self.stats['uploaded_new']} "
            f"updated={self.stats['updated']} "
            f"skipped={self.stats['skipped_identical']} "
            f"failed={self.stats['failed']}"
        )
        return dict(self.stats)


def upload_tree(
    local_root: str | Path,
    folder_name: str = "processed_dataset",
    parent_folder_name: str = "Capstone",
    credentials_file: str = "credentials.json",
    log: Callable[[str], None] = print,
) -> dict:
    """Upload an already-processed local tree (used by --upload-only)."""
    root = Path(local_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    uploader = DriveUploader(
        root,
        folder_name=folder_name,
        parent_folder_name=parent_folder_name,
        credentials_file=credentials_file,
        log=log,
    )
    uploader.start()

    skip_names = {uploader.manifest_path.name}
    count = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name in skip_names:
            continue
        # Never upload hidden files/dirs (manifests, caches, OS noise).
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        uploader.put(path)
        count += 1
    log(f"[upload] queued {count} file(s) from {root}")
    return uploader.finish()
