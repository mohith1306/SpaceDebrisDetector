"""Command-line entry point for tracking_using_drive.

Phase 1 implements ``status`` and ``verify``. Later phases register additional
subcommands here (``sync``, ``synth``, ``track``, ``stream``, ``orbit``); the
roadmap is shown in ``--help`` so the intended surface is visible before it
exists.

Run as::

    python -m tracking_using_drive.main status
    python -m tracking_using_drive.main verify
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import PACKAGE_DIR, __version__

ROADMAP = """\
roadmap (see tracking_using_drive/phases.md):
  status        resolved paths, device, environment          [Phase 1 - now]
  verify        assert no library path escapes the package   [Phase 1 - now]
  sync          selective Google Drive sync                  [Phase 2 - now]
  source        resolve a source (incl. gdrive://)           [Phase 2 - now]
  synth         still -> synthetic sequence + ground truth   [Phase 3]
  track         detect + track, with missed-detection fix    [Phase 4]
  stream        full pipeline: track -> classify -> orbit    [Phase 4]
  orbit         offline orbit report from a track JSONL      [Phase 4]
"""


def _ensure_utf8_stdio() -> None:
    """Force UTF-8 on stdio where possible.

    The Windows console defaults to cp1252, which cannot encode characters the
    library prints (``space_debris.config.verify_paths`` emits U+2713/U+2717 and
    crashes there). We are the entrypoint, so reconfiguring here protects every
    subsequent print. ``errors="replace"`` keeps an unencodable glyph from
    taking the process down.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # stream has no reconfigure (e.g. redirected to a pipe wrapper)


def _cmd_status(args: argparse.Namespace) -> int:
    from . import config

    info = {
        "package_version": __version__,
        "package_dir": str(PACKAGE_DIR),
        "capstone_dir_env": __import__("os").environ.get("CAPSTONE_DIR"),
        "device": config.DEVICE,
        "num_classes": config.NUM_CLASSES,
        "class_names": config.CLASS_NAMES,
        "debris_classes": config.DEBRIS_CLASSES,
        "paths": config.scoped_paths(),
        "scoped": config.verify_scoped_paths(),
        "library_data_present": config.data_presence(),
    }

    if args.json:
        print(json.dumps(info, indent=2))
        return 0

    print(f"tracking_using_drive {__version__}")
    print(f"  package dir   {info['package_dir']}")
    print(f"  CAPSTONE_DIR  {info['capstone_dir_env']}")
    print(f"  device        {info['device']}")
    print(f"  classes       {info['num_classes']}  debris={info['debris_classes']}")
    print()
    print("scoped paths (must all live inside tracking_using_drive/):")
    for name, path in info["paths"].items():
        print(f"  {name:18s} {path}")
    print()
    ok = info["scoped"]["ok"]
    print(f"R0 isolation: {'OK' if ok else 'BREACHED'}")
    if not ok:
        for esc in info["scoped"]["escapes"]:
            print(f"  ESCAPED {esc['name']} -> {esc['path']}")
    print()
    print("library data present locally:")
    for name, entry in info["library_data_present"].items():
        state = "yes" if entry["exists"] else "no"
        print(f"  {name:10s} {state:3s} {entry['path']}")
    return 0 if ok else 1


def _cmd_verify(args: argparse.Namespace) -> int:
    from . import config

    result = config.verify_scoped_paths()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["ok"]:
            print(f"OK - all {len(result['paths'])} paths inside {result['boundary']}")
        else:
            print(f"BREACHED - {len(result['escapes'])} path(s) escaped:")
            for esc in result["escapes"]:
                print(f"  {esc['name']:18s} {esc['path']}")
            print(f"expected boundary: {result['boundary']}")
    return 0 if result["ok"] else 1


def _cmd_sync(args: argparse.Namespace) -> int:
    """Selective Drive sync. Anonymous by default (D15); no credentials needed."""
    from .drive import DriveError, TRACKING_ALLOWLIST, sync_allowlist

    allowlist = tuple(args.path) if args.path else TRACKING_ALLOWLIST
    try:
        report = sync_allowlist(
            allowlist,
            folder_id=args.folder_id,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            no_sync=args.no_sync,
            workers=args.workers,
        )
    except DriveError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({
            **report.as_metadata(),
            "root": str(report.root),
            "downloaded": report.downloaded,
            "cache_hits": report.cache_hits,
            "bytes_downloaded": report.bytes_downloaded,
            "network_calls": report.network_calls,
        }, indent=2))
        return 0

    mode = "cache (no network)" if report.from_cache else "synced"
    print(f"Drive sync ({mode})")
    print(f"  root           {report.root}")
    print(f"  folder_id      {report.folder_id}")
    print(f"  files          {len(report.files)}")
    print(f"  downloaded     {report.downloaded}")
    print(f"  cache hits     {report.cache_hits}")
    print(f"  bytes          {report.bytes_downloaded:,}")
    print(f"  network calls  {report.network_calls}")
    if report.truncated_folders:
        print("  TRUNCATED      these folders hit the public listing cap (G17):")
        for name in report.truncated_folders:
            print(f"                   {name}")
        print("                   A full sync of these requires OAuth (--oauth).")
    return 0


def _cmd_source(args: argparse.Namespace) -> int:
    """Resolve a source, including gdrive:// paths, and report where it lands."""
    from .drive import DriveError
    from .source import resolve_source

    try:
        resolved = resolve_source(args.source, no_sync=args.no_sync)
    except (DriveError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"input": args.source, "resolved": str(resolved)}, indent=2))
        return 0

    print(f"source   {args.source or '(default)'}")
    print(f"resolved {resolved}")
    if isinstance(resolved, int):
        print("kind     camera index")
    else:
        from .source import is_still, is_video
        p = Path(resolved)
        if p.is_dir():
            n = sum(1 for x in p.iterdir() if x.is_file())
            print(f"kind     directory ({n} files)")
        elif is_video(p):
            print("kind     video")
        elif is_still(p):
            print("kind     still — must be synthesised before tracking (D8/R9)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracking_using_drive",
        description="Track space debris using the Google Drive-hosted dataset.",
        epilog=ROADMAP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_status = sub.add_parser("status", help="show resolved paths, device and environment")
    p_status.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p_status.set_defaults(func=_cmd_status)

    p_verify = sub.add_parser("verify", help="assert no library path escapes the package (R0)")
    p_verify.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p_verify.set_defaults(func=_cmd_verify)

    p_sync = sub.add_parser(
        "sync",
        help="selectively sync a slice of the shared Drive folder (anonymous, no credentials)",
    )
    p_sync.add_argument(
        "path", nargs="*",
        help="folder-relative path(s) to sync; default is the tracking allowlist",
    )
    p_sync.add_argument(
        "--folder-id", default=None,
        help="Drive folder id (default: the shared Capstone folder)",
    )
    p_sync.add_argument(
        "--cache-dir", default=None,
        help="destination root (default: tracking_using_drive/_data/gdrive_cache)",
    )
    p_sync.add_argument(
        "--no-sync", action="store_true",
        help="resolve from the warm cache only; error if it cannot (FR-2.4/R10)",
    )
    p_sync.add_argument(
        "--workers", type=int, default=4,
        help="concurrent downloads (default 4; the anonymous endpoint rate-limits)",
    )
    p_sync.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p_sync.set_defaults(func=_cmd_sync)

    p_source = sub.add_parser(
        "source",
        help="resolve a source (local path, camera index, or gdrive://subpath)",
    )
    p_source.add_argument(
        "source", nargs="?", default=None,
        help="e.g. gdrive://yolo_dataset/images/val, /path/to/dir, or 0 for a camera",
    )
    p_source.add_argument(
        "--no-sync", action="store_true",
        help="resolve gdrive:// from the warm cache only",
    )
    p_source.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p_source.set_defaults(func=_cmd_source)

    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
