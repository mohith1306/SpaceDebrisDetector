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
  synth         still -> synthetic sequence + ground truth   [Phase 3 - now]
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


def _synth_one(args: argparse.Namespace, image: Path, out_dir: Path) -> "object":
    """Synthesise a single sequence; returns its ``GroundTruth``.

    Separated from the argument plumbing so batch mode (``--count``) can call
    it repeatedly without re-resolving the source each time.
    """
    from .sequence import Motion, Occlusion, find_label_for, synthesise_sequence

    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    if args.label is None and labels_dir is None and image.parent.parent.name == "images":
        # Convention: .../images/<split>/<img>.jpg -> .../labels/<split>/
        guess = image.parent.parent.parent / "labels" / image.parent.name
        if guess.is_dir():
            labels_dir = guess
    if args.label:
        label = Path(args.label)
    elif labels_dir is not None:
        label = find_label_for(image, labels_dir)
    else:
        label = None

    occlusion = (
        Occlusion(args.occlude_start, args.occlude_end)
        if args.occlude_end is not None else None
    )

    _, gt = synthesise_sequence(
        image,
        out_dir,
        frames=args.frames,
        motion=Motion(args.mode, args.dx, args.dy, args.ddx, args.ddy),
        fps=args.fps,
        fov_deg=args.fov,
        occlusion=occlusion,
        label_path=label,
        labels_dir=labels_dir,
        write_video=args.video,
    )
    gt.save(out_dir / "ground_truth.json")
    return gt


def _cmd_synth(args: argparse.Namespace) -> int:
    """Synthesise an ordered sequence plus ground truth from a Drive still (D8).

    The dataset holds independent stills, so tracking cannot be measured on it
    directly. This renders a still into a known motion and records the exact
    answer, which is what R9 and R11 are scored against.

    ``--count N`` renders N sequences from N distinct stills into subdirectories
    of ``--out``. D8 says success is a *measured error bar*, and an error bar
    needs more than one sample — a single sequence yields one number with no
    spread, which cannot distinguish a real 8% from a lucky 8%.
    """
    from .drive import DriveError
    from .source import resolve_source

    try:
        resolved = resolve_source(args.image, no_sync=args.no_sync)
    except (DriveError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    root = Path(resolved)
    if root.is_dir():
        from .source import is_still

        stills = sorted(p for p in root.iterdir() if p.is_file() and is_still(p))
        if not stills:
            print(f"ERROR: no still images under {root}", file=sys.stderr)
            return 2
    else:
        stills = [root]

    if args.count < 1:
        print(f"ERROR: --count must be >= 1, got {args.count}", file=sys.stderr)
        return 2
    if args.count > len(stills):
        print(
            f"ERROR: --count {args.count} but only {len(stills)} still(s) available "
            f"under {root}",
            file=sys.stderr,
        )
        return 2
    if args.count > 1 and args.label:
        print(
            "ERROR: --label pins one label to every sequence; omit it when --count > 1",
            file=sys.stderr,
        )
        return 2

    out_root = Path(args.out)
    batch = args.count > 1
    runs = []
    for i, still in enumerate(stills[: args.count]):
        target = out_root / f"seq_{i:03d}" if batch else out_root
        try:
            gt = _synth_one(args, still, target)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"ERROR: {still.name}: {exc}", file=sys.stderr)
            return 2
        runs.append((still, target, gt))

    # Batch mode writes an index so downstream scoring can iterate the spread
    # without globbing; single mode already has its own ground_truth.json.
    if batch:
        index = {
            "synthetic": True,   # R7
            "count": len(runs),
            "fps": args.fps,
            "frames": args.frames,
            "fov_deg": args.fov,
            "sequences": [
                {
                    "source_image": str(img),
                    "ground_truth": str(tgt / "ground_truth.json"),
                    "frame_dir": str(tgt),
                    "angular_rate_deg_s": list(gt.angular_rate_deg_s),
                }
                for img, tgt, gt in runs
            ],
        }
        (out_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    if args.json:
        if batch:
            print(json.dumps({
                "index": str(out_root / "index.json"),
                "count": len(runs),
                "synthetic": True,
                "sequences": [
                    {"frame_dir": str(t), "angular_rate_deg_s": list(g.angular_rate_deg_s)}
                    for _, t, g in runs
                ],
            }, indent=2))
        else:
            _, out_dir, gt = runs[0]
            print(json.dumps({
                "frame_dir": str(out_dir),
                "ground_truth": str(out_dir / "ground_truth.json"),
                "frames": gt.frames,
                "synthetic": gt.synthetic,           # R7
                "angular_rate_deg_s": gt.angular_rate_deg_s,
                "occluded_frames": gt.occluded_frames,
                "source_image": gt.source_image,
            }, indent=2))
        return 0

    if batch:
        rates = [g.angular_rate_deg_s for _, _, g in runs]
        print(f"Synthetic batch: {len(runs)} sequences "
              f"({runs[0][2].width}x{runs[0][2].height}, "
              f"{runs[0][2].frames} frames @ {args.fps} fps)")
        print(f"  out root        {out_root}")
        print(f"  index           {out_root / 'index.json'}")
        print(f"  motion          {args.mode} dx={args.dx} dy={args.dy}")
        print(f"  angular rate    x: {min(r[0] for r in rates):.5f} .. {max(r[0] for r in rates):.5f}  "
              f"y: {min(r[1] for r in rates):.5f} .. {max(r[1] for r in rates):.5f} deg/s")
        print(f"  synthetic       True  <- validation instrument, never a measured orbit (R7)")
        return 0

    _, out_dir, gt = runs[0]
    print(f"Synthetic sequence ({gt.width}x{gt.height}, {gt.frames} frames @ {gt.fps} fps)")
    print(f"  source image    {gt.source_image}")
    print(f"  label           {gt.label_path or '(none — image centre used)'}")
    print(f"  frames          {out_dir}")
    print(f"  ground truth    {out_dir / 'ground_truth.json'}")
    print(f"  motion          {args.mode} dx={args.dx} dy={args.dy}")
    print(f"  angular rate    [{gt.angular_rate_deg_s[0]:.5f}, {gt.angular_rate_deg_s[1]:.5f}] deg/s")
    if args.fov:
        exact = gt.angular_rate_deg_s[1]
        small = (args.dy / gt.height) * (args.fov * gt.height / gt.width) * args.fps
        if abs(small - exact) > 1e-12:
            print(f"                  (small-angle approx would be {small:.5f} deg/s on y)")
    if gt.occluded_frames:
        print(f"  occlusion       frames {args.occlude_start}-{args.occlude_end - 1} "
              f"({len(gt.occluded_frames)} erased)")
    print(f"  synthetic       {gt.synthetic}  <- a validation instrument, never a measured orbit (R7)")
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

    p_synth = sub.add_parser(
        "synth",
        help="turn a Drive still into an ordered sequence with exact ground truth (D8)",
    )
    p_synth.add_argument(
        "image", nargs="?", default=None,
        help="still image, a directory of stills, or gdrive://subpath (default: first cached still)",
    )
    p_synth.add_argument("--out", required=True, help="output directory for frames + ground_truth.json")
    p_synth.add_argument("--frames", type=int, default=30, help="number of frames (default 30)")
    p_synth.add_argument("--fps", type=float, default=5.0, help="frame rate used for the deg/s truth")
    p_synth.add_argument(
        "--fov", type=float, default=60.0,
        help="horizontal FOV in degrees (must match the tracker's; exact arctan form is used)",
    )
    p_synth.add_argument("--mode", choices=("linear", "accel", "orbit_arc"), default="linear")
    p_synth.add_argument("--dx", type=float, default=6.0, help="pixels/frame in x")
    p_synth.add_argument("--dy", type=float, default=-3.0, help="pixels/frame in y")
    p_synth.add_argument("--ddx", type=float, default=0.0, help="x acceleration (mode=accel)")
    p_synth.add_argument("--ddy", type=float, default=0.0, help="y acceleration (mode=accel)")
    p_synth.add_argument(
        "--occlude-start", type=int, default=0,
        help="first erased frame; requires --occlude-end and a label (D12)",
    )
    p_synth.add_argument(
        "--occlude-end", type=int, default=None,
        help="first frame after the gap, exclusive; omit for no occlusion",
    )
    p_synth.add_argument("--label", default=None, help="explicit YOLO label path")
    p_synth.add_argument("--labels-dir", default=None, help="directory searched for <stem>.txt")
    p_synth.add_argument(
        "--count", type=int, default=1,
        help="synthesise this many sequences from distinct stills into "
             "out/seq_NNN/ plus index.json (D8: an error bar needs a spread, "
             "not one sample)",
    )
    p_synth.add_argument("--video", action="store_true", help="also write sequence.mp4")
    p_synth.add_argument(
        "--no-sync", action="store_true",
        help="resolve gdrive:// from the warm cache only",
    )
    p_synth.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p_synth.set_defaults(func=_cmd_synth)

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
