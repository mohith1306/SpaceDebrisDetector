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
  sync-api      big-folder sync via Drive API, no 5,500 cap  [Phase 5 - now]
  source        resolve a source (incl. gdrive://)           [Phase 2 - now]
  synth         still -> synthetic sequence + ground truth   [Phase 3 - now]
  stream        detect + track, with missed-detection fix    [Phase 4 - now]
  orbit         offline orbit report from a track JSONL      [Phase 4 - now]
  track         detect + track (library stage, unmodified)   [Phase 4 - now]
  validate      full chain end-to-end + R7/R9 audit          [Phase 6 - now]
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
        print("                   For the full listing use: sync-api (Drive API).")
    return 0


def _cmd_sync_api(args: argparse.Namespace) -> int:
    """Sync folders the anonymous listing truncates (G17), via the Drive API.

    Phase 5's requirement: ``train/`` holds 66,000 flat images, so the
    anonymous endpoint can only ever show 5,500 of them. The official API
    paginates without a cap; downloads stay on the anonymous endpoint by
    default so the API quota is spent only on listings.
    """
    from .drive import (
        DEFAULT_FOLDER_ID,
        GDRIVE_CACHE_DIR,
        DriveError,
        TRACKING_ALLOWLIST,
    )
    from .drive_api import sync_paths

    paths = tuple(args.path) if args.path else TRACKING_ALLOWLIST
    try:
        report = sync_paths(
            paths,
            folder_id=args.folder_id or DEFAULT_FOLDER_ID,
            dest_root=Path(args.cache_dir) if args.cache_dir
            else Path(GDRIVE_CACHE_DIR),
            api_key=args.api_key,
            workers=args.workers,
            download_mode=args.download_mode,
        )
    except DriveError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 1 if report["errors"] else 0

    print(f"Drive sync via API ({report['download_mode']} downloads)")
    print(f"  root           {report['folder_id']}")
    print(f"  paths          {', '.join(report['paths'])}")
    print(f"  files listed   {report['n_files']}")
    print(f"  downloaded     {report['downloaded']}")
    print(f"  already there  {report['skipped']}")
    print(f"  bytes          {report['bytes']:,}")
    print(f"  api list calls {report['api_list_calls']}")
    if report["errors"]:
        print(f"  ERRORS         {report['errors']} file(s) failed:")
        for sample in report["error_samples"]:
            print(f"                   {sample}")
        print("                   Rerun the same command — existing files "
              "are skipped, so it only fetches what is missing.")
        return 1
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


def _cmd_stream(args: argparse.Namespace) -> int:
    """Run detect -> track -> classify -> orbit, or the D13 injected path.

    With ``--ground-truth`` no weights are loaded: fps, FOV and the detection
    stream all come from a Phase 3 ``ground_truth.json``, so the tracking layer
    is exercised on its own (D13). Without it the real detector runs.
    """
    from .drive import DriveError
    from .sequence import GroundTruth, detections_from_ground_truth
    from .source import resolve_source

    injected = args.ground_truth is not None

    gt = None
    fps = args.fps
    fov_deg = args.fov
    detections = None
    if injected:
        try:
            gt = GroundTruth.load(Path(args.ground_truth))
        except (OSError, ValueError, KeyError) as exc:
            print(f"ERROR: cannot read ground truth: {exc}", file=sys.stderr)
            return 2
        # Truth and estimate must share fps/FOV, or R9 would be measuring an
        # argument mismatch rather than tracking error.
        fps = gt.fps
        fov_deg = gt.fov_deg
        if args.source is None:
            args.source = gt.frame_dir
        if args.source is None:
            print("ERROR: ground truth has no frame_dir; pass --source",
                  file=sys.stderr)
            return 2
        detections = detections_from_ground_truth(gt)

    try:
        resolved = resolve_source(args.source, no_sync=args.no_sync)
    except (DriveError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if isinstance(resolved, int):
        if injected:
            print("ERROR: injected detections cannot drive a camera index",
                  file=sys.stderr)
            return 2
        print(f"ERROR: camera sources are not supported yet: {resolved}",
              file=sys.stderr)
        return 2

    from .stream import run_stream

    try:
        summary = run_stream(
            source=str(resolved),
            detections=detections,
            fps=fps,
            fov_deg=fov_deg,
            range_km=args.range_km,
            orbit=not args.no_orbit,
            out_jsonl=args.out,
            orbit_report_path=args.orbit_report,
            annotate_path=args.annotate,
            max_frames=args.max_frames,
            max_missed=args.max_missed,
            model_path=args.model,
            conf=args.conf,
            tracker=None if args.no_tracker else args.tracker,
            classifier_path=args.classifier,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Stream ({summary['detection_source']}"
          f"{', SYNTHETIC' if summary['synthetic'] else ''})")
    print(f"  source          {summary['source']}")
    print(f"  frames          {summary['frames']}")
    print(f"  fps assumed     {summary['fps_assumed']}")
    print(f"  detections      {summary['detections']}")
    print(f"  tracks          {summary['num_tracks']} "
          f"(expired {summary['expired_tracks']})")
    print(f"  missed frames   {summary['missed_frames_total']}")
    if summary["tracks"]:
        for t in summary["tracks"]:
            print(f"    #{t['track_id']}: obs={t['observations']} "
                  f"missed={t['n_missed']} lifespan={t['lifespan_s']}s "
                  f"expired={t['expired']}")
    print(f"  orbit tracks    {summary['orbit_tracks']}")
    if summary["orbit_report"]:
        print(f"  orbit report    {summary['orbit_report']}")
    if summary["jsonl"]:
        print(f"  jsonl           {summary['jsonl']}")
    if summary["synthetic"]:
        print("  synthetic       True  <- validation run, not a measured "
              "debris orbit (R7)")
    return 0


def _cmd_orbit(args: argparse.Namespace) -> int:
    """Offline orbit report from a track JSONL (library stage, unmodified)."""
    from space_debris.utils.orbit import orbit_from_jsonl

    try:
        result = orbit_from_jsonl(
            args.jsonl, fov_deg=args.fov, range_km=args.range_km,
            out_json=args.out,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"Orbit from {args.jsonl}")
    print(f"  frames      {result['frames']}")
    print(f"  image size  {result['image_size']}")
    print(f"  tracks      {result['tracks']}")
    for entry in result["report"]:
        traj = entry["trajectory"]
        rate = traj.get("angular_rate_deg_s")
        print(f"    #{entry['track_id']}: n={traj.get('n_points')} "
              f"rate={rate} missing={traj.get('missing')}")
    if args.out:
        print(f"  written     {args.out}")
    return 0


def _cmd_track(args: argparse.Namespace) -> int:
    """Run the library's track stage unmodified (R0: read, never write).

    Exposed rather than reimplemented so the JSONL it produces can be compared
    against ours — ``orbit`` then reads either file unchanged.
    """
    from space_debris.models.track import track as library_track

    try:
        summary = library_track(
            model_path=args.model, source=args.source, conf=args.conf,
            tracker=args.tracker, max_frames=args.max_frames, fps=args.fps,
            out_jsonl=args.out,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Track ({summary['source']})")
    for key in ("frames", "fps_assumed", "num_tracks", "tracker", "jsonl"):
        print(f"  {key:14s} {summary.get(key)}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Run the full Phase 6 chain and audit the result (D13, no weights).

    Drive -> sync -> synthesise -> inject -> associate -> Kalman -> orbit,
    then compare against the degenerate baseline and sweep the FOV/range
    assumptions. Writes ``validation_report.json`` flagged ``synthetic: true``
    (R7) and exits non-zero if the acceptance criteria fail, so it can gate a
    commit rather than merely print numbers.
    """
    from .drive import DriveError
    from .source import resolve_source
    from .validate import audit_synthetic_flags, run_validation

    if args.image is None:
        # Warm the cache first so the chain has a real Drive still to use.
        try:
            resolve_source(None, no_sync=False)
        except (DriveError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    try:
        report = run_validation(
            args.out, frames=args.frames, fps=args.fps,
            occlusion=(args.occlude_start, args.occlude_end),
            max_missed=args.max_missed, still=args.image,
            no_sync=args.no_sync,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    violations = audit_synthetic_flags(report)
    rate = report["rate_recovery"]
    continuity = report["continuity"]
    sweep = report["sweep"]

    checks = [
        ("R7 synthetic flags", not violations),
        ("R9 rate within 10%", rate["within_tolerance"]),
        ("continuity improved", continuity["improved"]),
        ("missing range reason recorded", sweep["all_missing_range_reason"]),
        ("speed null without range", sweep["all_speed_null_without_range"]),
        ("speed linear in range", sweep["speed_linear_in_range"]),
        ("rate linear in fov", sweep["rate_linear_in_fov"]),
    ]
    failed = [name for name, ok in checks if not ok]

    if args.json:
        payload = dict(report)
        payload["checks"] = [{"name": n, "ok": ok} for n, ok in checks]
        payload["passed"] = not failed
        print(json.dumps(payload, indent=2))
        return 1 if failed else 0

    print("Validation (SYNTHETIC — a validation instrument, never a measured "
          "debris orbit; R7)")
    print(f"  report          {report['artifacts'].get('report')}")
    print(f"  chain           {' -> '.join(report['chain'])}")
    print(f"  frames          {report['frames']} @ {report['fps']} fps "
          f"(occlusion {report['occlusion']})")
    print(f"  rate recovered  {rate['recovered_deg_s']} vs truth "
          f"{rate['truth_deg_s']}")
    print(f"  rate error      {rate['error_pct']}%  (tolerance "
          f"{rate['tolerance_pct']}%)")
    print("  continuity")
    print(f"    baseline      {continuity['baseline']}")
    print(f"    synthetic     {continuity['synthetic_sequence']}")
    print(f"    improved      {continuity['improved']}")
    print(f"  sweep           {len(sweep['rows'])} points "
          f"(speed linear in range: {sweep['speed_linear_in_range']}, "
          f"rate linear in fov: {sweep['rate_linear_in_fov']})")
    for entry in report["orbit"]:
        print(f"    #{entry['track_id']}: {entry['final_class']} "
              f"is_debris={entry['is_debris']} "
              f"obs={entry['observations']} missed={entry['n_missed']} "
              f"speed={entry['speed_km_s']} missing={entry['missing']}")
    if violations:
        print("  R7 violations:")
        for v in violations:
            print(f"    - {v}")
    print()
    print(f"  checks: {len(checks) - len(failed)}/{len(checks)} passed")
    for name in failed:
        print(f"    FAIL {name}")
    return 1 if failed else 0


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

    p_sync_api = sub.add_parser(
        "sync-api",
        help="sync big folders via the Drive API — no 5,500 listing cap (G17)",
    )
    p_sync_api.add_argument(
        "path", nargs="*",
        help="folder-relative path(s) to sync; default is the tracking allowlist",
    )
    p_sync_api.add_argument(
        "--api-key", default=None,
        help="Google Drive API key (or set GDRIVE_API_KEY); needed because "
             "the anonymous listing caps at 5,500 children per folder",
    )
    p_sync_api.add_argument(
        "--folder-id", default=None,
        help="Drive folder id (default: the shared Capstone folder)",
    )
    p_sync_api.add_argument(
        "--cache-dir", default=None,
        help="destination root (default: tracking_using_drive/_data/gdrive_cache)",
    )
    p_sync_api.add_argument(
        "--workers", type=int, default=4,
        help="concurrent downloads (default 4; the anonymous endpoint rate-limits)",
    )
    p_sync_api.add_argument(
        "--download-mode", choices=("anonymous", "api"), default="anonymous",
        help="anonymous = public endpoint, no API quota (default); "
             "api = alt=media, consumes quota but survives IP rate limits",
    )
    p_sync_api.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p_sync_api.set_defaults(func=_cmd_sync_api)

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

    p_stream = sub.add_parser(
        "stream",
        help="run detect -> track -> classify -> orbit, or the D13 injected path",
    )
    p_stream.add_argument(
        "source", nargs="?", default=None,
        help="frames directory, video, or gdrive://subpath "
             "(default: the frame dir named by --ground-truth)",
    )
    p_stream.add_argument(
        "--ground-truth", default=None,
        help="Phase 3 ground_truth.json: use its fps/FOV and inject its "
             "detections instead of a detector (D13 — needs no weights)",
    )
    p_stream.add_argument("--out", default=None, help="per-frame JSONL output")
    p_stream.add_argument("--orbit-report", default=None, help="per-track orbit JSON")
    p_stream.add_argument("--annotate", default=None, help="annotated video output")
    p_stream.add_argument("--max-frames", type=int, default=None)
    p_stream.add_argument(
        "--max-missed", type=int, default=30,
        help="consecutive absent frames before a track expires (default 30)",
    )
    p_stream.add_argument("--fov", type=float, default=60.0, help="horizontal FOV (deg)")
    p_stream.add_argument("--range-km", type=float, default=None,
                          help="observer-to-object range; omit to leave speed null")
    p_stream.add_argument("--fps", type=float, default=None,
                          help="frame rate (required for correct deg/s; "
                               "--ground-truth supplies it)")
    p_stream.add_argument("--model", default=None, help="weights path")
    p_stream.add_argument("--conf", type=float, default=0.25)
    p_stream.add_argument("--tracker", default="bytetrack.yaml")
    p_stream.add_argument("--no-tracker", action="store_true",
                          help="run predict() instead of track() (no association)")
    p_stream.add_argument("--classifier", default=None,
                          help="second-stage classifier weights")
    p_stream.add_argument("--no-orbit", action="store_true")
    p_stream.add_argument("--no-sync", action="store_true",
                          help="resolve gdrive:// from the warm cache only")
    p_stream.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p_stream.set_defaults(func=_cmd_stream)

    p_orbit = sub.add_parser(
        "orbit",
        help="offline orbit report from an existing track JSONL",
    )
    p_orbit.add_argument("jsonl", help="JSONL produced by `stream --out`")
    p_orbit.add_argument("--fov", type=float, default=60.0)
    p_orbit.add_argument("--range-km", type=float, default=None)
    p_orbit.add_argument("--out", default=None, help="orbit JSON to write")
    p_orbit.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p_orbit.set_defaults(func=_cmd_orbit)

    p_track = sub.add_parser(
        "track",
        help="library track stage (unmodified space_debris.models.track)",
    )
    p_track.add_argument("source", nargs="?", default=None)
    p_track.add_argument("--model", default=None)
    p_track.add_argument("--conf", type=float, default=0.25)
    p_track.add_argument("--tracker", default="bytetrack.yaml")
    p_track.add_argument("--max-frames", type=int, default=None)
    p_track.add_argument("--fps", type=float, default=None)
    p_track.add_argument("--out", default=None, help="per-frame JSONL output")
    p_track.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p_track.set_defaults(func=_cmd_track)

    p_validate = sub.add_parser(
        "validate",
        help="Phase 6: run the full chain end-to-end and audit the result",
    )
    p_validate.add_argument(
        "--out", required=True,
        help="directory for frames, JSONL, orbit reports and the report",
    )
    p_validate.add_argument(
        "--image", default=None,
        help="source still (default: first cached Drive still; syncs if needed)",
    )
    p_validate.add_argument("--frames", type=int, default=30)
    p_validate.add_argument("--fps", type=float, default=5.0)
    p_validate.add_argument("--occlude-start", type=int, default=10)
    p_validate.add_argument("--occlude-end", type=int, default=20)
    p_validate.add_argument("--max-missed", type=int, default=30)
    p_validate.add_argument("--no-sync", action="store_true",
                            help="use the warm cache only, never the network")
    p_validate.add_argument("--json", action="store_true",
                            help="emit the full report as machine-readable JSON")
    p_validate.set_defaults(func=_cmd_validate)

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
