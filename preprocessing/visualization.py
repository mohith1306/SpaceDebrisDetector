"""Visualization & channel-analysis utilities for the 3-channel pipeline.

Produces, for each sample image, a 6-panel comparison:

    1. original satellite image        4. CLAHE channel (ch2)
    2. grayscale                       5. background residual (ch3)
    3. percentile-normalized (ch1)     6. final 3-channel image

and computes per-channel pixel distributions plus pairwise identity /
correlation checks so it can be proven the three channels are NOT accidental
duplicates of each other.

Matplotlib runs headless (Agg) so everything works on JarvisLabs VMs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after Agg backend is set)

from .physics_pipeline import (  # noqa: E402
    DEFAULT_PIPELINE_CONFIG,
    PhysicsPipelineConfig,
    background_residual_channel,
    clahe_channel,
    percentile_normalize,
    to_grayscale,
)

# Deterministic pixel cap for correlation math on huge images
CORR_SAMPLE_PIXELS = 100_000


def _flatten_sample(channel: np.ndarray) -> np.ndarray:
    flat = channel.reshape(-1).astype(np.float64)
    if flat.size > CORR_SAMPLE_PIXELS:
        # deterministic stride subsample (no RNG -> reproducible reports)
        step = flat.size // CORR_SAMPLE_PIXELS
        flat = flat[::step][:CORR_SAMPLE_PIXELS]
    return flat


def channel_statistics(processed: np.ndarray) -> dict:
    """Distribution stats + pairwise identity/correlation of the 3 channels."""
    if processed.ndim != 3 or processed.shape[2] != 3:
        raise ValueError(f"expected (H,W,3), got {processed.shape}")

    channels = [processed[:, :, i] for i in range(3)]
    names = ["ch1_percentile", "ch2_clahe", "ch3_residual"]

    distributions = {}
    for name, ch in zip(names, channels):
        distributions[name] = {
            "mean": round(float(ch.mean()), 3),
            "std": round(float(ch.std()), 3),
            "min": int(ch.min()),
            "max": int(ch.max()),
            "p1": round(float(np.percentile(ch, 1)), 2),
            "p50": round(float(np.percentile(ch, 50)), 2),
            "p99": round(float(np.percentile(ch, 99)), 2),
        }

    pairs = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = channels[i], channels[j]
        key = f"{names[i]}__vs__{names[j]}"
        exact_equal_frac = float(np.mean(a == b))
        fa, fb = _flatten_sample(a), _flatten_sample(b)
        if fa.std() > 0 and fb.std() > 0:
            corr = float(np.corrcoef(fa, fb)[0, 1])
        else:
            corr = float("nan")
        pairs[key] = {
            "exact_equal_fraction": round(exact_equal_frac, 6),
            "pearson_correlation": round(corr, 6),
            # "identical" means byte-for-byte equal, NOT merely correlated:
            # the channels are derived from the same source, so a moderate
            # positive correlation is expected and healthy.
            "identical": bool(exact_equal_frac > 0.999),
        }

    return {"distributions": distributions, "channel_pairs": pairs}


def aggregate_channel_report(sample_stats: Sequence[dict]) -> dict:
    """Aggregate per-image channel statistics over many samples."""
    if not sample_stats:
        return {"samples": 0}
    identical_pairs = 0
    max_equal_frac: dict[str, float] = {}
    corrs: dict[str, list[float]] = {}
    for s in sample_stats:
        for key, val in s["channel_pairs"].items():
            if val["identical"]:
                identical_pairs += 1
            max_equal_frac[key] = max(
                max_equal_frac.get(key, 0.0), val["exact_equal_fraction"]
            )
            if not np.isnan(val["pearson_correlation"]):
                corrs.setdefault(key, []).append(val["pearson_correlation"])
    return {
        "samples": len(sample_stats),
        "identical_channel_pairs_total": identical_pairs,
        "max_exact_equal_fraction_per_pair": {
            k: round(v, 6) for k, v in max_equal_frac.items()
        },
        "mean_pearson_correlation_per_pair": {
            k: round(float(np.mean(v)), 4) for k, v in corrs.items()
        },
        "verdict": (
            "PASS: no channel pair is byte-identical"
            if identical_pairs == 0
            else "FAIL: at least one channel pair is byte-identical"
        ),
    }


def _source_index(source_root: Path) -> dict[str, Path]:
    """stem -> original source image (searched across both source splits)."""
    index: dict[str, Path] = {}
    images_dir = Path(source_root) / "images"
    if not images_dir.is_dir():
        return index
    for split in ("train", "val"):
        d = images_dir / split
        if d.is_dir():
            for p in d.iterdir():
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    index[p.stem] = p
    if not index:
        for p in images_dir.iterdir():
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                index[p.stem] = p
    return index


def visualize_samples(
    image_paths: Sequence[str | Path],
    out_dir: str | Path,
    pipeline_cfg: PhysicsPipelineConfig = DEFAULT_PIPELINE_CONFIG,
    source_root: str | Path | None = None,
    prefix: str = "sample",
) -> dict:
    """Render 6-panel comparisons for the PROCESSED images; return stats.

    Panels 1-5 (original / grayscale / Ch1 / Ch2 / Ch3) are derived from the
    ORIGINAL source image found via `source_root` (matched by filename stem);
    panel 6 shows the actual stored processed file. Channel statistics are
    computed on the stored processed file (what training will consume).

    Writes <out_dir>/<prefix>_NN.png for each sample plus
    <out_dir>/channel_statistics.json.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_index = _source_index(Path(source_root)) if source_root else {}

    per_sample_stats: list[dict] = []
    written: list[str] = []
    source_missing: list[str] = []

    for idx, path in enumerate(image_paths, start=1):
        path = Path(path)
        processed = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if processed is None:
            continue
        if processed.ndim == 2 or processed.shape[2] == 1:
            # grayscale fallback (should not happen for this experiment)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        if processed.shape[2] == 4:
            processed = processed[:, :, :3]

        stats = channel_statistics(processed)
        stats["image"] = path.name
        per_sample_stats.append(stats)

        # panels 1-5 come from the ORIGINAL image, panel 6 from the stored file
        orig_path = src_index.get(path.stem)
        if orig_path is not None:
            orig_bgr = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
        else:
            orig_bgr = None
            source_missing.append(path.name)
        input_rgb = (
            cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            if orig_bgr is not None
            else processed
        )

        gray = to_grayscale(input_rgb)
        ch1 = percentile_normalize(gray, pipeline_cfg)
        ch2 = clahe_channel(gray, pipeline_cfg)
        ch3 = background_residual_channel(ch1, pipeline_cfg)
        final = processed  # exactly what was written to disk

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(
            f"3-Channel Physics-Motivated Pipeline - {path.name}",
            fontsize=14,
        )
        panels = [
            (input_rgb, "1. Original source image (RGB)", "rgb"),
            (gray, "2. Grayscale", "gray"),
            (ch1, "3. Ch1: percentile-normalized", "gray"),
            (ch2, "4. Ch2: CLAHE contrast", "gray"),
            (ch3, "5. Ch3: background residual (128=0)", "gray"),
            (final, "6. Stored 3-channel output\n(engineered, not RGB colour)", None),
        ]
        for ax, (data, title, mode) in zip(axes.flat, panels):
            if mode == "rgb":
                ax.imshow(data)
            elif mode == "gray":
                ax.imshow(data, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(data)  # false-colour composite of the 3 channels
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = out_dir / f"{prefix}_{idx:02d}.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        written.append(str(out_path))

    report = aggregate_channel_report(per_sample_stats)
    report["images"] = written
    if source_missing:
        report["source_images_missing"] = source_missing
    report_path = out_dir / "channel_statistics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
