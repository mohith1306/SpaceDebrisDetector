"""3-Channel Physics-Motivated Preprocessing Pipeline.

Each output channel is an ENGINEERED FEATURE MAP that pre-solves one physical
imaging problem of ground-based space-object imagery. The three channels are
NOT conventional R/G/B color channels and must never be interpreted as such:

    Channel 0 : percentile-normalized intensity   (robust global appearance)
    Channel 1 : CLAHE local contrast              (faint-object visibility)
    Channel 2 : background residual / high-pass   (compact targets, sky removed)

Why each channel exists
-----------------------
* Space imagery is dominated by sensor noise, hot pixels and a few extremely
  bright stars/satellites. Plain min-max normalization lets a single outlier
  pixel squash the useful dynamic range, so Channel 0 clips to the 1st/99th
  intensity percentiles before stretching to 0-255.
* Faint debris can differ from its local background by only a few DN. A global
  transform cannot reveal it, so Channel 1 applies Contrast Limited Adaptive
  Histogram Equalization (clipLimit=1.5, 8x8 tiles) after gentle Gaussian
  denoising (3x3, sigma=0 -> OpenCV derives sigma from the kernel size).
* Ground-based sky backgrounds carry large-scale gradients and illumination
  vignetting. Channel 2 estimates that background with a large-sigma Gaussian
  blur (sigmaX=25) of the percentile-normalized image, subtracts it, shifts the
  signed residual by +128 so zero-residual maps to mid-grey, clips to 0-255 and
  applies a small 3x3 blur to suppress isolated star/hot-pixel noise.

The denoising method and parameters used before CLAHE are therefore explicit:
Gaussian blur, kernel 3x3, sigmaX=0 (sigma computed by OpenCV from kernel).

Compatibility
-------------
Output is uint8, H x W x 3, contiguous, saved through PIL with JPEG quality 95
and 4:4:4 chroma subsampling (PNG stays lossless) so the independent channel
content is not destroyed by chroma subsampling. Consumable directly by
OpenCV / NumPy / Ultralytics YOLO.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PhysicsPipelineConfig:
    """Parameters of the 3-channel pipeline (documented, immutable)."""

    # Channel 1: percentile normalization
    percentiles: tuple[float, float] = (1.0, 99.0)

    # Denoising applied before CLAHE (documented explicitly)
    denoise_kernel: tuple[int, int] = (3, 3)
    denoise_sigma: float = 0.0  # 0 => OpenCV computes sigma from kernel size

    # Channel 2: CLAHE
    clahe_clip_limit: float = 1.5
    clahe_tile_grid: tuple[int, int] = (8, 8)

    # Channel 3: background residual
    background_sigma: float = 25.0
    residual_shift: int = 128
    residual_blur_kernel: tuple[int, int] = (3, 3)

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable dict of the parameters (for logs/summaries)."""
        d = asdict(self)
        d["percentiles"] = list(self.percentiles)
        d["denoise_kernel"] = list(self.denoise_kernel)
        d["clahe_tile_grid"] = list(self.clahe_tile_grid)
        d["residual_blur_kernel"] = list(self.residual_blur_kernel)
        return d


DEFAULT_PIPELINE_CONFIG = PhysicsPipelineConfig()


# ---------------------------------------------------------------------------
# Step 1: grayscale
# ---------------------------------------------------------------------------
def to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    """RGB (H,W,3) uint8 -> grayscale (H,W) uint8.

    Focuses on intensity/brightness/structure; colour carries no useful signal
    in space-object imagery and chroma noise is discarded here.
    """
    if img_rgb.ndim == 2:
        return img_rgb
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 1:
        return img_rgb[:, :, 0]
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------------------
# Step 2: Channel 1 - percentile normalization
# ---------------------------------------------------------------------------
def percentile_normalize(gray: np.ndarray, cfg: PhysicsPipelineConfig) -> np.ndarray:
    """Clip to [p_lo, p_hi] percentiles and rescale to 0-255.

    Robust against single bright/dark outliers (stars, hot pixels) that would
    otherwise dominate a plain min-max stretch.

    The degenerate case (percentile range == 0, i.e. a flat image) is handled:
    no stretch is mathematically possible, so the image passes through
    unchanged instead of dividing by zero.
    """
    lo_p, hi_p = cfg.percentiles
    lo, hi = np.percentile(gray, (lo_p, hi_p))
    lo = float(lo)
    hi = float(hi)

    if hi <= lo:
        # Flat / near-flat image: percentile range is zero -> pass through.
        return gray.astype(np.uint8, copy=True)

    clipped = np.clip(gray, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    return np.round(scaled).astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 3: Channel 2 - denoised CLAHE
# ---------------------------------------------------------------------------
def clahe_channel(gray: np.ndarray, cfg: PhysicsPipelineConfig) -> np.ndarray:
    """Denoise (Gaussian 3x3, sigma=0) then CLAHE(1.5, 8x8).

    The denoising step is explicit and documented: a gentle Gaussian blur with
    a 3x3 kernel and sigmaX=0 (OpenCV derives sigma from the kernel size).
    It suppresses sensor noise before CLAHE amplifies local contrast.
    """
    denoised = cv2.GaussianBlur(
        gray, cfg.denoise_kernel, cfg.denoise_sigma
    )
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=cfg.clahe_tile_grid,
    )
    return clahe.apply(denoised)


# ---------------------------------------------------------------------------
# Step 4: Channel 3 - background residual
# ---------------------------------------------------------------------------
def background_residual_channel(
    channel1: np.ndarray, cfg: PhysicsPipelineConfig
) -> np.ndarray:
    """Large-sigma background estimate subtracted from Channel 1.

    1. Gaussian blur (sigmaX=25) of the percentile-normalized image -> sky
       background estimate (small objects cannot survive such a large blur).
    2. residual = normalized - background  (signed).
    3. + shift (128) so zero residual maps to mid-grey.
    4. clip to 0-255, uint8.
    5. small 3x3 Gaussian blur to reduce isolated pixel noise.
    """
    background = cv2.GaussianBlur(
        channel1, (0, 0), sigmaX=cfg.background_sigma
    )
    residual = channel1.astype(np.int16) - background.astype(np.int16)
    residual = residual + cfg.residual_shift
    residual = np.clip(residual, 0, 255).astype(np.uint8)
    residual = cv2.GaussianBlur(residual, cfg.residual_blur_kernel, 0)
    return residual


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def preprocess_image_3ch(
    img_rgb: np.ndarray,
    cfg: PhysicsPipelineConfig = DEFAULT_PIPELINE_CONFIG,
) -> np.ndarray:
    """Apply the full 3-channel physics-motivated pipeline.

    Parameters
    ----------
    img_rgb : np.ndarray
        Input image as RGB uint8 (H,W,3). A single-channel input is also
        accepted and used as-is for the grayscale conversion.
    cfg : PhysicsPipelineConfig
        Pipeline parameters (defaults are the documented experiment values).

    Returns
    -------
    np.ndarray
        uint8, shape (H, W, 3), channel order:
            [:, :, 0] = percentile-normalized grayscale
            [:, :, 1] = CLAHE-enhanced grayscale
            [:, :, 2] = background residual (128 = zero-residual)

        These are three engineered image representations, NOT RGB colour
        channels. All three carry independent information.
    """
    if img_rgb is None:
        raise ValueError("preprocess_image_3ch received None image")

    gray = to_grayscale(img_rgb)

    channel1 = percentile_normalize(gray, cfg)
    channel2 = clahe_channel(gray, cfg)
    channel3 = background_residual_channel(channel1, cfg)

    processed = cv2.merge([channel1, channel2, channel3])

    # Guarantee contiguous uint8 HxWx3 for PIL / ultralytics consumption.
    processed = np.ascontiguousarray(processed, dtype=np.uint8)

    if processed.ndim != 3 or processed.shape[2] != 3:
        raise RuntimeError(
            f"pipeline produced shape {processed.shape}, expected (H, W, 3)"
        )
    return processed


# ---------------------------------------------------------------------------
# Saving (preserves the independent channels)
# ---------------------------------------------------------------------------
def save_processed_image(path, img_hwc: np.ndarray) -> None:
    """Save a 3-channel processed image without degrading channel content.

    JPEG is written with quality=95 and 4:4:4 chroma subsampling (subsampling=0)
    because default 4:2:0 subsampling would average the engineered channels
    together. PNG is written losslessly. Uses PIL (already a project
    dependency via convert_labels.py).
    """
    from PIL import Image

    if img_hwc.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {img_hwc.dtype}")
    if img_hwc.ndim != 3 or img_hwc.shape[2] != 3:
        raise ValueError(f"expected (H,W,3), got {img_hwc.shape}")

    pil = Image.fromarray(img_hwc, mode="RGB")
    suffix = str(path).lower().rsplit(".", 1)[-1]
    if suffix in ("jpg", "jpeg"):
        pil.save(path, format="JPEG", quality=95, subsampling=0)
    else:
        pil.save(path)  # png / anything else: lossless default
