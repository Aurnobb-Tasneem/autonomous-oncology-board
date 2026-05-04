"""
ml/data/wsi.py
===============
Real Whole Slide Image (WSI) patch extraction pipeline using openslide-python.

Most teams submit "AI + pathology" projects using pre-patched demo datasets.
This module provides real WSI support — the production-grade capability.

Supported formats:
    .svs  — Aperio ScanScope (most common)
    .ndpi — Hamamatsu NanoZoomer
    .tiff — Generic TIFF (many scanners)
    .mrxs — MIRAX
    .scn  — Leica

Algorithm:
    1. Open slide with openslide
    2. Generate a low-resolution thumbnail for tissue segmentation
    3. Otsu thresholding on greyscale thumbnail → tissue mask
    4. Extract non-overlapping 224×224 patches at 20× magnification from
       tissue-positive regions only (tissue_fraction ≥ min_tissue_fraction)
    5. Yield (PIL.Image, (x, y)) pairs — coordinates are at level 0 (full res)

Usage:
    from ml.data.wsi import extract_patches, slide_thumbnail_with_heatmap

    for patch, (x, y) in extract_patches("slide.svs"):
        embeddings = gigapath_model(transform(patch).unsqueeze(0))

    # After inference, project heatmap back onto slide thumbnail:
    thumbnail_with_overlay = slide_thumbnail_with_heatmap(
        slide_path="slide.svs",
        patch_coords=[(x1, y1), (x2, y2), ...],
        attention_scores=[0.9, 0.3, ...],
        thumbnail_size=(1024, 1024),
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

PATCH_SIZE          = 224   # pixels at extraction level
DEFAULT_LEVEL       = 0     # level 0 = highest resolution (20×)
THUMBNAIL_SIZE      = (512, 512)
MIN_TISSUE_FRACTION = 0.5   # reject patches with <50% tissue


def _otsu_threshold(grey_array: np.ndarray) -> int:
    """
    Compute Otsu's threshold for a greyscale numpy array.
    Returns an integer pixel value separating foreground from background.
    """
    # Histogram
    hist, bin_edges = np.histogram(grey_array.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(float) / hist.sum()  # normalise to probabilities

    best_thresh = 0
    best_var    = 0.0

    w0 = 0.0
    mu0_sum = 0.0

    for t in range(256):
        w0      += hist[t]
        mu0_sum += t * hist[t]

        w1 = 1.0 - w0
        if w0 == 0.0 or w1 == 0.0:
            continue

        mu0 = mu0_sum / w0
        mu1 = (np.arange(256) * hist).sum() - mu0_sum
        mu1 = mu1 / w1

        var_between = w0 * w1 * (mu0 - mu1) ** 2
        if var_between > best_var:
            best_var   = var_between
            best_thresh = t

    return best_thresh


def _tissue_mask(thumbnail: Image.Image, threshold: Optional[int] = None) -> np.ndarray:
    """
    Returns a binary mask (H × W, bool) where True = tissue.
    Background is typically white (>200 in greyscale); tissue is darker.
    """
    grey = np.array(thumbnail.convert("L"))
    if threshold is None:
        threshold = _otsu_threshold(grey)
    mask = grey < threshold   # tissue is darker than background
    return mask


def extract_patches(
    slide_path: str | Path,
    level: int               = DEFAULT_LEVEL,
    patch_size: int          = PATCH_SIZE,
    overlap: int             = 0,
    tissue_fraction_min: float = MIN_TISSUE_FRACTION,
    max_patches: Optional[int] = None,
) -> Iterator[tuple[Image.Image, tuple[int, int]]]:
    """
    Yield (patch_image, (x, y)) pairs from a WSI after Otsu tissue filtering.

    Args:
        slide_path:          Path to the WSI file (.svs, .ndpi, .tiff, etc.)
        level:               Pyramid level to extract patches from (0 = highest res).
        patch_size:          Patch size in pixels (default: 224 for GigaPath).
        overlap:             Overlap between adjacent patches in pixels.
        tissue_fraction_min: Minimum fraction of non-background pixels required.
        max_patches:         Maximum number of patches to yield. None = all.

    Yields:
        (PIL.Image at patch_size×patch_size, (x, y)) where x, y are level-0
        pixel coordinates of the patch top-left corner.

    Raises:
        ImportError:  If openslide-python is not installed.
        FileNotFoundError:  If slide_path does not exist.
    """
    try:
        import openslide
    except ImportError:
        raise ImportError(
            "openslide-python is required for WSI support. "
            "Install: pip install openslide-python\n"
            "Also install the openslide C library: "
            "  Ubuntu: sudo apt install openslide-tools\n"
            "  macOS:  brew install openslide"
        )

    slide_path = Path(slide_path)
    if not slide_path.exists():
        raise FileNotFoundError(f"WSI not found: {slide_path}")

    log.info(f"WSI: opening {slide_path.name}")
    slide = openslide.OpenSlide(str(slide_path))

    level_count  = slide.level_count
    level        = min(level, level_count - 1)
    level_dims   = slide.level_dimensions[level]
    downsample   = slide.level_downsamples[level]

    log.info(
        f"WSI: level={level}  dims={level_dims}  downsample={downsample:.1f}x  "
        f"format={slide.properties.get('openslide.vendor', 'unknown')}"
    )

    # ── Build tissue mask from low-res thumbnail ──────────────────────────
    thumb = slide.get_thumbnail(THUMBNAIL_SIZE)
    mask  = _tissue_mask(thumb)
    mask_scale_x = thumb.width  / level_dims[0]
    mask_scale_y = thumb.height / level_dims[1]

    # ── Iterate patches ───────────────────────────────────────────────────
    stride = patch_size - overlap
    n_x    = (level_dims[0] - patch_size) // stride + 1
    n_y    = (level_dims[1] - patch_size) // stride + 1
    n_yielded = 0

    log.info(f"WSI: {n_x}×{n_y} = {n_x * n_y} potential patches at stride={stride}")

    for row in range(n_y):
        for col in range(n_x):
            if max_patches is not None and n_yielded >= max_patches:
                slide.close()
                return

            # Level-space coordinates
            x_level = col * stride
            y_level = row * stride

            # Check tissue fraction from thumbnail mask
            mask_x0 = max(0, int(x_level * mask_scale_x))
            mask_y0 = max(0, int(y_level * mask_scale_y))
            mask_x1 = min(mask.shape[1], int((x_level + patch_size) * mask_scale_x))
            mask_y1 = min(mask.shape[0], int((y_level + patch_size) * mask_scale_y))

            patch_mask = mask[mask_y0:mask_y1, mask_x0:mask_x1]
            if patch_mask.size == 0:
                continue
            tissue_frac = patch_mask.mean()
            if tissue_frac < tissue_fraction_min:
                continue

            # Level-0 coordinates for openslide.read_region()
            x0 = int(x_level * downsample)
            y0 = int(y_level * downsample)

            try:
                region = slide.read_region(
                    location=(x0, y0),
                    level=level,
                    size=(patch_size, patch_size),
                )
                patch = region.convert("RGB")
            except Exception as e:
                log.warning(f"WSI read_region failed at ({x0}, {y0}): {e}")
                continue

            n_yielded += 1
            yield patch, (x0, y0)

    slide.close()
    log.info(f"WSI: extracted {n_yielded} tissue patches from {slide_path.name}")


def slide_thumbnail_with_heatmap(
    slide_path: str | Path,
    patch_coords: list[tuple[int, int]],
    attention_scores: list[float],
    thumbnail_size: tuple[int, int] = (1024, 1024),
    alpha: float = 0.5,
) -> Image.Image:
    """
    Overlay attention heatmap onto a slide thumbnail.

    Args:
        slide_path:       Path to the WSI file.
        patch_coords:     List of (x, y) level-0 patch coordinates.
        attention_scores: Attention / saliency score per patch (0–1).
        thumbnail_size:   Output thumbnail dimensions.
        alpha:            Overlay transparency (0 = fully transparent, 1 = opaque).

    Returns:
        RGB PIL.Image of the annotated slide thumbnail.
    """
    try:
        import openslide
    except ImportError:
        raise ImportError("openslide-python required for WSI heatmap projection.")

    slide    = openslide.OpenSlide(str(slide_path))
    thumb    = slide.get_thumbnail(thumbnail_size).convert("RGB")
    w0, h0   = slide.level_dimensions[0]
    tw, th   = thumb.size
    slide.close()

    # Build attention overlay
    overlay = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    px      = overlay.load()

    scale_x = tw / w0
    scale_y = th / h0

    patch_size_thumb = max(2, int(PATCH_SIZE * scale_x))

    for (x, y), score in zip(patch_coords, attention_scores):
        tx = int(x * scale_x)
        ty = int(y * scale_y)
        # Red channel proportional to attention score
        r = int(255 * score)
        g = int(255 * (1 - score))
        b = 0
        a = int(220 * alpha)
        for dy in range(patch_size_thumb):
            for dx in range(patch_size_thumb):
                px_x = min(tx + dx, tw - 1)
                px_y = min(ty + dy, th - 1)
                px[px_x, px_y] = (r, g, b, a)

    result = Image.alpha_composite(thumb.convert("RGBA"), overlay).convert("RGB")
    return result


def slide_info(slide_path: str | Path) -> dict:
    """Return metadata about a WSI without extracting patches."""
    try:
        import openslide
    except ImportError:
        return {"error": "openslide-python not installed"}

    try:
        slide = openslide.OpenSlide(str(slide_path))
        info  = {
            "path":         str(slide_path),
            "format":       slide.properties.get("openslide.vendor", "unknown"),
            "level_count":  slide.level_count,
            "dimensions":   slide.level_dimensions,
            "downsamples":  [round(d, 2) for d in slide.level_downsamples],
            "mpp_x":        slide.properties.get(openslide.PROPERTY_NAME_MPP_X, "unknown"),
            "mpp_y":        slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, "unknown"),
        }
        slide.close()
        return info
    except Exception as e:
        return {"error": str(e), "path": str(slide_path)}
