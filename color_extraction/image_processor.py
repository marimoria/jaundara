"""
image_processor.py
==================
Handles everything for a single image:

  1. Load from disk.
  2. Crop the center 50 % (removes the NeoJaundice color-card border).
  3. Build a neonatal skin mask on the cropped region.
  4. Extract valid skin pixels.
  5. Compute 14 color features.

The result is a flat dict ready to be appended to a DataFrame row.

Zone mapping (NeoJaundice filename suffix convention):
  -1  →  head / forehead   (Kramer Zone 1)
  -2  →  face              (Kramer Zone 1)
  -3  →  chest / sternum   (Kramer Zone 2)
"""

import os
import logging
from pathlib import Path

import numpy as np
import cv2

from .skin_mask import (
    build_neonatal_skin_mask,
    extract_valid_skin_pixels_rgb,
    skin_coverage_fraction,
)
from .feature_extractor import compute_features_from_skin_pixels

LOG = logging.getLogger("jaundice_extractor")

# Warn when fewer than this fraction of pixels are classified as skin
_MIN_ACCEPTABLE_COVERAGE = 0.05


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _crop_center_half(image: np.ndarray) -> np.ndarray:
    """
    Return the central 40 % of the image (width and height).
    Tightened from 50 % to reliably avoid the NeoJaundice color-card border.
    """
    h, w = image.shape[:2]
    
    # 30% to 70% covers the central 40% of the image
    y0 = int(h * 0.3)
    y1 = int(h * 0.7)
    x0 = int(w * 0.3)
    x1 = int(w * 0.7)
    
    return image[y0:y1, x0:x1]


def _load_bgr_image(image_path: str) -> np.ndarray:
    """Load an image from disk in BGR format, raising on failure."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    return img


def _patient_id_from_filename(filename: str) -> str:
    """
    Derive the patient identifier from the image filename.
    Assumes the convention '<patient_id>-<zone>.jpg' (e.g., '0003-1.jpg').
    """
    return Path(filename).stem.split("-")[0]


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def process_single_image(
    image_path: str,
    *,
    debug: bool = False,
    debug_dir: str = "debug",
) -> dict:
    """
    Full pipeline for one image: load → crop → mask → features.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.
    debug : bool
        When True, a diagnostic figure is saved to debug_dir.
    debug_dir : str
        Root folder for debug output sub-directories.

    Returns
    -------
    dict with keys:
        patient_id, image_idx,
        and all 14 feature names from feature_extractor.FEATURE_NAMES.
    """
    filename   = os.path.basename(image_path)
    patient_id = _patient_id_from_filename(filename)

    LOG.info(f"  Processing: {filename}")

    # ── Load ─────────────────────────────────────────────────
    raw_bgr = _load_bgr_image(image_path)
    h, w    = raw_bgr.shape[:2]
    LOG.debug(f"    Original size: {w}×{h} px")

    # ── Crop calibration card ─────────────────────────────────
    cropped_bgr = _crop_center_half(raw_bgr)
    ch, cw      = cropped_bgr.shape[:2]
    LOG.debug(f"    Cropped size:  {cw}×{ch} px")

    # ── Skin mask ─────────────────────────────────────────────
    skin_mask  = build_neonatal_skin_mask(cropped_bgr)
    coverage   = skin_coverage_fraction(skin_mask)
    n_skin     = int((skin_mask > 0).sum())
    LOG.debug(f"    Skin coverage: {n_skin:,} px ({coverage * 100:.1f}%)")

    if coverage < _MIN_ACCEPTABLE_COVERAGE:
        LOG.warning(
            f"    LOW COVERAGE ({coverage * 100:.1f}%) in {filename} — "
            "skin mask found very little skin. Features will be NaN. "
            "Check image quality or crop parameters."
        )

    # ── Valid pixels ──────────────────────────────────────────
    skin_pixels_rgb = extract_valid_skin_pixels_rgb(cropped_bgr, skin_mask)
    LOG.debug(f"    Valid pixel count: {len(skin_pixels_rgb):,}")

    # ── Feature computation ───────────────────────────────────
    features = compute_features_from_skin_pixels(skin_pixels_rgb)
    LOG.debug(
        f"    R_mean={features.get('R_mean'):.2f}  "
        f"Cr_mean={features.get('Cr_mean'):.2f}  "
        f"Lab_b_mean={features.get('Lab_b_mean'):.2f}  "
        f"H_mean={features.get('H_mean'):.2f}"
    )

    # ── Optional debug figure ─────────────────────────────────
    if debug:
        from .debug_visualizer import save_debug_figure
        image_debug_dir = os.path.join(debug_dir, Path(image_path).stem)
        save_debug_figure(
            image_path       = image_path,
            cropped_bgr      = cropped_bgr,
            skin_mask        = skin_mask,
            skin_pixels_rgb  = skin_pixels_rgb,
            features         = features,
            output_dir       = image_debug_dir,
        )

    return {"patient_id": patient_id, "image_idx": filename, **features}