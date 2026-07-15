"""
image_processor.py
Handles everything for a single image:

  1. Load from disk.
  2. Crop the center 40% (removes the NeoJaundice color-card border).
  3. Build a neonatal skin mask on the cropped region.
  4. Extract valid skin pixels.
  5. Compute 14 color features.

Zone mapping (NeoJaundice filename suffix convention):
  -1  :  head / forehead   (Kramer Zone 1)
  -2  :  face              (Kramer Zone 1)
  -3  :  chest / sternum   (Kramer Zone 2)
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from .feature_extractor import compute_features_from_skin_pixels
from .skin_mask import (
    build_neonatal_skin_mask,
    extract_valid_skin_pixels_rgb,
    skin_coverage_fraction,
)

LOG = logging.getLogger("jaundice_extractor")

_MIN_ACCEPTABLE_COVERAGE = 0.05


def _crop_center_half(image: np.ndarray) -> np.ndarray:
    """Return the central 40% of the image (30%–70% on each axis) to avoid the color-card border."""
    h, w = image.shape[:2]
    return image[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]


def _load_bgr_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    return img


def _patient_id_from_filename(filename: str) -> str:
    return Path(filename).stem.split("-")[0]


def process_single_image(
    image_path: str,
    *,
    debug: bool = False,
    debug_dir: str = "debug",
) -> dict:
    """
    Full pipeline for one image: load : crop : mask : features.

    Parameters
    ----------
    image_path : str   Absolute or relative path to the image file.
    debug      : bool  When True, a diagnostic figure is saved to debug_dir.
    debug_dir  : str   Root folder for debug output.

    Returns
    -------
    dict with keys: patient_id, image_idx, and all 14 FEATURE_NAMES.
    """
    filename   = os.path.basename(image_path)
    patient_id = _patient_id_from_filename(filename)

    LOG.info("processing: %s", filename)

    raw_bgr    = _load_bgr_image(image_path)
    LOG.debug("original size: %dx%d", raw_bgr.shape[1], raw_bgr.shape[0])

    cropped_bgr = _crop_center_half(raw_bgr)
    LOG.debug("cropped size: %dx%d", cropped_bgr.shape[1], cropped_bgr.shape[0])

    skin_mask  = build_neonatal_skin_mask(cropped_bgr)
    coverage   = skin_coverage_fraction(skin_mask)
    LOG.debug("skin coverage: %d px (%.1f%%)", int((skin_mask > 0).sum()), coverage * 100)

    if coverage < _MIN_ACCEPTABLE_COVERAGE:
        LOG.warning("low coverage (%.1f%%) in %s — features will be NaN", coverage * 100, filename)

    skin_pixels_rgb = extract_valid_skin_pixels_rgb(cropped_bgr, skin_mask)
    LOG.debug("valid pixels: %d", len(skin_pixels_rgb))

    features = compute_features_from_skin_pixels(skin_pixels_rgb)
    LOG.debug("R_mean=%.2f  Cr_mean=%.2f  Lab_b_mean=%.2f  H_mean=%.2f",
              features.get("R_mean"), features.get("Cr_mean"),
              features.get("Lab_b_mean"), features.get("H_mean"))

    if debug:
        from .debug_visualizer import save_debug_figure
        save_debug_figure(
            image_path=image_path,
            cropped_bgr=cropped_bgr,
            skin_mask=skin_mask,
            skin_pixels_rgb=skin_pixels_rgb,
            features=features,
            output_dir=os.path.join(debug_dir, Path(image_path).stem),
        )

    return {"patient_id": patient_id, "image_idx": filename, **features}