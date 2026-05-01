"""
augmentation.py
===============
Brightness-only augmentation for neonatal jaundice training images.

Why brightness only?
--------------------
The models operate on explicit color statistics (means and stds) extracted
from skin pixels — not on raw pixel tensors. Jittering hue, saturation, or
chrominance channels (Cr, b*) would write incorrect feature values into rows
that still carry their original jaundiced/normal label, introducing systematic
label noise into the training set.

Only the L (lightness) channel in HLS is varied. This simulates the brightness
differences introduced by different smartphone cameras and ambient lighting
conditions while keeping every diagnostically critical channel intact:

  H  (hue)                 — primary yellowing indicator     → FIXED
  Cr (YCrCb chrominance)   — direct bilirubin signal         → FIXED
  b* (CIELAB blue-yellow)  — strongest jaundice indicator    → FIXED
  L  (HLS lightness)       — lighting proxy only             → VARIED

Reference: Çağır (2025) on endoscopic disease classification confirms that
jittering diagnostically critical color channels degrades accuracy, while
brightness augmentation improves robustness to lighting variation.

Pipeline outcome:
  745 patients * 3 zones * (1 original + 3 augmented) = 8.940 images
"""

import random
from pathlib import Path

import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

_BRIGHTNESS_FACTOR_RANGE = (0.8, 1.2)
_DEFAULT_VARIANTS_PER_IMAGE = 3


# ──────────────────────────────────────────────────────────────
# Core augmentation
# ──────────────────────────────────────────────────────────────

def apply_brightness_shift(bgr_image: np.ndarray, factor: float) -> np.ndarray:
    """
    Multiply the L channel of an HLS image by `factor`, clamping to [0, 255].

    H (hue) and S (saturation) are left completely unchanged, ensuring that
    the yellowing signal and its intensity are preserved exactly.

    Parameters
    ----------
    bgr_image : np.ndarray   Source image in BGR format (uint8).
    factor : float           Multiplier for the L channel. Values in [0.8, 1.2]
                             simulate realistic camera/lighting variation.

    Returns
    -------
    np.ndarray  Augmented BGR image (uint8), same shape as input.
    """
    hls = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HLS).astype(np.float32)

    # Channel order in OpenCV HLS: [H, L, S]
    hls[:, :, 1] = np.clip(hls[:, :, 1] * factor, 0, 255)

    return cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2BGR)


def generate_brightness_augmented_variants(
    bgr_image: np.ndarray,
    n_variants: int = _DEFAULT_VARIANTS_PER_IMAGE,
    factor_range: tuple[float, float] = _BRIGHTNESS_FACTOR_RANGE,
    seed: int | None = None,
) -> list[np.ndarray]:
    """
    Generate `n_variants` brightness-shifted copies of bgr_image.

    Each variant receives an independently drawn random factor from
    `factor_range`. The original image is NOT included in the returned list —
    callers are responsible for keeping it alongside the variants.

    Parameters
    ----------
    bgr_image    : np.ndarray  Source BGR image.
    n_variants   : int         Number of augmented copies to produce (default 3).
    factor_range : tuple       (min, max) brightness multiplier (default 0.8–1.2).
    seed         : int | None  Optional random seed for reproducibility.

    Returns
    -------
    list[np.ndarray]  List of n_variants augmented BGR images.
    """
    rng = random.Random(seed)
    return [
        apply_brightness_shift(bgr_image, rng.uniform(*factor_range))
        for _ in range(n_variants)
    ]


# ──────────────────────────────────────────────────────────────
# Disk-based helpers (used by dataset_pipeline)
# ──────────────────────────────────────────────────────────────

def save_augmented_variants_to_disk(
    source_image_path: str,
    output_dir: str,
    n_variants: int = _DEFAULT_VARIANTS_PER_IMAGE,
    factor_range: tuple[float, float] = _BRIGHTNESS_FACTOR_RANGE,
    seed: int | None = None,
) -> list[str]:
    """
    Load `source_image_path`, produce brightness-shifted variants, and save
    them to `output_dir` with filenames like `<stem>_aug0.jpg`.

    Returns a list of the saved file paths.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    bgr = cv2.imread(source_image_path)
    if bgr is None:
        raise ValueError(f"Cannot read: {source_image_path}")

    stem     = Path(source_image_path).stem
    ext      = Path(source_image_path).suffix or ".jpg"
    variants = generate_brightness_augmented_variants(bgr, n_variants, factor_range, seed)

    saved_paths = []
    for i, variant in enumerate(variants):
        out_path = os.path.join(output_dir, f"{stem}_aug{i}{ext}")
        cv2.imwrite(out_path, variant)
        saved_paths.append(out_path)

    return saved_paths