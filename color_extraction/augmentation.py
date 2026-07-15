"""
Brightness-only augmentation for neonatal jaundice training images.
Only the L (lightness) channel in HLS is varied, simulating brightness
differences from different cameras and lighting conditions while preserving
every diagnostically critical channel:

  H  (hue)                - primary yellowing indicator    : FIXED
  Cr (YCrCb chrominance)  - direct bilirubin signal        : FIXED
  b* (CIELAB blue-yellow) - strongest jaundice indicator   : FIXED
  L  (HLS lightness)      - lighting proxy only            : VARIED

  Output:
  745 patients * 3 zones * (1 original + 3 augmented) = 8,940 images
"""

import random
from pathlib import Path

import cv2
import numpy as np

_BRIGHTNESS_FACTOR_RANGE = (0.8, 1.2)
_DEFAULT_VARIANTS_PER_IMAGE = 3


def apply_brightness_shift(bgr_image: np.ndarray, factor: float) -> np.ndarray:
    """
    Multiply the L channel of an HLS image by `factor`, clamping to [0, 255].

    H and S are unchanged, preserving the yellowing signal and its intensity.

    Parameters
    ----------
    bgr_image : np.ndarray  Source image in BGR format (uint8).
    factor    : float       L-channel multiplier; [0.8, 1.2] simulates realistic variation.

    Returns
    -------
    np.ndarray  Augmented BGR image (uint8), same shape as input.
    """
    hls = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HLS).astype(np.float32)
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

    Each variant receives an independently drawn random factor from `factor_range`.
    The original image is NOT included — callers are responsible for keeping it.

    Parameters
    ----------
    bgr_image    : np.ndarray       Source BGR image.
    n_variants   : int              Number of augmented copies (default 3).
    factor_range : tuple[float, float]  (min, max) brightness multiplier (default 0.8–1.2).
    seed         : int | None       Optional seed for reproducibility.

    Returns
    -------
    list[np.ndarray]  n_variants augmented BGR images.
    """
    rng = random.Random(seed)
    return [
        apply_brightness_shift(bgr_image, rng.uniform(*factor_range))
        for _ in range(n_variants)
    ]


def save_augmented_variants_to_disk(
    source_image_path: str,
    output_dir: str,
    n_variants: int = _DEFAULT_VARIANTS_PER_IMAGE,
    factor_range: tuple[float, float] = _BRIGHTNESS_FACTOR_RANGE,
    seed: int | None = None,
) -> list[str]:
    """
    Load source_image_path, produce brightness-shifted variants, and save
    them to output_dir as <stem>_aug0.jpg, <stem>_aug1.jpg, etc.

    Returns a list of saved file paths.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    bgr = cv2.imread(source_image_path)
    if bgr is None:
        raise ValueError(f"Cannot read: {source_image_path}")

    stem     = Path(source_image_path).stem
    ext      = Path(source_image_path).suffix or ".jpg"
    variants = generate_brightness_augmented_variants(bgr, n_variants, factor_range, seed)

    saved = []
    for i, variant in enumerate(variants):
        out_path = os.path.join(output_dir, f"{stem}_aug{i}{ext}")
        cv2.imwrite(out_path, variant)
        saved.append(out_path)
    return saved