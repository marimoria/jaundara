"""
skin_mask.py
HSV-based two-range skin segmentation for neonatal images.

Covers light : dark skin tones found across the populations
represented in NeoJaundice, NJN (Iraq/Iran), and Indonesian cohorts.
"""

import cv2
import numpy as np

# Range 1 — light to medium neonatal skin (Chinese, Indonesian, light Middle Eastern)
_HSV_LIGHT_SKIN_LOWER = np.array([0, 15, 60], dtype=np.uint8)
_HSV_LIGHT_SKIN_UPPER = np.array([25, 255, 255], dtype=np.uint8)

# Range 2 — darker neonatal skin tones (lower hue wrap, lower saturation floor)
_HSV_DARK_SKIN_LOWER  = np.array([0, 10, 40], dtype=np.uint8)
_HSV_DARK_SKIN_UPPER  = np.array([10, 200, 200], dtype=np.uint8)

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def build_neonatal_skin_mask(bgr_image: np.ndarray) -> np.ndarray:
    """
    Produce a binary skin mask for a neonatal BGR image.

    Two HSV ranges are unioned to cover lighter and darker neonatal skin tones.
    Morphological close then open operations fill small holes and remove noise.

    Parameters
    ----------
    bgr_image : np.ndarray  BGR image as loaded by cv2.imread.

    Returns
    -------
    np.ndarray (uint8, same H×W as input) — 255 where skin detected, 0 elsewhere.
    """
    hsv      = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    combined = cv2.bitwise_or(
        cv2.inRange(hsv, _HSV_LIGHT_SKIN_LOWER, _HSV_LIGHT_SKIN_UPPER),
        cv2.inRange(hsv, _HSV_DARK_SKIN_LOWER,  _HSV_DARK_SKIN_UPPER),
    )
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    cleaned = cv2.morphologyEx(cleaned,  cv2.MORPH_OPEN,  _MORPH_KERNEL)
    return cleaned


def extract_valid_skin_pixels_rgb(
    bgr_image: np.ndarray,
    skin_mask: np.ndarray,
) -> np.ndarray:
    """
    Return mask-selected pixels converted to RGB.

    Parameters
    ----------
    bgr_image : np.ndarray  Source image in BGR.
    skin_mask : np.ndarray  Binary mask (255 = skin) from build_neonatal_skin_mask.

    Returns
    -------
    np.ndarray, shape (N, 3), dtype uint8, RGB order. May be empty (shape (0, 3)).
    """
    return bgr_image[skin_mask > 0][:, ::-1]


def skin_coverage_fraction(skin_mask: np.ndarray) -> float:
    """Return the proportion of image pixels classified as skin [0.0, 1.0]."""
    return float((skin_mask > 0).sum()) / skin_mask.size