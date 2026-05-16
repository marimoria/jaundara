"""
skin_mask.py
============
HSV-based two-range skin segmentation for neonatal images.

Covers light → dark skin tones found across the populations
represented in NeoJaundice, NJN (Iraq/Iran), and Indonesian cohorts.
"""

import numpy as np
import cv2


# HSV range constants


# Range 1 — light to medium neonatal skin (Chinese, Indonesian, light Middle Eastern)
_HSV_LIGHT_SKIN_LOWER = np.array([0, 15, 60], dtype=np.uint8)
_HSV_LIGHT_SKIN_UPPER = np.array([25, 255, 255], dtype=np.uint8)

# Range 2 — darker neonatal skin tones (lower hue wrap, lower saturation floor)
_HSV_DARK_SKIN_LOWER = np.array([0, 10, 40], dtype=np.uint8)
_HSV_DARK_SKIN_UPPER = np.array([10, 200, 200], dtype=np.uint8)

# Morphological kernel for closing holes and removing speck noise
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


# Public API


def build_neonatal_skin_mask(bgr_image: np.ndarray) -> np.ndarray:
    """
    Produce a binary skin mask for a neonatal BGR image.

    Two HSV ranges are unioned to cover both lighter and darker
    neonatal skin tones. Morphological close then open operations
    fill small holes and remove isolated noise pixels.

    Parameters
    ----------
    bgr_image : np.ndarray
        Image in BGR format as loaded by cv2.imread.

    Returns
    -------
    mask : np.ndarray (uint8, same H×W as input)
        255 where skin was detected, 0 elsewhere.
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    mask_light = cv2.inRange(hsv, _HSV_LIGHT_SKIN_LOWER, _HSV_LIGHT_SKIN_UPPER)
    mask_dark = cv2.inRange(hsv, _HSV_DARK_SKIN_LOWER, _HSV_DARK_SKIN_UPPER)
    combined = cv2.bitwise_or(mask_light, mask_dark)

    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, _MORPH_KERNEL)

    return cleaned


def extract_valid_skin_pixels_rgb(
    bgr_image: np.ndarray,
    skin_mask: np.ndarray,
) -> np.ndarray:
    """
    Return only the pixels selected by skin_mask, converted to RGB.

    Parameters
    ----------
    bgr_image : np.ndarray   Source image in BGR.
    skin_mask : np.ndarray   Binary mask (255 = skin) from build_neonatal_skin_mask.

    Returns
    -------
    pixels : np.ndarray, shape (N, 3), dtype uint8, in RGB channel order.
             N is the number of mask-positive pixels. May be empty (shape (0, 3)).
    """
    valid_bgr = bgr_image[skin_mask > 0]
    return valid_bgr[:, ::-1]  # BGR → RGB


def skin_coverage_fraction(skin_mask: np.ndarray) -> float:
    """
    Return the proportion of image pixels classified as skin [0.0, 1.0].
    """
    return float((skin_mask > 0).sum()) / skin_mask.size
