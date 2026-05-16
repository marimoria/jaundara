"""
feature_extractor.py
====================
Extracts the 14 color statistics used as model input from a set of
valid skin pixels.

Feature table (14 features per image zone):

  Color space   Feature       What it captures
  ─────────────────────────────────────────────────────────────────
  RGB           R_mean        Baseline red channel signal
                G_mean        Baseline green channel signal
                B_mean        Baseline blue channel signal
                R_std         Skin color uniformity (patchiness)
  YCrCb         Y_mean        Luminance / brightness proxy
                Cr_mean       Red-yellow chrominance — direct jaundice signal
                Cb_mean       Blue-yellow chrominance — complementary signal
  HSL           H_mean        Core yellowing indicator (hue)
                S_mean        Intensity of color shift (saturation)
                L_mean        Lighting condition proxy (lightness)
  CIELAB        Lab_L_mean    Device-independent perceptual lightness
                Lab_a_mean    Green–red axis — secondary skin tone signal
                Lab_b_mean    Blue–yellow axis — strongest single jaundice indicator
                Lab_L_std     Lightness variation / skin uniformity

Two zones are captured per patient (forehead = Zone 1, sternum = Zone 2),
so the full per-patient feature vector is 28 values.
"""

import numpy as np
import cv2

from .color_math import rgb_to_hsl, rgb_to_lab

# Ordered list of feature names — kept here as the single source of truth
FEATURE_NAMES: list[str] = [
    "R_mean",
    "G_mean",
    "B_mean",
    "R_std",
    "Y_mean",
    "Cr_mean",
    "Cb_mean",
    "H_mean",
    "S_mean",
    "L_mean",
    "Lab_L_mean",
    "Lab_a_mean",
    "Lab_b_mean",
    "Lab_L_std",
]


def compute_features_from_skin_pixels(skin_pixels_rgb: np.ndarray) -> dict:
    """
    Compute 14 color statistics from a set of valid skin pixels.

    Parameters
    ----------
    skin_pixels_rgb : np.ndarray, shape (N, 3), dtype uint8
        Skin pixels in RGB order. May be empty (N == 0).

    Returns
    -------
    dict mapping each name in FEATURE_NAMES to a float value.
    If skin_pixels_rgb is empty, all values are np.nan.
    """
    if len(skin_pixels_rgb) == 0:
        return {k: np.nan for k in FEATURE_NAMES}

    pixels_f = skin_pixels_rgb.astype(np.float32) / 255.0
    R, G, B = pixels_f[:, 0], pixels_f[:, 1], pixels_f[:, 2]

    hsl = np.array([rgb_to_hsl(r, g, b) for r, g, b in pixels_f])
    H, S, L = hsl[:, 0], hsl[:, 1], hsl[:, 2]

    lab = np.array([rgb_to_lab(r, g, b) for r, g, b in pixels_f])
    Lab_L, Lab_a, Lab_b = lab[:, 0], lab[:, 1], lab[:, 2]

    # YCrCb via OpenCV (faster than a pure-Python loop)
    px_uint8 = skin_pixels_rgb.reshape(-1, 1, 3)
    px_bgr = cv2.cvtColor(px_uint8, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(px_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb_f = ycrcb.reshape(-1, 3).astype(np.float32)
    Y_ch, Cr, Cb = ycrcb_f[:, 0], ycrcb_f[:, 1], ycrcb_f[:, 2]

    return {
        # ── RGB ──────────────────────────────────────────────────
        "R_mean": round(float(R.mean() * 255), 4),
        "G_mean": round(float(G.mean() * 255), 4),
        "B_mean": round(float(B.mean() * 255), 4),
        "R_std": round(float(R.std() * 255), 4),
        # ── YCrCb ────────────────────────────────────────────────
        "Y_mean": round(float(Y_ch.mean()), 4),
        "Cr_mean": round(float(Cr.mean()), 4),
        "Cb_mean": round(float(Cb.mean()), 4),
        # ── HSL ──────────────────────────────────────────────────
        "H_mean": round(float(H.mean()), 4),
        "S_mean": round(float(S.mean()), 4),
        "L_mean": round(float(L.mean()), 4),
        # ── CIELAB ───────────────────────────────────────────────
        "Lab_L_mean": round(float(Lab_L.mean()), 4),
        "Lab_a_mean": round(float(Lab_a.mean()), 4),
        "Lab_b_mean": round(float(Lab_b.mean()), 4),
        "Lab_L_std": round(float(Lab_L.std()), 4),
    }
