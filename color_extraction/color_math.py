"""
color_math.py
Pure color-space conversion functions.
All inputs are floats in [0, 1] unless stated otherwise.
No I/O, no side effects, no dependencies beyond the standard library.
"""

import colorsys


def rgb_to_hsl(r: float, g: float, b: float) -> tuple[float, float, float]:
    """
    Convert a normalised RGB triplet to HSL.

    Returns h in degrees [0, 360), s and l in percent [0, 100].
    """
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return round(h * 360, 4), round(s * 100, 4), round(l * 100, 4)


def _linearise_srgb_channel(c: float) -> float:
    return (c / 12.92) if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rgb_to_xyz(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Convert normalised sRGB to CIE XYZ (D65). Returns X, Y, Z each in [0, ~1.09]."""
    r_l = _linearise_srgb_channel(r)
    g_l = _linearise_srgb_channel(g)
    b_l = _linearise_srgb_channel(b)
    X = r_l * 0.4124564 + g_l * 0.3575761 + b_l * 0.1804375
    Y = r_l * 0.2126729 + g_l * 0.7151522 + b_l * 0.0721750
    Z = r_l * 0.0193339 + g_l * 0.1191920 + b_l * 0.9503041
    return X, Y, Z


def _lab_f(t: float) -> float:
    d = 6 / 29
    return t ** (1 / 3) if t > d**3 else t / (3 * d**2) + 4 / 29


def xyz_to_lab(X: float, Y: float, Z: float) -> tuple[float, float, float]:
    """
    Convert CIE XYZ (D65) to CIELAB.

    Returns L* in [0, 100], a* on the green-red axis, b* on the blue-yellow axis.
    b* is the strongest single jaundice indicator.
    """
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    fx, fy, fz = _lab_f(X / Xn), _lab_f(Y / Yn), _lab_f(Z / Zn)
    return (
        round(116 * fy - 16, 4),
        round(500 * (fx - fy), 4),
        round(200 * (fy - fz), 4),
    )


def rgb_to_lab(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Convenience wrapper: normalised RGB : CIELAB (L*, a*, b*) in one call."""
    return xyz_to_lab(*rgb_to_xyz(r, g, b))