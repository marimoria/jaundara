"""
debug_visualizer.py
===================
Saves a multi-panel diagnostic PNG for a single processed image.

Panels
------
  1. Original image (after crop — card already removed)
  2. Skin mask with coverage percentage
  3. Valid-pixel overlay (non-skin pixels darkened to 20 % brightness)
  4. Extracted feature values as monospace text
  5. RGB histogram of valid skin pixels
  6. Lab b* histogram — the strongest single jaundice channel
  7. Cr histogram — the YCrCb jaundice channel

This module is imported lazily (only when debug=True) so it never slows
down normal batch processing runs.
"""

import os
import logging
from pathlib import Path

import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")  # headless — saves to file, never opens a window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .color_math import rgb_to_lab

LOG = logging.getLogger("jaundice_extractor")


# Colour palette (dark theme)

_BG_DARK = "#1a1a2e"
_BG_PANEL = "#16213e"
_SPINE_COL = "#444466"
_TEXT_COL = "#e0e0f0"
_TICK_COL = "gray"


# Public API


def save_debug_figure(
    image_path: str,
    cropped_bgr: np.ndarray,
    skin_mask: np.ndarray,
    skin_pixels_rgb: np.ndarray,
    features: dict,
    output_dir: str,
) -> None:
    """
    Render and save a 7-panel diagnostic figure for one processed image.

    Parameters
    ----------
    image_path      : str           Original path (used for title / filename).
    cropped_bgr     : np.ndarray    Image AFTER card crop, in BGR format.
    skin_mask       : np.ndarray    Binary mask (255 = skin) from skin segmentation.
    skin_pixels_rgb : np.ndarray    Valid pixels in RGB order, shape (N, 3).
    features        : dict          14 computed feature values.
    output_dir      : str           Directory where the PNG is written.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{Path(image_path).stem}_debug.png")

    img_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    coverage = float((skin_mask > 0).sum()) / skin_mask.size * 100

    overlay = _build_darkened_overlay(img_rgb, skin_mask)

    fig = plt.figure(figsize=(18, 10), facecolor=_BG_DARK)
    fig.suptitle(
        f"Debug: {Path(image_path).name}   |   Valid skin pixels: {len(skin_pixels_rgb):,}",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_mask = fig.add_subplot(gs[0, 1])
    ax_overlay = fig.add_subplot(gs[0, 2])
    ax_stats = fig.add_subplot(gs[0, 3])
    ax_rgb = fig.add_subplot(gs[1, 0:2])
    ax_lab_b = fig.add_subplot(gs[1, 2])
    ax_cr = fig.add_subplot(gs[1, 3])

    _style_axes([ax_orig, ax_mask, ax_overlay, ax_stats, ax_rgb, ax_lab_b, ax_cr])

    _draw_original_image(ax_orig, img_rgb)
    _draw_skin_mask(ax_mask, skin_mask, coverage)
    _draw_pixel_overlay(ax_overlay, overlay)
    _draw_feature_text(ax_stats, features)
    _draw_rgb_histogram(ax_rgb, skin_pixels_rgb)
    _draw_lab_b_histogram(ax_lab_b, skin_pixels_rgb)
    _draw_cr_histogram(ax_cr, skin_pixels_rgb)

    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    LOG.info(f"    Debug figure saved → {out_path}")


# Internal helpers


def _build_darkened_overlay(img_rgb: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    """Return image with non-skin pixels dimmed to 20 % of original brightness."""
    overlay = img_rgb.copy()
    dark = (overlay * 0.2).astype(np.uint8)
    mask_3ch = np.stack([skin_mask] * 3, axis=-1) > 0
    return np.where(mask_3ch, overlay, dark)


def _style_axes(axes: list) -> None:
    for ax in axes:
        ax.set_facecolor(_BG_PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(_SPINE_COL)


def _draw_original_image(ax, img_rgb: np.ndarray) -> None:
    ax.imshow(img_rgb)
    ax.set_title("Cropped (card removed)", color="white", fontsize=9)
    ax.axis("off")


def _draw_skin_mask(ax, skin_mask: np.ndarray, coverage: float) -> None:
    ax.imshow(skin_mask, cmap="gray")
    ax.set_title(f"Skin mask  ({coverage:.1f}% coverage)", color="white", fontsize=9)
    ax.axis("off")


def _draw_pixel_overlay(ax, overlay: np.ndarray) -> None:
    ax.imshow(overlay)
    ax.set_title("Valid skin pixels only", color="white", fontsize=9)
    ax.axis("off")


def _draw_feature_text(ax, features: dict) -> None:
    ax.axis("off")
    if any(not np.isnan(v) for v in features.values()):
        lines = [f"{'Feature':<14} {'Value':>8}", "─" * 24]
        for k, v in features.items():
            lines.append(
                f"{k:<14} {v:>8.3f}" if not np.isnan(v) else f"{k:<14} {'NaN':>8}"
            )
        text = "\n".join(lines)
    else:
        text = "⚠  No valid skin pixels\nAll features = NaN"

    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        fontfamily="monospace",
        color=_TEXT_COL,
    )
    ax.set_title("Extracted Features", color="white", fontsize=9)


def _draw_rgb_histogram(ax, skin_pixels_rgb: np.ndarray) -> None:
    if len(skin_pixels_rgb) > 0:
        for ch, col, lbl in [
            (0, "#ff6b6b", "R"),
            (1, "#6bff9e", "G"),
            (2, "#6bb5ff", "B"),
        ]:
            ax.hist(
                skin_pixels_rgb[:, ch],
                bins=50,
                color=col,
                alpha=0.6,
                label=lbl,
                density=True,
            )
        ax.legend(fontsize=8, facecolor=_BG_PANEL, labelcolor="white")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="gray")

    ax.set_title("RGB distribution (valid pixels)", color="white", fontsize=9)
    ax.tick_params(colors=_TICK_COL, labelsize=7)
    ax.set_xlabel("Pixel value", color=_TICK_COL, fontsize=8)


def _draw_lab_b_histogram(ax, skin_pixels_rgb: np.ndarray) -> None:
    if len(skin_pixels_rgb) > 0:
        pixels_f = skin_pixels_rgb.astype(np.float32) / 255.0
        lab_b = np.array([rgb_to_lab(r, g, b)[2] for r, g, b in pixels_f])
        ax.hist(lab_b, bins=40, color="#f9ca24", alpha=0.85, density=True)
        ax.axvline(
            lab_b.mean(),
            color="white",
            linestyle="--",
            linewidth=1.2,
            label=f"mean={lab_b.mean():.2f}",
        )
        ax.legend(fontsize=8, facecolor=_BG_PANEL, labelcolor="white")

    ax.set_title("Lab b*  (blue–yellow / jaundice)", color="white", fontsize=9)
    ax.tick_params(colors=_TICK_COL, labelsize=7)
    ax.set_xlabel("b* value  (+yellow)", color=_TICK_COL, fontsize=8)


def _draw_cr_histogram(ax, skin_pixels_rgb: np.ndarray) -> None:
    if len(skin_pixels_rgb) > 0:
        px_u8 = skin_pixels_rgb.reshape(-1, 1, 3)
        px_bgr = cv2.cvtColor(px_u8, cv2.COLOR_RGB2BGR)
        cr_ch = cv2.cvtColor(px_bgr, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)[:, 1]
        ax.hist(cr_ch.astype(float), bins=40, color="#ff9f43", alpha=0.85, density=True)
        ax.axvline(
            cr_ch.mean(),
            color="white",
            linestyle="--",
            linewidth=1.2,
            label=f"mean={cr_ch.mean():.2f}",
        )
        ax.legend(fontsize=8, facecolor=_BG_PANEL, labelcolor="white")

    ax.set_title("Cr channel (YCrCb)", color="white", fontsize=9)
    ax.tick_params(colors=_TICK_COL, labelsize=7)
    ax.set_xlabel("Cr value", color=_TICK_COL, fontsize=8)
