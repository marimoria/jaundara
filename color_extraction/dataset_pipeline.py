"""
dataset_pipeline.py
===================
Orchestrates the full NeoJaundice training-data preparation pipeline:

  Step 1  Discover images across all three training zones (-1 head, -2 face/cheek, -3 chest).
  Step 2  For each image: crop → mask → extract 14 features.
  Step 3  For each original image: produce 3 brightness-augmented variants
          and extract features from those too.
  Step 4  Merge the feature table with the clinical metadata CSV.
  Step 5  Write a training-ready CSV (42 color features + 3 metadata fields
          + TSB label + binary jaundice label).

Augmentation multiplier:
  745 patients * 3 zones * (1 original + 3 augmented) = 8,940 feature rows

Training zones:
  Suffix -1  →  head / forehead  (Kramer Zone 1)
  Suffix -2  →  face / cheek     (Kramer Zone 1 — additional Zone 1 signal)
  Suffix -3  →  chest / sternum  (Kramer Zone 2)

CSV schema expected from the NeoJaundice dataset:
  patient_id, image_idx, gender, gestational_age, age(day),
  weight, blood(mg/dL), Treatment

Model input columns written to the output CSV:
  42 color features (14 * 3 zones, prefixed zone1_ / zone2_ / zone3_)
  + gestational_age, postnatal_age_days, weight   (available to home user)
  + blood_mg_dl                                    (TSB — regression label)
  + jaundice_label                                 (binary — classification label)
  + treatment                                      (retained for subgroup analysis only)
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

from .image_processor import process_single_image
from .augmentation import generate_brightness_augmented_variants
from .feature_extractor import compute_features_from_skin_pixels, FEATURE_NAMES
from .skin_mask import (
    build_neonatal_skin_mask,
    extract_valid_skin_pixels_rgb,
    skin_coverage_fraction,
)

LOG = logging.getLogger("jaundice_extractor")

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

_TRAINING_ZONE_SUFFIXES = {"-1", "-2", "-3"}   # head, face/cheek, chest
_VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_TSB_JAUNDICE_THRESHOLD  = 12.9          # mg/dL — per NeoJaundice clinical protocol
_N_AUGMENTED_VARIANTS    = 3

# Rename map: raw CSV column → normalised name used throughout this codebase
_CSV_COLUMN_RENAMES = {
    "age(day)":    "postnatal_age_days",
    "blood(mg/dL)": "blood_mg_dl",
    "Treatment":   "treatment",
}

# Clinical metadata available to a mother at home (model inputs)
_HOME_USER_METADATA_COLS = ["gestational_age", "postnatal_age_days", "weight"]


# ──────────────────────────────────────────────────────────────
# Step 1 — Image discovery
# ──────────────────────────────────────────────────────────────

def discover_training_zone_images(image_dir: str) -> list[str]:
    """
    Return sorted paths of all images whose filename suffix matches a
    training zone (-1, -2, or -3).
    """
    all_paths = [
        os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir))
        if Path(f).suffix.lower() in _VALID_IMAGE_EXTENSIONS
    ]

    training_paths = [
        p for p in all_paths
        if any(Path(p).stem.endswith(s) for s in _TRAINING_ZONE_SUFFIXES)
    ]

    LOG.info(
        f"Image discovery: {len(all_paths)} total images in '{image_dir}', "
        f"{len(training_paths)} in training zones (-1 head, -2 face, -3 chest)"
    )
    return training_paths


# ──────────────────────────────────────────────────────────────
# Step 2 + 3 — Feature extraction with augmentation
# ──────────────────────────────────────────────────────────────

def _extract_features_from_bgr_in_memory(
    bgr_image: np.ndarray,
    patient_id: str,
    image_idx: str,
    is_augmented: bool,
    aug_variant_index: int | None = None,
) -> dict:
    """
    Run the crop → mask → feature pipeline on a BGR array already in memory.
    Used for augmented variants so we never write them to disk.
    """
    from .image_processor import _crop_center_half  # reuse the same crop logic

    cropped         = _crop_center_half(bgr_image)
    skin_mask       = build_neonatal_skin_mask(cropped)
    skin_pixels_rgb = extract_valid_skin_pixels_rgb(cropped, skin_mask)
    features        = compute_features_from_skin_pixels(skin_pixels_rgb)

    aug_label = f"_aug{aug_variant_index}" if is_augmented else ""
    return {
        "patient_id":   patient_id,
        "image_idx":    image_idx,
        "is_augmented": is_augmented,
        **features,
    }


def extract_features_with_augmentation(
    image_paths: list[str],
    n_variants: int = _N_AUGMENTED_VARIANTS,
    checkpoint_file: str | None = None,
    save_every: int = 50,
    debug: bool = False,
    debug_dir: str = "debug",
    start_from: str | None = None,
) -> pd.DataFrame:
    """
    For every image path:
      - Extract features from the original image.
      - Generate `n_variants` brightness-shifted copies in memory and extract
        features from each.

    Augmented variants are never written to disk — only their feature rows
    are kept. This keeps the augmentation lightweight and deterministic.

    Parameters
    ----------
    image_paths    : list[str]   Paths from discover_training_zone_images().
    n_variants     : int         Augmented copies per original (default 3).
    checkpoint_file: str | None  If given, intermediate rows are flushed here
                                 every `save_every` originals so the run is
                                 resumable.
    save_every     : int         Checkpoint frequency (originals processed).
    debug          : bool        Save debug figures for original images.
    debug_dir      : str         Root folder for debug output.
    start_from     : str | None  Skip all files alphabetically before this filename.

    Returns
    -------
    pd.DataFrame  One row per image×augmentation variant.
                  Columns: patient_id, image_idx, is_augmented, + 14 features.
    """
    if not image_paths:
        raise ValueError("No image paths provided to extract_features_with_augmentation.")

    # 1. Setup variables
    already_done: set[str] = set()
    rows: list[dict]       = []

    # 2. Checkpoint logic
    if checkpoint_file and os.path.exists(checkpoint_file):
        existing = pd.read_csv(checkpoint_file)
        if "image_idx" in existing.columns:
            already_done = set(existing["image_idx"].tolist())
            LOG.info(f"Resuming from checkpoint — {len(already_done)} original images done")

    # 3. Create the pending list
    pending = [p for p in image_paths if os.path.basename(p) not in already_done]
    
    # 4. Apply start_from override
    if start_from:
        pending = [p for p in pending if os.path.basename(p) >= start_from]
        LOG.info(f"Manual override: Starting from {start_from} (Pending: {len(pending)})")
        
    LOG.info(f"Images to process: {len(pending)} (skipping {len(already_done)} done)")

    # 5. Extraction Loop
    n_ok = n_fail = 0
    flush_buffer: list[dict] = []
    write_mode   = "a" if already_done else "w"
    write_header = not bool(already_done)

    for i, path in enumerate(pending, 1):
        filename   = os.path.basename(path)
        patient_id = Path(path).stem.split("-")[0]

        try:
            # ── Original image ────────────────────────────────
            original_row = process_single_image(path, debug=debug, debug_dir=debug_dir)
            original_row["is_augmented"] = False
            flush_buffer.append(original_row)

            # ── Augmented variants (in-memory only) ───────────
            raw_bgr  = cv2.imread(path)
            variants = generate_brightness_augmented_variants(raw_bgr, n_variants) # type: ignore

            for aug_idx, aug_bgr in enumerate(variants):
                aug_row = _extract_features_from_bgr_in_memory(
                    bgr_image         = aug_bgr,
                    patient_id        = patient_id,
                    image_idx         = filename,
                    is_augmented      = True,
                    aug_variant_index = aug_idx,
                )
                flush_buffer.append(aug_row)

            n_ok += 1

        except Exception as exc:
            LOG.error(f"  FAIL: {filename} — {exc}")
            n_fail += 1

        # Progress
        pct = i / len(pending) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r  [{bar}] {pct:5.1f}%  ok={n_ok}  fail={n_fail}", end="", flush=True)

        # Checkpoint flush
        if checkpoint_file and (i % save_every == 0 or i == len(pending)):
            chunk = pd.DataFrame(flush_buffer)
            chunk.to_csv(checkpoint_file, mode=write_mode, header=write_header, index=False)
            flush_buffer  = []
            write_mode    = "a"
            write_header  = False
            LOG.debug(f"\n  Checkpoint saved ({i}/{len(pending)})")

    print()
    LOG.info(f"Extraction complete — ok={n_ok}  fail={n_fail}")

    # Combine checkpoint + any unflushed in-memory rows
    if checkpoint_file and os.path.exists(checkpoint_file):
        saved  = pd.read_csv(checkpoint_file)
        extras = pd.DataFrame(flush_buffer)
        return pd.concat([saved, extras], ignore_index=True) if flush_buffer else saved

    return pd.DataFrame(flush_buffer)


# ──────────────────────────────────────────────────────────────
# Step 4 — Merge features with clinical CSV
# ──────────────────────────────────────────────────────────────

def _load_and_normalise_clinical_csv(csv_path: str) -> pd.DataFrame:
    """Load the NeoJaundice CSV and rename columns to consistent internal names."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns=_CSV_COLUMN_RENAMES)
    return df


def pivot_zones_into_patient_row(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the per-image feature rows into one row per patient,
    with zone-specific column prefixes:

      zone1_* (14 features) — head / forehead  (suffix -1)
      zone2_* (14 features) — face / cheek     (suffix -2)
      zone3_* (14 features) — chest / sternum  (suffix -3)

    This produces the 42-feature patient vector (3 zones × 14 features).
    """
    def zone_label(image_idx: str) -> str:
        stem = Path(image_idx).stem
        if stem.endswith("-1"):
            return "zone1"
        if stem.endswith("-2"):
            return "zone2"
        if stem.endswith("-3"):
            return "zone3"
        return "unknown"

    features_df = features_df.copy()
    features_df["zone"] = features_df["image_idx"].apply(zone_label)

    pivoted_parts = {}
    for zone_name in ("zone1", "zone2", "zone3"):
        zone_df = (
            features_df[features_df["zone"] == zone_name]
            .drop(columns=["zone", "image_idx"])
            .rename(columns={col: f"{zone_name}_{col}" for col in FEATURE_NAMES})
        )
        pivoted_parts[zone_name] = zone_df

    zone1 = pivoted_parts["zone1"].set_index(["patient_id", "is_augmented"])
    zone2 = pivoted_parts["zone2"].set_index(["patient_id", "is_augmented"])
    zone3 = pivoted_parts["zone3"].set_index(["patient_id", "is_augmented"])
    merged = zone1.join(zone2, how="inner").join(zone3, how="inner").reset_index()

    LOG.info(
        f"Zone pivot: {len(merged)} patient×augmentation rows "
        f"({merged['is_augmented'].sum()} augmented, "
        f"{(~merged['is_augmented']).sum()} originals)"
    )
    return merged


def merge_features_with_clinical_metadata(
    patient_feature_df: pd.DataFrame,
    clinical_csv_path: str,
) -> pd.DataFrame:
    """
    Join the 28-feature patient rows with the clinical CSV.

    Adds TSB (regression label), binary jaundice label, and the three
    metadata fields a mother can provide at home. Treatment is retained
    for subgroup analysis but is NOT used as a model input.

    Parameters
    ----------
    patient_feature_df : pd.DataFrame  Output of pivot_zones_into_patient_row.
    clinical_csv_path  : str           Path to the NeoJaundice metadata CSV.

    Returns
    -------
    pd.DataFrame  Training-ready table.
    """
    clinical = _load_and_normalise_clinical_csv(clinical_csv_path)

    # One clinical record per patient (all three zone rows are identical)
    clinical_per_patient = (
        clinical[["patient_id"] + _HOME_USER_METADATA_COLS + ["blood_mg_dl", "treatment"]]
        .drop_duplicates(subset="patient_id")
    )

    merged = patient_feature_df.merge(
        clinical_per_patient,
        on="patient_id",
        how="left",
    )

    # Binary jaundice label
    merged["jaundice_label"] = (merged["blood_mg_dl"] >= _TSB_JAUNDICE_THRESHOLD).astype(int)

    matched = merged["blood_mg_dl"].notna().sum()
    LOG.info(
        f"Clinical merge: {matched}/{len(merged)} rows matched "
        f"({matched / len(merged) * 100:.1f}%)"
    )

    unmatched_ids = merged[merged["blood_mg_dl"].isna()]["patient_id"].unique().tolist()
    if unmatched_ids:
        LOG.warning(
            f"  {len(unmatched_ids)} patient_ids without clinical data: "
            f"{unmatched_ids[:5]}{'...' if len(unmatched_ids) > 5 else ''}"
        )

    return merged


# ──────────────────────────────────────────────────────────────
# Step 5 — Write training CSV
# ──────────────────────────────────────────────────────────────

def write_training_csv(training_df: pd.DataFrame, output_path: str) -> None:
    """
    Save the training-ready DataFrame to `output_path`.

    Column order:
      patient_id, is_augmented,
      zone1_* (14 features), zone2_* (14 features), zone3_* (14 features),
      gestational_age, postnatal_age_days, weight,   ← home-user metadata
      blood_mg_dl, jaundice_label,                   ← labels
      treatment                                       ← subgroup analysis only
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    # Enforce column order
    zone_cols     = (
        [f"zone1_{f}" for f in FEATURE_NAMES] +
        [f"zone2_{f}" for f in FEATURE_NAMES] +
        [f"zone3_{f}" for f in FEATURE_NAMES]
    )
    meta_cols     = _HOME_USER_METADATA_COLS
    label_cols    = ["blood_mg_dl", "jaundice_label"]
    subgroup_cols = ["treatment"]

    ordered = ["patient_id", "is_augmented"] + zone_cols + meta_cols + label_cols + subgroup_cols
    # Keep any extra columns at the end (future-proof)
    extras  = [c for c in training_df.columns if c not in ordered]
    final   = training_df[[c for c in ordered + extras if c in training_df.columns]]

    final.to_csv(output_path, index=False)
    LOG.info(f"Training CSV saved → {output_path}  ({final.shape[0]} rows × {final.shape[1]} cols)")


# ──────────────────────────────────────────────────────────────
# High-level convenience runner
# ──────────────────────────────────────────────────────────────

def run_full_training_data_pipeline(
    image_dir: str,
    clinical_csv: str,
    output_csv: str,
    *,
    n_augmented_variants: int = _N_AUGMENTED_VARIANTS,
    checkpoint_file: str | None = None,
    save_every: int = 50,
    debug: bool = False,
    debug_dir: str = "debug",
    start_from: str | None = None
) -> pd.DataFrame:
    """
    End-to-end pipeline from raw images to a training-ready CSV.

    Parameters
    ----------
    image_dir            : str   Folder containing NeoJaundice images.
    clinical_csv         : str   Path to the NeoJaundice metadata CSV.
    output_csv           : str   Destination path for the training CSV.
    n_augmented_variants : int   Brightness-augmented copies per image (default 3).
    checkpoint_file      : str   Optional intermediate checkpoint path.
    save_every           : int   Checkpoint flush frequency (originals).
    debug                : bool  Save per-image debug figures.
    debug_dir            : str   Root folder for debug figures.
    start_from           : str | None Skip all files alphabetically before this filename.

    Returns
    -------
    pd.DataFrame  The final training table (also written to output_csv).
    """
    LOG.info("=" * 60)
    LOG.info("NeoJaundice Training Data Pipeline")
    LOG.info("=" * 60)

    # Step 1 — Find training images
    image_paths = discover_training_zone_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No training-zone images found in: {image_dir}")

    # Steps 2 + 3 — Extract + augment
    feature_df = extract_features_with_augmentation(
        image_paths     = image_paths,
        n_variants      = n_augmented_variants,
        checkpoint_file = checkpoint_file,
        save_every      = save_every,
        debug           = debug,
        debug_dir       = debug_dir,
        start_from      = start_from
    )

    # Step 3b — Pivot zones into per-patient rows
    patient_df = pivot_zones_into_patient_row(feature_df)

    # Step 4 — Merge with clinical metadata
    training_df = merge_features_with_clinical_metadata(patient_df, clinical_csv)

    # Step 5 — Write training CSV
    write_training_csv(training_df, output_csv)

    # Clean up checkpoint
    if checkpoint_file and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        LOG.info(f"Checkpoint removed: {checkpoint_file}")

    LOG.info("Pipeline complete.")
    return training_df