"""
dataset_pipeline.py
Orchestrates the full NeoJaundice training-data preparation pipeline:

  1  Discover images across all three training zones (-1 head, -2 face/cheek, -3 chest).
  2  For each image: crop : mask : extract 14 features.
  3  For each original image: produce 3 brightness-augmented variants and extract features.
  4  Merge the feature table with the clinical metadata CSV.
  5  Write a training-ready CSV (42 color features + 3 metadata fields + TSB + binary label).

Augmentation multiplier:
  745 patients * 3 zones * (1 original + 3 augmented) = 8,940 feature rows

Training zones:
  Suffix -1  :  head / forehead  (Kramer Zone 1)
  Suffix -2  :  face / cheek     (Kramer Zone 1 — additional signal)
  Suffix -3  :  chest / sternum  (Kramer Zone 2)

CSV schema expected from the NeoJaundice dataset:
  patient_id, image_idx, gender, gestational_age, age(day),
  weight, blood(mg/dL), jaundice_label

Output CSV columns:
  42 color features (14 x 3 zones, prefixed zone1_ / zone2_ / zone3_)
  + gestational_age, postnatal_age_days, weight
  + blood_mg_dl       (TSB — regression label)
  + jaundice_label    (binary — clinician-assigned ground truth)
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .augmentation import generate_brightness_augmented_variants
from .feature_extractor import FEATURE_NAMES, compute_features_from_skin_pixels
from .image_processor import process_single_image
from .skin_mask import build_neonatal_skin_mask, extract_valid_skin_pixels_rgb

LOG = logging.getLogger("jaundice_extractor")

_TRAINING_ZONE_SUFFIXES  = {"-1", "-2", "-3"}
_VALID_IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_TSB_JAUNDICE_THRESHOLD  = 12.9
_N_AUGMENTED_VARIANTS    = 3

_CSV_COLUMN_RENAMES = {
    "age(day)":     "postnatal_age_days",
    "blood(mg/dL)": "blood_mg_dl",
    "jaundice_label": "jaundice_label",
}

_HOME_USER_METADATA_COLS = ["gestational_age", "postnatal_age_days", "weight"]


def discover_training_zone_images(image_dir: str) -> list[str]:
    """Return sorted paths of all images whose filename suffix matches a training zone."""
    all_paths = [
        os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir))
        if Path(f).suffix.lower() in _VALID_IMAGE_EXTENSIONS
    ]
    training_paths = [
        p for p in all_paths
        if any(Path(p).stem.endswith(s) for s in _TRAINING_ZONE_SUFFIXES)
    ]
    LOG.info("discovered %d training-zone images out of %d total in '%s'",
             len(training_paths), len(all_paths), image_dir)
    return training_paths


def _extract_features_from_bgr_in_memory(
    bgr_image: np.ndarray,
    patient_id: str,
    image_idx: str,
    is_augmented: bool,
    aug_variant_index: int | None = None,
) -> dict:
    """Crop : mask : extract features from a BGR array already in memory."""
    from .image_processor import _crop_center_half

    cropped         = _crop_center_half(bgr_image)
    skin_mask       = build_neonatal_skin_mask(cropped)
    skin_pixels_rgb = extract_valid_skin_pixels_rgb(cropped, skin_mask)
    features        = compute_features_from_skin_pixels(skin_pixels_rgb)
    return {"patient_id": patient_id, "image_idx": image_idx, "is_augmented": is_augmented, **features}


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
    Extract features from each image and its brightness-augmented variants.

    Augmented variants are never written to disk — only their feature rows are kept.

    Parameters
    ----------
    image_paths     : list[str]   Paths from discover_training_zone_images().
    n_variants      : int         Augmented copies per original (default 3).
    checkpoint_file : str | None  Flush intermediate rows here every `save_every` originals.
    save_every      : int         Checkpoint frequency in originals processed.
    debug           : bool        Save debug figures for original images.
    debug_dir       : str         Root folder for debug output.
    start_from      : str | None  Skip all files alphabetically before this filename.

    Returns
    -------
    pd.DataFrame  One row per image × augmentation variant.
    """
    if not image_paths:
        raise ValueError("No image paths provided to extract_features_with_augmentation.")

    already_done: set[str] = set()
    if checkpoint_file and os.path.exists(checkpoint_file):
        existing = pd.read_csv(checkpoint_file)
        if "image_idx" in existing.columns:
            already_done = set(existing["image_idx"].tolist())
            LOG.info("resuming from checkpoint — %d originals done", len(already_done))

    pending = [p for p in image_paths if os.path.basename(p) not in already_done]
    if start_from:
        pending = [p for p in pending if os.path.basename(p) >= start_from]
        LOG.info("start_from=%s — %d images pending", start_from, len(pending))

    LOG.info("images to process: %d (skipping %d)", len(pending), len(already_done))

    n_ok = n_fail = 0
    flush_buffer: list[dict] = []
    write_mode   = "a" if already_done else "w"
    write_header = not bool(already_done)

    for i, path in enumerate(pending, 1):
        filename   = os.path.basename(path)
        patient_id = Path(path).stem.split("-")[0]

        try:
            original_row = process_single_image(path, debug=debug, debug_dir=debug_dir)
            original_row["is_augmented"] = False
            flush_buffer.append(original_row)

            raw_bgr  = cv2.imread(path)
            variants = generate_brightness_augmented_variants(raw_bgr, n_variants)  # type: ignore
            for aug_idx, aug_bgr in enumerate(variants):
                flush_buffer.append(_extract_features_from_bgr_in_memory(
                    bgr_image=aug_bgr, patient_id=patient_id,
                    image_idx=filename, is_augmented=True, aug_variant_index=aug_idx,
                ))
            n_ok += 1

        except Exception as exc:
            LOG.error("FAIL: %s — %s", filename, exc)
            n_fail += 1

        pct = i / len(pending) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r  [{bar}] {pct:5.1f}%  ok={n_ok}  fail={n_fail}", end="", flush=True)

        if checkpoint_file and (i % save_every == 0 or i == len(pending)):
            pd.DataFrame(flush_buffer).to_csv(
                checkpoint_file, mode=write_mode, header=write_header, index=False
            )
            flush_buffer  = []
            write_mode    = "a"
            write_header  = False
            LOG.debug("checkpoint saved (%d/%d)", i, len(pending))

    print()
    LOG.info("extraction complete — ok=%d  fail=%d", n_ok, n_fail)

    if checkpoint_file and os.path.exists(checkpoint_file):
        saved = pd.read_csv(checkpoint_file)
        return pd.concat([saved, pd.DataFrame(flush_buffer)], ignore_index=True) if flush_buffer else saved
    return pd.DataFrame(flush_buffer)


def _load_and_normalise_clinical_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path).rename(columns=_CSV_COLUMN_RENAMES)


def pivot_zones_into_patient_row(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot per-image feature rows into one row per patient with zone-prefixed columns:
      zone1_* (head), zone2_* (face/cheek), zone3_* (chest) — 14 features each.
    """
    def zone_label(image_idx: str) -> str:
        stem = Path(image_idx).stem
        if stem.endswith("-1"): return "zone1"
        if stem.endswith("-2"): return "zone2"
        if stem.endswith("-3"): return "zone3"
        return "unknown"

    features_df = features_df.copy()
    features_df["zone"] = features_df["image_idx"].apply(zone_label)

    parts = {}
    for zone in ("zone1", "zone2", "zone3"):
        parts[zone] = (
            features_df[features_df["zone"] == zone]
            .drop(columns=["zone", "image_idx"])
            .rename(columns={col: f"{zone}_{col}" for col in FEATURE_NAMES})
        )

    merged = (
        parts["zone1"].set_index(["patient_id", "is_augmented"])
        .join(parts["zone2"].set_index(["patient_id", "is_augmented"]), how="inner")
        .join(parts["zone3"].set_index(["patient_id", "is_augmented"]), how="inner")
        .reset_index()
    )
    LOG.info("zone pivot: %d rows (%d augmented, %d original)",
             len(merged), merged["is_augmented"].sum(), (~merged["is_augmented"]).sum())
    return merged


def merge_features_with_clinical_metadata(
    patient_feature_df: pd.DataFrame,
    clinical_csv_path: str,
) -> pd.DataFrame:
    """
    Join zone-feature patient rows with the clinical CSV.

    jaundice_label is the clinician-assigned ground truth, not a derived threshold.
    blood_mg_dl is retained as the TSB regression label.
    """
    clinical = _load_and_normalise_clinical_csv(clinical_csv_path)
    clinical_per_patient = (
        clinical[["patient_id"] + _HOME_USER_METADATA_COLS + ["blood_mg_dl", "jaundice_label"]]
        .drop_duplicates(subset="patient_id")
    )
    merged  = patient_feature_df.merge(clinical_per_patient, on="patient_id", how="left")
    matched = merged["blood_mg_dl"].notna().sum()
    LOG.info("clinical merge: %d/%d rows matched (%.1f%%)",
             matched, len(merged), matched / len(merged) * 100)

    unmatched = merged[merged["blood_mg_dl"].isna()]["patient_id"].unique().tolist()
    if unmatched:
        LOG.warning("%d patient_ids without clinical data: %s%s",
                    len(unmatched), unmatched[:5], "..." if len(unmatched) > 5 else "")
    return merged


def write_training_csv(training_df: pd.DataFrame, output_path: str) -> None:
    """
    Save the training DataFrame to output_path with enforced column order:
      patient_id, is_augmented, zone1_* (14), zone2_* (14), zone3_* (14),
      gestational_age, postnatal_age_days, weight, blood_mg_dl, jaundice_label.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    zone_cols = [f"{z}_{f}" for z in ("zone1", "zone2", "zone3") for f in FEATURE_NAMES]
    ordered   = ["patient_id", "is_augmented"] + zone_cols + _HOME_USER_METADATA_COLS + ["blood_mg_dl", "jaundice_label"]
    extras    = [c for c in training_df.columns if c not in ordered]
    final     = training_df[[c for c in ordered + extras if c in training_df.columns]]
    final.to_csv(output_path, index=False)
    LOG.info("training CSV saved: %s  (%d rows × %d cols)", output_path, *final.shape)


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
    start_from: str | None = None,
) -> pd.DataFrame:
    """
    End-to-end pipeline from raw images to a training-ready CSV.

    Parameters
    ----------
    image_dir            : str        Folder containing NeoJaundice images.
    clinical_csv         : str        Path to the NeoJaundice metadata CSV.
    output_csv           : str        Destination path for the training CSV.
    n_augmented_variants : int        Brightness-augmented copies per image (default 3).
    checkpoint_file      : str | None Optional intermediate checkpoint path.
    save_every           : int        Checkpoint flush frequency in originals processed.
    debug                : bool       Save per-image debug figures.
    debug_dir            : str        Root folder for debug figures.
    start_from           : str | None Skip all files alphabetically before this filename.

    Returns
    -------
    pd.DataFrame  The final training table (also written to output_csv).
    """
    image_paths = discover_training_zone_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No training-zone images found in: {image_dir}")

    feature_df  = extract_features_with_augmentation(
        image_paths=image_paths, n_variants=n_augmented_variants,
        checkpoint_file=checkpoint_file, save_every=save_every,
        debug=debug, debug_dir=debug_dir, start_from=start_from,
    )
    patient_df  = pivot_zones_into_patient_row(feature_df)
    training_df = merge_features_with_clinical_metadata(patient_df, clinical_csv)
    write_training_csv(training_df, output_csv)

    if checkpoint_file and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        LOG.info("checkpoint removed: %s", checkpoint_file)

    LOG.info("pipeline complete.")
    return training_df