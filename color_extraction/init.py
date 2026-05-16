"""
color_extraction
================
Neonatal jaundice color feature extraction package.

Public surface
--------------
High-level pipeline (most users only need this):

    from color_extraction.dataset_pipeline import run_full_training_data_pipeline

Single-image processing:

    from color_extraction.image_processor import process_single_image

Lower-level building blocks:

    from color_extraction.feature_extractor import compute_features_from_skin_pixels
    from color_extraction.skin_mask import build_neonatal_skin_mask
    from color_extraction.augmentation import generate_brightness_augmented_variants
    from color_extraction.color_math import rgb_to_lab, rgb_to_hsl

CLI entry point:

    python -m color_extraction training --help
    python -m color_extraction debug --help
"""

from .color_math import rgb_to_hsl, rgb_to_lab
from .skin_mask import build_neonatal_skin_mask, extract_valid_skin_pixels_rgb
from .feature_extractor import compute_features_from_skin_pixels, FEATURE_NAMES
from .image_processor import process_single_image
from .augmentation import (
    apply_brightness_shift,
    generate_brightness_augmented_variants,
)
from .dataset_pipeline import (
    discover_training_zone_images,
    extract_features_with_augmentation,
    pivot_zones_into_patient_row,
    merge_features_with_clinical_metadata,
    write_training_csv,
    run_full_training_data_pipeline,
)

__all__ = [
    # color_math
    "rgb_to_hsl",
    "rgb_to_lab",
    # skin_mask
    "build_neonatal_skin_mask",
    "extract_valid_skin_pixels_rgb",
    # feature_extractor
    "compute_features_from_skin_pixels",
    "FEATURE_NAMES",
    # image_processor
    "process_single_image",
    # augmentation
    "apply_brightness_shift",
    "generate_brightness_augmented_variants",
    # dataset_pipeline
    "discover_training_zone_images",
    "extract_features_with_augmentation",
    "pivot_zones_into_patient_row",
    "merge_features_with_clinical_metadata",
    "write_training_csv",
    "run_full_training_data_pipeline",
]
