"""
cli.py
======
Command-line entry point for the color_extraction package.

Two modes
---------
  training   Build the full training-ready CSV from a folder of images
             and the NeoJaundice clinical metadata CSV.

  debug      Process a single image with full diagnostic output.
             Useful for validating that the pipeline behaves correctly
             on a given image before running the full batch.

Usage examples
--------------
  # Full training pipeline
  python -m color_extraction training 
      --image_dir  data/neo/images 
      --clinical_csv data/neo/neo.csv 
      --output     out/training_data.csv

  # Training starting from a specific file (manual override)
  python -m color_extraction training
      --image_dir    data/neo/images
      --clinical_csv data/neo/neo.csv
      --output       out/training_data.csv
      --start_from   0032-3.jpg

  # Single-image debug mode
  python -m color_extraction debug 
      --image      data/neo/images/0003-1.jpg 
      --debug_dir  out/debug

  # Training with checkpoint and debug figures
  python -m color_extraction training 
      --image_dir    data/neo/images 
      --clinical_csv data/neo/neo.csv 
      --output       out/training_data.csv 
      --debug 
      --debug_dir    out/debug 
      --save_every   100
"""

import argparse
import logging
import sys
from datetime import datetime


def _setup_root_logger(log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("jaundice_extractor")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────────────────────
# Sub-command: training
# ──────────────────────────────────────────────────────────────

def run_training_pipeline(args: argparse.Namespace, log: logging.Logger) -> None:
    from .dataset_pipeline import run_full_training_data_pipeline

    checkpoint = args.output.replace(".csv", "_checkpoint.csv")

    run_full_training_data_pipeline(
        image_dir            = args.image_dir,
        clinical_csv         = args.clinical_csv,
        output_csv           = args.output,
        n_augmented_variants = args.n_augmented_variants,
        checkpoint_file      = checkpoint,
        save_every           = args.save_every,
        debug                = args.debug,
        debug_dir            = args.debug_dir,
        start_from           = getattr(args, "start_from", None),
    )


# ──────────────────────────────────────────────────────────────
# Sub-command: debug (single image)
# ──────────────────────────────────────────────────────────────

def run_single_image_debug(args: argparse.Namespace, log: logging.Logger) -> None:
    from .image_processor import process_single_image
    import cv2
    import os
    from pathlib import Path
    from .augmentation import generate_brightness_augmented_variants

    log.info(f"Mode: SINGLE IMAGE DEBUG — {args.image}")
    
    # 1. Process the original image
    row = process_single_image(args.image, debug=True, debug_dir=args.debug_dir)

    log.info("\n── Extracted features (Original) ───────────────────────")
    for k, v in row.items():
        if k not in ("patient_id", "image_idx", "is_augmented"):
            log.info(f"  {k:<16} {v}")
    log.info("────────────────────────────────────────────────────────")

    # 2. Process augmented variants if the flag is provided
    if getattr(args, "augment", False):
        log.info("\nMode: SINGLE IMAGE DEBUG — AUGMENTED VARIANTS")
        
        raw_bgr = cv2.imread(args.image)
        variants = generate_brightness_augmented_variants(raw_bgr, n_variants=3) # type: ignore
        
        stem = Path(args.image).stem
        ext = Path(args.image).suffix
        os.makedirs(args.debug_dir, exist_ok=True)

        for i, aug_bgr in enumerate(variants):
            # Save the in-memory variant to disk temporarily so the visualizer can read it
            temp_path = os.path.join(args.debug_dir, f"{stem}_aug{i}{ext}")
            cv2.imwrite(temp_path, aug_bgr)

            aug_row = process_single_image(temp_path, debug=True, debug_dir=args.debug_dir)
            
            log.info(f"\n── Extracted features (Augment {i}) ────────────────────")
            for k, v in aug_row.items():
                if k not in ("patient_id", "image_idx", "is_augmented"):
                    log.info(f"  {k:<16} {v}")
            log.info("────────────────────────────────────────────────────────")


# ──────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="color_extraction",
        description="Neonatal jaundice color feature extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # ── training sub-command ──────────────────────────────────
    train_p = sub.add_parser(
        "training",
        help="Build the full training-ready CSV from a folder of images.",
    )
    train_p.add_argument(
        "--image_dir", required=True, metavar="DIR",
        help="Folder containing NeoJaundice images.",
    )
    train_p.add_argument(
        "--clinical_csv", required=True, metavar="CSV",
        help="NeoJaundice metadata CSV (patient_id, blood(mg/dL), etc.).",
    )
    train_p.add_argument(
        "--output", required=True, metavar="CSV",
        help="Output path for the training-ready CSV.",
    )
    train_p.add_argument(
        "--n_augmented_variants", type=int, default=3, metavar="N",
        help="Brightness-augmented copies per original image (default: 3).",
    )
    train_p.add_argument(
        "--save_every", type=int, default=50, metavar="N",
        help="Checkpoint flush frequency in number of originals (default: 50).",
    )
    train_p.add_argument(
        "--debug", action="store_true",
        help="Save per-image diagnostic figures for original images.",
    )
    train_p.add_argument(
        "--debug_dir", default="debug", metavar="DIR",
        help="Root folder for debug figures (default: ./debug).",
    )
    train_p.add_argument(
        "--log_file", metavar="PATH",
        help="Optional path to write the full log as a .txt file.",
    )
    # In build_argument_parser() under the training sub-command:
    train_p.add_argument(
        "--start_from", metavar="FILENAME",
        help="Skip all files alphabetically before this filename (e.g., 0032-3.jpg).",
    )

    # ── debug sub-command ─────────────────────────────────────
    debug_p = sub.add_parser(
        "debug",
        help="Process a single image with full diagnostic output.",
    )
    debug_p.add_argument(
        "--image", required=True, metavar="PATH",
        help="Path to the image to process.",
    )
    debug_p.add_argument(
        "--debug_dir", default="debug", metavar="DIR",
        help="Root folder for the debug figure (default: ./debug).",
    )
    debug_p.add_argument(
        "--log_file", metavar="PATH",
        help="Optional path to write the full log as a .txt file.",
    )
    # Add this new flag:
    debug_p.add_argument(
        "--augment", action="store_true",
        help="Also generate, debug, and save brightness-augmented variants.",
    )

    return parser


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_argument_parser()
    args   = parser.parse_args()

    log_file = getattr(args, "log_file", None)
    log      = _setup_root_logger(log_file)

    log.info("=" * 60)
    log.info("Neonatal Jaundice Color Feature Extractor")
    log.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    if args.mode == "training":
        run_training_pipeline(args, log)
    elif args.mode == "debug":
        run_single_image_debug(args, log)
    else:
        parser.print_help()
        sys.exit(1)

    log.info("Done.")


if __name__ == "__main__":
    main()