import os
import argparse
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import colorsys
from pathlib import Path

warnings.filterwarnings("ignore")

# Automatically resolve the default image path relative to this script
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = str(BASE_DIR / "__data__" / "neo" / "images")

def rgb_to_hsl(r: float, g: float, b: float) -> tuple[float, float, float]:
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return round(h * 360, 4), round(s * 100, 4), round(l * 100, 4)

def rgb_to_xyz(r: float, g: float, b: float) -> tuple[float, float, float]:
    def linearise(c: float) -> float:
        return (c / 12.92) if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_lin = linearise(r)
    g_lin = linearise(g)
    b_lin = linearise(b)

    X = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    Y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    Z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
    return X, Y, Z

def xyz_to_lab(X: float, Y: float, Z: float) -> tuple[float, float, float]:
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    def f(t: float) -> float:
        delta = 6 / 29
        return t ** (1 / 3) if t > delta ** 3 else t / (3 * delta ** 2) + 4 / 29

    fx = f(X / Xn)
    fy = f(Y / Yn)
    fz = f(Z / Zn)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return round(L, 4), round(a, 4), round(b, 4)

def rgb_to_lab(r: float, g: float, b: float) -> tuple[float, float, float]:
    return xyz_to_lab(*rgb_to_xyz(r, g, b))

def get_center_crop(image: np.ndarray, crop_pct: float = 0.30) -> np.ndarray:
    h, w = image.shape[:2]
    margin_h = int(h * (1 - crop_pct) / 2)
    margin_w = int(w * (1 - crop_pct) / 2)
    return image[margin_h: h - margin_h, margin_w: w - margin_w]

def extract_color_stats(region: np.ndarray) -> dict:
    pixels = region.reshape(-1, 3).astype(np.float32) / 255.0

    R, G, B = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    hsl_pixels = np.array([rgb_to_hsl(r, g, b) for r, g, b in pixels])
    H, S, L = hsl_pixels[:, 0], hsl_pixels[:, 1], hsl_pixels[:, 2]

    lab_pixels = np.array([rgb_to_lab(r, g, b) for r, g, b in pixels])
    Lab_L, Lab_a, Lab_b = lab_pixels[:, 0], lab_pixels[:, 1], lab_pixels[:, 2]

    return {
        "R_mean":     round(float(R.mean() * 255), 4),
        "G_mean":     round(float(G.mean() * 255), 4),
        "B_mean":     round(float(B.mean() * 255), 4),
        "R_std":      round(float(R.std()  * 255), 4),
        "H_mean":     round(float(H.mean()), 4),
        "S_mean":     round(float(S.mean()), 4),
        "L_mean":     round(float(L.mean()), 4),
        "Lab_L_mean": round(float(Lab_L.mean()), 4),
        "Lab_a_mean": round(float(Lab_a.mean()), 4),
        "Lab_b_mean": round(float(Lab_b.mean()), 4),
        "Lab_L_std":  round(float(Lab_L.std()),  4),
    }

def process_image(image_path: str, crop_pct: float = 0.30) -> dict:
    filename = os.path.basename(image_path)
    patient_id = filename.split("-")[0]

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    region = get_center_crop(arr, crop_pct)
    stats  = extract_color_stats(region)

    return {"patient_id": patient_id, "image_idx": filename, **stats}

def collect_images(image_dir: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = []
    if not os.path.exists(image_dir):
        return paths
    for fname in sorted(os.listdir(image_dir)):
        if os.path.splitext(fname)[1].lower() in exts:
            paths.append(os.path.join(image_dir, fname))
    return paths

def batch_extract_robust(image_dir: str, checkpoint_file: str, crop_pct: float = 0.30, save_every: int = 50) -> pd.DataFrame:
    """Processes images with auto-save to prevent data loss on timeout."""
    paths = collect_images(image_dir)
    if not paths:
        raise FileNotFoundError(f"No supported images found in: {image_dir}")

    processed_files = set()
    if os.path.exists(checkpoint_file):
        existing_df = pd.read_csv(checkpoint_file)
        if "image_idx" in existing_df.columns:
            processed_files = set(existing_df["image_idx"].tolist())
            print(f"Resuming... Found {len(processed_files)} previously processed images.")

    pending_paths = [p for p in paths if os.path.basename(p) not in processed_files]
    
    if not pending_paths:
        print("All images already processed.")
        return pd.read_csv(checkpoint_file)

    print(f"Processing {len(pending_paths)} new images...")
    
    # Open file in append mode if it exists, otherwise write mode
    mode = 'a' if processed_files else 'w'
    header = not bool(processed_files)

    rows = []
    total_processed = 0

    for path in pending_paths:
        try:
            row = process_image(path, crop_pct)
            rows.append(row)
            total_processed += 1
            print(f"  OK: {os.path.basename(path)}")
        except Exception as exc:
            print(f"  FAIL: {os.path.basename(path)} - {exc}")

        # Save chunk and clear memory
        if total_processed % save_every == 0 or total_processed == len(pending_paths):
            chunk_df = pd.DataFrame(rows)
            chunk_df.to_csv(checkpoint_file, mode=mode, header=header, index=False)
            rows = []
            mode = 'a' 
            header = False 

    return pd.read_csv(checkpoint_file)

def merge_with_original(features_df: pd.DataFrame, original_csv: str) -> pd.DataFrame:
    orig = pd.read_csv(original_csv)

    if "image_idx" not in orig.columns:
        raise ValueError("Original CSV must have an 'image_idx' column.")

    # Drop patient_id from features to avoid duplicate columns during merge
    feat_cols = [c for c in features_df.columns if c != "patient_id"]
    merged = orig.merge(features_df[feat_cols], on="image_idx", how="left")

    matched = merged["R_mean"].notna().sum()
    print(f"\nMerge summary: {matched}/{len(orig)} rows matched color data.")
    return merged

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR)
    p.add_argument("--output", required=True)
    p.add_argument("--original_csv", required=True)
    p.add_argument("--crop_pct", type=float, default=0.30)
    return p.parse_args()

def main():
    args = parse_args()
    
    # Use a temporary checkpoint file to prevent timeout losses
    checkpoint_file = args.output.replace(".csv", "_checkpoint.csv")

    features_df = batch_extract_robust(args.image_dir, checkpoint_file, args.crop_pct, save_every=50)

    output_df = merge_with_original(features_df, args.original_csv)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    output_df.to_csv(args.output, index=False)
    
    # Clean up checkpoint if successful
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        
    print(f"\nSaved final merged dataset -> {args.output}")
    print(f"Shape: {output_df.shape[0]} rows x {output_df.shape[1]} columns\n")

if __name__ == "__main__":
    main()