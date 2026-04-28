import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, snapshot_download, create_repo
from huggingface_hub.errors import RepositoryNotFoundError

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
HF_TOKEN    = os.getenv("HF_TOKEN")
HF_REPO_ID  = "mariaamandadevina/neojaundice-dataset"
HF_REPO_TYPE = "dataset"

BASE_DIR    = Path(__file__).resolve().parent
DATA_NEO_DIR = Path(os.getenv("DATA_NEO_DIR", str(BASE_DIR / "__data__" / "neo")))

# NeoJaundice (China): 2235 images from 745 babies.
# Regions: head, face, chest.
# Confirmed TSB (total serum bilirubin, mg/dL) + clinical metadata.
# Metadata columns expected in the accompanying CSV/Excel:
METADATA_COLS = [
    "baby_id",           # unique baby identifier
    "image_path",        # relative path to image file
    "region",            # head | face | chest
    "tsb",               # total serum bilirubin in mg/dL
    "gestational_age",   # weeks
    "age_days",          # postnatal age in days
    "weight",            # grams
    "gender",            # M / F
    "treatment",         # treatment status / label
]

# TSB clinical thresholds (used in subgroup summary)
TSB_MILD_MAX   = 12.0   # mg/dL — below this: monitor
TSB_SEVERE_MIN = 17.0   # mg/dL — above this: urgent referral


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def separator(title=""):
    if title:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print(f"{'=' * 50}")
    else:
        print("=" * 50)


def ensure_repo_exists(api: HfApi):
    try:
        api.repo_info(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, token=HF_TOKEN)
        print(f"  Repo found: {HF_REPO_ID}")
    except RepositoryNotFoundError:
        print(f"  Repo not found. Creating: {HF_REPO_ID} ...")
        create_repo(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            private=False,
            token=HF_TOKEN,
        )
        print(f"  Repo created: https://huggingface.co/datasets/{HF_REPO_ID}")


def upload_files(api: HfApi, file_pairs: list[tuple[Path, str]]):
    ensure_repo_exists(api)
    any_uploaded = False
    for local_path, repo_path in file_pairs:
        if not local_path.exists():
            print(f"  Skipping (not found locally): {local_path.name}")
            continue
        print(f"  Uploading {local_path.name} → {repo_path} ...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            token=HF_TOKEN,
        )
        print(f"  Done: {local_path.name}")
        any_uploaded = True
    if any_uploaded:
        print(f"\n  🔗 https://huggingface.co/datasets/{HF_REPO_ID}")


def collect_all_files(directory: Path) -> list[Path]:
    return sorted([f for f in directory.rglob("*") if f.is_file()])


def pick_files(all_files: list[Path]) -> list[Path]:
    if not all_files:
        print(f"  No files found in {DATA_NEO_DIR}")
        return []

    print(f"\n  Files found in {DATA_NEO_DIR}:")
    for i, f in enumerate(all_files, 1):
        rel = f.relative_to(DATA_NEO_DIR)
        size_kb = f.stat().st_size / 1024
        print(f"    [{i}] {rel}  ({size_kb:.1f} KB)")

    print("\n  Enter file numbers to select (e.g. 1 2 3), or 'all':")
    choice = input("  > ").strip().lower()

    if choice == "all":
        return all_files
    else:
        try:
            indices = [int(x) - 1 for x in choice.split()]
            return [all_files[i] for i in indices]
        except (ValueError, IndexError):
            print("  Invalid selection.")
            return []


def find_metadata_file() -> Path | None:
    """Locate the metadata CSV or Excel file inside DATA_NEO_DIR."""
    candidate_exts = {".csv", ".xlsx", ".xls"}
    candidates = [
        f for f in DATA_NEO_DIR.rglob("*")
        if f.suffix.lower() in candidate_exts and f.is_file()
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    print(f"\n  Multiple tabular files found — pick the metadata file:")
    for i, f in enumerate(candidates, 1):
        print(f"    [{i}] {f.relative_to(DATA_NEO_DIR)}")
    choice = input("  > ").strip()
    try:
        return candidates[int(choice) - 1]
    except (ValueError, IndexError):
        print("  Invalid selection.")
        return None


def load_metadata(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        print(f"  Loaded: {path.name}  ({len(df):,} rows × {len(df.columns)} cols)")
        return df
    except Exception as e:
        print(f"  Error reading {path.name}: {e}")
        return None


# ─────────────────────────────────────────
# OPTION 1 — Upload local files to HuggingFace
# ─────────────────────────────────────────
def upload_local_files():
    separator("OPTION 1 — Upload local files to HuggingFace")

    if not DATA_NEO_DIR.exists():
        print(f"  DATA_NEO_DIR not found: {DATA_NEO_DIR}")
        return

    all_files = collect_all_files(DATA_NEO_DIR)
    selected  = pick_files(all_files)
    if not selected:
        return

    api   = HfApi()
    pairs = [(f, str(f.relative_to(DATA_NEO_DIR))) for f in selected]
    upload_files(api, pairs)


# ─────────────────────────────────────────
# OPTION 2 — Download ALL files from HuggingFace
# ─────────────────────────────────────────
def download_all_files():
    separator("OPTION 2 — Download all files from HuggingFace")
    DATA_NEO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading full repo snapshot → {DATA_NEO_DIR} ...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        local_dir=str(DATA_NEO_DIR),
        token=HF_TOKEN,
    )
    print(f"  All files downloaded to: {DATA_NEO_DIR}")


# ─────────────────────────────────────────
# OPTION 3 — Download specific file(s) from HuggingFace
# ─────────────────────────────────────────
def download_specific_files():
    separator("OPTION 3 — Download specific file(s) from HuggingFace")
    DATA_NEO_DIR.mkdir(parents=True, exist_ok=True)

    print("  Enter filename(s) as they appear in the repo (e.g. metadata.csv).")
    print("  Separate multiple filenames with spaces:")
    raw       = input("  > ").strip()
    filenames = raw.split()

    if not filenames:
        print("  No filenames entered.")
        return

    for filename in filenames:
        print(f"  Downloading {filename} ...")
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type=HF_REPO_TYPE,
                token=HF_TOKEN,
                local_dir=str(DATA_NEO_DIR),
            )
            print(f"  Saved: {DATA_NEO_DIR / filename}")
        except Exception as e:
            print(f"  Failed: {filename} — {e}")


# ─────────────────────────────────────────
# OPTION 4 — Inspect metadata & summarise dataset
# ─────────────────────────────────────────
def inspect_metadata():
    separator("OPTION 4 — Inspect metadata & summarise dataset")

    if not DATA_NEO_DIR.exists():
        print(f"  DATA_NEO_DIR not found: {DATA_NEO_DIR}")
        return

    meta_path = find_metadata_file()
    if meta_path is None:
        print(f"  No CSV/Excel metadata file found in {DATA_NEO_DIR}")
        return

    df = load_metadata(meta_path)
    if df is None:
        return

    # ── Column audit ────────────────────────────────────────────
    separator()
    print("  Column audit:")
    all_cols = list(df.columns)
    found    = [c for c in METADATA_COLS if c in all_cols]
    missing  = [c for c in METADATA_COLS if c not in all_cols]
    extra    = [c for c in all_cols if c not in METADATA_COLS]
    print(f"    Expected columns found : {found or 'none'}")
    print(f"    Expected but missing   : {missing or 'none'}")
    print(f"    Extra columns present  : {extra or 'none'}")

    # ── Image inventory ─────────────────────────────────────────
    separator()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = [f for f in DATA_NEO_DIR.rglob("*") if f.suffix.lower() in image_exts]
    print(f"  Image inventory:")
    print(f"    Total image files found  : {len(images):,}")

    # Group by subfolder (proxy for region)
    region_counts: dict[str, int] = {}
    for img in images:
        folder = img.parent.name
        region_counts[folder] = region_counts.get(folder, 0) + 1
    for folder, count in sorted(region_counts.items()):
        print(f"      {folder:<20} {count:>6,} images")

    # ── TSB summary (if column exists) ──────────────────────────
    tsb_col = next((c for c in df.columns if "tsb" in c.lower() or "bilirubin" in c.lower()), None)
    if tsb_col:
        separator()
        print(f"  TSB summary  (column: '{tsb_col}'):")
        tsb = df[tsb_col].dropna()
        print(f"    Count   : {len(tsb):,}")
        print(f"    Mean    : {tsb.mean():.2f} mg/dL")
        print(f"    Std     : {tsb.std():.2f}")
        print(f"    Min     : {tsb.min():.2f}")
        print(f"    Max     : {tsb.max():.2f}")
        mild   = (tsb < TSB_MILD_MAX).sum()
        moderate = ((tsb >= TSB_MILD_MAX) & (tsb < TSB_SEVERE_MIN)).sum()
        severe = (tsb >= TSB_SEVERE_MIN).sum()
        print(f"\n    Severity buckets (for subgroup analysis):")
        print(f"      < {TSB_MILD_MAX} mg/dL  (mild / monitor)      : {mild:>5,} rows ({100*mild/len(tsb):.1f}%)")
        print(f"      {TSB_MILD_MAX}–{TSB_SEVERE_MIN} mg/dL (moderate)             : {moderate:>5,} rows ({100*moderate/len(tsb):.1f}%)")
        print(f"      ≥ {TSB_SEVERE_MIN} mg/dL (severe / refer)      : {severe:>5,} rows ({100*severe/len(tsb):.1f}%)")
    else:
        print("\n  No TSB / bilirubin column detected — skipping TSB summary.")

    # ── Clinical metadata summaries ─────────────────────────────
    separator()
    print("  Clinical metadata summaries:")

    age_col = next((c for c in df.columns if "age_day" in c.lower() or c.lower() == "age"), None)
    if age_col:
        age = df[age_col].dropna()
        peak = ((age >= 2) & (age <= 5)).sum()
        print(f"    Age (days)  — mean: {age.mean():.1f}, range: {age.min():.0f}–{age.max():.0f}")
        print(f"      Peak risk window (days 2–5): {peak:,} rows ({100*peak/len(age):.1f}%)")

    ga_col = next((c for c in df.columns if "gestational" in c.lower() or c.lower() == "ga"), None)
    if ga_col:
        ga = df[ga_col].dropna()
        preterm = (ga < 37).sum()
        print(f"    Gestational age — mean: {ga.mean():.1f} wks, preterm (<37 wks): {preterm:,} ({100*preterm/len(ga):.1f}%)")

    wt_col = next((c for c in df.columns if "weight" in c.lower() or "wt" in c.lower()), None)
    if wt_col:
        wt = df[wt_col].dropna()
        lbw = (wt < 2500).sum()
        print(f"    Weight       — mean: {wt.mean():.0f} g, LBW (<2500 g): {lbw:,} ({100*lbw/len(wt):.1f}%)")

    gender_col = next((c for c in df.columns if "gender" in c.lower() or "sex" in c.lower()), None)
    if gender_col:
        print(f"    Gender distribution:")
        print(df[gender_col].value_counts().to_string())

    region_col = next((c for c in df.columns if "region" in c.lower()), None)
    if region_col:
        print(f"    Region distribution:")
        print(df[region_col].value_counts().to_string())

    # ── Baby-level count ────────────────────────────────────────
    id_col = next((c for c in df.columns if "baby" in c.lower() or "id" in c.lower()), None)
    if id_col:
        separator()
        n_babies = df[id_col].nunique()
        print(f"  Unique babies ('{id_col}'): {n_babies:,}")
        print(f"  Avg rows per baby        : {len(df)/n_babies:.1f}")

    separator()
    print("  Inspection complete.")


# ─────────────────────────────────────────
# OPTION 5 — Update / re-upload files to HuggingFace
# ─────────────────────────────────────────
def update_files():
    separator("OPTION 5 — Update / re-upload local files to HuggingFace")
    upload_local_files()


# ─────────────────────────────────────────
# MENU
# ─────────────────────────────────────────
def main():
    separator()
    print("       NEOJAUNDICE DATASET MANAGER")
    print("       China · 745 babies · 2235 images · TSB + clinical metadata")
    separator()
    print("  1. Upload local files to HuggingFace")
    print("  2. Download ALL files from HuggingFace")
    print("  3. Download specific file(s) from HuggingFace")
    print("  4. Inspect metadata & summarise dataset")
    print("  5. Update / re-upload local files to HuggingFace")
    print("  0. Exit")
    separator()

    choice = input("  Choose an option [0-5]: ").strip()

    actions = {
        "1": upload_local_files,
        "2": download_all_files,
        "3": download_specific_files,
        "4": inspect_metadata,
        "5": update_files,
    }

    if choice == "0":
        print("  Goodbye.")
        sys.exit(0)
    elif choice in actions:
        actions[choice]()
    else:
        print("  Invalid option. Please choose 0-5.")


if __name__ == "__main__":
    main()