import os
import sys
from pathlib import Path

import pandas as pd
import pyreadstat
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, snapshot_download, create_repo
from huggingface_hub.errors import RepositoryNotFoundError

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
HF_TOKEN     = os.getenv("HF_TOKEN")
HF_REPO_ID   = "mariaamandadevina/dhs-dataset"
HF_REPO_TYPE = "dataset"

BASE_DIR = Path(__file__).resolve().parent
DATA_DHS_DIR = Path(os.getenv("DATA_DHS_DIR", str(BASE_DIR / "__data__")))

OUT_CSV = DATA_DHS_DIR / "dhs_combined.csv"

COUNTRIES = {
    "Bangladesh": "BD",
    "India":      "IA",
    "Indonesia":  "ID",
    "Nepal":      "NP",
    "TimorLeste": "TL",
}

FILE_VARIABLES = {
    "KR": ["m18", "m19", "m19a", "m17", "b4", "m4", "m5"],
    "IR": ["v453", "v454", "v455", "v456", "v457"],
    "BR": ["b11", "b12"],
}


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
        print(f"  ✓ Repo found: {HF_REPO_ID}")
    except RepositoryNotFoundError:
        print(f"  ℹ Repo not found. Creating: {HF_REPO_ID} ...")
        create_repo(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            private=False,
            token=HF_TOKEN,
        )
        print(f"  ✓ Repo created: https://huggingface.co/datasets/{HF_REPO_ID}")


def upload_files(api: HfApi, file_pairs: list[tuple[Path, str]]):
    ensure_repo_exists(api)
    any_uploaded = False
    for local_path, repo_path in file_pairs:
        if not local_path.exists():
            print(f"  ⚠ Skipping (not found locally): {local_path.name}")
            continue
        print(f"  ↑ Uploading {local_path.name} → {repo_path} ...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            token=HF_TOKEN,
        )
        print(f"  ✓ Done: {local_path.name}")
        any_uploaded = True
    if any_uploaded:
        print(f"\n  🔗 https://huggingface.co/datasets/{HF_REPO_ID}")


def collect_all_files(directory: Path) -> list[Path]:
    return sorted([f for f in directory.rglob("*") if f.is_file()])


def pick_files(all_files: list[Path]) -> list[Path]:
    if not all_files:
        print(f"  ✗ No files found in {DATA_DHS_DIR}")
        return []

    print(f"\n  Files found in {DATA_DHS_DIR}:")
    for i, f in enumerate(all_files, 1):
        rel = f.relative_to(DATA_DHS_DIR)
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
            print("  ✗ Invalid selection.")
            return []


def find_dta(country_dir: Path, country_code: str, file_type: str):
    folders = list(country_dir.glob(f"{country_code}{file_type}*DT"))
    if not folders:
        print(f"  ⚠ Folder not found: {country_code}{file_type}*DT in {country_dir}")
        return None
    for folder in folders:
        dta_files = list(folder.glob(f"{country_code}{file_type}*FL.DTA"))
        if dta_files:
            return str(dta_files[0])
    print(f"  ⚠ Folder found but no DTA inside: {folders[0]}")
    return None


def read_dta(filepath: str, variables: list[str]):
    try:
        df, _ = pyreadstat.read_dta(filepath, usecols=variables)
        existing = [v for v in variables if v in df.columns] # type: ignore
        missing  = [v for v in variables if v not in df.columns] # type: ignore
        if missing:
            print(f"    ⚠ Missing columns: {missing}")
        return df[existing] # type: ignore
    except Exception as e:
        print(f"    ✗ Error reading {filepath}: {e}")
        return None


# ─────────────────────────────────────────
# OPTION 1 — Upload local files to HuggingFace
# ─────────────────────────────────────────
def upload_local_files():
    separator("OPTION 1 — Upload local files to HuggingFace")

    if not DATA_DHS_DIR.exists():
        print(f"  ✗ DATA_DHS_DIR not found: {DATA_DHS_DIR}")
        return

    all_files = collect_all_files(DATA_DHS_DIR)
    selected  = pick_files(all_files)
    if not selected:
        return

    api   = HfApi()
    pairs = [(f, str(f.relative_to(DATA_DHS_DIR))) for f in selected]
    upload_files(api, pairs)


# ─────────────────────────────────────────
# OPTION 2 — Download ALL files from HuggingFace
# ─────────────────────────────────────────
def download_all_files():
    separator("OPTION 2 — Download all files from HuggingFace")
    DATA_DHS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  ↓ Downloading full repo snapshot → {DATA_DHS_DIR} ...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        local_dir=str(DATA_DHS_DIR),
        token=HF_TOKEN,
    )
    print(f"  ✓ All files downloaded to: {DATA_DHS_DIR}")


# ─────────────────────────────────────────
# OPTION 3 — Download specific file(s) from HuggingFace
# ─────────────────────────────────────────
def download_specific_files():
    separator("OPTION 3 — Download specific file(s) from HuggingFace")
    DATA_DHS_DIR.mkdir(parents=True, exist_ok=True)

    print("  Enter filename(s) as they appear in the repo (e.g. dhs_combined.csv).")
    print("  Separate multiple filenames with spaces:")
    raw       = input("  > ").strip()
    filenames = raw.split()

    if not filenames:
        print("  ✗ No filenames entered.")
        return

    for filename in filenames:
        print(f"  ↓ Downloading {filename} ...")
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type=HF_REPO_TYPE,
                token=HF_TOKEN,
                local_dir=str(DATA_DHS_DIR),
            )
            print(f"  ✓ Saved: {DATA_DHS_DIR / filename}")
        except Exception as e:
            print(f"  ✗ Failed: {filename} — {e}")


# ─────────────────────────────────────────
# OPTION 4 — Extract DTA variables → dhs_combined.csv
# ─────────────────────────────────────────
def combine_dta_to_csv():
    separator("OPTION 4 — Extract DTA variables → dhs_combined.csv")

    all_dfs = []

    for country_name, country_code in COUNTRIES.items():
        print(f"\n  Processing: {country_name} ({country_code})")
        separator()

        country_dir = DATA_DHS_DIR / country_name
        country_dfs = []

        for file_type, variables in FILE_VARIABLES.items():
            print(f"\n    [{file_type}] Looking for: {variables}")

            dta_path = find_dta(country_dir, country_code, file_type)
            if not dta_path:
                continue

            print(f"    ✓ Found: {dta_path}")
            df = read_dta(dta_path, variables)
            if df is None or df.empty:
                continue

            print(f"    ✓ Rows: {len(df):,} | Columns: {list(df.columns)}")
            df["file_type"] = file_type
            country_dfs.append(df)

        if not country_dfs:
            print(f"  ✗ No data found for {country_name}")
            continue

        country_combined = pd.concat(country_dfs, axis=0, ignore_index=True)
        country_combined.insert(0, "country",      country_name)
        country_combined.insert(1, "country_code", country_code)
        all_dfs.append(country_combined)
        print(f"\n  ✓ {country_name} total rows: {len(country_combined):,}")

    if not all_dfs:
        print("\n  ✗ No data extracted. Check your file paths.")
        return

    separator()
    print("  Combining all countries...")
    final_df = pd.concat(all_dfs, axis=0, ignore_index=True)

    DATA_DHS_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUT_CSV, index=False)

    print(f"\n  ✓ Total rows : {len(final_df):,}")
    print(f"  ✓ Columns    : {list(final_df.columns)}")
    print(f"  ✓ Saved      : {OUT_CSV}")

    print("\n  Summary:")
    print(
        final_df.groupby(["country", "file_type"])
        .size()
        .reset_index(name="rows")
        .to_string(index=False)
    )


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
    print("       DHS DATA MANAGER")
    separator()
    print("  1. Upload local files to HuggingFace")
    print("  2. Download ALL files from HuggingFace")
    print("  3. Download specific file(s) from HuggingFace")
    print("  4. Extract DTA variables → dhs_combined.csv")
    print("  5. Update / re-upload local files to HuggingFace")
    print("  0. Exit")
    separator()

    choice = input("  Choose an option [0–5]: ").strip()

    actions = {
        "1": upload_local_files,
        "2": download_all_files,
        "3": download_specific_files,
        "4": combine_dta_to_csv,
        "5": update_files,
    }

    if choice == "0":
        print("  Goodbye.")
        sys.exit(0)
    elif choice in actions:
        actions[choice]()
    else:
        print("  ✗ Invalid option. Please choose 0–5.")


if __name__ == "__main__":
    main()