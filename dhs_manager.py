import os
import glob
import sys
from pathlib import Path

import pandas as pd
import pyreadstat
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download, hf_hub_download

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = "mariaamandadevina/dhs-dataset"
HF_REPO_TYPE = "dataset"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "__data__"))
OUTPUT_CSV = os.path.join(DATA_DIR, "dhs_combined.csv")

COUNTRIES = {
    "Bangladesh": "BD",
    "India": "IA",
    "Indonesia": "ID",
    "Nepal": "NP",
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
def find_dta(country_dir, country_code, file_type):
    base = Path(country_dir)
    folders = list(base.glob(f"{country_code}{file_type}*DT"))
    if not folders:
        print(f"  ⚠ Folder not found: {country_code}{file_type}*DT in {base}")
        return None
    for folder in folders:
        dta_files = list(folder.glob(f"{country_code}{file_type}*FL.DTA"))
        if dta_files:
            return str(dta_files[0])
    print(f"  ⚠ Folder found but no DTA inside: {folders[0]}")
    return None


def read_dta(filepath, variables):
    try:
        df, meta = pyreadstat.read_dta(filepath, usecols=variables)  # type: ignore
        existing = [v for v in variables if v in df.columns]  # type: ignore
        missing = [v for v in variables if v not in df.columns]  # type: ignore
        if missing:
            print(f"    ⚠ Missing columns: {missing}")
        return df[existing]  # type: ignore
    except Exception as e:
        print(f"    ✗ Error reading {filepath}: {e}")
        return None


def separator():
    print("=" * 40)


# ─────────────────────────────────────────
# OPTION 1: Download full DHS from HuggingFace
# ─────────────────────────────────────────
def download_full_dhs():
    separator()
    print("Downloading full DHS dataset from HuggingFace...")
    print(f"Destination: {DATA_DIR}")
    separator()

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        local_dir=DATA_DIR,
        token=HF_TOKEN,
    )

    print("✓ Full DHS dataset downloaded!")


# ─────────────────────────────────────────
# OPTION 2: Extract variables and save CSV
# ─────────────────────────────────────────
def extract_to_csv():
    separator()
    print("Extracting variables from DTA files...")
    separator()

    all_dfs = []

    for country_name, country_code in COUNTRIES.items():
        print(f"\nProcessing: {country_name} ({country_code})")
        separator()

        country_dir = os.path.join(DATA_DIR, country_name)
        country_dfs = []

        for file_type, variables in FILE_VARIABLES.items():
            print(f"\n  [{file_type}] Looking for: {variables}")

            dta_path = find_dta(country_dir, country_code, file_type)
            if not dta_path:
                continue

            print(f"  ✓ Found: {dta_path}")
            df = read_dta(dta_path, variables)
            if df is None or df.empty:
                continue

            print(f"  ✓ Rows: {len(df):,} | Columns: {list(df.columns)}")
            df["file_type"] = file_type
            country_dfs.append(df)

        if not country_dfs:
            print(f"  ✗ No data found for {country_name}")
            continue

        country_combined = pd.concat(country_dfs, axis=0, ignore_index=True)
        country_combined.insert(0, "country", country_name)
        country_combined.insert(1, "country_code", country_code)
        all_dfs.append(country_combined)
        print(f"\n  ✓ {country_name} total rows: {len(country_combined):,}")

    if not all_dfs:
        print("\n✗ No data extracted. Check your file paths.")
        return

    separator()
    print("Combining all countries...")
    final_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    print(f"✓ Total rows: {len(final_df):,}")
    print(f"✓ Columns: {list(final_df.columns)}")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved to: {OUTPUT_CSV}")

    separator()
    print("Summary:")
    print(
        final_df.groupby(["country", "file_type"])
        .size()
        .reset_index(name="rows")
        .to_string(index=False)
    )


# ─────────────────────────────────────────
# OPTION 3: Upload combined CSV to HuggingFace
# ─────────────────────────────────────────
def upload_csv():
    separator()
    print("Uploading combined CSV to HuggingFace...")
    separator()

    if not os.path.exists(OUTPUT_CSV):
        print(f"✗ CSV not found at: {OUTPUT_CSV}")
        print("  Run option 2 first to generate the CSV.")
        return

    api = HfApi()
    api.upload_file(
        path_or_fileobj=OUTPUT_CSV,
        path_in_repo="dhs_combined.csv",
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        token=HF_TOKEN,
    )

    print(f"✓ Uploaded to: https://huggingface.co/datasets/{HF_REPO_ID}")


# ─────────────────────────────────────────
# OPTION 4: Download combined CSV from HuggingFace
# ─────────────────────────────────────────
def download_csv():
    separator()
    print("Downloading combined CSV from HuggingFace...")
    separator()

    local_dir = os.path.dirname(OUTPUT_CSV)
    os.makedirs(local_dir, exist_ok=True)

    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="dhs_combined.csv",
        repo_type=HF_REPO_TYPE,
        token=HF_TOKEN,
        local_dir=local_dir,
    )

    print(f"✓ CSV downloaded to: {local_dir}/dhs_combined.csv")


# ─────────────────────────────────────────
# MENU
# ─────────────────────────────────────────
def main():
    separator()
    print("       DHS DATA MANAGER")
    separator()
    print("1. Download full DHS data from HuggingFace")
    print("2. Extract variables from DTA → save CSV")
    print("3. Upload combined CSV to HuggingFace")
    print("4. Download combined CSV from HuggingFace")
    print("0. Exit")
    separator()

    choice = input("Choose an option: ").strip()

    if choice == "1":
        download_full_dhs()
    elif choice == "2":
        extract_to_csv()
    elif choice == "3":
        upload_csv()
    elif choice == "4":
        download_csv()
    elif choice == "0":
        print("Terminated")
        sys.exit(0)
    else:
        print("Invalid option. Please choose 0-4.")


if __name__ == "__main__":
    main()