import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, snapshot_download, create_repo
from huggingface_hub.errors import RepositoryNotFoundError

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_NJN = Path(os.getenv("DATA_NJN_DIR", str(BASE_DIR / "__data__" / "njn")))
DEFAULT_JAMIA = Path(os.getenv("DATA_JAMIA_DIR", str(BASE_DIR / "__data__" / "jamia")))
DEFAULT_NEO = Path(os.getenv("DATA_NEO_DIR", str(BASE_DIR / "__data__" / "neo")))

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_TYPE = "dataset"
HF_REPO_ID = ""
DATA_DIR: Path = DEFAULT_NJN

def separator(title=""):
    if title:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print(f"{'=' * 50}")
    else:
        print("=" * 50)

def setup_session():
    global HF_REPO_ID, DATA_DIR
    separator("SETUP SESSION")

    print("  Enter HuggingFace Repository ID (e.g. mariaamandadevina/jaundice-neojaundice)")
    repo = input("  Repo ID: ").strip()
    HF_REPO_ID = repo if repo else "mariaamandadevina/jaundice-neojaundice"

    print("\n  Select local data directory:")
    print(f"  1. NJN Dataset   ({DEFAULT_NJN})")
    print(f"  2. JAMIA Dataset ({DEFAULT_JAMIA})")
    print(f"  3. NEO Dataset   ({DEFAULT_NEO})")
    print("  4. Custom Path")
    
    dir_choice = input("  Choice [1-4]: ").strip()
    
    if dir_choice == "1":
        DATA_DIR = DEFAULT_NJN
    elif dir_choice == "2":
        DATA_DIR = DEFAULT_JAMIA
    elif dir_choice == "3":
        DATA_DIR = DEFAULT_NEO
    else:
        custom = input("  Enter full path to directory: ").strip()
        DATA_DIR = Path(custom)

    print(f"\n  Active Repo : {HF_REPO_ID}")
    print(f"  Active Dir  : {DATA_DIR}")

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

def collect_all_files(directory: Path) -> list[Path]:
    return sorted([f for f in directory.rglob("*") if f.is_file()])

def pick_files(all_files: list[Path]) -> list[Path]:
    if not all_files:
        print(f"  No files found in {DATA_DIR}")
        return []

    print(f"\n  Files found in {DATA_DIR}:")
    for i, f in enumerate(all_files, 1):
        rel = f.relative_to(DATA_DIR)
        size_kb = f.stat().st_size / 1024
        print(f"    [{i}] {rel}  ({size_kb:.1f} KB)")

    print("\n  Enter file numbers to select (e.g. 1 2 3), or 'all':")
    choice = input("  ").strip().lower()

    if choice == "all":
        return all_files
    else:
        try:
            indices = [int(val) - 1 for val in choice.split()]
            return [all_files[i] for i in indices]
        except (ValueError, IndexError):
            print("  Invalid selection.")
            return []

def upload_files(api: HfApi, selected_files: list[Path]):
    ensure_repo_exists(api)
    
    rel_paths = [str(f.relative_to(DATA_DIR)).replace("\\", "/") for f in selected_files]
    
    print(f"  Batch uploading {len(rel_paths)} files to {HF_REPO_ID} (Large Folder Mode)...")
    
    try:
        api.upload_large_folder(
            folder_path=str(DATA_DIR),
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            allow_patterns=rel_paths
        )
        print(f"  Batch upload complete!")
        print(f"\n  https://huggingface.co/datasets/{HF_REPO_ID}")
    except Exception as e:
        print(f"  Upload failed: {e}")

def list_local_files():
    separator("OPTION 1 — List local files")

    if not DATA_DIR.exists():
        print(f"  Directory not found: {DATA_DIR}")
        return

    all_files = collect_all_files(DATA_DIR)
    if not all_files:
        print(f"  No files found in {DATA_DIR}")
        return

    total_kb = 0
    print(f"\n  {DATA_DIR}")
    for f in all_files:
        rel = f.relative_to(DATA_DIR)
        size_kb = f.stat().st_size / 1024
        total_kb += size_kb
        print(f"    {rel}  ({size_kb:.1f} KB)")

    print(f"\n  Total: {len(all_files)} files  ({total_kb / 1024:.1f} MB)")

def upload_local_files():
    separator("OPTION 2 — Upload local files to HuggingFace")

    if not DATA_DIR.exists():
        print(f"  Directory not found: {DATA_DIR}")
        return

    all_files = collect_all_files(DATA_DIR)
    selected  = pick_files(all_files)
    if not selected:
        return

    api = HfApi(token=HF_TOKEN)
    upload_files(api, selected)

def download_all_files():
    separator("OPTION 3 — Download all files from HuggingFace")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading full repo snapshot to {DATA_DIR} ...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        local_dir=str(DATA_DIR),
        token=HF_TOKEN,
    )
    print(f"  All files downloaded to: {DATA_DIR}")

def download_specific_files():
    separator("OPTION 4 — Download specific file(s) from HuggingFace")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("  Enter filename(s) as they appear in the repo (e.g. metadata.csv).")
    print("  Separate multiple filenames with spaces:")
    raw = input("  ").strip()
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
                local_dir=str(DATA_DIR),
            )
            print(f"  Saved: {DATA_DIR / filename}")
        except Exception as e:
            print(f"  Failed: {filename} - {e}")

def update_files():
    separator("OPTION 5 — Update / re-upload local files to HuggingFace")
    upload_local_files()

def main():
    setup_session()

    while True:
        separator()
        print("      DATASET MANAGER")
        print(f"      Active Repo : {HF_REPO_ID}")
        print(f"      Active Dir  : {DATA_DIR}")
        separator()
        print("  1. List local files")
        print("  2. Upload local files to HuggingFace")
        print("  3. Download ALL files from HuggingFace")
        print("  4. Download specific file(s) from HuggingFace")
        print("  5. Update / re-upload local files to HuggingFace")
        print("  6. Change Repository and Directory")
        print("  0. Exit")
        separator()

        choice = input("  Choose an option [0-6]: ").strip()

        if choice == "0":
            print("  Goodbye.")
            sys.exit(0)
        elif choice == "1":
            list_local_files()
        elif choice == "2":
            upload_local_files()
        elif choice == "3":
            download_all_files()
        elif choice == "4":
            download_specific_files()
        elif choice == "5":
            update_files()
        elif choice == "6":
            setup_session()
        else:
            print("  Invalid option. Please choose 0-6.")

if __name__ == "__main__":
    main()