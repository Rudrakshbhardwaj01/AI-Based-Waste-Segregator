"""
AI-Powered Waste Segregation - Dataset Download Script
======================================================
Helper script to download the TrashNet dataset automatically.
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

TRASHNET_KAGGLE_URL = "https://www.kaggle.com/datasets/techsash/waste-classification-data"
TRASHNET_GITHUB_URL = "https://github.com/garythung/trashnet"

def download_kaggle_dataset(kaggle_path="techsash/waste-classification-data", output_dir="data"):
    """
    Download dataset from Kaggle using kaggle API.
    Requires: pip install kaggle
    Requires: Kaggle API credentials (setup instructions in README)
    """
    
    try:
        import kaggle
        print("Downloading dataset from Kaggle...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            kaggle_path,
            path=output_dir,
            unzip=True
        )
        
        print(f"Dataset downloaded to: {output_dir}/")
        
    except ImportError:
        print(" Kaggle API not installed.")
        print("Install with: pip install kaggle")
        print("\nAlternative: Download manually from:")
        print(f"  {TRASHNET_KAGGLE_URL}")
        
    except Exception as e:
        print(f" Error downloading: {e}")
        print("\nAlternative: Download manually from:")
        print(f"  {TRASHNET_KAGGLE_URL}")

def setup_instructions():
    """Print setup instructions for manual download."""
    
    print("=" * 60)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("\nThe TrashNet dataset can be downloaded from:")
    print(f"  Kaggle: {TRASHNET_KAGGLE_URL}")
    print(f"  GitHub: {TRASHNET_GITHUB_URL}")
    
    print("\n Expected directory structure:")
    print("""
data/
├── cardboard/    (~400 images)
├── glass/        (~500 images)
├── metal/        (~400 images)
├── paper/        (~595 images)
├── plastic/      (~482 images)
└── trash/        (~140 images)
    """)
    
    print("\n Steps to download:")
    print("1. Visit the Kaggle dataset page")
    print("2. Click 'Download' button")
    print("3. Extract the ZIP file")
    print("4. Rename the extracted folder to 'data'")
    print("5. Place 'data' folder in the project root")
    
    print("\n Quick start (if you have kaggle API installed):")
    print("  python download_dataset.py --kaggle")

def main():
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--kaggle":
        download_kaggle_dataset()
    else:
        setup_instructions()

if __name__ == "__main__":
    main()

