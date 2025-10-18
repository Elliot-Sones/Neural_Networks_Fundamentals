"""
Setup script for RNN doodle classification dataset.

This script automatically downloads and processes the Quick Draw dataset
to create train/test splits for the 10 animal classes needed for RNN training.

Usage:
    python 3.RNN/setup_data.py [--force] [--samples N] [--insecure]

This script:
1. Downloads Quick Draw dataset files for the 10 animal classes
2. Processes the JSON stroke data into the required format
3. Creates stratified train/test splits (85%/15%)
4. Saves as animal_doodles_10_train.csv and animal_doodles_10_test.csv
5. Creates a combined animal_doodles.csv for backward compatibility

The script follows the same pattern as the MLP and CNN setup scripts,
automatically downloading data so users can clone and run without manual setup.
"""

import sys
import json
import gzip
import urllib.request
import ssl
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Ten target animal classes (matching class.py exactly)
ANIMALS = [
    "butterfly",
    "cow", 
    "elephant",
    "giraffe",
    "monkey",
    "octopus",
    "scorpion",
    "shark",
    "snake",
    "spider",
]

# Directory setup
BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive"
RAW_DIR = ARCHIVE_DIR / "raw"

# Output files that training/test scripts expect
TRAIN_OUTPUT = ARCHIVE_DIR / "animal_doodles_10_train.csv"
TEST_OUTPUT = ARCHIVE_DIR / "animal_doodles_10_test.csv"
COMBINED_OUTPUT = ARCHIVE_DIR / "animal_doodles.csv"  # For backward compatibility

# Quick Draw dataset URLs - use the public "simplified" .ndjson per-class files
QUICK_DRAW_RAW_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"


def get_ssl_context(insecure: bool = False) -> Optional[ssl.SSLContext]:
    """
    Build an SSL context using certifi if available.
    
    Args:
        insecure: Whether to use insecure SSL connections
        
    Returns:
        SSL context or None for HTTP URLs
    """
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def download_quickdraw_file(animal: str, dest_dir: Path, insecure: bool = False) -> Path:
    """
    Download the simplified Quick Draw data file (.ndjson) for a specific animal.
    
    Args:
        animal: Animal class name
        dest_dir: Destination directory
        insecure: Whether to use insecure SSL
        
    Returns:
        Path to the downloaded file
        
    Raises:
        RuntimeError: If download fails
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Simplified files are plain .ndjson (no gzip)
    filename = f"{animal}.ndjson"
    url = f"{QUICK_DRAW_RAW_URL}{filename}"
    dest_path = dest_dir / filename

    if dest_path.exists():
        print(f"Already present: {dest_path}")
        return dest_path

    print(f"Downloading {animal} data from {url}...")

    try:
        ctx = get_ssl_context(insecure=insecure)
        with urllib.request.urlopen(url, context=ctx) as resp, open(dest_path, "wb") as out:
            out.write(resp.read())

        print(f"Downloaded: {dest_path}")
        return dest_path

    except Exception as e:
        print(f"Failed to download {animal}: {e}")
        raise RuntimeError(f"Could not download {animal} data: {e}")


def parse_ndjson_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single line from the Quick Draw NDJSON file.
    
    Args:
        line: JSON line from the file
        
    Returns:
        Parsed data dictionary or None if parsing fails
    """
    try:
        data = json.loads(line.strip())
        return {
            'word': data.get('word', ''),
            'drawing': json.dumps(data.get('drawing', [])),
            'recognized': data.get('recognized', True),
            'countrycode': data.get('countrycode', ''),
            'key_id': data.get('key_id', ''),
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Error parsing line: {e}")
        return None


def load_animal_data(animal: str, raw_dir: Path, max_samples: int = 0) -> pd.DataFrame:
    """
    Load and process data for a specific animal class.
    
    Args:
        animal: Animal class name
        raw_dir: Directory containing raw files
        max_samples: Maximum number of samples to load per class
        
    Returns:
        DataFrame with animal data
        
    Raises:
        FileNotFoundError: If raw data file doesn't exist
    """
    file_path = raw_dir / f"{animal}.ndjson"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")
    
    print(f"Processing {animal} data...")
    
    samples = []
    limit = None if int(max_samples) <= 0 else int(max_samples)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if limit is not None and len(samples) >= limit:
                break
                
            parsed = parse_ndjson_line(line)
            if parsed and parsed['recognized']:
                samples.append(parsed)
    
    df = pd.DataFrame(samples)
    print(f"  Loaded {len(df)} samples for {animal}")
    
    return df


def create_sample_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a sample dataset when downloads fail or for testing.
    
    Returns:
        Tuple of (train_df, test_df) with sample data
    """
    print("Creating sample dataset for testing...")
    print("Note: This is a minimal dataset for testing purposes only.")
    print("For full training, ensure internet connection for data download.")
    
    sample_data = []
    for i, animal in enumerate(ANIMALS):
        for j in range(100):  # 100 samples per animal for testing
            # Create simple stroke patterns (just for demo/testing)
            stroke_data = [
                [[10, 20, 30, 40], [10, 15, 20, 25]],  # Simple line
                [[40, 35, 30, 25], [25, 30, 35, 40]]   # Another line
            ]
            
            sample_data.append({
                'countrycode': 'US',
                'drawing': json.dumps(stroke_data),
                'key_id': i * 100 + j,
                'recognized': True,
                'word': animal,
            })
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Create train/test split
    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["word"]
    )
    
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"Created sample dataset: {len(train_df)} train, {len(test_df)} test samples")
    return train_df, test_df


def download_and_process_data(force: bool = False, max_samples_per_class: int = 0, insecure: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download Quick Draw data and process it into train/test splits.
    
    Args:
        force: Whether to re-download existing files
        max_samples_per_class: Maximum samples to download per class
        insecure: Whether to use insecure SSL connections
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Check if output files already exist
    if not force and TRAIN_OUTPUT.exists() and TEST_OUTPUT.exists():
        print("Output files already exist. Use --force to regenerate.")
        try:
            return pd.read_csv(TRAIN_OUTPUT), pd.read_csv(TEST_OUTPUT)
        except Exception as e:
            print(f"Error loading existing files: {e}")
            print("Will regenerate files...")
    
    all_data = []
    successful_downloads = 0
    
    try:
        # Download and process each animal class
        for animal in ANIMALS:
            try:
                # Download raw data
                download_quickdraw_file(animal, RAW_DIR, insecure=insecure)
                
                # Load and process data
                animal_df = load_animal_data(animal, RAW_DIR, max_samples_per_class)
                all_data.append(animal_df)
                successful_downloads += 1
                
            except Exception as e:
                print(f"Failed to process {animal}: {e}")
                print("Continuing with other classes...")
                continue
        
        if successful_downloads == 0:
            raise RuntimeError("No animal data could be downloaded. Creating sample dataset.")
        elif successful_downloads < len(ANIMALS):
            print(f"Warning: Only {successful_downloads}/{len(ANIMALS)} animal classes downloaded successfully.")
        
        # Combine all animal data
        df_all = pd.concat(all_data, ignore_index=True)
        
        # Keep only required columns for training
        df_clean = df_all[["word", "drawing", "recognized"]].copy()
        
        print(f"\nTotal dataset: {len(df_clean)} samples")
        print("Class distribution:")
        class_counts = df_clean["word"].value_counts()
        for animal in ANIMALS:
            count = class_counts.get(animal, 0)
            print(f"  {animal}: {count} samples")
        
        # Create stratified train/test split (matching class.py exactly)
        train_df, test_df = train_test_split(
            df_clean, test_size=0.15, random_state=42, stratify=df_clean["word"]
        )
        
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        return train_df, test_df
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Falling back to sample dataset...")
        return create_sample_dataset()


def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Save the train and test splits to CSV files atomically to avoid partial files.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
    """
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity checks before writing
    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            f"Refusing to write empty splits: train={len(train_df)}, test={len(test_df)}"
        )

    # Write to temp files first, then atomically replace
    train_tmp = TRAIN_OUTPUT.with_suffix(TRAIN_OUTPUT.suffix + ".tmp")
    test_tmp = TEST_OUTPUT.with_suffix(TEST_OUTPUT.suffix + ".tmp")
    combined_tmp = COMBINED_OUTPUT.with_suffix(COMBINED_OUTPUT.suffix + ".tmp")

    # Train
    train_df.to_csv(train_tmp, index=False)
    # Validate temp file has content
    if not train_tmp.exists() or train_tmp.stat().st_size == 0:
        raise RuntimeError("Train temp file was not written correctly.")

    # Test
    test_df.to_csv(test_tmp, index=False)
    if not test_tmp.exists() or test_tmp.stat().st_size == 0:
        raise RuntimeError("Test temp file was not written correctly.")

    # Combined
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df.to_csv(combined_tmp, index=False)
    if not combined_tmp.exists() or combined_tmp.stat().st_size == 0:
        raise RuntimeError("Combined temp file was not written correctly.")

    # Atomic replace
    import os
    os.replace(train_tmp, TRAIN_OUTPUT)
    os.replace(test_tmp, TEST_OUTPUT)
    os.replace(combined_tmp, COMBINED_OUTPUT)

    print(f"✅ Saved training data: {TRAIN_OUTPUT}")
    print(f"✅ Saved test data: {TEST_OUTPUT}")
    print(f"✅ Saved combined data: {COMBINED_OUTPUT}")

    # Verify saved files
    print(f"\nFile verification:")
    print(f"  Train file size: {TRAIN_OUTPUT.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Test file size: {TEST_OUTPUT.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Combined file size: {COMBINED_OUTPUT.stat().st_size / 1024 / 1024:.1f} MB")

    # Verify data integrity
    print(f"\nData verification:")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Classes in train: {sorted(train_df['word'].unique())}")
    print(f"  Classes in test: {sorted(test_df['word'].unique())}")


def cleanup_raw_files() -> None:
    """Clean up temporary raw files to save disk space."""
    if RAW_DIR.exists():
        print("\nCleaning up temporary files...")
        # Remove both legacy .ndjson.gz and new .ndjson if present
        for pattern in ("*.ndjson", "*.ndjson.gz"):
            for file in RAW_DIR.glob(pattern):
                try:
                    file.unlink()
                    print(f"Removed: {file}")
                except Exception:
                    pass
        
        try:
            RAW_DIR.rmdir()
            print(f"Removed empty directory: {RAW_DIR}")
        except OSError:
            pass  # Directory not empty or doesn't exist


def main(argv: list[str]) -> int:
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup RNN doodle dataset from Quick Draw",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 3.RNN/setup_data.py                    # Basic setup
  python 3.RNN/setup_data.py --force           # Force re-download
  python 3.RNN/setup_data.py --samples 10000   # More samples per class
  python 3.RNN/setup_data.py --insecure        # Use insecure SSL if behind proxy
        """
    )
    parser.add_argument("--force", action="store_true", 
                       help="Force re-download and regenerate files")
    parser.add_argument("--samples", type=int, default=0,
                       help="Maximum samples per class (0 = all, default: 0)")
    parser.add_argument("--insecure", action="store_true",
                       help="Use insecure SSL connections (for proxy issues)")
    args = parser.parse_args(argv)
    
    try:
        print("🚀 Setting up RNN doodle classification dataset...")
        print(f"Target classes: {ANIMALS}")
        print(f"Max samples per class: {'ALL' if int(args.samples) <= 0 else int(args.samples)}")
        print(f"Archive directory: {ARCHIVE_DIR}")
        print(f"Output files will be:")
        print(f"  - {TRAIN_OUTPUT}")
        print(f"  - {TEST_OUTPUT}")
        print(f"  - {COMBINED_OUTPUT}")
        
        # Download and process data
        train_df, test_df = download_and_process_data(
            force=args.force, 
            max_samples_per_class=args.samples,
            insecure=args.insecure
        )
        
        # Save splits
        save_splits(train_df, test_df)
        
        # Clean up temporary files
        cleanup_raw_files()
        
        print("\n🎉 Setup complete!")
        print("\nNext steps:")
        print("  1. Train the model: python 3.RNN/training-doodle.py")
        print("  2. Test the model: python 3.RNN/test_model.py")
        print("  3. Run the demo: python 3.RNN/app.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  - Check your internet connection")
        print("  - Try --insecure flag if behind a proxy")
        print("  - Try reducing --samples if download is too slow")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))