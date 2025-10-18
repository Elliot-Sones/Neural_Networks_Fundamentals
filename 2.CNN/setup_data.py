"""
MNIST-100 dataset setup utility for the CNN project.

This script downloads the official MNIST IDX files, creates paired two-digit
combinations (00-99), and saves them as a compressed .npz file compatible
with the CNN training pipeline.
"""

from __future__ import annotations

import sys
import gzip
import urllib.request
import ssl
import shutil
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd


# Directories
BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive"
RAW_DIR = ARCHIVE_DIR / "raw"

# Mirrors (tried in order)
MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist",
    "http://yann.lecun.com/exdb/mnist",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz", 
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def get_ssl_context(insecure: bool = False) -> Optional[ssl.SSLContext]:
    """Build an SSL context using certifi if available."""
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


def download_with_fallbacks(filename: str, dest: Path, insecure: bool = False) -> None:
    """Try downloading a file from multiple mirrors with robust SSL handling."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if dest.exists():
        print(f"Already present: {dest}")
        return
        
    last_err: Optional[Exception] = None
    for base in MIRRORS:
        url = f"{base}/{filename}"
        print(f"Downloading {url} -> {dest} ...")
        try:
            ctx = get_ssl_context(insecure=insecure) if url.startswith("https://") else None
            with urllib.request.urlopen(url, context=ctx) as resp, open(dest, "wb") as out:
                shutil.copyfileobj(resp, out)
            print(f"Downloaded: {dest}")
            return
        except Exception as e:
            print(f"  Mirror failed: {e}")
            last_err = e
    raise RuntimeError(f"All mirrors failed for {filename}: {last_err}")


def read_idx_images_gz(path: Path) -> np.ndarray:
    """Read an IDX3 images .gz file into shape (num_examples, 28, 28) uint8 array."""
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError(f"Unexpected magic for images: {magic} in {path}")
        num = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        buf = f.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        arr = arr.reshape(num, rows, cols)
        return arr


def read_idx_labels_gz(path: Path) -> np.ndarray:
    """Read an IDX1 labels .gz file into shape (num_examples,) uint8 array."""
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError(f"Unexpected magic for labels: {magic} in {path}")
        num = int.from_bytes(f.read(4), "big")
        buf = f.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        if arr.shape[0] != num:
            raise ValueError("Label count mismatch")
        return arr


def to_csv(labels: np.ndarray, images: np.ndarray, dest_csv: Path) -> None:
    """Write label + pixels CSV with header compatible with training.py.

    - Column 0: label (int)
    - Columns 1..1568: p0..p1567 (uint8 pixel intensities for 28x56 images)
    """
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    # Flatten images from (N, 28, 56) to (N, 1568)
    images_flat = images.reshape(images.shape[0], -1)
    pixel_cols = [f"p{i}" for i in range(images_flat.shape[1])]
    df = pd.DataFrame(images_flat, columns=pixel_cols)
    df.insert(0, "label", labels.astype(np.int64))
    df.to_csv(dest_csv, index=False)
    print(f"Wrote {dest_csv} ({len(df)} rows)")


def create_mnist100_dataset(train_images, train_labels, test_images, test_labels):
    """Create MNIST-100 dataset by pairing single digits into two-digit numbers."""
    print("Creating MNIST-100 dataset...")
    
    # Set random seed for reproducible dataset creation
    np.random.seed(42)
    
    # Create all possible two-digit combinations (00-99)
    train_pairs = []
    train_labels_pairs = []
    test_pairs = []
    test_labels_pairs = []
    
    # For each possible two-digit number (00-99)
    for tens in range(10):
        for ones in range(10):
            label = tens * 10 + ones
            
            # Find images for tens and ones digits
            tens_indices = np.where(train_labels == tens)[0]
            ones_indices = np.where(train_labels == ones)[0]
            
            if len(tens_indices) > 0 and len(ones_indices) > 0:
                # Randomly sample pairs for training
                n_pairs = min(500, len(tens_indices), len(ones_indices))  # Limit to avoid huge dataset
                tens_sample = np.random.choice(tens_indices, n_pairs, replace=False)
                ones_sample = np.random.choice(ones_indices, n_pairs, replace=False)
                
                for t_idx, o_idx in zip(tens_sample, ones_sample):
                    # Concatenate horizontally: tens digit + ones digit
                    pair = np.concatenate([train_images[t_idx], train_images[o_idx]], axis=1)
                    train_pairs.append(pair)
                    train_labels_pairs.append(label)
            
            # Same for test set
            tens_indices_test = np.where(test_labels == tens)[0]
            ones_indices_test = np.where(test_labels == ones)[0]
            
            if len(tens_indices_test) > 0 and len(ones_indices_test) > 0:
                n_pairs_test = min(100, len(tens_indices_test), len(ones_indices_test))
                tens_sample_test = np.random.choice(tens_indices_test, n_pairs_test, replace=False)
                ones_sample_test = np.random.choice(ones_indices_test, n_pairs_test, replace=False)
                
                for t_idx, o_idx in zip(tens_sample_test, ones_sample_test):
                    pair = np.concatenate([test_images[t_idx], test_images[o_idx]], axis=1)
                    test_pairs.append(pair)
                    test_labels_pairs.append(label)
    
    train_images_100 = np.array(train_pairs, dtype=np.uint8)
    train_labels_100 = np.array(train_labels_pairs, dtype=np.int64)
    test_images_100 = np.array(test_pairs, dtype=np.uint8)
    test_labels_100 = np.array(test_labels_pairs, dtype=np.int64)
    
    print(f"Created MNIST-100: {len(train_images_100)} train, {len(test_images_100)} test samples")
    return train_images_100, train_labels_100, test_images_100, test_labels_100


def prepare_dataset(force: bool = False) -> Tuple[Path, Path]:
    """Ensure MNIST-100 CSV files exist; download/convert if needed."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    train_csv = ARCHIVE_DIR / "mnist_train.csv"
    test_csv = ARCHIVE_DIR / "mnist_test.csv"
    
    if train_csv.exists() and test_csv.exists() and not force:
        print("CSV files already exist. Use --force to regenerate.")
        return train_csv, test_csv
    
    # Download raw MNIST files
    for key, fname in FILES.items():
        download_with_fallbacks(fname, RAW_DIR / fname, insecure=("--insecure" in sys.argv))
    
    # Load MNIST data
    train_images = read_idx_images_gz(RAW_DIR / FILES["train_images"])  # (60000, 28, 28)
    train_labels = read_idx_labels_gz(RAW_DIR / FILES["train_labels"])  # (60000,)
    test_images = read_idx_images_gz(RAW_DIR / FILES["test_images"])    # (10000, 28, 28)
    test_labels = read_idx_labels_gz(RAW_DIR / FILES["test_labels"])    # (10000,)
    
    # Create MNIST-100 dataset
    train_images_100, train_labels_100, test_images_100, test_labels_100 = create_mnist100_dataset(
        train_images, train_labels, test_images, test_labels
    )
    
    # Save as CSV files only (much faster and simpler)
    to_csv(train_labels_100, train_images_100, train_csv)
    to_csv(test_labels_100, test_images_100, test_csv)
    
    # Clean up temporary files immediately
    print("\nCleaning up temporary files...")
    for fname in FILES.values():
        temp_file = RAW_DIR / fname
        if temp_file.exists():
            temp_file.unlink()
            print(f"Removed: {temp_file}")
    
    try:
        RAW_DIR.rmdir()
        print(f"Removed empty directory: {RAW_DIR}")
    except OSError:
        pass
    
    return train_csv, test_csv


def main(argv: list[str]) -> int:
    force = "--force" in argv
    try:
        train_csv, test_csv = prepare_dataset(force=force)
    except Exception as e:
        print("\n[ERROR] Failed to prepare MNIST-100 dataset:", e)
        print("If you are behind a proxy, you can manually create the CSV files\n"
              "and place them as '2.CNN/archive/mnist_train.csv' and '2.CNN/archive/mnist_test.csv'")
        return 1
    
    print("\nAll set! CSV files ready:")
    print(f" - {train_csv}")
    print(f" - {test_csv}")
    print("\nNext steps:")
    print("  python 2.CNN/training-100.py")
    print("  python 2.CNN/test_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
