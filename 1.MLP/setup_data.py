"""
MNIST dataset setup utility for the MLP project.

This script downloads the official MNIST IDX files, converts them to CSV
compatible with this repository's training pipeline, and places them under
`1.MLP/archive/` as `mnist_train.csv` and `mnist_test.csv`.

Why CSV? The training code expects CSV with the label in the first column
followed by the 784 pixel values. Keeping the format explicit avoids any
hidden preprocessing and makes the data easy to inspect.
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


# Mirrors (tried in order). First is HTTPS (Google), second is HTTP (LeCun fallback).
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
    """
    Build an SSL context using certifi if available.
    Returns None for insecure or for non-HTTPS URLs.
    """
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    try:
        import certifi  # type: ignore
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        # Fallback to default system context
        return ssl.create_default_context()


def download_with_fallbacks(filename: str, dest: Path, insecure: bool = False) -> None:
    """Try downloading a file from multiple mirrors with robust SSL handling."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Create more descriptive filenames
    better_names = {
        "train-images-idx3-ubyte.gz": "mnist_train_images_60k.gz",
        "train-labels-idx1-ubyte.gz": "mnist_train_labels_60k.gz",
        "t10k-images-idx3-ubyte.gz": "mnist_test_images_10k.gz", 
        "t10k-labels-idx1-ubyte.gz": "mnist_test_labels_10k.gz"
    }
    
    better_name = better_names.get(filename, filename)
    better_dest = dest.parent / better_name
    
    if better_dest.exists():
        print(f"Already present: {better_dest}")
        return
        
    last_err: Optional[Exception] = None
    for base in MIRRORS:
        url = f"{base}/{filename}"
        print(f"Downloading {url} -> {better_dest} ...")
        try:
            ctx = get_ssl_context(insecure=insecure) if url.startswith("https://") else None
            with urllib.request.urlopen(url, context=ctx) as resp, open(dest, "wb") as out:
                shutil.copyfileobj(resp, out)
            # Rename to better filename after successful download
            dest.rename(better_dest)
            print(f"Downloaded and renamed to: {better_dest}")
            return
        except Exception as e:
            print(f"  Mirror failed: {e}")
            last_err = e
    raise RuntimeError(f"All mirrors failed for {filename}: {last_err}")


def read_idx_images_gz(path: Path) -> np.ndarray:
    """Read an IDX3 images .gz file into shape (num_examples, 784) uint8 array."""
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError(f"Unexpected magic for images: {magic} in {path}")
        num = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        buf = f.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        arr = arr.reshape(num, rows * cols)
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
    - Columns 1..784: p0..p783 (uint8 pixel intensities)
    """
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    pixel_cols = [f"p{i}" for i in range(images.shape[1])]
    df = pd.DataFrame(images, columns=pixel_cols)
    df.insert(0, "label", labels.astype(np.int64))
    df.to_csv(dest_csv, index=False)
    print(f"Wrote {dest_csv} ({len(df)} rows)")


def prepare_csvs(force: bool = False) -> Tuple[Path, Path]:
    """Ensure MNIST CSVs exist; download/convert if needed.

    Returns
    -------
    (train_csv, test_csv)
    """
    train_csv = ARCHIVE_DIR / "mnist_train.csv"
    test_csv = ARCHIVE_DIR / "mnist_test.csv"

    if train_csv.exists() and test_csv.exists() and not force:
        print("CSV files already exist. Use --force to regenerate.")
        return train_csv, test_csv

    # Download raw IDX.gz files (try mirrors, HTTPS first, then HTTP fallback)
    better_names = {
        "train-images-idx3-ubyte.gz": "mnist_train_images_60k.gz",
        "train-labels-idx1-ubyte.gz": "mnist_train_labels_60k.gz",
        "t10k-images-idx3-ubyte.gz": "mnist_test_images_10k.gz", 
        "t10k-labels-idx1-ubyte.gz": "mnist_test_labels_10k.gz"
    }
    
    for key, fname in FILES.items():
        download_with_fallbacks(fname, RAW_DIR / fname, insecure=("--insecure" in sys.argv))

    # Load and convert using better filenames
    train_images = read_idx_images_gz(RAW_DIR / better_names[FILES["train_images"]])  # (60000,784)
    train_labels = read_idx_labels_gz(RAW_DIR / better_names[FILES["train_labels"]])  # (60000,)
    test_images = read_idx_images_gz(RAW_DIR / better_names[FILES["test_images"]])    # (10000,784)
    test_labels = read_idx_labels_gz(RAW_DIR / better_names[FILES["test_labels"]])    # (10000,)

    # Basic sanity checks
    assert train_images.shape[0] == train_labels.shape[0] == 60000
    assert test_images.shape[0] == test_labels.shape[0] == 10000
    assert train_images.shape[1] == test_images.shape[1] == 784

    to_csv(train_labels, train_images, train_csv)
    to_csv(test_labels, test_images, test_csv)
    
    # Clean up temporary files - we only need the CSV files
    print("\nCleaning up temporary files...")
    for fname in FILES.values():
        temp_file = RAW_DIR / fname
        if temp_file.exists():
            temp_file.unlink()
            print(f"Removed: {temp_file}")
    
    # Also remove the better-named files
    for better_name in better_names.values():
        better_file = RAW_DIR / better_name
        if better_file.exists():
            better_file.unlink()
            print(f"Removed: {better_file}")
    
    # Remove the raw directory if empty
    try:
        RAW_DIR.rmdir()
        print(f"Removed empty directory: {RAW_DIR}")
    except OSError:
        pass  # Directory not empty or doesn't exist
    
    return train_csv, test_csv


def main(argv: list[str]) -> int:
    force = "--force" in argv
    try:
        train_csv, test_csv = prepare_csvs(force=force)
    except Exception as e:
        print("\n[ERROR] Failed to prepare MNIST CSVs:", e)
        print("If you are behind a proxy or need to use a different mirror,\n"
              "you can manually place 'mnist_train.csv' and 'mnist_test.csv'\n"
              "under '1.MLP/archive/' and rerun training.")
        return 1

    print("\nAll set! Files ready:")
    print(f" - {train_csv}")
    print(f" - {test_csv}")
    print("\nNext steps:")
    print("  python 1.MLP/training.py")
    print("  python 1.MLP/test_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


