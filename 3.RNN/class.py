from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

# Ten target animal classes
animals: List[str] = [
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

archive_dir = Path(__file__).resolve().parent / "archive"
archive_dir.mkdir(parents=True, exist_ok=True)
local_master = archive_dir / "master_doodle_dataframe.csv"

if not local_master.exists():
    raise FileNotFoundError(
        f"Expected local dataset at {local_master}. Please place the master CSV there."
    )

print(f"Loading master dataset from: {local_master}")
df = pd.read_csv(local_master)

# Keep only required columns
needed_cols = [c for c in ["word", "drawing", "recognized"] if c in df.columns]
df = df[needed_cols].dropna(subset=["word", "drawing"]).reset_index(drop=True)

# Filter to requested animals (intersection only)
present = set(df["word"].unique())
selected = [a for a in animals if a in present]
missing = [a for a in animals if a not in present]
if missing:
    print("Warning: missing classes with no samples:", missing)
print("Using classes:", selected)

df_animals = df[df["word"].isin(selected)].reset_index(drop=True)
print("Class distribution (all):")
print(df_animals["word"].value_counts())

# Stratified split train/test
train_df, test_df = train_test_split(
    df_animals, test_size=0.15, random_state=42, stratify=df_animals["word"]
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
print("Train distribution:")
print(train_df["word"].value_counts())
print("Test distribution:")
print(test_df["word"].value_counts())

# Save splits
train_path = archive_dir / "animal_doodles_10_train.csv"
test_path = archive_dir / "animal_doodles_10_test.csv"
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print(f"Saved: {train_path}")
print(f"Saved: {test_path}")

# Remove previous combined dataset if present
legacy = archive_dir / "animal_doodles.csv"
if legacy.exists():
    legacy.unlink()
    print(f"Removed old dataset: {legacy}")
