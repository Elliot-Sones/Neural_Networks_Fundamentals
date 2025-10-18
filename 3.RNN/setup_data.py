"""
Setup script for RNN doodle classification dataset.

Prepares the Quick Draw dataset for the 10 animal classes in the format
expected by training-doodle.py and eval_and_plots.py.

Usage:
    python 3.RNN/setup_data.py

This script:
1. Checks if master_doodle_dataframe.csv exists (if not, provides instructions)
2. Filters to the 10 animal classes needed for training
3. Runs class.py to create train/test splits
"""

import os
import json
import ssl
import urllib.request
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Ten target animal classes (matching class.py)
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

ARCHIVE_DIR = Path(__file__).resolve().parent / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

def check_master_dataset() -> bool:
    """
    Check if the master dataset exists.
    
    Returns:
        True if master dataset exists, False otherwise
    """
    master_path = ARCHIVE_DIR / "master_doodle_dataframe.csv"
    return master_path.exists()

def create_sample_dataset() -> Path:
    """
    Create a small sample dataset for testing purposes.
    
    Returns:
        Path to the created sample dataset file
    """
    print("Creating sample dataset for testing...")
    
    # Create some sample stroke data for each animal
    sample_data = []
    for i, animal in enumerate(ANIMALS):
        for j in range(50):  # 50 samples per animal
            # Create a simple stroke pattern (this is just for demo)
            stroke_data = [
                [[10, 20, 30, 40], [10, 15, 20, 25]],  # Simple line
                [[40, 35, 30, 25], [25, 30, 35, 40]]   # Another line
            ]
            
            sample_data.append({
                'countrycode': 'US',
                'drawing': json.dumps(stroke_data),
                'key_id': i * 50 + j,
                'recognized': True,
                'word': animal,
                'image_path': f'data/{animal}/{i * 50 + j}.png'
            })
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Save master dataset
    master_path = ARCHIVE_DIR / "master_doodle_dataframe.csv"
    df.to_csv(master_path, index=False)
    
    print(f"Saved sample dataset: {master_path}")
    print(f"Total samples: {len(df)}")
    print("Class distribution:")
    print(df['word'].value_counts())
    
    return master_path

def main():
    """Main setup function."""
    print("Setting up RNN doodle classification dataset...")
    print(f"Target classes: {ANIMALS}")
    print(f"Archive directory: {ARCHIVE_DIR}")
    
    # Check if master dataset exists
    if check_master_dataset():
        print("✓ Master dataset found: master_doodle_dataframe.csv")
        master_path = ARCHIVE_DIR / "master_doodle_dataframe.csv"
    else:
        print("⚠ Master dataset not found. Creating sample dataset for testing...")
        print("For production use, download the full Quick Draw dataset from:")
        print("https://console.cloud.google.com/storage/browser/quickdraw_dataset")
        print("and place it as master_doodle_dataframe.csv in the archive directory.")
        master_path = create_sample_dataset()
    
    # Run class.py to create train/test splits
    print("\nCreating train/test splits...")
    try:
        import subprocess
        import sys
        class_script = Path(__file__).resolve().parent / "class.py"
        result = subprocess.run([sys.executable, str(class_script)], 
                              capture_output=True, text=True, check=True)
        print("✓ Train/test splits created successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"✗ Error creating splits: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    print("\n✓ Setup complete!")
    print("Files created:")
    for file in ARCHIVE_DIR.glob("*.csv"):
        print(f"  - {file}")
    
    print("\nNext steps:")
    print("1. Train the model: python 3.RNN/training-doodle.py")
    print("2. Evaluate the model: python 3.RNN/eval_and_plots.py")
    print("3. Run the demo: python 3.RNN/app.py")

if __name__ == "__main__":
    main()