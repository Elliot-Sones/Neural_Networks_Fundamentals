from __future__ import annotations

"""
Plot loss, accuracy, and generalization gap from a training history CSV.

Outputs three PNGs suitable for quick before/after comparisons in the README.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_history_csv(path: Path):
    epochs, loss, train_acc, dev_acc = [], [], [], []
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            loss.append(float(row["loss"]))
            train_acc.append(float(row["train_acc"]))
            dev_acc.append(float(row["dev_acc"]))
    return epochs, loss, train_acc, dev_acc


def plot_curves(epochs, loss, train_acc, dev_acc, outdir: Path, prefix: str):
    outdir.mkdir(parents=True, exist_ok=True)

    # Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_loss.png", dpi=140)
    plt.close()

    # Accuracy curves
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_acc, label="Train acc")
    plt.plot(epochs, dev_acc, label="Dev acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Accuracy Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_accuracy.png", dpi=140)
    plt.close()

    # Generalization gap
    gap = [ta - da for ta, da in zip(train_acc, dev_acc)]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, gap, label="Train − Dev")
    plt.xlabel("Epoch")
    plt.ylabel("Gap")
    plt.title("CNN Generalization Gap")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_gap.png", dpi=140)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot history CSV into comparison images.")
    parser.add_argument("--history", type=Path, required=True, help="Path to train_history.csv")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory to save images")
    parser.add_argument("--prefix", type=str, default="cnn_iteration1", help="Filename prefix")
    args = parser.parse_args()

    epochs, loss, tr, dv = load_history_csv(args.history)
    plot_curves(epochs, loss, tr, dv, args.outdir, args.prefix)
    print(f"Saved images to: {args.outdir}")


if __name__ == "__main__":
    main()

