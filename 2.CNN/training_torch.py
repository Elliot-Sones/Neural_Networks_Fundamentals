"""
PyTorch training script for MNIST-100 (28x56) that mirrors the from-scratch CNN:

Architecture: Conv(3x3,16) -> ReLU -> MaxPool(2x2) -> Conv(3x3,32) -> ReLU -> MaxPool(2x2)
             -> Flatten -> FC(256) -> ReLU -> Dropout(0.4) -> FC(100) -> Softmax

Key goals:
- Keep the architecture, normalization, regularization, and training loop as close as possible
  to `training-100.py` while leveraging PyTorch for significant CPU speed-ups.
- Preserve per-feature standardization (mean/std computed on training set over 28x56 pixels),
  matching the from-scratch pipeline for fair apples-to-apples behavior.
- Provide early stopping on dev accuracy and save the best checkpoint.

Outputs:
- archive/trained_model_mnist100_torch.pt (Torch checkpoint: state_dict, mean, std, config)
- Optional history CSVs (epoch, loss, train_acc, dev_acc) when --history-dir is set

This script trains quickly on CPU; if CUDA/MPS is available, you can opt-in via --device.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Constants (match scratch)
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive"
TRAIN_CSV_PATH = ARCHIVE_DIR / "mnist_train.csv"
TEST_CSV_PATH = ARCHIVE_DIR / "mnist_test.csv"

IMAGE_CHANNELS = 1
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 56
OUTPUT_DIM = 100

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4  # approximate L2
DEFAULT_DROPOUT = 0.4
DEV_SIZE = 10_000
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-3

MAX_SHIFT_PIXELS = 2
CONTRAST_JITTER_STD = 0.1


@dataclass
class TrainConfig:
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    dropout: float = DEFAULT_DROPOUT
    dev_size: int = DEV_SIZE
    early_stop_patience: int = EARLY_STOP_PATIENCE
    early_stop_min_delta: float = EARLY_STOP_MIN_DELTA
    device: str = "cpu"  # set to "cuda" or "mps" to use accelerators
    history_dir: Path | None = None
    output_model: Path = ARCHIVE_DIR / "trained_model_mnist100_torch.pt"


def _ensure_paths():
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _load_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    labels = df["label"].values.astype(np.int64)
    pixels = df.drop("label", axis=1).values.astype(np.float32)
    return pixels, labels


def load_dataset(dev_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not TRAIN_CSV_PATH.exists() or not TEST_CSV_PATH.exists():
        raise FileNotFoundError(
            "MNIST-100 CSVs not found. Run 'python setup_data.py' first to generate them."
        )
    train_pixels, train_labels = _load_csv(TRAIN_CSV_PATH)
    # Shuffle before split to form the dev set
    rng = np.random.default_rng(42)
    idx = rng.permutation(train_pixels.shape[0])
    train_pixels = train_pixels[idx]
    train_labels = train_labels[idx]

    dev_pixels = train_pixels[:dev_size]
    dev_labels = train_labels[:dev_size]
    train_pixels = train_pixels[dev_size:]
    train_labels = train_labels[dev_size:]
    return train_pixels, train_labels, dev_pixels, dev_labels


def compute_feature_normalization(train_pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Inputs come in [0,255]; convert to [0,1] then standardize per pixel
    X = train_pixels / 255.0
    # Mean/std over samples, per-feature
    mean = X.mean(axis=0, keepdims=False).astype(np.float32)
    std = X.std(axis=0, keepdims=False).astype(np.float32)
    std = np.maximum(std, 1e-8)
    return mean, std


def standardize(pixels: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = pixels / 255.0
    Z = (X - mean) / std
    return Z.astype(np.float32, copy=False)


class Mnist100Dataset(Dataset):
    def __init__(self, data_z: np.ndarray, labels: np.ndarray):
        self.data = data_z.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        # Return as 1x28x56 torch tensor already standardized
        x = torch.from_numpy(self.data[idx]).unsqueeze(0)  # (1,H,W)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def augment_batch_inplace(batch: torch.Tensor) -> None:
    """
    Lightweight augmentation operating on standardized inputs:
    - Horizontal shifts in [-MAX_SHIFT_PIXELS, MAX_SHIFT_PIXELS] with zero-padding
    - Contrast/brightness jitter with Gaussian noise (σ=CONTRAST_JITTER_STD)

    batch: (B, 1, H, W) float32
    """
    if MAX_SHIFT_PIXELS <= 0 and CONTRAST_JITTER_STD <= 0.0:
        return

    B, C, H, W = batch.shape
    device = batch.device

    if MAX_SHIFT_PIXELS > 0:
        shifts = torch.randint(
            low=-MAX_SHIFT_PIXELS, high=MAX_SHIFT_PIXELS + 1, size=(B,), device=device
        )
        for i in range(B):
            dx = int(shifts[i].item())
            if dx == 0:
                continue
            rolled = torch.roll(batch[i], shifts=(0, dx), dims=(1, 2))
            if dx > 0:
                # zero-out newly wrapped columns on the left
                rolled[:, :, :dx] = 0.0
            else:
                # zero-out newly wrapped columns on the right
                rolled[:, :, dx:] = 0.0
            batch[i].copy_(rolled)

    if CONTRAST_JITTER_STD > 0.0:
        scale = 1.0 + torch.randn((B, 1, 1, 1), device=device) * CONTRAST_JITTER_STD
        bias = torch.randn((B, 1, 1, 1), device=device) * CONTRAST_JITTER_STD
        batch.mul_(scale).add_(bias)
        batch.clamp_(-3.0, 3.0)


class ScratchLikeCNN(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # After two 2x2 pools: 28->7, 56->14, channels=32
        flattened_dim = 32 * (IMAGE_HEIGHT // 4) * (IMAGE_WIDTH // 4)
        self.fc1 = nn.Linear(flattened_dim, 256)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(256, OUTPUT_DIM)

        # He initialization akin to scratch
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().item())


def save_history_to_csv(history, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("epoch", "loss", "train_acc", "dev_acc"))
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def train_and_eval(cfg: TrainConfig):
    _ensure_paths()

    print("Loading dataset from CSV files...")
    train_px, train_y, dev_px, dev_y = load_dataset(cfg.dev_size)
    mean, std = compute_feature_normalization(train_px)

    train_z = standardize(train_px, mean, std)
    dev_z = standardize(dev_px, mean, std)

    train_ds = Mnist100Dataset(train_z, train_y)
    dev_ds = Mnist100Dataset(dev_z, dev_y)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=max(512, cfg.batch_size),
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
    )

    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    model = ScratchLikeCNN(dropout=cfg.dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best_dev_acc = -float("inf")
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        num_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            # in-place augmentation on standardized inputs
            augment_batch_inplace(xb)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item() * xb.size(0))
            num_seen += xb.size(0)

        epoch_loss = running_loss / max(1, num_seen)

        # Full-batch evaluation on train/dev
        model.eval()
        with torch.no_grad():
            # train accuracy
            train_acc_sum = 0.0
            train_n = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                train_acc_sum += accuracy_from_logits(logits, yb) * xb.size(0)
                train_n += xb.size(0)
            train_acc = train_acc_sum / max(1, train_n)

            # dev accuracy
            dev_acc_sum = 0.0
            dev_n = 0
            for xb, yb in dev_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                dev_acc_sum += accuracy_from_logits(logits, yb) * xb.size(0)
                dev_n += xb.size(0)
            dev_acc = dev_acc_sum / max(1, dev_n)

        print(
            f"Epoch {epoch:02d} - loss: {epoch_loss:.4f} - train_acc: {train_acc:.4f} - dev_acc: {dev_acc:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "train_acc": train_acc,
                "dev_acc": dev_acc,
            }
        )

        # Early stopping on dev accuracy (improvement threshold)
        if dev_acc > best_dev_acc + cfg.early_stop_min_delta:
            best_dev_acc = dev_acc
            best_state = {
                "model": model.state_dict(),
                "mean": mean,
                "std": std,
                "config": cfg.__dict__,
            }
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(
                    f"Early stopping triggered at epoch {epoch:02d}. Best dev_acc={best_dev_acc:.4f}"
                )
                break

    # Save history and best checkpoint
    if cfg.history_dir is not None:
        save_history_to_csv(history, Path(cfg.history_dir) / "train_history_torch.csv")

    if best_state is None:
        # Fallback in case no improvement was registered
        best_state = {
            "model": model.state_dict(),
            "mean": mean,
            "std": std,
            "config": cfg.__dict__,
        }

    cfg.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, cfg.output_model)
    print(f"\nSaved best model to '{cfg.output_model}'. Best dev_acc={best_dev_acc:.4f}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="PyTorch training for MNIST-100 (CPU-friendly)")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--dev-size", type=int, default=DEV_SIZE)
    p.add_argument("--early-stop-patience", type=int, default=EARLY_STOP_PATIENCE)
    p.add_argument("--early-stop-min-delta", type=float, default=EARLY_STOP_MIN_DELTA)
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Training device. Defaults to CPU for broad compatibility.",
    )
    p.add_argument("--history-dir", type=Path)
    p.add_argument(
        "--output-model",
        type=Path,
        default=ARCHIVE_DIR / "trained_model_mnist100_torch.pt",
    )
    args = p.parse_args()

    # Validate dropout range
    if not (0.0 <= float(args.dropout) < 1.0):
        raise ValueError("Dropout rate must be in [0, 1).")

    return TrainConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        dropout=float(args.dropout),
        dev_size=int(args.dev_size),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_min_delta=float(args.early_stop_min_delta),
        device=str(args.device),
        history_dir=Path(args.history_dir) if args.history_dir is not None else None,
        output_model=Path(args.output_model),
    )


def main():
    cfg = parse_args()
    train_and_eval(cfg)


if __name__ == "__main__":
    main()


