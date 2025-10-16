from __future__ import annotations

"""
PyTorch implementation of the MNIST-100 CNN trainer (kept separate from the
NumPy-from-scratch version). Mirrors the original architecture and training
setup for comparable accuracy and faster iteration.

Key features:
- Loads 28x56 paired MNIST from archive/mnist_compressed.npz
- Per-feature standardization using training-set mean/std (like the original)
- Lightweight augmentation: horizontal shifts (±2px) and contrast/brightness jitter (σ=0.1)
- Architecture: Conv3x3(16) → ReLU → MaxPool2 → Conv3x3(32) → ReLU → MaxPool2 → FC(256) → ReLU → Dropout(0.4) → FC(100)
- Adam with weight decay (L2) and early stopping on dev accuracy
- CSV history logging compatible with the existing plot scripts
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive"
DATASET_PATH = ARCHIVE_DIR / "mnist_compressed.npz"


# Defaults aligned with the from-scratch implementation
IMAGE_CHANNELS = 1
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 56
OUTPUT_DIM = 100
KERNEL_SIZE = 3
POOL_SIZE = 2
CONV_FILTERS = (16, 32)
FC_HIDDEN_DIM = 256


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # L2
    dropout: float = 0.4
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-3
    dev_size: int = 10_000
    max_shift_pixels: int = 2
    contrast_jitter_std: float = 0.1
    seed: int | None = 42


def set_seed(seed: int | None):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_npz_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at '{path}'")
    with np.load(path) as data:
        train_images = data["train_images"].astype(np.float32)
        train_labels = data["train_labels"].astype(np.int64)
        test_images = data["test_images"].astype(np.float32)
        test_labels = data["test_labels"].astype(np.int64)
    return train_images, train_labels, test_images, test_labels


def make_splits(train_images: np.ndarray, train_labels: np.ndarray, dev_size: int, seed: int | None):
    # Flatten to (features, samples) to mimic original pre-processing
    X_full = train_images.reshape(train_images.shape[0], -1).T
    Y_full = train_labels.copy()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X_full.shape[1])
    X_full = X_full[:, perm]
    Y_full = Y_full[perm]

    X_dev = X_full[:, :dev_size]
    Y_dev = Y_full[:dev_size]
    X_train = X_full[:, dev_size:]
    Y_train = Y_full[dev_size:]
    return X_train, Y_train, X_dev, Y_dev


def compute_norm_stats(X_train: np.ndarray):
    X_scaled = X_train / 255.0
    mean = np.mean(X_scaled, axis=1, keepdims=True)
    std = np.std(X_scaled, axis=1, keepdims=True) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X_scaled = X / 255.0
    return ((X_scaled - mean) / std).astype(np.float32)


class TorchMNIST100(Dataset):
    def __init__(
        self,
        X_flat: np.ndarray,
        Y: np.ndarray,
        *,
        mean: np.ndarray,
        std: np.ndarray,
        training: bool,
        max_shift: int,
        contrast_jitter_std: float,
    ):
        self.X = apply_standardize(X_flat, mean, std)  # (features, samples)
        self.Y = Y
        self.training = training
        self.max_shift = max_shift
        self.contrast_jitter_std = contrast_jitter_std

    def __len__(self) -> int:
        return self.Y.shape[0]

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        # img shape: (1, 28, 56) standardized
        if self.max_shift > 0:
            shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
            if shift != 0:
                img = torch.roll(img, shifts=(shift,), dims=(2,))
                if shift > 0:
                    img[:, :, :shift] = 0.0
                else:
                    img[:, :, shift:] = 0.0
        if self.contrast_jitter_std > 0.0:
            scale = 1.0 + torch.randn((), dtype=img.dtype) * self.contrast_jitter_std
            bias = torch.randn((), dtype=img.dtype) * self.contrast_jitter_std
            img = img * scale + bias
            img = torch.clamp(img, -3.0, 3.0)
        return img

    def __getitem__(self, idx: int):
        x_col = self.X[:, idx]  # (features,)
        img = torch.from_numpy(x_col).view(1, IMAGE_HEIGHT, IMAGE_WIDTH)
        if self.training:
            img = self._augment(img)
        label = int(self.Y[idx])
        return img, label


class CNNModel(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv2d(IMAGE_CHANNELS, CONV_FILTERS[0], kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE // 2)
        self.conv2 = nn.Conv2d(CONV_FILTERS[0], CONV_FILTERS[1], kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE // 2)
        # After two 2x2 pools on 28x56: 7 x 14
        self.flatten_dim = CONV_FILTERS[1] * (IMAGE_HEIGHT // (POOL_SIZE ** 2)) * (IMAGE_WIDTH // (POOL_SIZE ** 2))
        self.fc1 = nn.Linear(self.flatten_dim, FC_HIDDEN_DIM)
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(FC_HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=POOL_SIZE, stride=POOL_SIZE)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=POOL_SIZE, stride=POOL_SIZE)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def save_history(history, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("epoch", "loss", "train_acc", "dev_acc"))
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def train_torch(config: TrainConfig, *, history_out: Path, model_out: Path):
    set_seed(config.seed)

    device = (
        torch.device("mps") if torch.backends.mps.is_available() else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    )

    train_images, train_labels, _, _ = load_npz_dataset(DATASET_PATH)
    X_train, Y_train, X_dev, Y_dev = make_splits(train_images, train_labels, config.dev_size, config.seed)
    mean, std = compute_norm_stats(X_train)

    ds_train = TorchMNIST100(
        X_train, Y_train, mean=mean, std=std, training=True,
        max_shift=config.max_shift_pixels, contrast_jitter_std=config.contrast_jitter_std,
    )
    ds_dev = TorchMNIST100(
        X_dev, Y_dev, mean=mean, std=std, training=False,
        max_shift=config.max_shift_pixels, contrast_jitter_std=config.contrast_jitter_std,
    )

    dl_train = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True, num_workers=0)
    dl_dev = DataLoader(ds_dev, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = CNNModel(dropout=config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_dev = -np.inf
    best_state = None
    patience = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        for x, y in dl_train:
            x = x.to(device)
            y = torch.as_tensor(y, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            running_correct += int((preds == y).sum().item())
            total += batch_size
        train_loss = running_loss / max(1, total)
        train_acc = running_correct / max(1, total)

        # Eval on dev only (to speed up epochs)
        model.eval()
        with torch.no_grad():
            dv_correct = 0
            dv_count = 0
            for x, y in dl_dev:
                x = x.to(device)
                y = torch.as_tensor(y, device=device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                dv_correct += int((preds == y).sum().item())
                dv_count += y.size(0)
            dev_acc = dv_correct / max(1, dv_count)

        print(f"Epoch {epoch:02d} - loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - dev_acc: {dev_acc:.4f}")
        history.append({"epoch": epoch, "loss": float(train_loss), "train_acc": float(train_acc), "dev_acc": float(dev_acc)})

        # Early stopping on dev accuracy
        if dev_acc > best_dev + config.early_stop_min_delta:
            best_dev = dev_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch:02d}. Best dev_acc={best_dev:.4f}")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Persist
    save_history(history, history_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean.squeeze().reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH),
        "std": std.squeeze().reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH),
        "config": config.__dict__,
    }, model_out)
    print(f"Saved history -> {history_out}\nSaved model -> {model_out}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch trainer for MNIST-100 CNN (separate from NumPy version)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--dev-size", type=int, default=10_000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--max-shift", type=int, default=2)
    parser.add_argument("--contrast-jitter-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-dir", type=Path, default=BASE_DIR / "history_torch")
    parser.add_argument("--output-model", type=Path, default=ARCHIVE_DIR / "trained_model_mnist100_torch.pt")
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        early_stop_patience=args.patience,
        early_stop_min_delta=args.min_delta,
        dev_size=args.dev_size,
        max_shift_pixels=args.max_shift,
        contrast_jitter_std=args.contrast_jitter_std,
        seed=args.seed,
    )
    history_out = Path(args.history_dir) / "train_history.csv"
    train_torch(cfg, history_out=history_out, model_out=Path(args.output_model))


if __name__ == "__main__":
    main()
