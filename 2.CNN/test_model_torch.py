from __future__ import annotations

"""
Evaluate the PyTorch MNIST-100 CNN model on the held-out test set and report
top-line accuracy plus a lightweight per-class breakdown (mirrors the NumPy
test_model output style).
"""

import argparse
from pathlib import Path

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive"
DATASET_PATH = ARCHIVE_DIR / "mnist_compressed.npz"


def load_npz_test(path: Path | None = None):
    dataset_path = Path(path) if path is not None else DATASET_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset archive not found at '{dataset_path}'")
    with np.load(dataset_path) as data:
        images = data["test_images"].astype(np.float32)
        labels = data["test_labels"].astype(np.int64)
    X = images.reshape(images.shape[0], -1).T  # (features, samples)
    return X, labels


class CNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 7 * 14, 256)
        self.drop = torch.nn.Dropout(p=0.4)
        self.fc2 = torch.nn.Linear(256, 100)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


def normalize_with_stats(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    X_scaled = X / 255.0
    std_safe = np.maximum(std, 1e-8)
    return (X_scaled - mean) / std_safe


def per_class_accuracy(preds: np.ndarray, labels: np.ndarray, num_classes: int = 100):
    counts = np.zeros(num_classes, dtype=np.int32)
    correct = np.zeros(num_classes, dtype=np.int32)
    for p, y in zip(preds, labels):
        counts[y] += 1
        if p == y:
            correct[y] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.divide(correct, counts, out=np.zeros_like(correct, dtype=np.float32), where=counts > 0)
    return counts, correct, acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch MNIST-100 CNN on test split.")
    parser.add_argument("--model", type=Path, default=ARCHIVE_DIR / "trained_model_mnist100_torch.pt", help="Path to .pt model")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Path to mnist_compressed.npz")
    parser.add_argument("--topk", type=int, default=5, help="Report the K lowest-accuracy classes")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    mean = checkpoint["mean"].astype(np.float32).reshape(1, -1)  # (1, 28*56)
    std = checkpoint["std"].astype(np.float32).reshape(1, -1)

    X_test, y_test = load_npz_test(args.dataset)
    X_norm = normalize_with_stats(X_test, mean, std)
    images = torch.from_numpy(X_norm.T.reshape(-1, 1, 28, 56))

    model = CNNModel()
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()
        acc = float((preds == y_test).mean())

    print(f"\nTest accuracy: {acc * 100:.2f}% ({preds.size} samples)")

    counts, correct, class_acc = per_class_accuracy(preds, y_test)
    hardest = np.argsort(class_acc)[: args.topk]
    print("\nHardest classes (lowest accuracy):")
    for label in hardest:
        total = counts[label]
        if total == 0:
            continue
        print(f"  {label:02d}: {class_acc[label] * 100:5.2f}% ({correct[label]}/{total})")

    # Show a quick sanity check
    with torch.no_grad():
        for idx in range(min(3, preds.size)):
            p = int(preds[idx])
            t = int(y_test[idx])
            conf = float(probs[idx, p].cpu().item())
            print(f"Sample {idx:04d}: pred={p:02d}, target={t:02d}, confidence={conf:.3f}")


if __name__ == "__main__":
    main()

