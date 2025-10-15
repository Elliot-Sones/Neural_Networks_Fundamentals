"""
Evaluation utilities for the MNIST-100 model.

This script mirrors the MNIST `test_model.py`, loading the trained parameters,
normalizing the held-out test split with the stored training statistics, and
reporting top-line accuracy plus a lightweight per-class breakdown.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import importlib.util


def _load_training_module():
    module_path = Path(__file__).resolve().parent / "training-100.py"
    spec = importlib.util.spec_from_file_location("mnist100_training", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


training_mod = _load_training_module()

ARCHIVE_DIR = training_mod.ARCHIVE_DIR
DATASET_PATH = training_mod.DATASET_PATH
forward_prop = training_mod.forward_prop
get_predictions = training_mod.get_predictions

STD_FLOOR = 1e-8


def load_model(filepath: Path | None = None):
    target = Path(filepath) if filepath is not None else ARCHIVE_DIR / "trained_model_mnist100.npz"
    if not target.exists():
        raise FileNotFoundError(
            f"Model file '{target}' not found. Train the model first with 'python training-100.py'."
        )
    loaded = np.load(target)
    params = {key: loaded[key] for key in loaded.files if key not in {"mean", "std"}}
    mean = loaded["mean"]
    std = loaded["std"]
    return params, mean, std


def load_test_split(path: Path | None = None):
    dataset_path = Path(path) if path is not None else DATASET_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset archive not found at '{dataset_path}'")

    with np.load(dataset_path) as data:
        images = data["test_images"].astype(np.float32)
        labels = data["test_labels"].astype(np.int64)

    X = images.reshape(images.shape[0], -1).T  # (features, samples)
    return X, labels


def normalize_with_stats(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    X_scaled = X / 255.0
    std_safe = np.maximum(std, STD_FLOOR)
    return (X_scaled - mean) / std_safe


def evaluate_on_test(params, mean, std, X_test, y_test):
    X_norm = normalize_with_stats(X_test, mean, std)
    _, probs = forward_prop(X_norm, params, training=False)
    predictions = get_predictions(probs)
    accuracy = float(np.mean(predictions == y_test))
    return accuracy, predictions, probs


def per_class_accuracy(predictions: np.ndarray, labels: np.ndarray, num_classes: int = 100):
    counts = np.zeros(num_classes, dtype=np.int32)
    correct = np.zeros(num_classes, dtype=np.int32)
    for pred, label in zip(predictions, labels):
        counts[label] += 1
        if pred == label:
            correct[label] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.divide(correct, counts, out=np.zeros_like(correct, dtype=np.float32), where=counts > 0)
    return counts, correct, acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate the MNIST-100 model on the held-out test split.")
    parser.add_argument("--model", type=Path, default=None, help="Path to the trained model .npz file.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the mnist_compressed.npz dataset (defaults to archive/mnist_compressed.npz).",
    )
    parser.add_argument("--topk", type=int, default=5, help="Show the top-K classes with lowest accuracy.")
    args = parser.parse_args()

    params, mean, std = load_model(args.model)
    X_test, y_test = load_test_split(args.dataset)

    accuracy, predictions, probs = evaluate_on_test(params, mean, std, X_test, y_test)
    print(f"\nTest accuracy: {accuracy * 100:.2f}% ({predictions.size} samples)")

    counts, correct, class_acc = per_class_accuracy(predictions, y_test)
    hardest = np.argsort(class_acc)[: args.topk]
    print("\nHardest classes (lowest accuracy):")
    for label in hardest:
        total = counts[label]
        if total == 0:
            continue
        print(f"  {label:02d}: {class_acc[label] * 100:5.2f}% ({correct[label]}/{total})")

    # Show a quick sanity check of model confidence on a few samples
    for idx in range(min(3, predictions.size)):
        pred = predictions[idx]
        target = y_test[idx]
        confidence = probs[pred, idx]
        print(f"Sample {idx:04d}: pred={pred:02d}, target={target:02d}, confidence={confidence:.3f}")


if __name__ == "__main__":
    main()
