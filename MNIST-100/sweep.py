"""
Hyperparameter sweep driver for MNIST-100 training.

Loads the training module dynamically (its filename contains a hyphen) and
evaluates several learning rate / regularization combinations, reporting the
best dev accuracy observed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TRAINING_MODULE_PATH = BASE_DIR / "training-100.py"

# Candidate grids – adjust or extend as needed.
LEARNING_RATES = [1e-4, 5e-4, 1e-3]
REG_LAMBDAS = [1e-4, 5e-4, 1e-3]
SWEEP_EPOCHS = 5


def load_training_module():
    """Import the training script despite the hyphenated filename."""
    spec = importlib.util.spec_from_file_location("training_100", TRAINING_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    training = load_training_module()

    results = []
    best = None

    for lr in LEARNING_RATES:
        for reg in REG_LAMBDAS:
            print(f"\n=== Running lr={lr:.1e}, reg={reg:.1e} ===")
            params, dev_acc, mean, std, history = training.train_once(
                learning_rate=lr,
                reg_lambda=reg,
                epochs=SWEEP_EPOCHS,
            )
            results.append((lr, reg, dev_acc))
            print(f"Dev accuracy: {dev_acc:.4f}")

            if not best or dev_acc > best["acc"]:
                best = {
                    "lr": lr,
                    "reg": reg,
                    "acc": dev_acc,
                    "params": params,
                    "mean": mean,
                    "std": std,
                    "history": history,
                }

    print("\n=== Summary ===")
    for lr, reg, acc in sorted(results, key=lambda item: item[2], reverse=True):
        print(f"lr={lr:.1e}, reg={reg:.1e}, dev_acc={acc:.4f}")

    if best:
        print(
            "\nBest configuration -> "
            f"lr={best['lr']:.1e}, reg={best['reg']:.1e}, dev_acc={best['acc']:.4f}"
        )

        # Uncomment the lines below to retrain longer and save the model:
        # params, dev_acc, mean, std, history = training.train_once(
        #     learning_rate=best["lr"],
        #     reg_lambda=best["reg"],
        #     epochs=training.EPOCHS,
        # )
        # training.save_model(params, mean, std, training.ARCHIVE_DIR / "trained_model_best.npz")


if __name__ == "__main__":
    main()
