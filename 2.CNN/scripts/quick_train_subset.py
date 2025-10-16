from __future__ import annotations

"""
Quick subset training to generate history curves fast.

This script uses a small train/dev subset and a few epochs to produce a
compact training history CSV for plotting loss/accuracy/generalization gap.
"""

import argparse
from pathlib import Path
import importlib.util


def _load_training_module():
    base = Path(__file__).resolve().parent.parent
    module_path = base / "training-100.py"
    spec = importlib.util.spec_from_file_location("mnist100_training", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Quick subset train for CNN curves.")
    parser.add_argument("--train-size", type=int, default=2000, help="Number of training samples.")
    parser.add_argument("--dev-size", type=int, default=1000, help="Number of dev samples.")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs to run.")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size.")
    parser.add_argument(
        "--history-out",
        type=Path,
        default=Path("2.CNN/history_quick/train_history.csv"),
        help="Output CSV path for training history.",
    )
    args = parser.parse_args()

    tm = _load_training_module()

    # Load full dataset, then carve out small subsets.
    X_train, Y_train, X_dev, Y_dev, _, _ = tm.load_data(tm.DATASET_PATH)

    tsize = max(1, int(args.train_size))
    dsize = max(1, int(args.dev_size))
    X_train = X_train[:, :tsize]
    Y_train = Y_train[:tsize]
    X_dev = X_dev[:, :dsize]
    Y_dev = Y_dev[:dsize]

    X_train, X_dev, mean, std = tm.normalize_features(X_train, X_dev)

    params, history = tm.train_model(
        X_train,
        Y_train,
        X_dev,
        Y_dev,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=tm.LEARNING_RATE,
        reg_lambda=tm.REG_LAMBDA,
        dropout_rate=tm.DROP_RATE_FC,
        early_stop_patience=tm.EARLY_STOP_PATIENCE,
        early_stop_min_delta=tm.EARLY_STOP_MIN_DELTA,
    )

    # Persist history
    out_path = Path(args.history_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tm.save_history_to_csv(history, out_path)
    print(f"Saved training history to: {out_path}")


if __name__ == "__main__":
    main()

