"""
Test-only evaluation for the 10-class RNN doodle classifier.

This script loads a trained GRU checkpoint and evaluates it on the held-out
test split CSV. It reports top-1 and top-3 accuracy and can optionally print a
per-class breakdown. No plots are generated.

Defaults:
- Checkpoint: 3.RNN/archive/rnn_animals_last.pt
- Test CSV : 3.RNN/archive/animal_doodles_10_test.csv

Usage examples:
  python 3.RNN/test_model.py
  python 3.RNN/test_model.py --per_class
  python 3.RNN/test_model.py --ckpt 3.RNN/archive/rnn_animals_best.pt --test_csv 3.RNN/archive/animal_doodles_10_test.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# Import training utilities (model class, sequence parser, device helper)
import importlib.util as _importlib_util


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "training-doodle.py"
SPEC = _importlib_util.spec_from_file_location("rnn_train_mod", str(TRAIN_PATH))
rnn_mod = _importlib_util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(rnn_mod)

# Reused APIs from training
GRUClassifier = rnn_mod.GRUClassifier
parse_drawing_to_seq = rnn_mod.parse_drawing_to_seq
get_device = rnn_mod.get_device


def load_model(ckpt_path: Path) -> Tuple[torch.nn.Module, Dict[int, str]]:
    """Load a trained GRU model and index→class mapping from a checkpoint."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model_state"]
    class_to_idx = ckpt["class_to_idx"]
    cfg_dict: Dict[str, Any] = ckpt.get("config", {})
    num_classes = len(class_to_idx)

    model = GRUClassifier(
        input_size=int(cfg_dict.get("input_size", 3)),
        hidden_size=int(cfg_dict.get("hidden_size", 192)),
        num_layers=int(cfg_dict.get("num_layers", 2)),
        bidirectional=bool(cfg_dict.get("bidirectional", True)),
        dropout=float(cfg_dict.get("dropout", 0.2)),
        num_classes=num_classes,
        use_packing=bool(cfg_dict.get("use_packing", True)),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return model, idx_to_class


def pad_batch(seqs: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences [T,3] into [B,T_max,3] and return lengths."""
    lengths = [s.shape[0] for s in seqs]
    max_len = int(max(1, max(lengths) if lengths else 1))
    x = torch.zeros((len(seqs), max_len, 3), dtype=torch.float32)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        if L > 0:
            x[i, :L, :] = torch.from_numpy(s)
    return x, torch.tensor(lengths, dtype=torch.long)


@torch.no_grad()
def evaluate_on_csv(
    model: torch.nn.Module,
    idx_to_class: Dict[int, str],
    csv_path: Path,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Evaluate the model on a CSV with columns ["word", "drawing"].
    Returns metrics, predictions, and per-class stats.
    """
    df = pd.read_csv(csv_path)
    df = df[["word", "drawing"]].dropna().reset_index(drop=True)

    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    seqs: List[np.ndarray] = []
    labels: List[int] = []
    for _, row in df.iterrows():
        word = str(row["word"])  # ground-truth label
        if word not in class_to_idx:
            continue  # skip out-of-scope classes
        seq = parse_drawing_to_seq(row["drawing"])  # [T,3]
        seqs.append(seq)
        labels.append(class_to_idx[word])

    device = get_device()
    model = model.to(device)

    N = len(seqs)
    C = len(class_names)
    confidences: List[float] = []
    correct_flags: List[int] = []
    preds: List[int] = []
    trues: List[int] = labels.copy()
    cm = np.zeros((C, C), dtype=np.int32)

    top1_correct = 0
    top3_correct = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_seqs = seqs[start:end]
        batch_labels = labels[start:end]
        x, lengths = pad_batch(batch_seqs)
        x = x.to(device)
        lengths = lengths.to(device)
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        top1 = probs.argmax(axis=1)
        top3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
        for i, y_true in enumerate(batch_labels):
            y_pred = int(top1[i])
            p = float(probs[i, y_pred])
            confidences.append(p)
            is_correct = int(y_pred == y_true)
            correct_flags.append(is_correct)
            preds.append(y_pred)
            cm[y_true, y_pred] += 1
            top1_correct += is_correct
            if y_true in top3[i].tolist():
                top3_correct += 1

    acc1 = top1_correct / max(1, N)
    acc3 = top3_correct / max(1, N)

    # Per-class accuracy
    per_class_acc: Dict[str, Dict[str, float | int | None]] = {}
    for c in range(C):
        total = int(cm[c, :].sum())
        correct = int(cm[c, c])
        per_class_acc[class_names[c]] = {
            "total": total,
            "correct": correct,
            "accuracy": float(correct) / total if total > 0 else None,
        }

    return {
        "acc1": float(acc1),
        "acc3": float(acc3),
        "confidences": confidences,
        "correct_flags": correct_flags,
        "preds": preds,
        "trues": trues,
        "class_names": class_names,
        "confusion_matrix": cm,
        "per_class_acc": per_class_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RNN model on the test CSV (no plots).")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=BASE_DIR / "archive" / "rnn_animals_last.pt",
        help="Path to the trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=BASE_DIR / "archive" / "animal_doodles_10_test.csv",
        help="Path to the test CSV (word,drawing).",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument("--per_class", action="store_true", help="Print per-class accuracy breakdown.")
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{args.ckpt}'. Train or copy the model file first.")
    if not args.test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found at '{args.test_csv}'. Run data setup first.")

    model, idx_to_class = load_model(args.ckpt)
    results = evaluate_on_csv(model, idx_to_class, args.test_csv, batch_size=args.batch_size)

    acc1 = results["acc1"]
    acc3 = results["acc3"]
    num_samples = len(results["trues"]) if isinstance(results.get("trues"), list) else 0

    print(f"\nTest samples: {num_samples}")
    print(f"Top-1 accuracy: {acc1 * 100:.2f}%")
    print(f"Top-3 accuracy: {acc3 * 100:.2f}%")

    if args.per_class:
        per = results["per_class_acc"]
        rows: List[Tuple[str, float]] = []
        for cname, stats in per.items():
            acc = stats["accuracy"] if stats["accuracy"] is not None else 0.0
            rows.append((cname, float(acc)))
        rows.sort(key=lambda r: r[1])
        print("\nPer-class accuracy (ascending):")
        for cname, acc in rows:
            total = per[cname]["total"]
            correct = per[cname]["correct"]
            print(f"  {cname:12s}: {acc * 100:6.2f}%  ({correct}/{total})")


if __name__ == "__main__":
    main()


