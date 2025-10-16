"""
Evaluate the 10-class RNN on the held-out test split and generate plots:
- Confusion matrix
- Per-class accuracy bar chart
- Reliability diagram (confidence vs. accuracy)
- Confidence histogram

Outputs are written to 3.RNN/archive/plots/ and a JSON/CSV summary is saved.

Usage:
  python3 3.RNN/eval_and_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Import training utilities (model + parser)
import importlib.util as _importlib_util

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "training-doodle.py"
SPEC = _importlib_util.spec_from_file_location("rnn_train_mod", str(TRAIN_PATH))
rnn_mod = _importlib_util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(rnn_mod)

GRUClassifier = rnn_mod.GRUClassifier
parse_drawing_to_seq = rnn_mod.parse_drawing_to_seq
get_device = rnn_mod.get_device


def load_model(ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model_state"]
    class_to_idx = ckpt["class_to_idx"]
    cfg_dict = ckpt.get("config", {})
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
    lengths = [s.shape[0] for s in seqs]
    maxL = int(max(1, max(lengths)))
    B = len(seqs)
    x = torch.zeros((B, maxL, 3), dtype=torch.float32)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        if L > 0:
            x[i, :L, :] = torch.from_numpy(s)
    return x, torch.tensor(lengths, dtype=torch.long)


@torch.no_grad()
def evaluate_on_csv(
    model: torch.nn.Module,
    idx_to_class: dict[int, str],
    csv_path: Path,
    batch_size: int = 128,
) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    df = df[["word", "drawing"]].dropna().reset_index(drop=True)

    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    seqs: List[np.ndarray] = []
    labels: List[int] = []
    for _, row in df.iterrows():
        word = str(row["word"])  # label
        if word not in class_to_idx:
            # Skip any out-of-scope class just in case
            continue
        seq = parse_drawing_to_seq(row["drawing"])
        seqs.append(seq)
        labels.append(class_to_idx[word])

    device = get_device()
    model = model.to(device)

    N = len(seqs)
    C = len(class_names)
    confs: List[float] = []
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
            confs.append(p)
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
    per_class_acc = {}
    for c in range(C):
        total = int(cm[c, :].sum())
        correct = int(cm[c, c])
        per_class_acc[class_names[c]] = {
            "total": total,
            "correct": correct,
            "accuracy": float(correct / total) if total > 0 else None,
        }

    return {
        "acc1": float(acc1),
        "acc3": float(acc3),
        "confidences": confs,
        "correct_flags": correct_flags,
        "preds": preds,
        "trues": trues,
        "class_names": class_names,
        "confusion_matrix": cm,
        "per_class_acc": per_class_acc,
    }


def plot_confusion(cm: np.ndarray, classes: List[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    # Annotate for small matrices
    thresh = cm.max() * 0.5 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(j, i, str(val), ha="center", va="center", color="white" if val > thresh else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_class(per_class_acc: dict, out_path: Path):
    classes = list(per_class_acc.keys())
    accs = [per_class_acc[c]["accuracy"] if per_class_acc[c]["accuracy"] is not None else 0.0 for c in classes]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.bar(classes, accs, color="#4C78A8")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-class accuracy")
    ax.set_xticklabels(classes, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_reliability(confs: List[float], correct_flags: List[int], out_path: Path, bins: int = 10):
    confs = np.asarray(confs, dtype=np.float32)
    correct = np.asarray(correct_flags, dtype=np.int32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2.0
    bin_idx = np.digitize(confs, edges, right=True) - 1
    bin_acc = []
    bin_conf = []
    for b in range(bins):
        mask = bin_idx == b
        if not np.any(mask):
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
        else:
            bin_acc.append(float(correct[mask].mean()))
            bin_conf.append(float(confs[mask].mean()))

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot(bin_conf, bin_acc, marker="o", label="model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_conf_hist(confs: List[float], out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
    ax.hist(confs, bins=20, range=(0, 1), color="#72B7B2")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Top-1 confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence distribution")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    archive = BASE_DIR / "archive"
    ckpt_path = archive / "rnn_animals_best.pt"
    test_csv = archive / "animal_doodles_10_test.csv"
    plots_dir = archive / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model, idx_to_class = load_model(ckpt_path)
    results = evaluate_on_csv(model, idx_to_class, test_csv)

    # Save metrics
    summary = {
        "acc1": results["acc1"],
        "acc3": results["acc3"],
        "num_samples": len(results["trues"]),
        "classes": results["class_names"],
    }
    (archive / "rnn_eval_summary.json").write_text(json.dumps(summary, indent=2))
    # Per-class CSV
    per_rows = [
        {"class": c, **results["per_class_acc"][c]} for c in results["class_names"]
    ]
    pd.DataFrame(per_rows).to_csv(archive / "rnn_per_class_accuracy.csv", index=False)

    # Plots
    plot_confusion(results["confusion_matrix"], results["class_names"], plots_dir / "rnn_confusion_matrix.png")
    plot_per_class(results["per_class_acc"], plots_dir / "rnn_per_class_accuracy.png")
    plot_reliability(results["confidences"], results["correct_flags"], plots_dir / "rnn_reliability.png")
    plot_conf_hist(results["confidences"], plots_dir / "rnn_confidence_hist.png")

    print("Saved:")
    for p in (
        archive / "rnn_eval_summary.json",
        archive / "rnn_per_class_accuracy.csv",
        plots_dir / "rnn_confusion_matrix.png",
        plots_dir / "rnn_per_class_accuracy.png",
        plots_dir / "rnn_reliability.png",
        plots_dir / "rnn_confidence_hist.png",
    ):
        print(" -", p)
    print(f"acc@1={summary['acc1']:.3f}  acc@3={summary['acc3']:.3f}  samples={summary['num_samples']}")


if __name__ == "__main__":
    main()

