"""
Section 1: This section manages imports, config, and device selection
"""
import os
import ast
import math
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    csv_path: str = str((Path(__file__).resolve().parent / "archive" / "animal_doodles.csv"))
    recognized_only: bool = True
    min_seq_len: int = 6             # drop drawings with fewer than 6 moves
    max_len: int = 250               # cap sequence length for speed/memory (faster first run)
    per_class_limit: int = 1500      # limit per class for faster training; scale up later
    # Optionally reduce number of classes to speed up training.
    # If set, keeps only these classes (exact names in CSV's 'word' column).
    allowed_classes: Optional[List[str]] = None
    # Or keep the top-K most frequent classes. Ignored if allowed_classes is provided.
    num_classes_limit: Optional[int] = None
    test_size: float = 0.15          # stratified split fraction
    batch_size: int = 64
    num_workers: int = 2             # try 2 workers for faster loading on M2
    input_size: int = 3              # [dx, dy, pen_lift]
    hidden_size: int = 192
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.2
    lr: float = 3e-3
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    epochs: int = 12                 # fewer epochs for quicker first pass
    patience: int = 3                # earlier stop on plateau
    grad_clip: float = 1.0
    use_packing: bool = True         # if you see MPS issues, set to False
    out_dir: str = str((Path(__file__).resolve().parent / "archive"))
    # Speedups
    use_cache: bool = True
    cache_dir: str = str((Path(__file__).resolve().parent / "archive" / "seq_cache_v1"))
    n_buckets: int = 8               # length buckets for faster batches
    scheduler_type: str = "onecycle"  # 'onecycle' or 'plateau'
    resume_from_best: bool = True

"""
Section 2: This is a helper function to parse the drawing string into a sequence of [dx, dy, pen_lift] 
where dx and dy are the normmalized (-1, 1) movements of the pen from the previous point
and pen_lift is 1 at the end of a stroke, else 0
"""

def parse_drawing_to_seq(drawing_str: str) -> np.ndarray:

    # Prefer fast JSON parsing; fall back to ast for robustness
    try:
        strokes = json.loads(drawing_str)
    except Exception:
        try:
            strokes = ast.literal_eval(drawing_str)
        except Exception:
            return np.zeros((0, 3), dtype=np.float32)

    seq_parts = []
    for stroke in strokes:
        if not isinstance(stroke, (list, tuple)) or len(stroke) != 2:
            continue
        x, y = stroke
        n = min(len(x), len(y))
        if n < 2:
            continue
        x = np.asarray(x[:n], dtype=np.int16)
        y = np.asarray(y[:n], dtype=np.int16)

        dx = np.diff(x).astype(np.float32) / 255.0
        dy = np.diff(y).astype(np.float32) / 255.0
        if dx.size == 0:
            continue
        pen = np.zeros_like(dx, dtype=np.float32)
        pen[-1] = 1.0  # mark end-of-stroke

        seq_parts.append(np.stack([dx, dy, pen], axis=1))

    if not seq_parts:
        return np.zeros((0, 3), dtype=np.float32)

    seq = np.concatenate(seq_parts, axis=0)
    # Optional: clip extreme deltas for robustness
    seq[:, :2] = np.clip(seq[:, :2], -1.0, 1.0)
    return seq.astype(np.float32)

"""
Section 3: Prepare the data set for training. How to read one doodle and how to bundle them into batches.
"""
class SketchDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, class_to_idx: dict, max_len: int, min_seq_len: int = 6):
        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.max_len = max_len
        self.min_seq_len = min_seq_len

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        seq = parse_drawing_to_seq(row["drawing"])
        label = self.class_to_idx[row["word"]]
        return seq, label


class SequenceCache:
    """Memory-mapped cache of concatenated sequences with offsets.

    Stores truncated sequences (<= max_len) contiguously in a .npy file.
    Offsets/lengths are stored in a small .npz file.
    """
    def __init__(self, data_path: Path, meta_path: Path):
        self.data_path = data_path
        self.meta_path = meta_path
        meta = np.load(str(meta_path))
        self.offsets: np.ndarray = meta["offsets"]
        self.lengths: np.ndarray = meta["lengths"]
        self.input_size: int = int(meta["input_size"]) if "input_size" in meta else 3
        self.max_len_stored: int = int(meta["max_len"]) if "max_len" in meta else int(self.lengths.max(initial=0))
        # lazy memmap; load in read-only mode
        self.data = np.load(str(data_path), mmap_mode="r")

    def __len__(self):
        return self.offsets.shape[0]

    def get_seq(self, idx: int) -> np.ndarray:
        start = int(self.offsets[idx])
        L = int(self.lengths[idx])
        if L <= 0:
            return np.zeros((0, self.input_size), dtype=np.float32)
        return np.asarray(self.data[start:start + L], dtype=np.float32)


def build_or_load_cache(frame: pd.DataFrame, split: str, cfg: Config) -> SequenceCache:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / f"{split}_data.npy"
    meta_path = cache_dir / f"{split}_meta.npz"

    if data_path.exists() and meta_path.exists():
        # Validate compatibility
        meta = np.load(str(meta_path))
        meta_max_len = int(meta["max_len"]) if "max_len" in meta else None
        meta_input = int(meta["input_size"]) if "input_size" in meta else None
        # Also require the number of cached sequences to match current frame length
        meta_num_seqs = int(meta["lengths"].shape[0]) if "lengths" in meta else -1
        expected_num_seqs = int(len(frame))
        if not (
            meta_max_len == int(cfg.max_len)
            and meta_input == int(cfg.input_size)
            and meta_num_seqs == expected_num_seqs
        ):
            # Invalidate old cache
            try:
                data_path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                meta_path.unlink(missing_ok=True)
            except Exception:
                pass
            print(f"Cache incompatible for {split}, rebuilding…")
        else:
            print(f"Loaded sequence cache ({split}) from {cache_dir}")
            return SequenceCache(data_path, meta_path)

    # Build cache
    print(f"Building sequence cache ({split}) in {cache_dir} …")
    seq_lengths: List[int] = []
    # First pass: compute lengths after truncation
    for s in frame["drawing"].values:
        seq = parse_drawing_to_seq(s)
        L = min(int(seq.shape[0]), int(cfg.max_len))
        seq_lengths.append(L)

    total_len = int(sum(seq_lengths))
    data = np.zeros((total_len, cfg.input_size), dtype=np.float32)
    offsets = np.zeros((len(seq_lengths),), dtype=np.int64)

    cursor = 0
    for i, s in enumerate(frame["drawing"].values):
        offsets[i] = cursor
        seq = parse_drawing_to_seq(s)
        L = min(int(seq.shape[0]), int(cfg.max_len))
        if L > 0:
            data[cursor:cursor + L, :] = seq[:L, :]
            cursor += L

    # Save to disk: data as .npy (mmap-able), metadata as .npz
    np.save(str(data_path), data)
    np.savez(str(meta_path), offsets=offsets, lengths=np.asarray(seq_lengths, dtype=np.int32), input_size=np.int32(cfg.input_size), max_len=np.int32(cfg.max_len))
    print(f"Saved cache ({split}): {data_path.name}, {meta_path.name}")

    return SequenceCache(data_path, meta_path)


def collate_pad(batch: List[Tuple[np.ndarray, int]], max_len: int, min_seq_len: int | None = None):
    # Filter out too-short sequences to avoid zero-length packs
    filtered: List[Tuple[np.ndarray, int]] = []
    for seq, label in batch:
        L = min(seq.shape[0], max_len)
        if min_seq_len is not None and L < max(1, min_seq_len):
            continue
        filtered.append((seq, label))

    if not filtered:
        # Fallback: keep the longest from original batch; ensure at least length 1
        seq, label = max(batch, key=lambda t: t[0].shape[0])
        if seq.shape[0] == 0:
            seq = np.zeros((1, 3), dtype=np.float32)
        filtered = [(seq[:max_len], label)]

    sequences, labels = zip(*filtered)
    lengths = [min(s.shape[0], max_len) for s in sequences]
    maxL = max(1, min(max(lengths), max_len))
    B = len(sequences)
    x = torch.zeros((B, maxL, 3), dtype=torch.float32)
    for i, s in enumerate(sequences):
        L = min(s.shape[0], maxL)
        if L > 0:
            x[i, :L, :] = torch.from_numpy(s[:L, :])
    y = torch.tensor(labels, dtype=torch.long)
    lens = torch.tensor([min(l, maxL) for l in lengths], dtype=torch.long)
    return x, lens, y


class CollatePad:
    """Top-level callable collate for multiprocessing pickling safety.

    Avoids lambda/nested functions which break with num_workers>0 on macOS/Windows.
    """
    def __init__(self, max_len: int, min_seq_len: int | None = None):
        self.max_len = max_len
        self.min_seq_len = min_seq_len

    def __call__(self, batch: List[Tuple[np.ndarray, int]]):
        return collate_pad(batch, self.max_len, self.min_seq_len)


class CachedSketchDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, class_to_idx: dict, cache: SequenceCache):
        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.cache = cache

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        seq = self.cache.get_seq(idx)
        label = self.class_to_idx[row["word"]]
        return seq, label


class ListBatchSampler:
    def __init__(self, batches: List[List[int]]):
        self.batches = batches

    def __iter__(self):
        for b in self.batches:
            yield b

    def __len__(self):
        return len(self.batches)


def make_bucketed_batches(lengths: np.ndarray, batch_size: int, shuffle: bool = True) -> List[List[int]]:
    # Sort indices by length, then chunk, then shuffle batch order
    idx = np.argsort(lengths.astype(np.int64))
    batches: List[List[int]] = []
    for i in range(0, len(idx), batch_size):
        batches.append(idx[i:i + batch_size].tolist())
    if shuffle:
        rng = np.random.default_rng(42)
        rng.shuffle(batches)
    return batches

"""
Section 4: Gru classifier. It reads the sequence, form a memeory of the shape and then map the memory to a label
"""
class GRUClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 bidirectional: bool, dropout: float, num_classes: int, use_packing: bool = True):
        super().__init__()
        self.use_packing = use_packing
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: [B, T, 3], lengths: [B]
        if self.use_packing:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(x)

        # h_n: [num_layers * num_directions, B, hidden_size]
        if self.gru.bidirectional:
            # last layer, both directions
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]

        h = self.norm(h)
        logits = self.fc(h)
        return logits

"""
Section 5: Accuracy, training, evaluation and early stopping
"""
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 3)):
    maxk = max(ks)
    with torch.no_grad():
        _, pred = logits.topk(maxk, dim=1)
        pred = pred.t()  # [maxk, B]
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        res = []
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / targets.size(0)).item())
        return res


def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0, label_smoothing=0.05, log_interval: int = 0, batch_scheduler: Optional[object] = None, scheduler_type: str = ""):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss, total_correct, total_count = 0.0, 0, 0
    total_top3 = 0.0

    for batch_idx, (x, lengths, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if batch_scheduler is not None and scheduler_type == "onecycle":
            batch_scheduler.step()

        with torch.no_grad():
            acc1, acc3 = topk_accuracy(logits, y, ks=(1, 3))
        total_loss += loss.item() * y.size(0)
        total_correct += acc1 * y.size(0)
        total_top3 += acc3 * y.size(0)
        total_count += y.size(0)

        if log_interval and ((batch_idx + 1) % log_interval == 0):
            curr_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Batch {batch_idx + 1}/{len(loader)} | "
                f"loss={loss.item():.4f} acc1={acc1:.3f} lr={curr_lr:.3g}"
            )

    return {
        "loss": total_loss / total_count,
        "acc1": total_correct / total_count,
        "acc3": total_top3 / total_count,
    }


@torch.no_grad()
def evaluate(model, loader, device, label_smoothing=0.0):
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss, total_correct, total_count = 0.0, 0, 0
    total_top3 = 0.0

    for x, lengths, y in loader:
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.to(device)

        logits = model(x, lengths)
        loss = criterion(logits, y)

        acc1, acc3 = topk_accuracy(logits, y, ks=(1, 3))
        total_loss += loss.item() * y.size(0)
        total_correct += acc1 * y.size(0)
        total_top3 += acc3 * y.size(0)
        total_count += y.size(0)

    return {
        "loss": total_loss / total_count,
        "acc1": total_correct / total_count,
        "acc3": total_top3 / total_count,
    }

"""
Section 6: Build splits, DataLoaders, model, and training driver
"""

def build_frame_and_classes(cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(cfg.csv_path)
    if cfg.recognized_only and "recognized" in df.columns:
        df = df[df["recognized"] == True].reset_index(drop=True)

    # Only keep rows with 'word' and 'drawing'
    df = df[["word", "drawing"]].dropna().reset_index(drop=True)

    # Optionally filter to a specific subset of classes
    if cfg.allowed_classes:
        allowed = set(cfg.allowed_classes)
        before = len(df)
        df = df[df["word"].isin(allowed)].reset_index(drop=True)
        print(f"Filtered to allowed classes ({len(allowed)}): {sorted(list(allowed))} | rows: {before} -> {len(df)}")
    elif cfg.num_classes_limit is not None and cfg.num_classes_limit > 0:
        counts = df["word"].value_counts()
        topk = counts.head(int(cfg.num_classes_limit)).index.tolist()
        before = len(df)
        df = df[df["word"].isin(topk)].reset_index(drop=True)
        print(f"Kept top-{cfg.num_classes_limit} classes by frequency: {sorted(topk)} | rows: {before} -> {len(df)}")

    # Determine classes from filtered CSV content (unique labels)
    classes = sorted(df["word"].unique().tolist())

    return df, classes


def stratify_and_cap(df: pd.DataFrame, classes: List[str], cfg: Config):
    # Cap per class for faster training (optional)
    if cfg.per_class_limit is not None and cfg.per_class_limit > 0:
        parts = []
        for c in classes:
            sub = df[df["word"] == c]
            if len(sub) > cfg.per_class_limit:
                sub = sub.sample(cfg.per_class_limit, random_state=42)
            parts.append(sub)
        df = pd.concat(parts, axis=0).reset_index(drop=True)

    # Stratified split by label
    train_df, val_df = train_test_split(
        df, test_size=cfg.test_size, random_state=42, stratify=df["word"]
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def make_loaders(train_df, val_df, classes, cfg: Config):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    collator = CollatePad(cfg.max_len, cfg.min_seq_len)

    if cfg.use_cache:
        train_cache = build_or_load_cache(train_df, "train", cfg)
        val_cache = build_or_load_cache(val_df, "val", cfg)
        train_ds = CachedSketchDataset(train_df, class_to_idx, train_cache)
        val_ds = CachedSketchDataset(val_df, class_to_idx, val_cache)
        # Length bucketing for train
        train_batches = make_bucketed_batches(train_cache.lengths, cfg.batch_size, shuffle=True)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=ListBatchSampler(train_batches),
            num_workers=cfg.num_workers,
            collate_fn=collator,
            persistent_workers=(cfg.num_workers > 0),
        )
        # Validation: deterministic order, optional bucketing (no shuffle)
        val_batches = make_bucketed_batches(val_cache.lengths, cfg.batch_size, shuffle=False)
        val_loader = DataLoader(
            val_ds,
            batch_sampler=ListBatchSampler(val_batches),
            num_workers=cfg.num_workers,
            collate_fn=collator,
            persistent_workers=(cfg.num_workers > 0),
        )
    else:
        train_ds = SketchDataset(train_df, class_to_idx, max_len=cfg.max_len, min_seq_len=cfg.min_seq_len)
        val_ds = SketchDataset(val_df, class_to_idx, max_len=cfg.max_len, min_seq_len=cfg.min_seq_len)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collator,
            drop_last=False,
            persistent_workers=(cfg.num_workers > 0),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collator,
            drop_last=False,
            persistent_workers=(cfg.num_workers > 0),
        )

    return train_loader, val_loader, class_to_idx


def save_checkpoint(model, class_to_idx, cfg: Config, metrics: dict, fname: str = "rnn_animals.pt"):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    payload = {
        "model_state": model.state_dict(),
        "class_to_idx": class_to_idx,
        "config": cfg.__dict__,
        "metrics": metrics,
    }
    torch.save(payload, str(path))
    return str(path)


def run_training(cfg: Config):
    device = get_device()
    set_seed(42)

    print(f"Using device: {device}")
    df, classes = build_frame_and_classes(cfg)
    print(f"Classes ({len(classes)}): {classes}")

    train_df, val_df = stratify_and_cap(df, classes, cfg)
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    train_loader, val_loader, class_to_idx = make_loaders(train_df, val_df, classes, cfg)

    model = GRUClassifier(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        bidirectional=cfg.bidirectional,
        dropout=cfg.dropout,
        num_classes=len(classes),
        use_packing=cfg.use_packing,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Optionally resume from best checkpoint
    if cfg.resume_from_best:
        best_path = Path(cfg.out_dir) / "rnn_animals_best.pt"
        if best_path.exists():
            try:
                ckpt = torch.load(str(best_path), map_location="cpu")
                ckpt_mapping = ckpt.get("class_to_idx")
                if ckpt_mapping == class_to_idx:
                    model.load_state_dict(ckpt["model_state"], strict=True)
                    print(f"Resumed weights from {best_path}")
                else:
                    print("Checkpoint class mapping mismatch; skipping resume.")
            except Exception as e:
                print(f"Could not resume from checkpoint: {e}")

    # Scheduler selection
    scheduler_type = (cfg.scheduler_type or "").lower()
    batch_scheduler = None
    if scheduler_type == "onecycle":
        steps_per_epoch = len(train_loader)
        batch_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            div_factor=3.0,
            final_div_factor=10.0,
        )
    else:
        # Some Torch builds do not accept 'verbose' in ReduceLROnPlateau; omit it for compatibility
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

    def _get_lr(opt: torch.optim.Optimizer) -> float:
        for pg in opt.param_groups:
            return float(pg.get("lr", 0.0))
        return 0.0
    prev_lr = _get_lr(optimizer)

    best_val_loss = float("inf")
    best_metrics = None
    patience_left = cfg.patience

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            grad_clip=cfg.grad_clip, label_smoothing=cfg.label_smoothing,
            log_interval=100,
            batch_scheduler=batch_scheduler,
            scheduler_type=scheduler_type,
        )
        val_metrics = evaluate(
            model, val_loader, device, label_smoothing=0.0
        )
        if scheduler_type == "onecycle":
            curr_lr = _get_lr(optimizer)
        else:
            scheduler.step(val_metrics["loss"])  # plateau
            # Log when LR changes (since 'verbose' may be unsupported in some torch builds)
            curr_lr = _get_lr(optimizer)
            if curr_lr != prev_lr:
                print(f"LR reduced: {prev_lr:.6g} -> {curr_lr:.6g}")
                prev_lr = curr_lr
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc1={train_metrics['acc1']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc1={val_metrics['acc1']:.3f} "
            f"val_acc3={val_metrics['acc3']:.3f} "
            f"| {dt:.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = {"epoch": epoch, **val_metrics}
            save_checkpoint(model, class_to_idx, cfg, best_metrics, fname="rnn_animals_best.pt")
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    # Save final model as well (last epoch)
    final_path = save_checkpoint(model, class_to_idx, cfg, best_metrics or {}, fname="rnn_animals_last.pt")
    print(f"Saved model to: {final_path}")


if __name__ == "__main__":
    cfg = Config()
    # Prefer 10-class train split if present
    try:
        ten_train = Path(cfg.out_dir) / "animal_doodles_10_train.csv"
        if ten_train.exists():
            cfg.csv_path = str(ten_train)
            print(f"Detected 10-class train split: {ten_train}")
    except Exception:
        pass
    # Stronger defaults for a thorough 10-class run
    cfg.per_class_limit = 0          # use all available samples
    cfg.epochs = 15                  # longer run, early stopping still applies
    cfg.patience = 4                 # allow a few plateaus before stopping
    cfg.resume_from_best = True      # keep improving from best checkpoint
    cfg.num_workers = 2              # speed up data loading if supported
    run_training(cfg)
