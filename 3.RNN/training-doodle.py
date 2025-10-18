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


# Using GPU if available, otherwise use CPU
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# Set the seed for reproducibility within the epochs but not between epochs
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
# ones to sweep: lr, model capacity (hidden_size, num_layers), dropout,  regs (weight decay) , batch size, scheduler type 
# on small dataset, then slwoly scale up to full dataset with DSP
    # ----------------------------
    # Data paths and filtering
    # ----------------------------
    csv_path: str = str((Path(__file__).resolve().parent / "archive" / "animal_doodles_10_train.csv"))
    recognized_only: bool = True
    min_seq_len: int = 6             # drop drawings with fewer than 6 moves
    max_len: int = 250               # cap sequence length for speed/memory (faster first run)
    per_class_limit: int = 10000    # limit per class for faster training; scale up later (1.5% of the dataset)
    # Optional class filtering controls
    allowed_classes: Optional[List[str]] = None
    num_classes_limit: Optional[int] = None

    # ----------------------------
    # Dataset split
    # ----------------------------
    test_size: float = 0.15          # val set (or dev set) size 

    # ----------------------------
    # Batching and loading
    # ----------------------------
    batch_size: int = 192             # the number of samples until one weight updates
    num_workers: int = 4            # try 2 workers for faster loading on M2
    n_buckets: int = 8               # length buckets for faster batches
    use_packing: bool = True         # if you see MPS issues, set to False
    log_interval: int = 100          # per-batch logging interval (set 1 during LR sweep)

    # ----------------------------
    # Model architecture
    # ----------------------------
    input_size: int = 3              # [dx, dy, pen_lift]
    hidden_size: int = 192           # the number of hidden units in the GRU
    num_layers: int = 2              # the number of layers in the GRU
    bidirectional: bool = True       # if True, the GRU is bidirectional    

    # ----------------------------
    # Regularization
    # ----------------------------
    dropout: float = 0.2             # regularization 
    weight_decay: float = 1e-2       # L2 regularization 
    label_smoothing: float = 0.05    # soften hard labels
    grad_clip: float = 1.0           # gradient clipping to avoid exploding gradients

    # ----------------------------
    # Optimization & training loop
    # ----------------------------
    lr: float = 2e-3                 # learning rate
    epochs: int = 15                # fewer epochs for quicker first pass
    patience: int = 4                # earlier stop on plateau (if val doesnt improve for this amount of epochs, stop)
    # LR sweep settings (range test)
    do_lr_sweep: bool = False
    lr_sweep_min: float = 1e-5
    lr_sweep_max: float = 1e-1
    lr_sweep_steps: int = 300        # number of mini-batches to sweep over
    lr_sweep_early_stop: bool = True # stop sweep when loss diverges
    lr_sweep_smooth: float = 0.05    # EMA smoothing for plotting (0..1, lower is smoother)

    # ----------------------------
    # I/O and caching
    # ----------------------------
    out_dir: str = str((Path(__file__).resolve().parent / "archive"))
    use_cache: bool = True
    cache_dir: str = str((Path(__file__).resolve().parent / "archive" / "seq_cache_v1"))
    plots_dir: str = str((Path(__file__).resolve().parent / "assets" / "plots"))

    # ----------------------------
    # Scheduling / checkpoints
    # ----------------------------
    scheduler_type: str = "onecycle"  # 'onecycle' or 'plateau'
    resume_from_best: bool = False


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
        grad_norm_val = None
        if grad_clip is not None and grad_clip > 0:
            # Returns total grad norm BEFORE clipping; useful to track
            grad_norm_val = float(nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
        else:
            # Compute norm without clipping if requested
            total_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_sq += float(p.grad.detach().pow(2).sum().item())
            grad_norm_val = math.sqrt(total_sq) if total_sq > 0 else 0.0
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
                f"loss={loss.item():.4f} acc1={acc1:.3f} lr={curr_lr:.3g} grad_norm={grad_norm_val:.3g}"
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
            log_interval=cfg.log_interval,
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


def _ensure_dirs(cfg: Config):
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.plots_dir).mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def _quick_val_loss(model: nn.Module, val_loader: DataLoader, device: torch.device, max_batches: int | None = None) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    for i, (x, lengths, y) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.to(device)
        logits = model(x, lengths)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        total += int(y.size(0))
        if max_batches is not None and (i + 1) >= max_batches:
            break
    return total_loss / max(1, total)


def run_lr_range_sweep(cfg: Config):
    """Perform a Leslie Smith LR range test and plot loss/grad_norm vs LR.

    Records one optimizer step per training batch, increasing LR exponentially
    from cfg.lr_sweep_min to cfg.lr_sweep_max over cfg.lr_sweep_steps steps.
    Saves CSV and two plots under archive/ and assets/plots.
    """
    device = get_device()
    set_seed(42)
    print(f"Using device: {device}")

    df, classes = build_frame_and_classes(cfg)
    train_df, val_df = stratify_and_cap(df, classes, cfg)
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_sweep_min, weight_decay=cfg.weight_decay)

    # Precompute LR schedule (log-spaced)
    steps = int(cfg.lr_sweep_steps)
    lrs = np.logspace(math.log10(cfg.lr_sweep_min), math.log10(cfg.lr_sweep_max), steps)

    # Containers
    history = {
        "iter": [],
        "lr": [],
        "train_loss": [],
        "grad_norm": [],
    }

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    ema = None
    smooth = float(max(0.0, min(1.0, cfg.lr_sweep_smooth)))

    it = 0
    diverged = False
    for batch in train_loader:
        if it >= steps:
            break
        # Set LR for this step
        lr_val = float(lrs[it])
        for pg in optimizer.param_groups:
            pg["lr"] = lr_val

        x, lengths, y = batch
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        # Track grad norm (pre-clip)
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        else:
            total_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_sq += float(p.grad.detach().pow(2).sum().item())
            grad_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
        optimizer.step()

        # Optional EMA smoothing of loss for stability
        loss_val = float(loss.item())
        ema = loss_val if ema is None else (smooth * ema + (1.0 - smooth) * loss_val)

        history["iter"].append(it)
        history["lr"].append(lr_val)
        history["train_loss"].append(ema if smooth > 0 else loss_val)
        history["grad_norm"].append(float(grad_norm))

        if (it + 1) % max(1, cfg.log_interval) == 0:
            print(f"  iter={it+1}/{steps} lr={lr_val:.3g} loss={loss_val:.4f} ema={ema:.4f} grad_norm={grad_norm:.3g}")

        # Early stop on divergence
        if cfg.lr_sweep_early_stop:
            if not math.isfinite(loss_val) or (ema is not None and ema > 10.0):
                print("Loss diverged during sweep; stopping early.")
                diverged = True
                break

        it += 1

    # Compute a quick validation loss snapshot
    val_loss = _quick_val_loss(model, val_loader, device, max_batches=5)
    print(f"Validation loss (quick, ~5 batches): {val_loss:.4f}")

    # Save CSV
    _ensure_dirs(cfg)
    import pandas as _pd
    sweep_df = _pd.DataFrame(history)
    csv_path = Path(cfg.out_dir) / "lr_sweep.csv"
    sweep_df.to_csv(csv_path, index=False)
    print(f"Saved LR sweep log: {csv_path}")

    # Plots
    try:
        import matplotlib.pyplot as plt
        # Train loss vs LR
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(sweep_df["lr"], sweep_df["train_loss"], label="train_loss")
        ax1.set_xscale("log")
        ax1.set_xlabel("Learning rate (log)")
        ax1.set_ylabel("Train loss (smoothed)" if smooth > 0 else "Train loss")
        ax1.set_title("LR Range Test: Loss vs LR")
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        fig1.tight_layout()
        p1 = Path(cfg.plots_dir) / "lr_range_train_loss.png"
        fig1.savefig(p1, dpi=150)
        plt.close(fig1)

        # Grad norm vs LR
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(sweep_df["lr"], sweep_df["grad_norm"], color="tab:orange", label="grad_norm")
        ax2.set_xscale("log")
        ax2.set_xlabel("Learning rate (log)")
        ax2.set_ylabel("Grad norm (pre-clip)")
        ax2.set_title("LR Range Test: Grad Norm vs LR")
        ax2.grid(True, which="both", ls=":", alpha=0.4)
        fig2.tight_layout()
        p2 = Path(cfg.plots_dir) / "lr_range_grad_norm.png"
        fig2.savefig(p2, dpi=150)
        plt.close(fig2)
        print(f"Saved plots: {p1}, {p2}")
    except Exception as e:
        print(f"Plotting failed: {e}")

    # Heuristic suggestion: pick LR with lowest smoothed loss before gradient norm spikes
    try:
        gl = np.asarray(history["grad_norm"]) if len(history["grad_norm"]) else None
        tl = np.asarray(history["train_loss"]) if len(history["train_loss"]) else None
        lr_arr = np.asarray(history["lr"]) if len(history["lr"]) else None
        if gl is not None and tl is not None and lr_arr is not None:
            # Mask out clearly divergent region (grad norm > 2x median)
            med = np.median(gl)
            ok = gl < (2.0 * max(1e-8, med))
            if ok.any():
                idx = np.argmin(np.where(ok, tl, np.inf))
            else:
                idx = int(np.argmin(tl))
            best_lr = float(lr_arr[idx])
            print(f"Suggested LR (range test): ~{best_lr:.3g}")
    except Exception:
        pass


if __name__ == "__main__":
    # Lightweight CLI to pick between normal training and LR sweep
    import argparse as _argparse
    parser = _argparse.ArgumentParser(description="Train RNN doodle classifier or run an LR sweep.")
    parser.add_argument("--lr_sweep", action="store_true", help="Run LR range test instead of full training.")
    parser.add_argument("--lr_min", type=float, default=None, help="Min LR for sweep (overrides config).")
    parser.add_argument("--lr_max", type=float, default=None, help="Max LR for sweep (overrides config).")
    parser.add_argument("--steps", type=int, default=None, help="Steps (batches) for LR sweep.")
    parser.add_argument("--log_every", type=int, default=None, help="Per-batch log interval.")
    args = parser.parse_args()

    cfg = Config()
    if args.lr_min is not None:
        cfg.lr_sweep_min = float(args.lr_min)
    if args.lr_max is not None:
        cfg.lr_sweep_max = float(args.lr_max)
    if args.steps is not None:
        cfg.lr_sweep_steps = int(args.steps)
    if args.log_every is not None:
        cfg.log_interval = int(args.log_every)

    if args.lr_sweep or cfg.do_lr_sweep:
        run_lr_range_sweep(cfg)
    else:
        run_training(cfg)
