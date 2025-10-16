## rgef
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

"""RNN demo app.

This file is also used on Hugging Face Spaces, where the project layout may
change (e.g., the app can live at the repo root while the training module and
checkpoint stay under `RNN/` or `3.RNN/`). To make deployment robust, we
resolve those paths dynamically below.
"""

# Try to import the training module to reuse parsing and model classes
import importlib.util as _importlib_util


def _resolve_training_module_path() -> Path:
    """Find the training script in a few common locations.

    Supports running this app from:
    - 3.RNN/app.py (local dev)
    - repo-root/app.py with files under RNN/ (HF Space)
    - repo-root/app.py with files under 3.RNN/ (alt layout)
    """
    here = Path(__file__).resolve().parent
    env_override = Path(str(Path.cwd() / (os.environ.get("RNN_TRAIN_PATH", "")))) if "RNN_TRAIN_PATH" in os.environ else None
    candidates = [
        here / "training-doodle.py",
        here / "RNN" / "training-doodle.py",
        here / "3.RNN" / "training-doodle.py",
        Path.cwd() / "RNN" / "training-doodle.py",
        Path.cwd() / "3.RNN" / "training-doodle.py",
    ]
    if env_override and env_override.exists():
        return env_override
    for p in candidates:
        if p.exists():
            return p
    # Fall back to the expected local path so the error message is meaningful
    return here / "training-doodle.py"


def _import_training_module():
    train_path = _resolve_training_module_path()
    spec = _importlib_util.spec_from_file_location("rnn_train_mod", str(train_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training module from {train_path}")
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# Defer import until after helpers are defined
import os  # used in _resolve_training_module_path

rnn_mod = _import_training_module()

GRUClassifier = rnn_mod.GRUClassifier
parse_drawing_to_seq = rnn_mod.parse_drawing_to_seq
get_device = rnn_mod.get_device


def _load_checkpoint(path: Path):
    ckpt = torch.load(str(path), map_location="cpu")
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


def _resolve_checkpoint_path(base_dir: Path) -> Path:
    """Find the checkpoint in common layouts.

    Prefers a local `archive/rnn_animals_best.pt` next to this file, but also
    checks `RNN/archive/` and `3.RNN/archive/` when the app is at repo root.
    """
    candidates = [
        base_dir / "archive" / "rnn_animals_best.pt",
        base_dir / "RNN" / "archive" / "rnn_animals_best.pt",
        base_dir / "3.RNN" / "archive" / "rnn_animals_best.pt",
        Path.cwd() / "RNN" / "archive" / "rnn_animals_best.pt",
        Path.cwd() / "3.RNN" / "archive" / "rnn_animals_best.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Last resort: return the first candidate so the failure includes a path
    return candidates[0]


def _softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def _canvas_to_quickdraw_strokes(canvas_payload: Dict[str, Any]) -> List[List[List[int]]]:
    """Convert a gradio-canvas style payload to QuickDraw strokes [[xs],[ys], ...].

    The payload is expected to have keys like {"width", "height", "paths": [{"points": [{"x":..,"y":..}, ...]}]}
    We map coordinates into [0, 255] to align with training normalization.
    """
    if not isinstance(canvas_payload, dict):
        return []
    width = float(canvas_payload.get("width") or canvas_payload.get("w") or 256)
    height = float(canvas_payload.get("height") or canvas_payload.get("h") or 256)
    paths = canvas_payload.get("paths") or canvas_payload.get("strokes") or []
    strokes: List[List[List[int]]] = []
    if not isinstance(paths, list):
        return []
    for p in paths:
        pts = p.get("points") if isinstance(p, dict) else None
        if not isinstance(pts, list) or len(pts) < 2:
            continue
        xs: List[int] = []
        ys: List[int] = []
        for pt in pts:
            x = float(pt.get("x", 0.0))
            y = float(pt.get("y", 0.0))
            # scale to 0..255
            xi = int(np.clip(round(x / max(1.0, width) * 255.0), 0, 255))
            yi = int(np.clip(round(y / max(1.0, height) * 255.0), 0, 255))
            xs.append(xi)
            ys.append(yi)
        strokes.append([xs, ys])
    return strokes


def _prepare_sequence(input_obj: Any) -> np.ndarray:
    """Accepts either QuickDraw JSON string/list or a canvas dict and returns [T,3] float32 sequence.

    Note: Sketchpad provides a raster. We vectorize it then parse to [dx,dy,pen].
    The traced polyline typically has very small steps (~1px), which produces
    tiny normalized deltas (~1/255). The model was trained on QuickDraw strokes
    with larger average steps, so we apply a light gain calibration on (dx, dy)
    to better match the training distribution. This significantly improves
    predictions on raster-drawn inputs while keeping values in [-1, 1].
    """
    # Try to extract a raster first (works for Sketchpad dicts too)
    raster = _extract_raster(input_obj)
    if raster is not None:
        strokes = _raster_to_quickdraw_strokes(raster)
        if not strokes:
            return np.zeros((0, 3), dtype=np.float32)
        try:
            seq = parse_drawing_to_seq(json.dumps(strokes))
            seq = _limit_len(seq)
            return _calibrate_seq(seq)
        except Exception:
            return np.zeros((0, 3), dtype=np.float32)

    # Direct raster object
    if isinstance(input_obj, (np.ndarray, Image.Image)):
        strokes = _raster_to_quickdraw_strokes(input_obj)
        if not strokes:
            return np.zeros((0, 3), dtype=np.float32)
        try:
            seq = parse_drawing_to_seq(json.dumps(strokes))
            seq = _limit_len(seq)
            return _calibrate_seq(seq)
        except Exception:
            return np.zeros((0, 3), dtype=np.float32)
    # Direct QuickDraw list format
    if isinstance(input_obj, list):
        try:
            seq = parse_drawing_to_seq(json.dumps(input_obj))
            seq = _limit_len(seq)
            return _calibrate_seq(seq)
        except Exception:
            return np.zeros((0, 3), dtype=np.float32)

    # JSON string
    if isinstance(input_obj, str):
        if not input_obj.strip():
            return np.zeros((0, 3), dtype=np.float32)
        try:
            seq = parse_drawing_to_seq(input_obj)
            seq = _limit_len(seq)
            return _calibrate_seq(seq)
        except Exception:
            try:
                seq = parse_drawing_to_seq(json.dumps(json.loads(input_obj)))
                seq = _limit_len(seq)
                return _calibrate_seq(seq)
            except Exception:
                return np.zeros((0, 3), dtype=np.float32)

    # QuickDraw-style canvas dict (paths)
    if isinstance(input_obj, dict):
        strokes = _canvas_to_quickdraw_strokes(input_obj)
        seq = parse_drawing_to_seq(json.dumps(strokes))
        seq = _limit_len(seq)
        return _calibrate_seq(seq)

    return np.zeros((0, 3), dtype=np.float32)


def _calibrate_seq(seq: np.ndarray, target_mean: float = 0.04, max_gain: float = 12.0, min_gain: float = 0.5) -> np.ndarray:
    """Scale (dx, dy) so the mean step magnitude roughly matches `target_mean`.

    This makes raster-to-stroke conversion behave closer to QuickDraw stroke
    statistics the model saw during training. Clamps to keep values in [-1, 1].
    """
    try:
        if seq is None or seq.ndim != 2 or seq.shape[1] < 2 or seq.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        steps = np.sqrt((seq[:, 0] ** 2) + (seq[:, 1] ** 2))
        curr = float(steps.mean()) if steps.size else 0.0
        if curr <= 1e-6:
            return seq.astype(np.float32)
        gain = float(np.clip(target_mean / curr, min_gain, max_gain))
        out = seq.astype(np.float32).copy()
        out[:, 0:2] = np.clip(out[:, 0:2] * gain, -1.0, 1.0)
        return out
    except Exception:
        return seq.astype(np.float32)


def _limit_len(seq: np.ndarray, max_len: int = 250) -> np.ndarray:
    """Downsample long sequences to <= max_len using uniform stride.

    The GRU was trained with sequences capped at ~250 steps. Extremely long
    traced paths from raster inputs can confuse the model; keeping T within the
    training range stabilizes predictions and speeds inference.
    """
    try:
        if seq is None or seq.ndim != 2 or seq.shape[0] <= max_len:
            return seq.astype(np.float32)
        step = int(np.ceil(seq.shape[0] / float(max_len)))
        return seq[::step][:max_len].astype(np.float32)
    except Exception:
        return seq.astype(np.float32)


def _extract_raster(input_obj: Any) -> Optional[Any]:
    """Extract a raster image/array from various Sketchpad/canvas payloads.

    Handles dicts like {"image"|"composite"|"background"|"value"} or a layers list.
    Returns a PIL Image or numpy array when possible, else None.
    """
    try:
        import numpy as _np
    except Exception:
        _np = None

    if isinstance(input_obj, Image.Image) or (hasattr(np, 'ndarray') and isinstance(input_obj, np.ndarray)):
        return input_obj
    if isinstance(input_obj, dict):
        # Common keys used by Gradio components
        for key in ("image", "composite", "background", "value"):
            payload = input_obj.get(key)
            if isinstance(payload, Image.Image) or (hasattr(np, 'ndarray') and isinstance(payload, np.ndarray)):
                return payload
        # Layers array e.g., {layers: [np.array | PIL.Image, ...]}
        layers = input_obj.get("layers")
        if isinstance(layers, list) and layers:
            for lay in reversed(layers):
                if isinstance(lay, Image.Image) or (hasattr(np, 'ndarray') and isinstance(lay, np.ndarray)):
                    return lay
                if isinstance(lay, dict):
                    for key in ("image", "data", "value"):
                        payload = lay.get(key)
                        if isinstance(payload, Image.Image) or (hasattr(np, 'ndarray') and isinstance(payload, np.ndarray)):
                            return payload
    return None


def _raster_to_quickdraw_strokes(img_input: Any) -> List[List[List[int]]]:
    """Convert a raster (Sketchpad) image to QuickDraw-like strokes.

    Steps: grayscale -> threshold -> skeletonize -> trace polylines -> scale coords to 0..255.
    This is a lightweight approximation sufficient for interactive demos.
    """
    try:
        import numpy as _np
        from skimage.morphology import skeletonize, remove_small_objects
        from skimage.filters import threshold_otsu
        from skimage import measure as _measure
    except Exception:
        return []

    # Normalize to grayscale uint8 [0,255]
    if isinstance(img_input, Image.Image):
        img = img_input
    elif isinstance(img_input, _np.ndarray):
        arr = img_input
        if arr.ndim == 3 and arr.shape[2] == 4:
            img = Image.fromarray(arr.astype(_np.uint8), mode="RGBA").convert("RGB")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            img = Image.fromarray(arr.astype(_np.uint8), mode="RGB")
        else:
            # Assume single-channel
            img = Image.fromarray(arr.astype(_np.uint8))
    else:
        return []

    gray = _np.array(img.convert("L"), dtype=_np.uint8)
    H, W = gray.shape
    if H == 0 or W == 0:
        return []

    # Invert so strokes have high value; robust binary threshold
    inv = 255 - gray
    try:
        thr_val = float(threshold_otsu(inv.astype(_np.float32)))
        mask = inv > max(8.0, thr_val * 0.85)
    except Exception:
        p90 = float(_np.percentile(inv, 90))
        thr = int(max(8.0, min(40.0, p90 * 0.2)))
        mask = inv > thr
    # Remove tiny speckles to stabilize skeleton
    mask = remove_small_objects(mask.astype(bool), min_size=max(16, (H * W) // 5000))
    if not _np.any(mask):
        return []

    # Skeletonize to 1-pixel width
    skel = skeletonize(mask).astype(_np.uint8)

    # Helper: neighbor iteration without wraparound
    def neighbors(y: int, x: int):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    yield ny, nx

    # Count 8-neighbors for endpoints detection
    def neighbor_count(y: int, x: int) -> int:
        c = 0
        for ny, nx in neighbors(y, x):
            if skel[ny, nx]:
                c += 1
        return c

    visited = _np.zeros_like(skel, dtype=bool)
    strokes: List[List[List[int]]] = []

    # Collect all foreground pixels
    ys, xs = _np.where(skel > 0)
    pixels = list(zip(map(int, ys.tolist()), map(int, xs.tolist())))
    pixel_set = set(pixels)

    # Build index for fast membership
    for (sy, sx) in pixels:
        if visited[sy, sx]:
            continue
        if skel[sy, sx] == 0:
            continue

        # Find a start: prefer endpoints; otherwise use this pixel
        start = None
        if neighbor_count(sy, sx) <= 1:
            start = (sy, sx)
        else:
            # attempt to find a nearby endpoint within a small radius
            found = False
            for (yy, xx) in pixels:
                if not visited[yy, xx] and skel[yy, xx] and neighbor_count(yy, xx) <= 1:
                    start = (yy, xx)
                    found = True
                    break
            if not found:
                start = (sy, sx)

        path: List[Tuple[int, int]] = []
        prev = None
        cy, cx = start
        while True:
            if visited[cy, cx]:
                break
            visited[cy, cx] = True
            path.append((cy, cx))
            # Find next neighbor not visited
            candidates = []
            for ny, nx in neighbors(cy, cx):
                if skel[ny, nx] and not visited[ny, nx]:
                    candidates.append((ny, nx))
            if not candidates:
                break
            if prev is None:
                ny, nx = candidates[0]
            else:
                # choose neighbor most aligned with current direction
                vy, vx = cy - prev[0], cx - prev[1]
                best = None
                best_dot = -1e9
                for (ay, ax) in candidates:
                    dy, dx = ay - cy, ax - cx
                    dot = dy * vy + dx * vx
                    if dot > best_dot:
                        best_dot = dot
                        best = (ay, ax)
                ny, nx = best
            prev = (cy, cx)
            cy, cx = ny, nx

        if len(path) < 2:
            continue

        # Downsample long paths to cap total points
        max_points = 200
        step = max(1, len(path) // max_points)
        sampled = path[::step]
        if sampled[-1] != path[-1]:
            sampled.append(path[-1])

        # Convert to scaled QuickDraw stroke arrays
        xs_out: List[int] = []
        ys_out: List[int] = []
        for (yy, xx) in sampled:
            xi = int(round(xx / max(1, W - 1) * 255.0))
            yi = int(round(yy / max(1, H - 1) * 255.0))
            xs_out.append(xi)
            ys_out.append(yi)
        strokes.append([xs_out, ys_out])

    # If no strokes traced from skeleton (e.g., very short or noisy), fall back to contours
    if not strokes:
        try:
            contours = _measure.find_contours(mask.astype(float), 0.5)
            for cnt in contours:
                if cnt.shape[0] < 2:
                    continue
                # Downsample
                max_points = 200
                step = max(1, int(cnt.shape[0] // max_points))
                cnt_ds = cnt[::step]
                xs_out = []
                ys_out = []
                for (yy, xx) in cnt_ds:
                    xi = int(round(xx / max(1, W - 1) * 255.0))
                    yi = int(round(yy / max(1, H - 1) * 255.0))
                    xs_out.append(xi)
                    ys_out.append(yi)
                if len(xs_out) >= 2:
                    strokes.append([xs_out, ys_out])
        except Exception:
            pass

    # Fallback 1: contour paths when skeleton tracing found nothing
    if not strokes:
        try:
            contours = _measure.find_contours(mask.astype(float), 0.5)
            for cnt in contours:
                if cnt.shape[0] < 2:
                    continue
                # Downsample
                max_points = 200
                step = max(1, int(cnt.shape[0] // max_points))
                cnt_ds = cnt[::step]
                xs_out = []
                ys_out = []
                for (yy, xx) in cnt_ds:
                    xi = int(round(xx / max(1, W - 1) * 255.0))
                    yi = int(round(yy / max(1, H - 1) * 255.0))
                    xs_out.append(xi)
                    ys_out.append(yi)
                if len(xs_out) >= 2:
                    strokes.append([xs_out, ys_out])
        except Exception:
            pass

    # Fallback 2: row-centroid backbone path to guarantee a stroke exists
    if not strokes:
        xs_out, ys_out = [], []
        for yy in range(H):
            xs = _np.where(mask[yy])[0]
            if xs.size == 0:
                continue
            xi = int(_np.clip(round(xs.mean()), 0, W - 1))
            xs_out.append(int(round(xi / max(1, W - 1) * 255.0)))
            ys_out.append(int(round(yy / max(1, H - 1) * 255.0)))
        if len(xs_out) >= 2:
            # Downsample to at most 200 points
            step = max(1, len(xs_out) // 200)
            xs_out = xs_out[::step]
            ys_out = ys_out[::step]
            strokes.append([xs_out, ys_out])

    return strokes


def _diagnose_raster(img_input: Any, strokes: List[List[List[int]]], seq_len: int, avg_step_override: Optional[float] = None):
    try:
        import numpy as _np
        from skimage.morphology import skeletonize
        from skimage.filters import threshold_otsu
    except Exception:
        return "Install scikit-image for diagnostics.", None, None, None

    if isinstance(img_input, Image.Image):
        img = img_input
    elif isinstance(img_input, _np.ndarray):
        arr = img_input
        if arr.ndim == 3 and arr.shape[2] == 4:
            img = Image.fromarray(arr.astype(_np.uint8), mode="RGBA").convert("RGB")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            img = Image.fromarray(arr.astype(_np.uint8), mode="RGB")
        else:
            img = Image.fromarray(arr.astype(_np.uint8))
    else:
        return "Unsupported input for diagnostics.", None, None, None

    gray = _np.array(img.convert("L"), dtype=_np.uint8)
    H, W = gray.shape
    inv = 255 - gray
    try:
        thr_val = float(threshold_otsu(inv.astype(_np.float32)))
        mask = inv > max(8.0, thr_val * 0.85)
    except Exception:
        thr = max(16, int(_np.percentile(inv, 90) * 0.3))
        mask = inv > thr
    mass_fraction = float(mask.mean())
    skel = skeletonize(mask).astype(_np.uint8)
    skel_count = int(skel.sum())

    # Previews
    mask_img = (mask.astype(_np.uint8) * 255)
    skel_img = (skel.astype(_np.uint8) * 255)

    # Stroke path preview on 256x256 canvas
    path_img = Image.new("L", (256, 256), color=255)
    draw = ImageDraw.Draw(path_img)
    for xs, ys in strokes:
        if len(xs) < 2:
            continue
        pts = [(int(round(x * 255 / 255)), int(round(y * 255 / 255))) for x, y in zip(xs, ys)]
        draw.line(pts, fill=0, width=2)
    path_img = _np.array(path_img, dtype=_np.uint8)

    # Compute additional path statistics if we have a sequence
    avg_step = avg_step_override
    if avg_step is None and seq_len and seq_len > 1:
        try:
            import json as _json
            seq = parse_drawing_to_seq(_json.dumps(strokes)) if strokes else None
            if seq is not None and seq.shape[0] > 1:
                steps = _np.sqrt((seq[:,0]**2) + (seq[:,1]**2))
                avg_step = float(steps.mean())
        except Exception:
            pass

    hint = []
    if seq_len < 6:
        hint.append("Sequence too short (<6). Draw longer, continuous strokes.")
    if mass_fraction < 0.01:
        hint.append("Very low ink coverage. Use thicker/longer strokes.")
    if skel_count < 50:
        hint.append("Skeleton is tiny. Try drawing bigger in the canvas.")
    if avg_step is not None and avg_step < 0.002:
        hint.append("Steps are very small. Draw with smoother, longer motions.")
    diag_lines = [
        f"Canvas: {W}x{H}",
        f"Ink fraction: {mass_fraction:.3f}",
        f"Skeleton pixels: {skel_count}",
        f"Strokes traced: {len(strokes)}", 
        f"Sequence length (T): {seq_len}",
        f"Avg step (norm): {avg_step:.4f}" if avg_step is not None else "Avg step (norm): n/a",
    ]
    if hint:
        diag_lines += ["\nSuggestions:"] + [f"- {h}" for h in hint]
    return "\n".join(diag_lines), mask_img, skel_img, path_img


class RNNPredictor:
    def __init__(self, ckpt_path: Optional[str] = None, conf_threshold: float = 0.8):
        base_dir = Path(__file__).resolve().parent
        default_ckpt = _resolve_checkpoint_path(base_dir)
        self.ckpt_path = Path(ckpt_path) if ckpt_path else default_ckpt
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {self.ckpt_path}. Ensure the model file is uploaded "
                f"(e.g., RNN/archive/rnn_animals_best.pt) and .hfignore does not exclude it."
            )
        self.model, self.idx_to_class = _load_checkpoint(self.ckpt_path)
        self.device = get_device()
        self.model.to(self.device)
        self.conf_threshold = float(conf_threshold)

    @torch.no_grad()
    def predict_from_input(self, input_obj: Any) -> Tuple[str, float, List[Tuple[str, float]]]:
        seq = _prepare_sequence(input_obj)
        if seq.shape[0] == 0:
            return "", 0.0, []
        x = torch.from_numpy(seq).unsqueeze(0).to(torch.float32)  # [1, T, 3]
        lengths = torch.tensor([seq.shape[0]], dtype=torch.long)
        x = x.to(self.device)
        lengths = lengths.to(self.device)
        logits = self.model(x, lengths)
        probs = _softmax(logits)[0].cpu().numpy()
        top_idx = int(probs.argmax())
        top_conf = float(probs[top_idx])
        top_label = self.idx_to_class.get(top_idx, str(top_idx))
        # top3
        top3_idx = probs.argsort()[-3:][::-1].tolist()
        top3 = [(self.idx_to_class.get(i, str(i)), float(probs[i])) for i in top3_idx]
        return top_label, top_conf, top3


def build_ui() -> gr.Blocks:
    predictor = RNNPredictor()
    MIN_CONF = predictor.conf_threshold

    with gr.Blocks(title="RNN Doodle Classifier (10 classes)") as demo:
        gr.Markdown("""
        Draw an animal. The model updates after each stroke. When confidence > 80%, it shows the predicted class.
        If confidence is lower, it shows top-3 suggestions.

        Game mode: you'll get a random target to draw. When the model predicts your target with ≥ 80% confidence, it counts as correct. Click "Next Target" or enable auto-advance.
        """)

        # Use built-in Sketchpad as the default canvas
        input_component = gr.Sketchpad(label="Draw here", brush=8, type="numpy", width=512, height=512)
        # Optional JSON input as an advanced alternative
        json_input = gr.Textbox(
            label="QuickDraw strokes JSON (optional)",
            lines=3,
            placeholder="Paste strokes JSON [[[x...],[y...]], ...] or draw above",
            visible=False,
        )

        # Game state
        classes = [predictor.idx_to_class[i] for i in range(len(predictor.idx_to_class))]
        def _pick_target(prev: Optional[str] = None) -> str:
            choices = [c for c in classes if c != prev] if prev in classes and len(classes) > 1 else classes
            return random.choice(choices)

        init_target = _pick_target(None)
        target_state = gr.State(init_target)
        score_state = gr.State(0)
        attempts_state = gr.State(0)

        with gr.Row():
            target_md = gr.Markdown(f"Target: {init_target}")
            score_md = gr.Markdown("Score: 0 / 0")
            auto_next = gr.Checkbox(value=True, label="Auto next on correct")

        with gr.Row():
            status = gr.Markdown("Waiting for input…")
        with gr.Row():
            label = gr.Label(label="Top prediction", num_top_classes=3)

        with gr.Accordion("Diagnostics", open=False):
            diag_text = gr.Markdown("Draw to see diagnostics.")
            with gr.Row():
                diag_mask = gr.Image(label="Binarized mask", image_mode="L")
                diag_skel = gr.Image(label="Skeleton", image_mode="L")
                diag_path = gr.Image(label="Stroke path preview", image_mode="L")

        def _fmt_target(t: str) -> str:
            return f"Target: {t}"

        def _fmt_score(s: int, a: int) -> str:
            return f"Score: {s} / {a}"

        def _predict_fn(payload: Any, target: str, score: int, attempts: int, auto_advance: bool):
            seq = _prepare_sequence(payload)
            # For diagnostics/use, extract raster robustly
            raster_src = _extract_raster(payload)
            strokes = _raster_to_quickdraw_strokes(raster_src) if raster_src is not None else []
            pred_label = ""
            conf = 0.0
            top3 = []
            if seq.shape[0] > 0:
                with torch.no_grad():
                    x = torch.from_numpy(seq).unsqueeze(0).to(torch.float32)
                    lengths = torch.tensor([seq.shape[0]], dtype=torch.long)
                    x = x.to(predictor.device)
                    lengths = lengths.to(predictor.device)
                    logits = predictor.model(x, lengths)
                    probs_t = _softmax(logits)[0].detach().cpu()
                probs = probs_t.numpy()
                top_idx = int(probs.argmax())
                conf = float(probs[top_idx])
                pred_label = predictor.idx_to_class.get(top_idx, str(top_idx))
                top3_idx = probs.argsort()[-3:][::-1].tolist()
                top3 = [(predictor.idx_to_class.get(i, str(i)), float(probs[i])) for i in top3_idx]

            # Build diagnostics
            # Pre-compute average step magnitude for the calibrated sequence to display in diagnostics
            avg_step = None
            if seq.shape[0] > 1:
                try:
                    import numpy as _np
                    avg_step = float(_np.sqrt((seq[:,0]**2) + (seq[:,1]**2)).mean())
                except Exception:
                    avg_step = None
            diag_md, mask_img, skel_img, path_img = _diagnose_raster(
                raster_src, strokes, int(seq.shape[0]), avg_step_override=avg_step
            ) if raster_src is not None else ("Provide a drawing for diagnostics.", None, None, None)
            if not pred_label:
                return (
                    gr.update(value="Draw or paste strokes to begin."),
                    {"": 0.0},
                    gr.update(value=diag_md),
                    mask_img,
                    skel_img,
                    path_img,
                    gr.update(value=_fmt_target(target)),
                    gr.update(value=_fmt_score(score, attempts)),
                    target,
                    score,
                    attempts,
                    gr.update(),
                )
            conf_pct = int(round(conf * 100))
            status_text = f"Prediction: {pred_label} ({conf_pct}%)"
            if conf < MIN_CONF:
                status_text = f"Low confidence ({conf_pct}%). Keep drawing…"
                scores = {name: float(p) for name, p in top3}
                return (
                    gr.update(value=status_text),
                    scores,
                    gr.update(value=diag_md),
                    mask_img,
                    skel_img,
                    path_img,
                    gr.update(value=_fmt_target(target)),
                    gr.update(value=_fmt_score(score, attempts)),
                    target,
                    score,
                    attempts,
                    gr.update(),
                )

            # High confidence path
            if pred_label == target:
                score += 1
                status_text = f"Correct! {pred_label} ({conf_pct}%)"
                if auto_advance:
                    attempts += 1
                    new_t = _pick_target(target)
                    scores = {name: float(p) for name, p in top3}
                    return (
                        gr.update(value=status_text),
                        scores,
                        gr.update(value=diag_md),
                        mask_img,
                        skel_img,
                        path_img,
                        gr.update(value=_fmt_target(new_t)),
                        gr.update(value=_fmt_score(score, attempts)),
                        new_t,
                        score,
                        attempts,
                        gr.update(value=None),
                    )
            # Else: high confidence but wrong target or no auto-advance
            scores = {name: float(p) for name, p in top3}
            return (
                gr.update(value=status_text),
                scores,
                gr.update(value=diag_md),
                mask_img,
                skel_img,
                path_img,
                gr.update(value=_fmt_target(target)),
                gr.update(value=_fmt_score(score, attempts)),
                target,
                score,
                attempts,
                gr.update(),
            )

        # Update on change (include game state)
        input_component.change(
            _predict_fn,
            inputs=[input_component, target_state, score_state, attempts_state, auto_next],
            outputs=[status, label, diag_text, diag_mask, diag_skel, diag_path, target_md, score_md, target_state, score_state, attempts_state, input_component],
        )
        # Some Spaces/Gradio versions emit edits via `.edit` instead of `.change`.
        # Hook both to be safe (idempotent since we update full outputs each call).
        if hasattr(input_component, "edit"):
            input_component.edit(
                _predict_fn,
                inputs=[input_component, target_state, score_state, attempts_state, auto_next],
                outputs=[status, label, diag_text, diag_mask, diag_skel, diag_path, target_md, score_md, target_state, score_state, attempts_state, input_component],
            )

        # Provide a manual predict button as well
        btn = gr.Button("Predict now")
        btn.click(
            _predict_fn,
            inputs=[input_component, target_state, score_state, attempts_state, auto_next],
            outputs=[status, label, diag_text, diag_mask, diag_skel, diag_path, target_md, score_md, target_state, score_state, attempts_state, input_component],
        )

        # Game controls
        def _next_target(curr_t: str, score: int, attempts: int):
            new_t = _pick_target(curr_t)
            attempts += 1
            return (
                gr.update(value=None),  # clear drawing
                gr.update(value=f"New target selected: {new_t}"),
                {"": 0.0},
                gr.update(value="Draw to see diagnostics."),
                None,
                None,
                None,
                gr.update(value=_fmt_target(new_t)),
                gr.update(value=_fmt_score(score, attempts)),
                new_t,
                score,
                attempts,
            )

        def _reset(_: Any):
            new_t = _pick_target(None)
            score = 0
            attempts = 0
            return (
                gr.update(value=None),
                gr.update(value="Reset. Draw the new target!"),
                {"": 0.0},
                gr.update(value="Draw to see diagnostics."),
                None,
                None,
                None,
                gr.update(value=_fmt_target(new_t)),
                gr.update(value=_fmt_score(score, attempts)),
                new_t,
                score,
                attempts,
            )

        with gr.Row():
            next_btn = gr.Button("Next Target", variant="secondary")
            reset_btn = gr.Button("Reset Game")

        next_btn.click(
            _next_target,
            inputs=[target_state, score_state, attempts_state],
            outputs=[input_component, status, label, diag_text, diag_mask, diag_skel, diag_path, target_md, score_md, target_state, score_state, attempts_state],
        )
        reset_btn.click(
            _reset,
            inputs=[input_component],
            outputs=[input_component, status, label, diag_text, diag_mask, diag_skel, diag_path, target_md, score_md, target_state, score_state, attempts_state],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch()
