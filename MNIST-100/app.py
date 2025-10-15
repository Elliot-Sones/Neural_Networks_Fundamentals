import numpy as np
import gradio as gr
from PIL import Image, ImageOps
from pathlib import Path
import importlib.util


OUTPUT_CLASSES = 100
TARGET_HEIGHT, TARGET_WIDTH = 28, 56
STD_FLOOR = 1e-8
METRIC_TARGETS = {
    "mass_fraction": (0.08, 0.35),
    "stroke_density": (0.12, 0.65),
    "center_offset": (0.0, 8.0),
    "mean_abs_z_score": (0.0, 2.5),
    "max_abs_z_score": (0.0, 6.0),
    "std_abs_z_score": (0.0, 1.5),
}


def _load_training_module():
    module_path = Path(__file__).resolve().parent / "training-100.py"
    spec = importlib.util.spec_from_file_location("mnist100_training", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


training_mod = _load_training_module()
forward_prop = training_mod.forward_prop
get_predictions = training_mod.get_predictions
softmax = training_mod.softmax


def _metric_status(name, value):
    target = METRIC_TARGETS.get(name)
    status = "not_tracked"
    target_dict = None
    if target is not None:
        low, high = target
        target_dict = {"min": low, "max": high}
        if value is None or np.isnan(value):
            status = "invalid"
        elif low <= value <= high:
            status = "ok"
        else:
            status = "out_of_range"
    return status, target_dict


def load_trained_artifacts(model_path=None):
    base_dir = Path(__file__).resolve().parent
    if model_path is None:
        resolved_path = base_dir / "archive" / "trained_model_mnist100.npz"
    else:
        candidate = Path(model_path)
        resolved_path = candidate if candidate.is_absolute() else base_dir / candidate
    if not resolved_path.exists():
        raise RuntimeError(
            f"Model file '{resolved_path}' not found. Train the MNIST-100 model first by running 'python training-100.py'."
        )
    loaded = np.load(resolved_path)
    params = {key: loaded[key] for key in loaded.files if key not in {"mean", "std"}}
    mean = loaded["mean"]
    std = loaded["std"]
    return params, mean, std


params, mean, std = None, None, None


def ensure_model_loaded():
    global params, mean, std
    if params is None or mean is None or std is None:
        params, mean, std = load_trained_artifacts()


def extract_canvas_array(img_input):
    if img_input is None:
        return None

    if isinstance(img_input, dict):
        for key in ("image", "composite", "background", "value"):
            payload = img_input.get(key)
            if payload is not None:
                img_input = payload
                break
        else:
            return None

    if isinstance(img_input, Image.Image):
        return img_input

    if isinstance(img_input, np.ndarray):
        arr_in = img_input
        if arr_in.dtype != np.uint8:
            max_val = float(arr_in.max()) if arr_in.size else 1.0
            if max_val <= 1.5:
                arr_in = (arr_in * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr_in = np.clip(arr_in, 0, 255).astype(np.uint8)
        if arr_in.ndim == 3 and arr_in.shape[2] == 4:
            return Image.fromarray(arr_in, mode="RGBA")
        return Image.fromarray(arr_in)

    return None


def shift_with_zero_pad(arr, shift_y=0, shift_x=0):
    if shift_y == 0 and shift_x == 0:
        return arr
    rolled = np.roll(arr, shift=shift_y, axis=0)
    rolled = np.roll(rolled, shift=shift_x, axis=1)
    out = rolled.copy()
    if shift_y > 0:
        out[:shift_y, :] = 0.0
    elif shift_y < 0:
        out[shift_y:, :] = 0.0
    if shift_x > 0:
        out[:, :shift_x] = 0.0
    elif shift_x < 0:
        out[:, shift_x:] = 0.0
    return out


def dilate_binary_like(arr, radius=1):
    pad = radius
    padded = np.pad(arr, pad, mode="constant", constant_values=0.0)
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            window = padded[i : i + 2 * pad + 1, j : j + 2 * pad + 1]
            out[i, j] = window.max()
    return out


def erode_binary_like(arr, radius=1):
    pad = radius
    padded = np.pad(arr, pad, mode="constant", constant_values=1.0)
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            window = padded[i : i + 2 * pad + 1, j : j + 2 * pad + 1]
            out[i, j] = window.min()
    return out


def generate_inference_variants(arr):
    variants = []
    # slight shifts
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            variants.append(shift_with_zero_pad(arr, dy, dx))
    # dilation/erosion to handle thin or thick strokes
    variants.append(dilate_binary_like(arr, radius=1))
    variants.append(erode_binary_like(arr, radius=1))
    return variants


def preprocess_image(img_input, stroke_scale: float = 1.0):
    ensure_model_loaded()
    img = extract_canvas_array(img_input)
    if img is None:
        return None

    try:
        bands = img.getbands()
    except Exception:
        bands = ()
    if "A" in bands:
        rgba = img.convert("RGBA")
        white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white_bg, rgba).convert("RGB")

    img = img.convert("L")
    img = ImageOps.invert(img)

    arr_u8 = np.array(img, dtype=np.uint8)
    original_canvas_shape = arr_u8.shape

    coords = np.column_stack(np.where(arr_u8 > 10))
    bbox = None
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        pad = 4
        y_min = max(0, y_min - pad)
        x_min = max(0, x_min - pad)
        y_max = min(arr_u8.shape[0], y_max + pad)
        x_max = min(arr_u8.shape[1], x_max + pad)
        bbox = (int(y_min), int(y_max), int(x_min), int(x_max))
        arr_u8 = arr_u8[y_min:y_max, x_min:x_max]

    if arr_u8.size == 0:
        return None

    h, w = arr_u8.shape
    target_ratio = TARGET_WIDTH / TARGET_HEIGHT
    if h == 0 or w == 0:
        return None
    current_ratio = w / h if h else target_ratio

    if current_ratio > target_ratio:
        new_height = int(round(w / target_ratio))
        pad_total = max(new_height - h, 0)
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        pad_left = pad_right = 0
    else:
        new_width = int(round(h * target_ratio))
        pad_total = max(new_width - w, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = pad_bottom = 0

    arr_padded = np.pad(
        arr_u8,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )

    resized = Image.fromarray(arr_padded).resize(
        (TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS
    )
    arr_resized = np.array(resized, dtype=np.float32) / 255.0

    mean_image = mean.reshape(TARGET_HEIGHT, TARGET_WIDTH)
    std_safe = np.maximum(std, STD_FLOOR)

    stroke_scale = float(stroke_scale)
    stroke_scale = max(0.3, min(stroke_scale, 1.5))
    arr_resized = np.clip(arr_resized * stroke_scale, 0.0, 1.0)

    augmented_arrays = [arr_resized, *generate_inference_variants(arr_resized)]
    augmented_standardized = [
        (arr.reshape(TARGET_HEIGHT * TARGET_WIDTH, 1) - mean) / std_safe
        for arr in augmented_arrays
    ]

    mean_diff = np.abs(arr_resized - mean_image)
    mean_diff_uint8 = (mean_diff / (mean_diff.max() + 1e-8) * 255.0).astype(np.uint8)

    diagnostics = compute_diagnostics(
        arr_resized,
        bbox,
        original_canvas_shape,
        mean_image,
        augmented_standardized[0],
        std_safe,
    )

    return augmented_standardized, arr_resized, mean_diff_uint8, diagnostics


def compute_diagnostics(arr_float, bbox, original_shape, mean_image, standardized, std_safe):
    mass = arr_float
    total_intensity = float(mass.sum())
    mass_threshold = mass > 0.05
    if mass_threshold.any():
        ys, xs = np.where(mass_threshold)
        bbox_est = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    else:
        bbox_est = None

    cy = cx = None
    if total_intensity > 1e-6:
        grid_y, grid_x = np.indices(mass.shape)
        weighted_sum = mass.sum()
        cy = float((grid_y * mass).sum() / weighted_sum)
        cx = float((grid_x * mass).sum() / weighted_sum)

    bbox_use = bbox_est or bbox
    if bbox_use:
        top, bottom, left, right = bbox_use
        height = bottom - top
        width = right - left
        bbox_area = height * width
        bbox_metrics = {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "height": height,
            "width": width,
            "aspect_ratio": float(width / height) if height else None,
            "area": bbox_area,
            "area_ratio": float(bbox_area / (TARGET_HEIGHT * TARGET_WIDTH)) if bbox_area else 0.0,
        }
    else:
        bbox_metrics = {
            "top": None,
            "bottom": None,
            "left": None,
            "right": None,
            "height": 0,
            "width": 0,
            "aspect_ratio": None,
            "area": 0,
            "area_ratio": 0.0,
        }

    density = 0.0
    bbox_area = bbox_metrics.get("area", 0)
    if bbox_area:
        density = float(total_intensity / bbox_area)

    center_offset = None
    if cy is not None and cx is not None:
        ideal_cy = (TARGET_HEIGHT - 1) / 2.0
        ideal_cx = (TARGET_WIDTH - 1) / 2.0
        center_offset = float(np.sqrt((cy - ideal_cy) ** 2 + (cx - ideal_cx) ** 2))

    standardized_flat = standardized.flatten()
    mean_flat = mean_image.flatten()
    arr_flat = arr_float.flatten()
    std_flat = std_safe.flatten()

    norm_input = np.linalg.norm(arr_flat)
    norm_mean = np.linalg.norm(mean_flat)
    cosine_similarity = None
    if norm_input > 0.0 and norm_mean > 0.0:
        cosine_similarity = float(np.dot(arr_flat, mean_flat) / (norm_input * norm_mean))

    mean_abs_z = float(np.mean(np.abs(standardized_flat)))
    max_abs_z = float(np.max(np.abs(standardized_flat)))
    std_of_z = float(np.std(standardized_flat))

    low_var_mask = std_flat <= STD_FLOOR + 1e-12
    activated_low_var = int(np.count_nonzero(low_var_mask & (np.abs(arr_flat - mean_flat) > 1e-3)))

    stats = {
        "total_intensity": total_intensity,
        "mass_fraction": float(total_intensity / (TARGET_HEIGHT * TARGET_WIDTH)),
        "center_of_mass": {"row": cy, "col": cx},
        "center_offset": center_offset,
        "bbox": bbox_metrics,
        "original_canvas_shape": original_shape,
        "stroke_density": density,
        "warnings": [],
        "mean_intensity": float(arr_float.mean()),
        "pixel_intensity_range": {
            "min": float(arr_float.min()),
            "max": float(arr_float.max()),
        },
        "cosine_similarity_vs_mean": cosine_similarity,
        "mean_abs_z_score": mean_abs_z,
        "max_abs_z_score": max_abs_z,
        "std_abs_z_score": std_of_z,
        "low_variance_pixels_triggered": activated_low_var,
        "low_variance_threshold": STD_FLOOR,
        "low_variance_pixels_fraction": float(activated_low_var / max(1, int(low_var_mask.sum()))),
    }

    if mean_image is not None:
        stats["distance_from_mean"] = float(np.linalg.norm(arr_float - mean_image))

    metric_checks = {}
    for metric_name in (
        "mass_fraction",
        "stroke_density",
        "center_offset",
        "mean_abs_z_score",
        "max_abs_z_score",
        "std_abs_z_score",
    ):
        value = stats.get(metric_name)
        if value is not None:
            value = float(value)
        status, target_dict = _metric_status(metric_name, value)
        entry = {"value": value, "status": status}
        if target_dict is not None:
            entry["target"] = target_dict
        metric_checks[metric_name] = entry
    stats["metric_checks"] = metric_checks

    return stats


def enrich_diagnostics(stats, probs):
    warnings = []
    bbox = stats.get("bbox", {})
    metric_checks = stats.get("metric_checks", {})

    for name, info in metric_checks.items():
        if info.get("status") == "out_of_range":
            target = info.get("target")
            value = info.get("value")
            value_str = "None" if value is None else f"{value:.4f}"
            if target is not None:
                warnings.append(
                    f"{name}: value={value_str}, target=[{target['min']:.4f},{target['max']:.4f}]"
                )
            else:
                warnings.append(f"{name}: value={value_str}")

    aspect_ratio = bbox.get("aspect_ratio")
    if aspect_ratio is not None and (aspect_ratio < 1.0 or aspect_ratio > 3.5):
        warnings.append(f"aspect_ratio: value={aspect_ratio:.4f}, expected≈[1.00,3.50]")

    confidences = np.sort(probs.flatten())[::-1]
    if confidences.size >= 2:
        margin = confidences[0] - confidences[1]
        stats_margin = {
            "value": float(margin),
            "status": "ok" if margin >= 0.05 else "low_margin",
            "target": {"min": 0.05, "max": 1.0},
        }
    else:
        margin = None
        stats_margin = {"value": None, "status": "insufficient_classes"}

    if margin is not None and margin < 0.05:
        warnings.append(f"prob_margin: value={margin:.4f}, target≥0.0500")

    stats = dict(stats)
    stats["warnings"] = warnings
    stats["top_confidence"] = float(confidences[0]) if confidences.size else None
    stats["second_confidence"] = float(confidences[1]) if confidences.size > 1 else None
    stats["prob_margin"] = stats_margin
    return stats


def predict_number(img_input, stroke_scale):
    ensure_model_loaded()
    result = preprocess_image(img_input, stroke_scale=stroke_scale)
    if result is None:
        blank_probs = {f"{i:02d}": 0.0 for i in range(OUTPUT_CLASSES)}
        empty_preview = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
        empty_diff = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
        diagnostics = {"warnings": ["Draw a number to see diagnostics."]}
        return None, blank_probs, empty_preview, empty_diff, diagnostics

    standardized_variants, preview, mean_diff, diagnostics = result

    variants_matrix = np.concatenate(standardized_variants, axis=1).astype(np.float32, copy=False)
    cache, probs_matrix = forward_prop(variants_matrix, params, training=False)
    logits_matrix = cache["Z_fc2"]
    avg_logits = np.mean(logits_matrix, axis=1, keepdims=True)
    probs = softmax(avg_logits)

    pred = int(get_predictions(probs)[0])

    prob_dict = {f"{i:02d}": float(probs[i, 0]) for i in range(OUTPUT_CLASSES)}
    diagnostics = enrich_diagnostics(diagnostics, probs)
    diagnostics["variants_used"] = int(probs_matrix.shape[1])
    diagnostics["variant_top_confidences"] = [
        float(probs_matrix[pred, idx]) for idx in range(probs_matrix.shape[1])
    ]
    return pred, prob_dict, (preview * 255).astype(np.uint8), mean_diff, diagnostics


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Elliot's MNIST-100 Classifier
        Draw a two-digit number (00-99). The model will predict the number, show the top class probabilities, and display diagnostics for the processed input.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Sketchpad()
            stroke_slider = gr.Slider(
                minimum=0.3,
                maximum=1.2,
                value=1.0,
                step=0.05,
                label="Stroke Intensity (scale)",
            )

        with gr.Column(scale=1):
            pred_box = gr.Number(label="Predicted Number", precision=0, value=None)
            label = gr.Label(
                num_top_classes=5,
                label="Class Probabilities",
                value={f"{i:02d}": 0.0 for i in range(OUTPUT_CLASSES)},
            )
            preview = gr.Image(label="Model Input Preview (28x56)", image_mode="L")
            mean_diff_view = gr.Image(label="Difference vs Training Mean", image_mode="L")
            diagnostics_box = gr.JSON(label="Diagnostics")
            predict_btn = gr.Button("Predict", variant="primary")
            clear_btn = gr.ClearButton(
                [canvas, stroke_slider, pred_box, label, preview, mean_diff_view, diagnostics_box]
            )

    predict_btn.click(
        fn=predict_number,
        inputs=[canvas, stroke_slider],
        outputs=[pred_box, label, preview, mean_diff_view, diagnostics_box],
    )
    canvas.change(
        fn=predict_number,
        inputs=[canvas, stroke_slider],
        outputs=[pred_box, label, preview, mean_diff_view, diagnostics_box],
    )


if __name__ == "__main__":
    demo.launch()
