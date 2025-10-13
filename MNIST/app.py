import numpy as np
import gradio as gr
from PIL import Image, ImageOps
from pathlib import Path

# Reuse existing helpers from your project
from training import load_model
from test_model import forward_prop, get_predictions


# Load trained model parameters and normalization stats on demand
def load_trained_artifacts(model_path=None):
    base_dir = Path(__file__).resolve().parent
    if model_path is None:
        resolved_path = base_dir / "archive" / "trained_model.npz"
    else:
        candidate = Path(model_path)
        resolved_path = candidate if candidate.is_absolute() else base_dir / candidate
    try:
        return load_model(resolved_path)
    except FileNotFoundError:
        raise RuntimeError(
            f"Model file '{resolved_path}' not found. Run 'python training.py' to train and create it before launching the app."
        )


params, mean, std = None, None, None


def ensure_model_loaded():
    global params, mean, std
    if params is None or mean is None or std is None:
        params, mean, std = load_trained_artifacts()


def preprocess_image(img_input):
    """
    Convert a canvas image (PIL or ndarray) into a normalized (784, 1) float32 vector
    using the same preprocessing as training: scale to [0,1], then standardize
    with the training mean and std saved in the model file.
    """
    if img_input is None:
        return None

    # Accept PIL, numpy, or dict payloads from Sketchpad/Image variants
    if isinstance(img_input, dict):
        # Try common keys Gradio returns in different versions without boolean evaluation of arrays
        img_data = None
        for key in ("image", "composite", "background", "value"):
            if key in img_input and img_input[key] is not None:
                img_data = img_input[key]
                break
        if img_data is None:
            return None
        img_input = img_data

    if isinstance(img_input, Image.Image):
        img = img_input
    elif isinstance(img_input, np.ndarray):
        arr_in = img_input
        # Normalize dtype/range
        if arr_in.dtype != np.uint8:
            max_val = float(arr_in.max()) if arr_in.size > 0 else 1.0
            if max_val <= 1.5:
                arr_in = (arr_in * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr_in = np.clip(arr_in, 0, 255).astype(np.uint8)
        # If RGBA provided, preserve mode
        if arr_in.ndim == 3 and arr_in.shape[2] == 4:
            img = Image.fromarray(arr_in, mode="RGBA")
        else:
            img = Image.fromarray(arr_in)
    else:
        return None

    # If the sketch has transparency, composite over white background first
    try:
        bands = img.getbands()
    except Exception:
        bands = ()
    if "A" in bands:
        # Ensure RGBA, then alpha-composite onto white background
        rgba = img.convert("RGBA")
        white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white_bg, rgba).convert("RGB")

    # Ensure grayscale and invert to match MNIST (white digit on black background)
    img = img.convert("L")

    # Canvas usually returns black strokes on white background; MNIST is white on black.
    img = ImageOps.invert(img)

    # Convert to numpy for cropping
    arr_u8 = np.array(img, dtype=np.uint8)
    original_canvas_shape = arr_u8.shape

    # Find bounding box of the digit using a low threshold
    coords = np.column_stack(np.where(arr_u8 > 10))  # (y, x) where stroke exists
    bbox = None
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        bbox = (int(y_min), int(y_max), int(x_min), int(x_max))
        # Add margin
        pad = 4
        y_min = max(0, y_min - pad)
        x_min = max(0, x_min - pad)
        y_max = min(arr_u8.shape[0], y_max + pad)
        x_max = min(arr_u8.shape[1], x_max + pad)
        arr_u8 = arr_u8[y_min:y_max, x_min:x_max]

    # Pad to square
    h, w = arr_u8.shape
    side = max(h, w)
    pad_top = (side - h) // 2
    pad_bottom = side - h - pad_top
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left
    arr_square = np.pad(arr_u8, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)

    # Resize to 20x20 then pad 4 pixels to reach 28x28 (mirrors MNIST formatting)
    img_20 = Image.fromarray(arr_square).resize((20, 20), Image.Resampling.LANCZOS)
    arr_20 = np.array(img_20, dtype=np.float32)
    arr_28 = np.pad(arr_20, ((4, 4), (4, 4)), mode="constant", constant_values=0)

    arr_28_float = arr_28.astype(np.float32) / 255.0

    ensure_model_loaded()

    # Diagnostics before standardization
    mean_image = mean.reshape(28, 28)
    diagnostics = compute_diagnostics(arr_28_float, bbox, original_canvas_shape, mean_image)

    # Normalize like training and flatten to (784, 1)
    arr = arr_28_float.reshape(28 * 28, 1)
    standardized = (arr - mean) / std
    return standardized, arr_28_float, diagnostics


def predict_digit(img_input):
    ensure_model_loaded()
    result = preprocess_image(img_input)
    if result is None:
        blank_probs = {str(i): 0.0 for i in range(10)}
        empty_preview = np.zeros((28, 28), dtype=np.uint8)
        diagnostics = {"warnings": ["Draw a digit to see diagnostics."]}
        return None, blank_probs, empty_preview, diagnostics

    x, preview, diagnostics = result

    # Forward pass with loaded params
    _, probs = forward_prop(x, params)
    pred = int(get_predictions(probs)[0])

    # Convert probabilities to dict for Label component
    prob_dict = {str(i): float(probs[i, 0]) for i in range(10)}
    diagnostics = enrich_diagnostics(diagnostics, probs)
    return pred, prob_dict, (preview * 255).astype(np.uint8), diagnostics


def compute_diagnostics(arr_28_float, bbox, original_shape, mean_image):
    mass = arr_28_float
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
    bbox_metrics = {}
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
            "area_ratio": float(bbox_area / (28 * 28)) if bbox_area else 0.0,
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

    stats = {
        "total_intensity": total_intensity,
        "mass_fraction": float(total_intensity / (28 * 28)),
        "center_of_mass": {"row": cy, "col": cx},
        "center_offset": None,
        "bbox": bbox_metrics,
        "original_canvas_shape": original_shape,
        "stroke_density": density,
        "warnings": [],
    }

    if cy is not None and cx is not None:
        stats["center_offset"] = float(np.sqrt((cy - 13.5) ** 2 + (cx - 13.5) ** 2))

    if mean_image is not None:
        stats["distance_from_mean"] = float(np.linalg.norm(arr_28_float - mean_image))

    return stats


def enrich_diagnostics(stats, probs):
    warnings = []
    mass_fraction = stats.get("mass_fraction", 0.0)
    bbox = stats.get("bbox", {})
    center_offset = stats.get("center_offset")
    stroke_density = stats.get("stroke_density", 0.0)

    if mass_fraction < 0.02:
        warnings.append("Digit covers very little area; try drawing larger.")
    elif mass_fraction > 0.4:
        warnings.append("Digit fills most of the canvas; try leaving some margin.")

    if stroke_density < 0.1:
        warnings.append("Strokes look very thin compared to MNIST digits.")
    elif stroke_density > 0.7:
        warnings.append("Strokes look very thick; consider easing pen pressure.")

    aspect_ratio = bbox.get("aspect_ratio")
    if aspect_ratio is not None:
        if aspect_ratio < 0.3 or aspect_ratio > 3.5:
            warnings.append("Digit bounding box aspect ratio is unusual for MNIST; vertical strokes might confuse the model.")

    if center_offset is not None and center_offset > 6.0:
        warnings.append("Digit is far from centered; recentre it for better predictions.")

    confidences = np.sort(probs.flatten())[::-1]
    if confidences.size >= 2:
        margin = confidences[0] - confidences[1]
        if margin < 0.2:
            warnings.append("Model is unsure (top probabilities close); redraw may help.")

    stats = dict(stats)  # shallow copy
    stats["warnings"] = warnings
    stats["top_confidence"] = float(confidences[0]) if confidences.size else None
    stats["second_confidence"] = float(confidences[1]) if confidences.size > 1 else None
    return stats


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Elliot's Digit Classifier
        Welcome to Elliot's Digit Classifier! Draw a digit (0-9) below. The model will predict the digit, show class probabilities, and display diagnostics for the processed input.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Sketchpad()

        with gr.Column(scale=1):
            pred_box = gr.Number(label="Predicted Digit", precision=0, value=None)
            label = gr.Label(num_top_classes=10, label="Class Probabilities", value={str(i): 0.0 for i in range(10)})
            preview = gr.Image(label="Model Input Preview (28x28)", image_mode="L")
            diagnostics_box = gr.JSON(label="Diagnostics")
            predict_btn = gr.Button("Predict", variant="primary")
            clear_btn = gr.ClearButton([canvas, pred_box, label, preview, diagnostics_box])

    # Wire interactions
    # Predict on demand
    predict_btn.click(fn=predict_digit, inputs=canvas, outputs=[pred_box, label, preview, diagnostics_box])
    # Also try updating on change (may be a no-op depending on version)
    canvas.change(fn=predict_digit, inputs=canvas, outputs=[pred_box, label, preview, diagnostics_box])
    # ClearButton handles resetting components


if __name__ == "__main__":
    demo.launch()
