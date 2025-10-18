"""
Section 1: Imports and network configurations
"""

from __future__ import annotations

import numpy as np
import argparse
import csv
from pathlib import Path
from copy import deepcopy
from numpy.lib.stride_tricks import sliding_window_view


BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive"
TRAIN_CSV_PATH = ARCHIVE_DIR / "mnist_train.csv"
TEST_CSV_PATH = ARCHIVE_DIR / "mnist_test.csv"

np.random.seed(42)


# Network configuration
IMAGE_CHANNELS = 1
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 56
INPUT_DIM = IMAGE_HEIGHT * IMAGE_WIDTH  # flattened input for compatibility
CONV_FILTERS = (16, 32)
KERNEL_SIZE = 3
POOL_SIZE = 2
FC_HIDDEN_DIM = 256
OUTPUT_DIM = 100
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
REG_LAMBDA = 1e-4
DROP_RATE_FC = 0.4
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-3
MAX_SHIFT_PIXELS = 2
CONTRAST_JITTER_STD = 0.1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
DEV_SIZE = 10_000  # held-out validation set size


def save_history_to_csv(history, filepath):
    target_path = Path(filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=("epoch", "loss", "train_acc", "dev_acc"))
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_sweep_summary(results, filepath, *, include_trial=False):
    target_path = Path(filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["learning_rate", "reg_lambda", "dev_acc"]
    if include_trial:
        fieldnames.insert(0, "trial")
    with target_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in results:
            row = {
                "learning_rate": float(entry["learning_rate"]),
                "reg_lambda": float(entry["reg_lambda"]),
                "dev_acc": float(entry["dev_acc"]),
            }
            if include_trial:
                row["trial"] = int(entry["trial"])
            writer.writerow(row)

"""
Section 2: Loads the input data, transposes (so arrays are feature x samples) and normalises it (scales features to 0-1)
"""
def load_data(train_csv_path: Path = None, test_csv_path: Path = None, dev_size: int = DEV_SIZE):
    """
    Load the MNIST-100 dataset from CSV files and return
    training / validation splits flattened to (features, samples).
    """
    if train_csv_path is None:
        train_csv_path = TRAIN_CSV_PATH
    if test_csv_path is None:
        test_csv_path = TEST_CSV_PATH
    
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found at '{train_csv_path}'")
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found at '{test_csv_path}'")

    # Load training data from CSV
    import pandas as pd
    train_df = pd.read_csv(train_csv_path)
    train_labels = train_df['label'].values.astype(np.int64)
    train_pixels = train_df.drop('label', axis=1).values.astype(np.float32)
    
    # Load test data from CSV
    test_df = pd.read_csv(test_csv_path)
    test_labels = test_df['label'].values.astype(np.int64)
    test_pixels = test_df.drop('label', axis=1).values.astype(np.float32)

    # Convert to column-major format (features, samples)
    X_full = train_pixels.T  # (input_dim, m)
    Y_full = train_labels

    # Shuffle before splitting to validation
    permutation = np.random.permutation(X_full.shape[1])
    X_full = X_full[:, permutation]
    Y_full = Y_full[permutation]

    X_dev = X_full[:, :dev_size]
    Y_dev = Y_full[:dev_size]
    X_train = X_full[:, dev_size:]
    Y_train = Y_full[dev_size:]

    # Test set is already flattened
    X_test = test_pixels.T

    return X_train, Y_train, X_dev, Y_dev, X_test, test_labels


"""
Section 3: Normalises the features [(0, 255)] to [(0, 1)]
"""
def normalize_features(X_train, X_dev):
    """
    Normalize features to zero mean and unit variance using the training set.
    """
    X_train /= 255.0
    X_dev /= 255.0

    mean = np.mean(X_train, axis=1, keepdims=True)
    std = np.std(X_train, axis=1, keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_dev = (X_dev - mean) / std

    return X_train, X_dev, mean, std


"""
Section 4: Initialises the parameters (layers, weights and biases) and adam optimizer
"""
def init_params():
    params = {}
    conv1_fan_in = IMAGE_CHANNELS * KERNEL_SIZE * KERNEL_SIZE
    params["conv1_W"] = (
        np.random.randn(CONV_FILTERS[0], IMAGE_CHANNELS, KERNEL_SIZE, KERNEL_SIZE) * np.sqrt(2.0 / conv1_fan_in)
    ).astype(np.float32)
    params["conv1_b"] = np.zeros((CONV_FILTERS[0], 1), dtype=np.float32)

    conv2_fan_in = CONV_FILTERS[0] * KERNEL_SIZE * KERNEL_SIZE
    params["conv2_W"] = (
        np.random.randn(CONV_FILTERS[1], CONV_FILTERS[0], KERNEL_SIZE, KERNEL_SIZE) * np.sqrt(2.0 / conv2_fan_in)
    ).astype(np.float32)
    params["conv2_b"] = np.zeros((CONV_FILTERS[1], 1), dtype=np.float32)

    height_after_pool1 = IMAGE_HEIGHT // POOL_SIZE
    width_after_pool1 = IMAGE_WIDTH // POOL_SIZE
    height_after_pool2 = height_after_pool1 // POOL_SIZE
    width_after_pool2 = width_after_pool1 // POOL_SIZE
    flattened_dim = CONV_FILTERS[1] * height_after_pool2 * width_after_pool2

    params["fc1_W"] = (
        np.random.randn(FC_HIDDEN_DIM, flattened_dim) * np.sqrt(2.0 / flattened_dim)
    ).astype(np.float32)
    params["fc1_b"] = np.zeros((FC_HIDDEN_DIM, 1), dtype=np.float32)

    params["fc2_W"] = (
        np.random.randn(OUTPUT_DIM, FC_HIDDEN_DIM) * np.sqrt(2.0 / FC_HIDDEN_DIM)
    ).astype(np.float32)
    params["fc2_b"] = np.zeros((OUTPUT_DIM, 1), dtype=np.float32)

    return params


def init_adam(params):
    v = {}
    s = {}
    for key, value in params.items():
        v[key] = np.zeros_like(value)
        s[key] = np.zeros_like(value)
    return v, s


"""
Section 5: ReLu activation function and backward ReLu function
"""
def relu(Z):
    return np.maximum(0.0, Z)


def relu_backward(Z):
    return (Z > 0).astype(np.float32)


"""
Section 6: Reshapes the flattened input to 4D tensors (batch, channels, height, width) for the convolutional layers
"""
def reshape_flat_to_images(X: np.ndarray, *, batch_size: int | None = None):
    """
    Convert flattened columns (features, batch) into 4D tensors (batch, channels, height, width).
    """
    _, m = X.shape
    if batch_size is not None and m != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got {m}")
    images = X.T.reshape(m, IMAGE_HEIGHT, IMAGE_WIDTH)
    return images[:, None, :, :]  # add channel dim


"""
Section 7: Convolutional layer forward pass and backward pass
"""

def im2col(X, kernel_h, kernel_w, stride, padding):
    X_padded = np.pad(
        X,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    windows = sliding_window_view(X_padded, (kernel_h, kernel_w), axis=(2, 3))
    # windows shape: (batch, channels, out_height, out_width, kernel_h, kernel_w)
    batch_size, channels, out_height, out_width, _, _ = windows.shape
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size * out_height * out_width, channels * kernel_h * kernel_w)
    return X_padded, cols, out_height, out_width


def col2im(cols, X_shape, kernel_h, kernel_w, stride, padding, out_height, out_width):
    batch_size, channels, height, width = X_shape
    cols_reshaped = cols.reshape(batch_size, out_height, out_width, channels, kernel_h, kernel_w)
    cols_reshaped = cols_reshaped.transpose(0, 3, 1, 2, 4, 5)
    X_padded = np.zeros((batch_size, channels, height + 2 * padding, width + 2 * padding), dtype=np.float32)

    for h_idx in range(out_height):
        h_start = h_idx * stride
        h_end = h_start + kernel_h
        for w_idx in range(out_width):
            w_start = w_idx * stride
            w_end = w_start + kernel_w
            X_padded[:, :, h_start:h_end, w_start:w_end] += cols_reshaped[:, :, h_idx, w_idx, :, :]

    if padding > 0:
        return X_padded[:, :, padding:-padding, padding:-padding]
    return X_padded


def conv_forward(X, W, b, *, stride: int = 1, padding: int = 0):
    batch_size, in_channels, height, width = X.shape
    num_filters, _, kernel_h, kernel_w = W.shape

    X_padded, cols, out_height, out_width = im2col(X, kernel_h, kernel_w, stride, padding)
    W_col = W.reshape(num_filters, -1)
    out_cols = cols @ W_col.T  # (batch*out_height*out_width, num_filters)
    out = out_cols.reshape(batch_size, out_height, out_width, num_filters).transpose(0, 3, 1, 2)
    out = out.astype(np.float32, copy=False)
    out += b.reshape(1, num_filters, 1, 1)

    cache = {
        "X": X,
        "X_padded": X_padded,
        "W": W,
        "stride": stride,
        "padding": padding,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "out_height": out_height,
        "out_width": out_width,
        "cols": cols,
        "W_col": W_col,
        "output_shape": out.shape,
    }
    return out, cache


def conv_backward(dout, cache):
    X = cache["X"]
    W = cache["W"]
    stride = cache["stride"]
    padding = cache["padding"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]
    out_height = cache["out_height"]
    out_width = cache["out_width"]
    cols = cache["cols"]
    W_col = cache["W_col"]

    batch_size, _, _, _ = X.shape
    num_filters = W.shape[0]

    dout_cols = dout.transpose(0, 2, 3, 1).reshape(batch_size * out_height * out_width, num_filters)
    dW_col = dout_cols.T @ cols
    dW = dW_col.reshape(W.shape)
    db = np.sum(dout, axis=(0, 2, 3)).reshape(num_filters, 1)

    dcols = dout_cols @ W_col
    dX = col2im(dcols, X.shape, kernel_h, kernel_w, stride, padding, out_height, out_width)

    return dX, dW, db



"""
Section 8: Max pooling layer forward pass and backward pass
"""
def maxpool_forward(X, *, pool_size: int = 2, stride: int = 2):
    batch_size, channels, height, width = X.shape
    out_height = (height - pool_size) // stride + 1
    out_width = (width - pool_size) // stride + 1

    out = np.zeros((batch_size, channels, out_height, out_width), dtype=np.float32)

    for h_idx in range(out_height):
        h_start = h_idx * stride
        h_end = h_start + pool_size
        for w_idx in range(out_width):
            w_start = w_idx * stride
            w_end = w_start + pool_size
            window = X[:, :, h_start:h_end, w_start:w_end]
            max_vals = np.max(window, axis=(2, 3))
            out[:, :, h_idx, w_idx] = max_vals

    cache = {
        "X": X,
        "pool_size": pool_size,
        "stride": stride,
        "output_shape": out.shape,
    }
    return out, cache


def maxpool_backward(dout, cache):
    X = cache["X"]
    pool_size = cache["pool_size"]
    stride = cache["stride"]
    batch_size, channels, out_height, out_width = dout.shape

    dX = np.zeros_like(X)
    for h_idx in range(out_height):
        h_start = h_idx * stride
        h_end = h_start + pool_size
        for w_idx in range(out_width):
            w_start = w_idx * stride
            w_end = w_start + pool_size
            window = X[:, :, h_start:h_end, w_start:w_end]
            max_vals = np.max(window, axis=(2, 3), keepdims=True)
            mask = (window == max_vals).astype(np.float32)
            mask_sum = np.sum(mask, axis=(2, 3), keepdims=True)
            mask /= np.maximum(mask_sum, 1.0)
            grad_slice = dout[:, :, h_idx, w_idx][:, :, None, None]
            dX[:, :, h_start:h_end, w_start:w_end] += mask * grad_slice
    return dX


def softmax(Z):
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=0, keepdims=True)


def one_hot(Y, num_classes=OUTPUT_DIM):
    one_hot_y = np.zeros((num_classes, Y.size), dtype=np.float32)
    one_hot_y[Y, np.arange(Y.size)] = 1.0
    return one_hot_y



"""
Section 9: Forward propagation and comptutes for loss
"""
def forward_prop(
    X,
    params,
    *,
    training: bool = False,
    dropout_rate: float = DROP_RATE_FC,
):
    batch_size = X.shape[1]
    images = reshape_flat_to_images(X, batch_size=batch_size)
    padding = KERNEL_SIZE // 2

    conv1_out, conv1_cache = conv_forward(images, params["conv1_W"], params["conv1_b"], stride=1, padding=padding)
    relu1 = relu(conv1_out)
    pool1_out, pool1_cache = maxpool_forward(relu1, pool_size=POOL_SIZE, stride=POOL_SIZE)

    conv2_out, conv2_cache = conv_forward(pool1_out, params["conv2_W"], params["conv2_b"], stride=1, padding=padding)
    relu2 = relu(conv2_out)
    pool2_out, pool2_cache = maxpool_forward(relu2, pool_size=POOL_SIZE, stride=POOL_SIZE)

    flattened = pool2_out.reshape(batch_size, -1).T  # (features_flat, batch)

    Z_fc1 = params["fc1_W"] @ flattened + params["fc1_b"]
    A_fc1 = relu(Z_fc1)

    dropout_mask = None
    keep_prob = 1.0 - dropout_rate
    if training and dropout_rate > 0.0:
        dropout_mask = (np.random.rand(*A_fc1.shape) >= dropout_rate).astype(np.float32)
        A_fc1 = (A_fc1 * dropout_mask) / keep_prob

    Z_fc2 = params["fc2_W"] @ A_fc1 + params["fc2_b"]
    probs = softmax(Z_fc2)

    cache = {
        "X": X,
        "images": images,
        "conv1_out": conv1_out,
        "conv1_cache": conv1_cache,
        "pool1_cache": pool1_cache,
        "conv2_out": conv2_out,
        "conv2_cache": conv2_cache,
        "pool2_cache": pool2_cache,
        "flattened": flattened,
        "Z_fc1": Z_fc1,
        "A_fc1": A_fc1,
        "dropout_mask": dropout_mask,
        "keep_prob": keep_prob,
        "dropout_rate": dropout_rate,
        "Z_fc2": Z_fc2,
        "probs": probs,
    }

    return cache, probs


def compute_loss(probs, Y_batch, params, reg_lambda):
    m = Y_batch.shape[1]
    log_likelihood = -np.log(probs + 1e-9) * Y_batch
    data_loss = np.sum(log_likelihood) / m

    l2_penalty = 0.0
    for key in ("conv1_W", "conv2_W", "fc1_W", "fc2_W"):
        l2_penalty += np.sum(np.square(params[key]))
    l2_loss = (reg_lambda / (2 * m)) * l2_penalty

    return data_loss + l2_loss


"""
Section 10: Back propagation for the CNN model
"""
def back_prop(cache, Y_batch, params, reg_lambda, dropout_rate):
    m = Y_batch.shape[1]
    grads = {}

    probs = cache["probs"]
    A_fc1 = cache["A_fc1"]
    Z_fc1 = cache["Z_fc1"]
    flattened = cache["flattened"]
    dropout_mask = cache["dropout_mask"]
    keep_prob = cache["keep_prob"]

    dZ_fc2 = probs - Y_batch
    grads["fc2_W"] = (dZ_fc2 @ A_fc1.T) / m + (reg_lambda / m) * params["fc2_W"]
    grads["fc2_b"] = np.sum(dZ_fc2, axis=1, keepdims=True) / m

    dA_fc1 = params["fc2_W"].T @ dZ_fc2
    if dropout_mask is not None:
        dA_fc1 = (dA_fc1 * dropout_mask) / keep_prob
    dZ_fc1 = dA_fc1 * relu_backward(Z_fc1)
    grads["fc1_W"] = (dZ_fc1 @ flattened.T) / m + (reg_lambda / m) * params["fc1_W"]
    grads["fc1_b"] = np.sum(dZ_fc1, axis=1, keepdims=True) / m

    dFlatten = params["fc1_W"].T @ dZ_fc1  # (flatten_dim, batch)
    pool2_shape = cache["pool2_cache"]["output_shape"]
    dPool2 = dFlatten.T.reshape(pool2_shape)

    dRelu2_input = maxpool_backward(dPool2, cache["pool2_cache"])
    dConv2 = dRelu2_input * relu_backward(cache["conv2_out"])
    dPool1_input, dConv2_W, dConv2_b = conv_backward(dConv2, cache["conv2_cache"])
    grads["conv2_W"] = dConv2_W / m + (reg_lambda / m) * params["conv2_W"]
    grads["conv2_b"] = dConv2_b / m

    dRelu1_input = maxpool_backward(dPool1_input, cache["pool1_cache"])
    dConv1 = dRelu1_input * relu_backward(cache["conv1_out"])
    _, dConv1_W, dConv1_b = conv_backward(dConv1, cache["conv1_cache"])
    grads["conv1_W"] = dConv1_W / m + (reg_lambda / m) * params["conv1_W"]
    grads["conv1_b"] = dConv1_b / m

    return grads


"""
Section 11: Updates the parameters using the adam optimizer
"""

def update_params_adam(params, grads, v, s, t, learning_rate):
    updated_params = {}
    for key in params:
        v[key] = BETA1 * v[key] + (1 - BETA1) * grads[key]
        s[key] = BETA2 * s[key] + (1 - BETA2) * (grads[key] ** 2)

        v_corrected = v[key] / (1 - BETA1 ** t)
        s_corrected = s[key] / (1 - BETA2 ** t)

        updated_params[key] = params[key] - learning_rate * v_corrected / (np.sqrt(s_corrected) + EPSILON)

    return updated_params, v, s


def get_predictions(probs):
    return np.argmax(probs, axis=0)


def get_accuracy(probs, labels):
    predictions = get_predictions(probs)
    return np.mean(predictions == labels)


"""
Section 12: Augments the batch with horizontal shifts and contrast/brightness jitter
"""

def augment_batch(
    X_batch,
    *,
    image_shape: tuple[int, int] = (28, 56),
    max_shift: int = MAX_SHIFT_PIXELS,
    contrast_jitter_std: float = CONTRAST_JITTER_STD,
):
    """
    Apply lightweight augmentation: horizontal shifts and contrast/brightness jitter.
    """
    if max_shift <= 0 and contrast_jitter_std <= 0.0:
        return X_batch

    batch_size = X_batch.shape[1]
    images = X_batch.T.reshape(batch_size, *image_shape)

    if max_shift > 0:
        shifts = np.random.randint(-max_shift, max_shift + 1, size=batch_size)
        for idx, shift in enumerate(shifts):
            if shift > 0:
                shifted = np.roll(images[idx], shift, axis=1)
                shifted[:, :shift] = 0.0
                images[idx] = shifted
            elif shift < 0:
                shift = -shift
                shifted = np.roll(images[idx], -shift, axis=1)
                shifted[:, -shift:] = 0.0
                images[idx] = shifted

    if contrast_jitter_std > 0.0:
        scale = 1.0 + np.random.normal(0.0, contrast_jitter_std, size=batch_size)
        bias = np.random.normal(0.0, contrast_jitter_std, size=batch_size)
        images *= scale[:, None, None]
        images += bias[:, None, None]
        np.clip(images, -3.0, 3.0, out=images)

    return images.reshape(batch_size, -1).T


"""
Section 13: Trains the model + evaluates the model
"""
def train_model(
    X_train,
    Y_train,
    X_dev,
    Y_dev,
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    reg_lambda: float = REG_LAMBDA,
    dropout_rate: float = DROP_RATE_FC,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    early_stop_min_delta: float = EARLY_STOP_MIN_DELTA,
    use_augmentation: bool = True,
):
    params = init_params()
    v, s = init_adam(params)
    m_train = X_train.shape[1]
    global_step = 0
    best_dev_acc = -np.inf
    best_params = deepcopy(params)
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        permutation = np.random.permutation(m_train)
        X_shuffled = X_train[:, permutation]
        Y_shuffled = Y_train[permutation]

        epoch_loss = 0.0

        for start in range(0, m_train, batch_size):
            end = min(start + batch_size, m_train)
            X_batch = X_shuffled[:, start:end]
            Y_batch_indices = Y_shuffled[start:end]
            Y_batch = one_hot(Y_batch_indices)

            if use_augmentation:
                X_batch = augment_batch(X_batch.copy())

            cache, probs = forward_prop(
                X_batch,
                params,
                training=True,
                dropout_rate=dropout_rate,
            )
            loss = compute_loss(probs, Y_batch, params, reg_lambda)
            grads = back_prop(cache, Y_batch, params, reg_lambda, dropout_rate)

            global_step += 1
            params, v, s = update_params_adam(params, grads, v, s, global_step, learning_rate)

            epoch_loss += loss * (end - start)

        epoch_loss /= m_train

        _, train_probs = forward_prop(X_train, params, training=False, dropout_rate=dropout_rate)
        train_accuracy = get_accuracy(train_probs, Y_train)

        _, dev_probs = forward_prop(X_dev, params, training=False, dropout_rate=dropout_rate)
        dev_accuracy = get_accuracy(dev_probs, Y_dev)

        print(
            f"Epoch {epoch:02d} - loss: {epoch_loss:.4f} "
            f"- train_acc: {train_accuracy:.4f} - dev_acc: {dev_accuracy:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "train_acc": train_accuracy,
                "dev_acc": dev_accuracy,
            }
        )

        if dev_accuracy > best_dev_acc + early_stop_min_delta:
            best_dev_acc = dev_accuracy
            best_params = deepcopy(params)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(
                    f"Early stopping triggered at epoch {epoch:02d}. "
                    f"Best dev_acc={best_dev_acc:.4f}"
                )
                break

    return best_params, history


def evaluate(params, X, Y):
    _, probs = forward_prop(X, params, training=False)
    predictions = get_predictions(probs)
    accuracy = np.mean(predictions == Y)
    return predictions, accuracy


"""
Section 14: Trains the model once
"""
def train_once(
    learning_rate: float,
    reg_lambda: float,
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    dropout_rate: float = DROP_RATE_FC,
    history_path: Path | None = None,
):
    """
    Convenience wrapper for hyperparameter sweeps. Returns trained params and dev accuracy.
    """
    X_train, Y_train, X_dev, Y_dev, _, _ = load_data()
    X_train, X_dev, mean, std = normalize_features(X_train, X_dev)

    params, history = train_model(
        X_train,
        Y_train,
        X_dev,
        Y_dev,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        dropout_rate=dropout_rate,
    )

    _, dev_accuracy = evaluate(params, X_dev, Y_dev)

    if history_path is not None:
        save_history_to_csv(history, history_path)

    return params, dev_accuracy, mean, std, history

"""
Section 15: Hyperparameter sweep for learning rate, regularization and dropout rate
"""

def lr_sweep(
    learning_rates: list[float],
    *,
    reg_lambda: float = REG_LAMBDA,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    dropout_rate: float = DROP_RATE_FC,
    history_dir: Path | None = None,
    summary_path: Path | None = None,
):
    results = []
    history_directory = Path(history_dir) if history_dir is not None else None
    if history_directory is not None:
        history_directory.mkdir(parents=True, exist_ok=True)

    for lr in learning_rates:
        history_path = None
        if history_directory is not None:
            safe_lr = f"{lr:.2e}".replace("+", "").replace("-", "m")
            history_path = history_directory / f"lr_{safe_lr}.csv"
        _, dev_acc, _, _, history = train_once(
            lr,
            reg_lambda,
            epochs=epochs,
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            history_path=history_path,
        )
        results.append(
            {
                "learning_rate": float(lr),
                "reg_lambda": float(reg_lambda),
                "dev_acc": float(dev_acc),
                "history": history,
            }
        )
    if summary_path is not None:
        save_sweep_summary(results, summary_path)
    return results


def random_search_hparams(
    num_trials: int,
    lr_bounds: tuple[float, float],
    reg_bounds: tuple[float, float],
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    dropout_rate: float = DROP_RATE_FC,
    seed: int | None = None,
    history_dir: Path | None = None,
    summary_path: Path | None = None,
):
    if num_trials <= 0:
        raise ValueError("num_trials must be positive")

    lr_min, lr_max = lr_bounds
    reg_min, reg_max = reg_bounds
    if lr_min <= 0 or lr_max <= 0:
        raise ValueError("Learning rate bounds must be positive")
    if reg_min <= 0 or reg_max <= 0:
        raise ValueError("Regularization bounds must be positive")

    rng = np.random.default_rng(seed)
    history_directory = Path(history_dir) if history_dir is not None else None
    if history_directory is not None:
        history_directory.mkdir(parents=True, exist_ok=True)

    results = []
    log_lr_min, log_lr_max = np.log(lr_min), np.log(lr_max)
    log_reg_min, log_reg_max = np.log(reg_min), np.log(reg_max)

    for trial in range(1, num_trials + 1):
        lr_sample = float(np.exp(rng.uniform(log_lr_min, log_lr_max)))
        reg_sample = float(np.exp(rng.uniform(log_reg_min, log_reg_max)))
        history_path = None
        if history_directory is not None:
            safe_lr = f"{lr_sample:.2e}".replace("+", "").replace("-", "m")
            safe_reg = f"{reg_sample:.2e}".replace("+", "").replace("-", "m")
            history_path = history_directory / f"trial_{trial:02d}_lr-{safe_lr}_reg-{safe_reg}.csv"

        _, dev_acc, _, _, history = train_once(
            lr_sample,
            reg_sample,
            epochs=epochs,
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            history_path=history_path,
        )

        results.append(
            {
                "trial": trial,
                "learning_rate": lr_sample,
                "reg_lambda": reg_sample,
                "dev_acc": float(dev_acc),
                "history": history,
            }
        )

    results.sort(key=lambda item: item["dev_acc"], reverse=True)
    if summary_path is not None:
        save_sweep_summary(results, summary_path, include_trial=True)
    return results


def auto_train_pipeline(
    *,
    trials: int,
    lr_bounds: tuple[float, float],
    reg_bounds: tuple[float, float],
    search_epochs: int,
    final_epochs: int,
    batch_size: int,
    dropout_rate: float,
    final_batch_size: int | None,
    final_dropout_rate: float | None,
    history_dir: Path | None,
    seed: int | None,
    output_model_path: Path | None,
):
    history_directory = Path(history_dir) if history_dir is not None else None
    if history_directory is not None:
        history_directory.mkdir(parents=True, exist_ok=True)

    search_summary_path = None
    if history_directory is not None:
        search_summary_path = history_directory / "random_search_summary.csv"

    results = random_search_hparams(
        trials,
        lr_bounds,
        reg_bounds,
        epochs=search_epochs,
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        seed=seed,
        history_dir=history_directory / "search_histories" if history_directory is not None else None,
        summary_path=search_summary_path,
    )
    best = results[0]
    print(
        f"\nBest search trial -> LR={best['learning_rate']:.3e}, "
        f"reg={best['reg_lambda']:.3e}, dev_acc={best['dev_acc']:.4f}"
    )

    final_dropout = final_dropout_rate if final_dropout_rate is not None else dropout_rate
    final_history_path = None
    if history_directory is not None:
        final_history_path = history_directory / "final_train_history.csv"

    params, final_dev_acc, mean, std, final_history = train_once(
        best["learning_rate"],
        best["reg_lambda"],
        epochs=final_epochs,
        batch_size=final_batch_size or batch_size,
        dropout_rate=final_dropout,
        history_path=final_history_path,
    )

    model_output_path = output_model_path if output_model_path is not None else ARCHIVE_DIR / "trained_model_mnist100.npz"
    save_model(params, mean, std, model_output_path)

    return {
        "best_trial": best,
        "final_dev_acc": final_dev_acc,
        "model_path": Path(model_output_path),
        "final_history": final_history,
    }


"""
Section 16: Saves the model
"""
def save_model(params, mean, std, filepath=None):
    target_path = Path(filepath) if filepath is not None else ARCHIVE_DIR / "trained_model_mnist100.npz"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving trained model to '{target_path}'...")
    np.savez(target_path, **params, mean=mean, std=std)
    print("Model saved successfully!")


"""
Section 17: Main function
"""

def main():
    parser = argparse.ArgumentParser(description="MNIST-100 training and tuning utilities.")
    parser.add_argument(
        "--mode",
        choices=("train", "lr-sweep", "random-search", "auto-train"),
        default="train",
        help="Select high-level action.",
    )
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Base learning rate.")
    parser.add_argument("--learning-rates", type=str, help="Comma-separated list for LR sweep.")
    parser.add_argument("--reg-lambda", type=float, default=REG_LAMBDA, help="L2 regularization strength.")
    parser.add_argument("--lr-min", type=float, default=1e-4, help="Min LR for random search (exclusive mode).")
    parser.add_argument("--lr-max", type=float, default=5e-3, help="Max LR for random search.")
    parser.add_argument("--reg-min", type=float, default=1e-5, help="Min lambda for random search.")
    parser.add_argument("--reg-max", type=float, default=1e-3, help="Max lambda for random search.")
    parser.add_argument("--trials", type=int, default=5, help="Number of random-search trials.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Train epochs per run.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=40,
        help="Epoch budget for the final training run in auto-train mode.",
    )
    parser.add_argument(
        "--final-batch-size",
        type=int,
        help="Mini-batch size for the final training run (defaults to --batch-size).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="Override dropout rate for the fully connected layer.",
    )
    parser.add_argument(
        "--final-dropout",
        type=float,
        help="Dropout rate for the final training pass in auto-train mode.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        help="Directory for saving training histories (CSV).",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        help="Path to save the trained model (.npz). Defaults to archive/trained_model_mnist100.npz.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for random search.")
    args = parser.parse_args()

    dropout_rate = DROP_RATE_FC if args.dropout is None else float(args.dropout)
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("Dropout rate must be in [0, 1).")

    final_dropout_rate = None
    if args.final_dropout is not None:
        final_dropout_rate = float(args.final_dropout)
        if not 0.0 <= final_dropout_rate < 1.0:
            raise ValueError("Final dropout rate must be in [0, 1).")

    history_dir = args.history_dir
    if history_dir is not None:
        history_dir = Path(history_dir)
        history_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        print(f"Loading dataset from CSV files...")
        X_train, Y_train, X_dev, Y_dev, _, _ = load_data()
        X_train, X_dev, mean, std = normalize_features(X_train, X_dev)

        print(
            f"Training samples: {X_train.shape[1]}, features: {X_train.shape[0]} "
            f"| Dev samples: {X_dev.shape[1]}"
        )

        params, history = train_model(
            X_train,
            Y_train,
            X_dev,
            Y_dev,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            reg_lambda=args.reg_lambda,
            dropout_rate=dropout_rate,
        )

        _, dev_accuracy = evaluate(params, X_dev, Y_dev)
        print(f"\nFinal Dev Accuracy: {dev_accuracy:.4f}")

        if history_dir is not None:
            save_history_to_csv(history, history_dir / "train_history.csv")

        save_model(params, mean, std, args.output_model or ARCHIVE_DIR / "trained_model_mnist100.npz")

    elif args.mode == "lr-sweep":
        if args.learning_rates is None:
            raise ValueError("LR sweep mode requires --learning-rates.")
        lr_values = [float(value.strip()) for value in args.learning_rates.split(",") if value.strip()]
        print(f"Running LR sweep over {lr_values}...")
        summary_path = history_dir / "lr_sweep_summary.csv" if history_dir is not None else None
        results = lr_sweep(
            lr_values,
            reg_lambda=args.reg_lambda,
            epochs=args.epochs,
            batch_size=args.batch_size,
            dropout_rate=dropout_rate,
            history_dir=history_dir,
            summary_path=summary_path,
        )
        for entry in results:
            print(
                f"LR={entry['learning_rate']:.3e} | reg={entry['reg_lambda']:.3e} "
                f"| dev_acc={entry['dev_acc']:.4f}"
            )

    elif args.mode == "random-search":
        print(
            f"Running random search ({args.trials} trials) "
            f"LR∈[{args.lr_min:.2e},{args.lr_max:.2e}], "
            f"λ∈[{args.reg_min:.2e},{args.reg_max:.2e}]..."
        )
        summary_path = history_dir / "random_search_summary.csv" if history_dir is not None else None
        results = random_search_hparams(
            args.trials,
            (args.lr_min, args.lr_max),
            (args.reg_min, args.reg_max),
            epochs=args.epochs,
            batch_size=args.batch_size,
            dropout_rate=dropout_rate,
            seed=args.seed,
            history_dir=history_dir,
            summary_path=summary_path,
        )
        for entry in results:
            print(
                f"Trial {entry['trial']:02d} | LR={entry['learning_rate']:.3e} "
                f"| reg={entry['reg_lambda']:.3e} | dev_acc={entry['dev_acc']:.4f}"
            )
        best = results[0]
        print(
            f"\nBest trial -> LR={best['learning_rate']:.3e}, "
            f"reg={best['reg_lambda']:.3e}, dev_acc={best['dev_acc']:.4f}"
        )

    elif args.mode == "auto-train":
        print(
            f"Auto-train pipeline: {args.trials} search trials "
            f"(epochs={args.epochs}) followed by final training (epochs={args.final_epochs})."
        )
        results = auto_train_pipeline(
            trials=args.trials,
            lr_bounds=(args.lr_min, args.lr_max),
            reg_bounds=(args.reg_min, args.reg_max),
            search_epochs=args.epochs,
            final_epochs=args.final_epochs,
            batch_size=args.batch_size,
            dropout_rate=dropout_rate,
            final_batch_size=args.final_batch_size,
            final_dropout_rate=final_dropout_rate,
            history_dir=history_dir,
            seed=args.seed,
            output_model_path=args.output_model,
        )
        best = results["best_trial"]
        print(
            f"\nAuto-train complete. "
            f"Best trial LR={best['learning_rate']:.3e}, reg={best['reg_lambda']:.3e}. "
            f"Final dev_acc={results['final_dev_acc']:.4f}. "
            f"Model saved to '{results['model_path']}'."
        )


if __name__ == "__main__":
    main()