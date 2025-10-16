"""
Section 1: Imports and training paramaters for the MLP model 
"""
import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive"

np.random.seed(42)


INPUT_DIM = 784
HIDDEN_DIMS = [256, 128, ]
OUTPUT_DIM = 10
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
REG_LAMBDA = 5e-4
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8


"""
Section 2: Loads the input data, transposes (so arrays are features x samples) and normalises it (scales features to 0-1)
"""
def load_data(path):
    path = Path(path)
    data = pd.read_csv(path).to_numpy(dtype=np.float32)
    m, n = data.shape
    np.random.shuffle(data)

    data_dev = data[:10000].T
    Y_dev = data_dev[0].astype(np.int64)
    X_dev = data_dev[1:n]

    data_train = data[10000:].T
    Y_train = data_train[0].astype(np.int64)
    X_train = data_train[1:n]

    return X_train, Y_train, X_dev, Y_dev


def normalize_features(X_train, X_dev):
    X_train /= 255.0
    X_dev /= 255.0

    mean = np.mean(X_train, axis=1, keepdims=True)
    std = np.std(X_train, axis=1, keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_dev = (X_dev - mean) / std

    return X_train, X_dev, mean, std


"""
Section 3: Initialises the parameters (layers, weights and biases) and adam optimizer
"""
def init_params():
    layer_dims = [INPUT_DIM, *HIDDEN_DIMS, OUTPUT_DIM]
    params = {}
    for idx in range(1, len(layer_dims)):
        fan_in = layer_dims[idx - 1]
        params[f"W{idx}"] = np.random.randn(layer_dims[idx], fan_in) * np.sqrt(2.0 / fan_in)
        params[f"b{idx}"] = np.zeros((layer_dims[idx], 1), dtype=np.float32)
    return params

def init_adam(params):
    v = {}
    s = {}
    for key, value in params.items():
        v[key] = np.zeros_like(value)
        s[key] = np.zeros_like(value)
    return v, s


"""
Section 4: Defines the relu activation function ReLu (and backward ReLu) function, softmax function and one hot encoding function
"""
def relu(Z):
    return np.maximum(0.0, Z)


def relu_backward(Z):
    return (Z > 0).astype(np.float32)


def softmax(Z):
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=0, keepdims=True)


def one_hot(Y, num_classes=OUTPUT_DIM):
    one_hot_y = np.zeros((num_classes, Y.size), dtype=np.float32)
    one_hot_y[Y, np.arange(Y.size)] = 1.0
    return one_hot_y


"""
Section 5: Forward propagation (foward pass and activation functions - ReLu and softmax) and returns the cache and the class probabilities
"""
def forward_prop(X, params):
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    W3, b3 = params["W3"], params["b3"]

    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = relu(Z2)
    Z3 = W3 @ A2 + b3
    A3 = softmax(Z3)

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "A3": A3,
    }

    return cache, A3


"""
Section 6: Computes the loss and l2 regularization and returns the total loss
"""
def compute_loss(probs, Y_batch, params):
    m = Y_batch.shape[1]
    log_likelihood = -np.log(probs + 1e-9) * Y_batch
    data_loss = np.sum(log_likelihood) / m

    l2_penalty = 0.0
    for key in ("W1", "W2", "W3"):
        l2_penalty += np.sum(np.square(params[key]))
    l2_loss = (REG_LAMBDA / (2 * m)) * l2_penalty

    return data_loss + l2_loss


"""
Section 7: Back propagation (backward pass and activation functions - ReLu and softmax) and returns the gradients
"""
def back_prop(cache, Y_batch, params):
    m = Y_batch.shape[1]
    grads = {}

    A3 = cache["A3"]
    A2 = cache["A2"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    Z1 = cache["Z1"]
    X = cache["X"]

    dZ3 = A3 - Y_batch
    grads["W3"] = (dZ3 @ A2.T) / m + (REG_LAMBDA / m) * params["W3"]
    grads["b3"] = np.sum(dZ3, axis=1, keepdims=True) / m

    dA2 = params["W3"].T @ dZ3
    dZ2 = dA2 * relu_backward(Z2)
    grads["W2"] = (dZ2 @ A1.T) / m + (REG_LAMBDA / m) * params["W2"]
    grads["b2"] = np.sum(dZ2, axis=1, keepdims=True) / m

    dA1 = params["W2"].T @ dZ2
    dZ1 = dA1 * relu_backward(Z1)
    grads["W1"] = (dZ1 @ X.T) / m + (REG_LAMBDA / m) * params["W1"]
    grads["b1"] = np.sum(dZ1, axis=1, keepdims=True) / m

    return grads


"""
Section 8: Updates the parameters using the adam optimizer usings the calculated gradients
"""
def update_params_adam(params, grads, v, s, t):
    updated_params = {}
    for key in params:
        v[key] = BETA1 * v[key] + (1 - BETA1) * grads[key]
        s[key] = BETA2 * s[key] + (1 - BETA2) * (grads[key] ** 2)

        v_corrected = v[key] / (1 - BETA1 ** t)
        s_corrected = s[key] / (1 - BETA2 ** t)

        updated_params[key] = params[key] - LEARNING_RATE * v_corrected / (np.sqrt(s_corrected) + EPSILON)

    return updated_params, v, s


"""
Section 9: Gets the predictions and accuracy
"""
def get_predictions(probs):
    return np.argmax(probs, axis=0)


def get_accuracy(probs, labels):
    predictions = get_predictions(probs)
    return np.mean(predictions == labels)


"""
Section 10: Trains the model using the adam optimizer and returns the trained parameters
"""
def train_model(X_train, Y_train, X_dev, Y_dev):
    params = init_params()
    v, s = init_adam(params)
    m_train = X_train.shape[1]
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        permutation = np.random.permutation(m_train)
        X_shuffled = X_train[:, permutation]
        Y_shuffled = Y_train[permutation]

        epoch_loss = 0.0

        for start in range(0, m_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, m_train)
            X_batch = X_shuffled[:, start:end]
            Y_batch_indices = Y_shuffled[start:end]
            Y_batch = one_hot(Y_batch_indices)

            cache, probs = forward_prop(X_batch, params)
            loss = compute_loss(probs, Y_batch, params)
            grads = back_prop(cache, Y_batch, params)

            global_step += 1
            params, v, s = update_params_adam(params, grads, v, s, global_step)

            epoch_loss += loss * (end - start)

        epoch_loss /= m_train

        _, train_probs = forward_prop(X_train, params)
        train_accuracy = get_accuracy(train_probs, Y_train)

        _, dev_probs = forward_prop(X_dev, params)
        dev_accuracy = get_accuracy(dev_probs, Y_dev)

        print(f"Epoch {epoch:02d} - loss: {epoch_loss:.4f} - train_acc: {train_accuracy:.4f} - dev_acc: {dev_accuracy:.4f}")

    return params


"""
Section 11: Evaluates and saves+loads the model
"""

def evaluate(params, X, Y):
    _, probs = forward_prop(X, params)
    predictions = get_predictions(probs)
    accuracy = np.mean(predictions == Y)
    return predictions, accuracy


def save_model(params, mean, std, filepath=None):
    target_path = Path(filepath) if filepath is not None else ARCHIVE_DIR / "trained_model.npz"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving trained model to '{target_path}'...")
    np.savez(target_path, **params, mean=mean, std=std)
    print(f"Model saved successfully!")


def load_model(filepath=None):
    source_path = Path(filepath) if filepath is not None else ARCHIVE_DIR / "trained_model.npz"
    print(f"Loading model from '{source_path}'...")
    loaded = np.load(source_path)
    params = {key: loaded[key] for key in loaded.files if key not in {"mean", "std"}}
    mean = loaded["mean"]
    std = loaded["std"]
    print(f"Model loaded successfully!")
    return params, mean, std


"""
Section 12: main function, trains the model and prints final dev accuracy and saves the model
"""
def main():
    X_train, Y_train, X_dev, Y_dev = load_data(ARCHIVE_DIR / "mnist_train.csv")
    X_train, X_dev, mean, std = normalize_features(X_train, X_dev)

    print(f"Training samples: {X_train.shape[1]}, features: {X_train.shape[0]}")

    params = train_model(X_train, Y_train, X_dev, Y_dev)

    dev_predictions, dev_accuracy = evaluate(params, X_dev, Y_dev)
    print(f"\nFinal Dev Accuracy: {dev_accuracy:.4f}")
    
    # Save the trained model
    save_model(params, mean, std, ARCHIVE_DIR / "trained_model.npz")


if __name__ == "__main__":
    main()
