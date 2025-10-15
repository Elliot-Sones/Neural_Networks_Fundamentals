import numpy as np
import pandas as pd
from pathlib import Path

# Configuration (must match training.py)
INPUT_DIM = 784
HIDDEN_DIMS = [256, 128]
OUTPUT_DIM = 10

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive"


def relu(Z):
    """ReLU activation function"""
    return np.maximum(0.0, Z)


def softmax(Z):
    """Numerically stable softmax"""
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=0, keepdims=True)


def forward_prop(X, params):
    """
    Forward propagation through the network
    
    Args:
        X: Input data (784, m)
        params: Dictionary containing W1, b1, W2, b2, W3, b3
    
    Returns:
        cache: Dictionary with intermediate values
        probs: Output probabilities (10, m)
    """
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


def get_predictions(probs):
    """Get predicted class from probabilities"""
    return np.argmax(probs, axis=0)


def get_accuracy(probs, labels):
    """Calculate accuracy"""
    predictions = get_predictions(probs)
    return np.mean(predictions == labels)


def load_model(filepath=None):
    """
    Load trained model parameters from disk
    
    Args:
        filepath: Path to the saved model file
    
    Returns:
        params: Dictionary of model parameters
    """
    source_path = Path(filepath) if filepath is not None else ARCHIVE_DIR / "trained_model.npz"
    print(f"Loading model from '{source_path}'...")
    loaded = np.load(source_path)
    params = {key: loaded[key] for key in loaded.files if key not in {"mean", "std"}}
    mean = loaded["mean"]
    std = loaded["std"]
    print(f"Model loaded successfully!")
    print(f"Model contains: {list(params.keys())}")
    return params, mean, std


def load_test_data(path=None):
    """
    Load and preprocess test data
    
    Args:
        path: Path to test CSV file
    
    Returns:
        X_test: Test features (784, m)
        Y_test: Test labels (m,)
    """
    source_path = Path(path) if path is not None else ARCHIVE_DIR / "mnist_test.csv"
    print(f"\nLoading test data from '{source_path}'...")
    data = pd.read_csv(source_path).to_numpy(dtype=np.float32)
    
    data_test = data.T
    Y_test = data_test[0].astype(np.int64)
    X_test = data_test[1:]
    
    print(f"Test set size: {X_test.shape[1]} examples")
    return X_test, Y_test


def normalize_with_stats(X, mean, std):
    """
    Normalize data using provided mean and std
    
    Args:
        X: Input data
        mean: Mean from training set
        std: Standard deviation from training set
    
    Returns:
        X_normalized: Normalized data
    """
    return (X - mean) / std


def predict_single_image(image, params, mean, std, show_probabilities=True):
    """
    Make prediction on a single image
    
    Args:
        image: Image data (784,) or (784, 1)
        params: Trained model parameters
        show_probabilities: Whether to print class probabilities
    
    Returns:
        prediction: Predicted digit
        probabilities: Probability for each class
    """
    # Ensure correct shape
    if image.ndim == 1:
        image = image.reshape(784, 1)
    image = image.astype(np.float32) / 255.0
    image = normalize_with_stats(image, mean, std)
    
    # Forward pass
    _, probs = forward_prop(image, params)
    prediction = get_predictions(probs)[0]
    
    if show_probabilities:
        print("\nClass probabilities:")
        for digit in range(10):
            prob = probs[digit, 0]
            bar = "█" * int(prob * 50)
            print(f"  {digit}: {prob:.4f} {bar}")
    
    return prediction, probs[:, 0]


def evaluate_test_set(params, X_test, Y_test):
    """
    Evaluate model on test set
    
    Args:
        params: Trained model parameters
        X_test: Test features
        Y_test: Test labels
    
    Returns:
        predictions: Predicted labels
        accuracy: Test accuracy
    """
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    _, probs = forward_prop(X_test, params)
    predictions = get_predictions(probs)
    accuracy = np.mean(predictions == Y_test)
    
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-digit accuracy
    print("\nPer-digit accuracy:")
    for digit in range(10):
        digit_mask = Y_test == digit
        digit_correct = np.sum((predictions == Y_test) & digit_mask)
        digit_total = np.sum(digit_mask)
        digit_accuracy = digit_correct / digit_total if digit_total > 0 else 0
        print(f"  Digit {digit}: {digit_accuracy*100:5.2f}% ({digit_correct}/{digit_total})")
    
    return predictions, accuracy


def show_sample_predictions(params, X_test, Y_test, num_samples=10):
    """
    Show sample predictions
    
    Args:
        params: Trained model parameters
        X_test: Test features
        Y_test: Test labels
        num_samples: Number of samples to show
    """
    print("\n" + "="*60)
    print(f"SAMPLE PREDICTIONS (first {num_samples} examples)")
    print("="*60)
    
    _, probs = forward_prop(X_test[:, :num_samples], params)
    predictions = get_predictions(probs)
    
    print(f"Predicted: {predictions}")
    print(f"Actual:    {Y_test[:num_samples]}")
    print(f"Match:     {predictions == Y_test[:num_samples]}")


def main():
    # Load the trained model
    params, mean, std = load_model()

    # Load test data
    X_test_raw, Y_test = load_test_data()
    X_test_scaled = X_test_raw / 255.0
    X_test = normalize_with_stats(X_test_scaled, mean, std)

    # Show sample predictions
    show_sample_predictions(params, X_test, Y_test, num_samples=20)

    # Evaluate on full test set
    predictions, accuracy = evaluate_test_set(params, X_test, Y_test)
    
    # Test single image prediction
    print("\n" + "="*60)
    print("SINGLE IMAGE PREDICTION EXAMPLE")
    print("="*60)
    print(f"Testing image #0 (actual label: {int(Y_test[0])})")
    prediction, probs = predict_single_image(X_test_raw[:, 0], params, mean, std)
    print(f"\nPredicted digit: {prediction}")
    
    print("\n" + "="*60)
    print(f"FINAL TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
