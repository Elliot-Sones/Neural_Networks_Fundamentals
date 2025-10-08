# Simple Neural Network for MNIST

A neural network implementation from scratch using NumPy to recognize handwritten digits from the MNIST dataset.

## Features

- **3-layer neural network**: 784 → 256 → 128 → 10
- **Adam optimizer**: Adaptive learning rates for faster convergence
- **L2 regularization**: Prevents overfitting
- **Mini-batch training**: Efficient batch processing (128 examples)
- **Model persistence**: Save and load trained models
- **Feature normalization**: Standardization for better performance

## Architecture

```
Input Layer:    784 neurons (28×28 pixels)
Hidden Layer 1: 256 neurons (ReLU activation)
Hidden Layer 2: 128 neurons (ReLU activation)
Output Layer:   10 neurons (Softmax activation)
```

## Expected Performance

- **Training Accuracy**: ~97-98%
- **Validation Accuracy**: ~96-97%
- **Test Accuracy**: ~96-97%

## Setup

### Requirements

```bash
pip install numpy pandas
```

### Download MNIST Data

Place the MNIST CSV files in the `archive/` directory:
- `archive/mnist_train.csv` (60,000 examples)
- `archive/mnist_test.csv` (10,000 examples)

## Usage

### 1. Train the Model

Train the neural network and save the weights:

```bash
python training.py
```

This will:
- Load and preprocess the training data
- Train for 15 epochs with mini-batch gradient descent
- Display loss and accuracy each epoch
- Save the trained model as `archive/trained_model.npz`

**Output:**
```
Epoch 01 - loss: 0.3245 - train_acc: 0.9021 - dev_acc: 0.8987
Epoch 02 - loss: 0.1876 - train_acc: 0.9456 - dev_acc: 0.9342
...
Epoch 15 - loss: 0.0823 - train_acc: 0.9756 - dev_acc: 0.9689

Final Dev Accuracy: 0.9689
Saving trained model to 'archive/trained_model.npz'...
Model saved successfully!
```

### 2. Test the Model

Evaluate the trained model on the test set:

```bash
python test_model.py
```

This will:
- Load the saved model weights
- Evaluate on the test dataset
- Show sample predictions
- Display per-digit accuracy
- Test single image prediction with probabilities

**Output:**
```
Loading model from 'archive/trained_model.npz'...
Model loaded successfully!

### 3. Run the Web App (Draw Digits)

This project includes a Gradio web app with a drawing canvas that loads the saved model and predicts digits in real time.

Install the extra dependency:

```bash
pip install gradio pillow
```

Launch the app:

```bash
python app.py
```

Then open the URL shown in the terminal (usually `http://127.0.0.1:7860`). Draw a digit and the app will display the predicted digit and class probabilities.

SAMPLE PREDICTIONS (first 20 examples)
Predicted: [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
Actual:    [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]

EVALUATING ON TEST SET
Test Accuracy: 0.9678 (96.78%)

Per-digit accuracy:
  Digit 0: 98.57% (966/980)
  Digit 1: 98.68% (1121/1135)
  ...
```

## Project Structure

```
simpleNeuralNetwork/
├── training.py          # Training script with model saving
├── test_model.py        # Testing script with model loading
├── archive/trained_model.npz    # Saved model weights (created after training)
├── archive/
│   ├── mnist_train.csv  # Training data (60,000 examples)
│   └── mnist_test.csv   # Test data (10,000 examples)
├── .gitignore
└── README.md
```

## Configuration

Edit the constants at the top of `training.py` to tune the model:

```python
INPUT_DIM = 784          # Input size (28×28 pixels)
HIDDEN_DIMS = [256, 128] # Hidden layer sizes
OUTPUT_DIM = 10          # Number of classes (digits 0-9)
EPOCHS = 15              # Training epochs
BATCH_SIZE = 128         # Mini-batch size
LEARNING_RATE = 1e-3     # Adam learning rate
REG_LAMBDA = 5e-4        # L2 regularization strength
```

## Key Improvements Over Basic Implementation

1. **Deeper network**: 3 layers vs 2 (more capacity)
2. **Adam optimizer**: Better than plain gradient descent
3. **Mini-batch training**: Faster and more stable
4. **L2 regularization**: Reduces overfitting
5. **Feature standardization**: Improves convergence
6. **Model persistence**: Save/load trained weights
7. **Better initialization**: He initialization for ReLU

## Workflow

### Standard workflow:
1. Train once: `python training.py` (saves `archive/trained_model.npz`)
2. Test many times: `python test_model.py` (loads saved model)

### No need to retrain unless:
- Changing network architecture
- Tuning hyperparameters
- Adding more training data
- Wanting better performance

## License

MIT

## Author

Built from scratch as a learning project for understanding neural networks.
