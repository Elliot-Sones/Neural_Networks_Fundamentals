# 2. Convolutional Neural Network — 0-99 Digit Classifier

<p align="center">
  <img src="https://img.shields.io/badge/NumPy-from--scratch-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/PyTorch-library--version-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/MNIST--100-0--99-blue" alt="MNIST-100">
</p>

[Try the live demo](https://huggingface.co/spaces/Eli181927/0-99_Classification) · [Training code (PyTorch)](training_torch.py) · [Training code (from scratch)](training-100.py)

---

## Goal

Scale the MLP approach to recognizing two-digit numbers using convolutional neural networks.

| | |
|---|---|
| **Dataset** | MNIST-100: 28×56 images of handwritten digits 0-99 (concatenated pairs) |
| **Architecture** | Conv(3×3,16) → ReLU → MaxPool → Conv(3×3,32) → ReLU → MaxPool → FC(256) → Dropout(0.4) → FC(100) |
| **Result** | **97.88% test accuracy** (10,000 samples) |

---

## Why CNNs?

CNNs are different from MLPs because they have **convolutional layers** designed to recognize spatial patterns in images.

<img src="assets/CNN_explanation/cnn.jpeg" alt="Convolutional layers" width="420"/>

**Kernels** (small 3×3 filters) slide across the image, building a feature map that shows where certain visual patterns occur.

<img src="assets/CNN_explanation/maxpool.jpg" alt="Max pooling" width="420"/>

**Pooling** layers (usually MaxPool) reduce spatial dimensions, keeping only the strongest signals. **Activation functions** (ReLU, GELU) determine which patterns actually matter. After spatial feature extraction, the data is flattened and passed into fully connected layers — just like an MLP.

Through deeper layers, CNNs build increasingly complex representations: from simple edges to high-level features.

---

## Process

### MLP Baseline

First tested the MLP from the previous section:
- ~99% train vs ~87% dev — clear overfitting
- The MLP treats pixels independently and lacks spatial bias, so small shifts degraded performance

<table>
  <tr>
    <td align="center">
      <img src="assets/CNN_1st_iteration/loss_curve.png" alt="Loss curve" width="420"/><br/>
      <em>Training loss over epochs</em>
    </td>
    <td align="center">
      <img src="assets/CNN_1st_iteration/accuracy_curves.png" alt="Accuracy curves" width="420"/><br/>
      <em>Train vs Dev accuracy — widening gap</em>
    </td>
  </tr>
</table>

<p align="center">
  <img src="assets/CNN_1st_iteration/generalization_gap.png" alt="Generalization gap" width="420"/><br/>
  <em>Generalization gap (train − dev)</em>
</p>

### CNN Implementation

- **Data & normalization**: scale to [0,1], standardize with training-set mean/std
- **Augmentation**: random horizontal shifts (±2 px), mild contrast/brightness jitter
- **Optimization**: Adam (lr=1e-3), He init, L2 regularization (λ=1e-4), batch size 256, up to 20 epochs
- **Early stopping**: patience=5, min_delta=1e-3 on dev accuracy
- **From-scratch ops**: NumPy-only conv via im2col/col2im, max-pooling, dropout, vectorized softmax cross-entropy, full backward pass

### CNN Results

**Test accuracy: 97.88%** (10,000 samples)

Hardest classes: 29 (90.2%), 97 (93.4%), 39 (93.8%), 33 (94.1%), 70 (94.1%)

<table>
  <tr>
    <td align="center">
      <img src="assets/CNN_2nd_iteration/loss.png" alt="CNN Loss" width="420"/><br/>
      <em>CNN training loss</em>
    </td>
    <td align="center">
      <img src="assets/CNN_2nd_iteration/accuracy.png" alt="CNN Accuracy" width="420"/><br/>
      <em>Train vs Dev accuracy</em>
    </td>
  </tr>
</table>

<p align="center">
  <img src="assets/CNN_2nd_iteration/gap.png" alt="CNN Generalization gap" width="420"/><br/>
  <em>Generalization gap (train − dev)</em>
</p>

---

## Quickstart

Two training options:

**Option A: From-scratch NumPy (GPU recommended)**

```bash
cd 2.CNN
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python setup_data.py
python training-100.py --epochs 20 --batch-size 256
```

**Option B: PyTorch (fast on CPU)**

```bash
cd 2.CNN
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python setup_data.py
python training_torch.py --epochs 20 --batch-size 256 --device cpu
```

Then evaluate and run the demo:

```bash
python test_model.py
python app.py
```

---

## Files

```
2.CNN/
├── training-100.py             # From-scratch NumPy CNN
├── training_torch.py           # PyTorch CNN implementation
├── app.py                      # Gradio demo app
├── test_model.py               # Test set evaluation
├── setup_data.py               # Download & prepare MNIST-100
├── requirements.txt
└── assets/                     # Training plots and diagrams
```
