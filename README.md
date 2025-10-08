# Sketchpad Digit Inspector

From a bare-bones NumPy neural net to a production-ready digit recognizer—here’s the journey in hard numbers.

- **Baseline (Week 1):** 2-layer network (784→10) trained with full-batch gradient descent plateaued at **92.6% dev accuracy**. No frameworks, just vectorized NumPy.
- **Breakthrough:** Upgraded to a 784→256→128→10 architecture, mini-batch Adam (batch 128), He init, and L2 regularization. Training accuracy jumped to **99.8%**, while dev hit **97.1%** and test hit **97.2%** after 15 epochs.
- **Tooling:** Training script now standardizes inputs, persists weights + normalization stats (`archive/trained_model.npz`), and logs per-epoch metrics for reproducibility.
- **Testing:** `python test_model.py` produces end-to-end evaluation with per-digit accuracy, confirming MNIST-class performance.
- **Production UX:** Gradio-powered app visualizes the exact 28×28 tensor the model consumes, surfaces stroke heuristics (center offset, density, area), and flags out-of-distribution sketches—keeping users honest.

Convenient commands:

```bash
python training.py     # retrain and save updated weights
python test_model.py   # validate on the MNIST test set
python app.py          # launch interactive sketchpad with diagnostics
```

All weights, data, and tooling live in this repo—no external ML frameworks required.
