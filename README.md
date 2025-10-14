# Building Neural Networks from scratch

Neural Networks are the foundation of modern machine learning so I wanted to build and push my understanding of Neural Networks and explain every step process and more. 

My goal was to build a simple MLP classification for digits 0-9 . Then push my understanding by implementing a CNN for classifying digits 00-99. 

## Quick Overview
- **Goal** – Understand core deep-learning mechanics by hand-coding training and inference without high-level frameworks.
- **Datasets** – Classic MNIST (28×28 grayscale digits 0–9) and a paired MNIST variant that stacks two digits side-by-side to form numbers 00–99.
- **Implementations** – A NumPy-based multilayer perceptron for single-digit recognition plus a convolutional pipeline that extends to two-digit classification with production-ready tooling.
- **Results** – The MLP reaches ~97% test accuracy on MNIST, while the CNN workflow auto-tunes to ~90% dev accuracy on the harder two-digit task and powers a live demo app.

# Simple Mutli-Layered Perceptron (Digits 0-9)
[Try the MLP here](https://huggingface.co/spaces/Eli181927/elliot_digit_classifier/)

Simple explanation:
- **Goal** – Prove that hand-built NumPy code can rival starter ML systems.
- **Dataset** – MNIST (60k train / 10k test grayscale digits).
- **Implementation** – 3-layer ReLU MLP trained with Adam, He init, and L2 regularization.
- **Result** – About 97% accuracy on the held-out test set.

For detailed implementation and results, see the [MNIST README](MNIST/README.md). 


# Convolutional Neural Network (Digits 0-99)
[Try the CNN here]()

Simple explanation:
- **Goal** – Scale the scratch-built approach to recognizing two-digit numbers.
- **Dataset** – Paired-MNIST where two 28×28 digits are concatenated into 28×56 images for 00–99 labels.
- **Implementation** – Stride-1 CNN with pooling, dropout, Adam, and auto-tuning to streamline training.
- **Result** – Consistently around 90% dev accuracy powering the production Gradio demo.

For detailed implementation, explanation and results see the [MNSIT-100 README](MNIST-100/README.md)
