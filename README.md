# Digit classification Neural Networks from scratch

Neural Networks are the foundation of modern machine learning so I wanted to build and push my understanding of Neural Networks by building them from scratch (no libraries except NumPy). 


My goal was to build a simple MLP classification for digits 0-9, then push my understanding by implementing a CNN for classifying digits 00-99 while explaining every step and getting the model into production.

## Mutli-Layered Perceptron: Handwritten 0-9 digits classification
<p align="center">
  <a href="https://www.youtube.com/watch?v=RzZ32FRI4nI">
    <img src="https://img.youtube.com/vi/RzZ32FRI4nI/hqdefault.jpg" width="300" />
  </a>
</p>

[Try the MLP here](https://huggingface.co/spaces/Eli181927/elliot_digit_classifier/)

#### Simple explanation:
- **Goal** – Accurately predict hand drawn digits in production
- **Dataset** – MNIST 28*28 pixel images(60k train / 10k test grayscale digits).
- **Implementation** – 3-layer ReLU MLP trained with Adam, He init, and L2 regularization.
- **Result** – About 97% accuracy on the held-out test set.

For detailed implementation and results, see the [MNIST README](MNIST/README.md). 





## Convolutional Neural Network: Handwritten 0-99  digits classification (MNIST-100)
[Try the CNN here]()

#### Simple explanation:
- **Goal** – Scale the scratch-built approach to recognizing two-digit numbers.
- **Dataset** – Paired-MNIST where two 28×28 digits are concatenated into 28×56 images for 00–99 labels.
- **Implementation** – Stride-1 CNN with pooling, dropout, Adam, and auto-tuning to streamline training.
- **Result** – Consistently around 90% dev accuracy powering the production Gradio demo.

For detailed implementation, explanation and results see the [MNSIT-100 README](MNIST-100/README.md)


## Recurrent Neural Network: Hand drawn doodles classification.
[Try the RNN here]()

#### Simple explanation
