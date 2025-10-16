# Classification Neural Networks from scratch

Neural Networks are the foundation of modern machine learning, thus understanding the main concepts like MLP, CNN and RNN are vital. In this repo I aim to guide through all of the main with simple and easy to follow examples to then put your own model into production. 

Who this repo is for: Have some prior knowledge of the theory and looking for simple ways to get hands on with projects

## 1. Mutli-Layered Perceptron: Handwritten 0-9 digits classification
[Try the MLP here](https://huggingface.co/spaces/Eli181927/elliot_digit_classifier/)

#### Simple explanation:
- **Goal** – Accurately predict hand drawn digits in production
- **Dataset** – MNIST 28*28 pixel images(60k train / 10k test grayscale digits).
- **Implementation** – 3-layer ReLU MLP trained with Adam, He init, and L2 regularization.
- **Result** – About 97% accuracy on the held-out test set.

For detailed implementation and results, see the [MLP README](1.MLP/README.md). 





## 2. Convolutional Neural Network: Handwritten 0-99  digits classification (MNIST-100)
[Try the CNN here](https://huggingface.co/spaces/Eli181927/0-99_Classification)

#### Simple explanation:
- **Goal** – Scale the scratch-built approach to recognizing two-digit numbers.
- **Dataset** – Paired-MNIST where two 28×28 digits are concatenated into 28×56 images for 00–99 labels.
- **Implementation** – Stride-1 CNN with pooling, dropout, Adam, and auto-tuning to streamline training.
- **Result** – Consistently around 90% dev accuracy powering the production Gradio demo.

For detailed implementation, explanation and results see the [CNN README](2.CNN//README.md)


## 3. Recurrent Neural Network: Hand drawn doodles classification.
[Try the RNN here](https://huggingface.co/spaces/Eli181927/animal_doodle_classifier)

#### Simple explanation



For detailed implementation and results, see the [RNN README](3.RNN/README.md)

