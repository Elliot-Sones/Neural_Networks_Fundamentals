# Classification Neural Networks from scratch

Neural Networks are the foundation of modern machine learning, thus understanding the main types of architectures like **Multi-Layered Perceptron**, **Convolutional Neural Networks** and **Recurrent Neural Networks** are crucial. 

In this project, I will explore these architectures from scratch, explaining how they work and to implement them in code, allowing for hands-on learning and quick and easy deployment of your model. 


## 1. Mutli-Layered Perceptron: MNIST classfication problem
[Try the MLP digit classifier Demo here](https://huggingface.co/spaces/Eli181927/elliot_digit_classifier/)

In this section I will be explaining how to implement a Multi-layered perceptron from scratch (no libraires) to solve the MNIST digit classification problem.

For detailed explanation and implementation, see the [MLP README](1.MLP/README.md) and the [training code](/1.MLP/training.py). 

#### MNIST Classification problem
- **Goal** – Accurately predict hand drawn digits in production. 
- **Dataset** – MNIST 28*28 pixel images(60k train / 10k test grayscale digits).
- **My implementation** – 4-layer ReLU MLP trained with Adam, He initialisation, and L2 regularization.
- **Result** – About 97% accuracy on test set 

#### MLP Quickstart

Minimal steps to download data, train, test, and run the app.

```bash
# 1) Navigate to project folder and create virtual env
cd 1.MLP
python -m venv .venv && source .venv/bin/activate

# 2) Install deps for the MLP
pip install -r requirements.txt

# 3) Download MNIST and prepare CSVs (writes to archive)
python setup_data.py

# 4) Train (saves model to archive/trained_model.npz)
python training.py

# 5) Evaluate on test set
python test_model.py

# 6) Optional: launch the local demo UI
python app.py
```

Notes:
- **Data**: 1.MLP/setup_data.py pulls MNIST from a reliable mirror and writes 'mnist_train.csv and mnist_test.csv under 1.MLP/archive in the exact format expected by training.py and test_model.py.
- **Model file**: Training creates 1.MLP/archive/trained_model.npz, which the demo app loads automatically.



## 2. Convolutional Neural Network: MNIST-100 (0-99  digits) classification problem 
[Try the CNN digit classifier here](https://huggingface.co/spaces/Eli181927/0-99_Classification)

In this section we will be evaluating where the MLP will not be good enough and 

#### Simple explanation:
- **Goal** – Scale the scratch-built approach to recognizing two-digit numbers.
- **Dataset** – Paired-MNIST where two 28×28 digits are concatenated into 28×56 images for 00–99 labels.
- **Implementation** – Stride-1 CNN with pooling, dropout, Adam, and auto-tuning to streamline training.
- **Result** – Test accuracy 97.88% (10000 samples)

#### CNN Quickstart

Minimal steps to download data, train, test, and run the app.

Two options:

1) From-scratch (GPU recommended)

```bash
cd 2.CNN
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python setup_data.py
python training-100.py --epochs 20 --batch-size 256
```

2) Libraries (fast on CPU)

```bash
cd 2.CNN
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python setup_data.py
python training_torch.py --epochs 20 --batch-size 256 --device cpu
```

Evaluate and run the app:

```bash
cd 2.CNN
python test_model.py
python app.py
```

Notes:
- **Data**: `2.CNN/setup_data.py` downloads MNIST and creates paired two-digit combinations (00-99), saving as `mnist_train.csv` and `mnist_test.csv` in `2.CNN/archive/`.
- **Model file**: Training creates `2.CNN/archive/trained_model_mnist100.npz`, which the demo app loads automatically.

For detailed implementation, explanation and results see the [CNN README](2.CNN/README.md)


## 3. Recurrent Neural Network: Hand drawn doodles classification.
[Try the RNN doodle classifier here](https://huggingface.co/spaces/Eli181927/animal_doodle_classifier)

#### Simple explanation:
- **Goal** – Classify hand-drawn doodles into 10 animal classes.
- **Dataset** – Google Quick, Draw! stroke sequences (dx, dy, pen-lift).
- **Implementation** – 2-layer bidirectional GRU with sequence packing, AdamW, dropout, label smoothing, and grad clipping.
- **Result** – 188 779 test samples: 94.36% accuracy.

#### RNN Quickstart

Minimal steps to download data, train, test, and run the app.

```bash
# 1) Navigate to project folder and create virtual env
cd 3.RNN
python -m venv .venv && source .venv/bin/activate

# 2) Install deps for the RNN
pip install -r requirements.txt

# 3) Download Quick Draw dataset and prepare splits (writes to archive)
python setup_data.py

# 4) Train (saves model to archive/rnn_animals_best.pt)
python training-doodle.py

# 5) Evaluate on test set and generate plots
python eval_and_plots.py

# 6) Optional: launch the local demo UI
python app.py
```

Notes:
- **Data**: `3.RNN/setup_data.py` downloads Quick Draw data for 10 animal classes and creates `animal_doodles_10_train.csv` and `animal_doodles_10_test.csv` in `3.RNN/archive/`.
- **Model file**: Training creates `3.RNN/archive/rnn_animals_best.pt`, which the demo app loads automatically.

For detailed implementation and results, see the [RNN README](3.RNN/README.md)
