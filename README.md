# Simple Neural Network to production-ready digit classifier from scratch, no libraries except NumPy

Neural networks form the foundation of modern machine learning, so I wanted to go beyond using libraries and truly understand how they work. To do that, I built a neural network completely from scratch using only NumPy—no frameworks, no shortcuts. This project gave me a hands-on understanding of core concepts like forward propagation, backpropagation, and gradient descent, and helped me see exactly how data flows and learns within a model.

From scratch to production-ready—all in NumPy. Here’s the path, step by step, from me explaining the math and the steps I took to the final demo. 



## Neural Network full walkthrough from Scratch: 

## Demo: 
**Try it for yourself:** 
https://huggingface.co/spaces/Eli181927/elliot_digit_classifier


**Or watch demo video:**
<p align="center">
  <a href="https://www.youtube.com/watch?v=RzZ32FRI4nI">
    <img src="https://img.youtube.com/vi/RzZ32FRI4nI/hqdefault.jpg" width="400" />
  </a>
</p>



## 1st iteration: Simple Multi Layer Perceptron
- Architecture: 784 → 10 (single linear layer) with softmax, trained via full-batch gradient descent.
- Result: **92.6% dev accuracy**, **91–92% test accuracy**.
- Takeaway: even a naive implementation works, but capacity and optimization limit headroom.

## Results after 1st iteration: 
- 92% on both train and dev set with loss stuck on 0.28. The capacity was too low. 
- Common errors on similar shapes (4vs9, 3vs5 and 7vs1) and a low confidence score demonstrated the classifier wasn't being precise enough with patterns and couldn't combine patterns for a more accurate prediction.
- Loss curve wobbled even late in training, demonstrating late loss curve uncertainty

## 2nd Iteration: Architecture and Training Improvements 
- Architecture upgrade: 784 → 256 → 128 → 10 with ReLU activations.
- Training changes: mini-batch Adam (batch 128), He initialization, L2 regularization (5e-4), 15 epochs.
- Metrics: **99.8% train accuracy**, **97.1% dev accuracy**, **97.2% test accuracy**.
- Insight: depth + adaptive optimization + regularization bridge the gap to state-of-the-art MNIST results.

## 3. Putting into Production 
- Interactive app (`python app.py`): Putting into production. Accuracy was very low from the start. 
- Added diagnostic: shows exact 28x28 tensor fed into the NN, stroke density (sum of pixel value), center offset and area ratio. Found that the production input was too different from the training data. 
- Passed heavy stress testing as shown in the video

## 4. Quick Start
```bash
python training.py     # retrain and persist weights/norm stats
python test_model.py   # validate on the 10k MNIST test set
python app.py          # launch the sketchpad digit inspector
```

Everything—from data ingestion to UI—runs in this repo with pure NumPy. No high-level ML frameworks, yet the model still delivers **97%+** accuracy and production-grade UX.
