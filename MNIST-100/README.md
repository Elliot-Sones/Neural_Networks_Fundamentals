Now building to a more complex NN to predict all 100 digits from 00-99

Dataset: MNIST-100
train: 60 000 (dev 10 000) 
test:10 000
Image: 28*56 (1568 total) grayscale pixels

First I implemented the same neural network but with 1568 input values and 100 output values with 512 in the first layer and 256 in the second . And got: 

train_acc: 99.09 and dev_acc: 87.43


12% diff -> overfitting  (model memorized the training)
Instability around epoch 10-13 -> learning rate too high or bad regularisation


First test -> find best learning rate and ergularisasation
