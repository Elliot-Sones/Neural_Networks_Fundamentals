import numpy as np
import pandas as pd

data = pd.read_csv("archive/mnist_train.csv")

data=np.array(data)
m , n = data.shape 
np.random.shuffle(data)

data_dev = data[0:10000].T 
Y_dev = data_dev[0] 
X_dev = data_dev[1:n] 
X_dev = X_dev / 255.0

data_train = data[10000:m].T 
Y_train = data_train[0] 
X_train = data_train[1:n] 
X_train = X_train / 255.0
_, m_train = X_train.shape

print(X_train.shape)

def init_params():
    W1 = np.random.randn(10, 784) * np.sqrt(2.0/784)
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * np.sqrt(2.0/10)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def ReLU(Z): 
    return np.maximum(0 , Z)

def softmax(Z):
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def foward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) +  b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def derivative_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W2, X, Y): 
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)  
    dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)  
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = foward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % 50 == 0):
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print("Accuracy: ", get_accuracy(predictions, Y))
    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.3)


# accuracy on dev set
_, _, _, A2_dev = foward_prop(W1, b1, W2, b2, X_dev)
dev_predictions = get_predictions(A2_dev)
dev_accuracy = get_accuracy(dev_predictions, Y_dev)
print(f"Dev Accuracy: {dev_accuracy:.4f}")