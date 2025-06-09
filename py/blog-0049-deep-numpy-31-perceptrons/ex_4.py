import numpy as np
from numpy.typing import NDArray
from typing import Union, Dict
from neural_network import (
    gradient_descent,
    binary_cross_entropy,
    sigmoid,
    forward_perceptron,
)


X = np.array([[i, j] for i in range(-2, 3) for j in range(-2, 3)])
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

# Initialize parameters
n_features = X.shape[1]
n_samples = X.shape[0]
W_init = np.zeros((n_features, 1))  # Initial weights
b_init = np.zeros((1, 1))  # Initial bias
lr = 0.1  # Learning rate
num_epochs = 200  # Number of epochs (high to attempt convergence)
batch_size = 5  # Full batch since dataset is small

# Train perceptron using gradient_descent with sigmoid activation
W_final, b_final, losses = gradient_descent(
    X,
    y,
    W_init,
    b_init,
    lr,
    num_epochs,
    batch_size,
    loss_fn=binary_cross_entropy,
    activation_fn=sigmoid,
)

# Evaluate the model
A = forward_perceptron(X, W_final, b_final)
predictions = (A > 0.5).astype(int)
accuracy = np.mean(predictions == y)
print("Final Predictions (probabilities):\n", A)
print("Final Predictions (binary):\n", predictions)
print("True Labels:\n", y)
print("Accuracy:", accuracy)
print("Final Loss:", losses[-1])
print("Loss History (first 5 and last 5):", losses[:5] + losses[-5:])
