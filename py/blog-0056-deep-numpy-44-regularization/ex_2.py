import numpy as np
from numpy.typing import NDArray
from typing import Union, List, cast
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_network import (
    normalize,
    relu,
    softmax,
    cross_entropy,
    forward_mlp_3layer,
    backward_mlp_3layer,
    l2_regularization,
    dropout,
)
import matplotlib.pyplot as plt

X = np.random.randn(5, 3)
y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]])
W1 = np.random.randn(3, 4) * 0.1
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 2) * 0.1
b2 = np.zeros((1, 2))
lr = 0.1
lambda_ = 0.01
# Forward pass
Z1 = X @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
A2 = softmax(Z2)
# Loss without regularization
data_loss = cross_entropy(A2, y)
print("Data Loss (without L2):", data_loss)
# Loss with L2 regularization
l2_penalty, l2_grads = l2_regularization([W1, W2], lambda_)
total_loss = data_loss + l2_penalty
print("Total Loss (with L2):", total_loss)
# Backpropagation with L2 gradients
delta2 = A2 - y
grad_W2 = (A1.T @ delta2) / X.shape[0] + l2_grads[1]
grad_b2 = np.mean(delta2, axis=0, keepdims=True)
delta1 = (delta2 @ W2.T) * (Z1 > 0)
grad_W1 = (X.T @ delta1) / X.shape[0] + l2_grads[0]
grad_b1 = np.mean(delta1, axis=0, keepdims=True)
# Update parameters
W1 -= lr * grad_W1
b1 -= lr * grad_b1
W2 -= lr * grad_W2
b2 -= lr * grad_b2
