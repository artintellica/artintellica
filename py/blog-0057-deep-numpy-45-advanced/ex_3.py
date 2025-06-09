import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, List, cast
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
    accuracy,
    momentum_update,
    accuracy,
)
import matplotlib.pyplot as plt

# Your code here
X = np.random.randn(5, 3)
y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]])
W1 = np.random.randn(3, 4) * 0.1
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 2) * 0.1
b2 = np.zeros((1, 2))
v_W1 = np.zeros_like(W1)
v_b1 = np.zeros_like(b1)
v_W2 = np.zeros_like(W2)
v_b2 = np.zeros_like(b2)
mu = 0.9
lr = 0.1
# Initial forward pass and loss
Z1 = X @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
A2 = softmax(Z2)
initial_loss = cross_entropy(A2, y)
print("Initial Loss:", initial_loss)
# Backpropagation
delta2 = A2 - y
grad_W2 = (A1.T @ delta2) / X.shape[0]
grad_b2 = np.mean(delta2, axis=0, keepdims=True)
delta1 = (delta2 @ W2.T) * (Z1 > 0)
grad_W1 = (X.T @ delta1) / X.shape[0]
grad_b1 = np.mean(delta1, axis=0, keepdims=True)
# Momentum updates
v_W1, update_W1 = momentum_update(v_W1, grad_W1, mu, lr)
v_b1, update_b1 = momentum_update(v_b1, grad_b1, mu, lr)
v_W2, update_W2 = momentum_update(v_W2, grad_W2, mu, lr)
v_b2, update_b2 = momentum_update(v_b2, grad_b2, mu, lr)
W1 += update_W1
b1 += update_b1
W2 += update_W2
b2 += update_b2
# Final forward pass and loss
Z1 = X @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
A2 = softmax(Z2)
final_loss = cross_entropy(A2, y)
print("Final Loss after one update:", final_loss)
