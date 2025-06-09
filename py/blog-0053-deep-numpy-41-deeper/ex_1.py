import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, Dict, cast
from neural_network import forward_mlp_3layer, relu, softmax
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_network import (
    normalize,
    relu,
    softmax,
    cross_entropy,
    forward_mlp,
    backward_mlp,
    forward_mlp_3layer,
    backward_mlp_3layer,
)

X = np.array([[1.0, 2.0], [3.0, 4.0]])
y = np.array([[1, 0], [0, 1]])
W1 = np.random.randn(2, 4) * 0.1
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 3) * 0.1
b2 = np.zeros((1, 3))
W3 = np.random.randn(3, 2) * 0.1
b3 = np.zeros((1, 2))
A1, A2, A3 = forward_mlp_3layer(X, W1, b1, W2, b2, W3, b3)
Z1 = X @ W1 + b1
Z2 = A1 @ W2 + b2
grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = backward_mlp_3layer(
    X, A1, A2, A3, y, W1, W2, W3, Z1, Z2
)
print("A1 shape:", A1.shape)
print("A2 shape:", A2.shape)
print("A3 shape:", A3.shape)
print("Gradient W1 shape:", grad_W1.shape)
print("Gradient b1 shape:", grad_b1.shape)
print("Gradient W2 shape:", grad_W2.shape)
print("Gradient b2 shape:", grad_b2.shape)
print("Gradient W3 shape:", grad_W3.shape)
print("Gradient b3 shape:", grad_b3.shape)
