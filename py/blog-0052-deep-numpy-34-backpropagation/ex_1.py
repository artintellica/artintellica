import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, Dict, cast
from neural_network import forward_mlp, backward_mlp

X = np.array([[1.0, 2.0], [3.0, 4.0]])
y = np.array([[1, 0], [0, 1]])
W1 = np.random.randn(2, 3) * 0.1
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 2) * 0.1
b2 = np.zeros((1, 2))
A1, A2 = forward_mlp(X, W1, b1, W2, b2)
Z1 = X @ W1 + b1  # Pre-activation for hidden layer
grad_W1, grad_b1, grad_W2, grad_b2 = backward_mlp(X, A1, A2, y, W1, W2, Z1)
print("Gradient W1 shape:", grad_W1.shape)
print("Gradient b1 shape:", grad_b1.shape)
print("Gradient W2 shape:", grad_W2.shape)
print("Gradient b2 shape:", grad_b2.shape)
