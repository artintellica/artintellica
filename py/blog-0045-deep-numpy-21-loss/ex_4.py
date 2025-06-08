import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import normalize, matrix_multiply, mse_loss, sigmoid

Z: NDArray[np.floating] = np.array([[-2.0], [2.0], [-1.0]])
A: NDArray[np.floating] = sigmoid(Z)
y_true: NDArray[np.floating] = np.array([[1.0], [0.0], [1.0]])
loss: np.floating = mse_loss(A, y_true)
print("Raw outputs Z (3x1):\n", Z)
print("Sigmoid of Z (1 / (1 + e^(-Z))):\n", A)
print("True labels y_true (3x1):\n", y_true)
print("Mean Squared Error Loss:", loss)
