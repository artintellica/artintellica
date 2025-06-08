import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import normalize, matrix_multiply, mse_loss

X: NDArray[np.floating] = np.array([[1.0], [2.0], [3.0]])
W: NDArray[np.floating] = np.array([[1.5]])
b: float = 0.5
y_pred: NDArray[np.floating] = X @ W + b
y_true: NDArray[np.floating] = np.array([[2.0], [3.5], [5.0]])
loss: np.floating = mse_loss(y_true, y_pred)
print("Input matrix X (3x1):\n", X)
print("Weights matrix W (1x1):\n", W)
print("Bias b (scalar):\n", b)
print("Predicted values y_pred (3x1):\n", y_pred)
print("True values y_true (3x1):\n", y_true)
print("Mean Squared Error Loss:", loss)
