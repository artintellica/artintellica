import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import normalize, matrix_multiply, sigmoid

Z: NDArray[np.floating] = np.array([[1, -1], [2, -2]], dtype=np.float64)
Z_manual_sigmoid: NDArray[np.floating] = 1 / (1 + np.exp(-Z))
Z_sigmoid: NDArray[np.floating] = sigmoid(Z)
print("Input matrix Z (2x2):\n", Z)
print("Manual sigmoid of Z (1 / (1 + e^(-Z))):\n", Z_manual_sigmoid)
print("Sigmoid of Z using the sigmoid function:\n", Z_sigmoid)
