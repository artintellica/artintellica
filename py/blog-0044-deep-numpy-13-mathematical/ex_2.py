import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import normalize, matrix_multiply, sigmoid

Z: NDArray[np.floating] = np.array([[-1, 0, 1], [-2, 2, 0.5]], dtype=np.float64)
Z_sigmoid: NDArray[np.floating] = sigmoid(Z)
print("Input matrix Z (2x3):\n", Z)
print("Sigmoid of Z (1 / (1 + e^(-Z))):\n", Z_sigmoid)
print(
    "Are all values in Z_sigmoid between 0 and 1?",
    np.all((Z_sigmoid >= 0) & (Z_sigmoid <= 1)),
)
