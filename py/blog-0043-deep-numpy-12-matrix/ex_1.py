import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import normalize, matrix_multiply

X: NDArray[np.floating] = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
W: NDArray[np.floating] = np.array(
    [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float64
)
Z: NDArray[np.floating] = matrix_multiply(X, W)
print("Input matrix X (3x2):\n", X)
print("Weight matrix W (2x4):\n", W)
print("Output matrix Z (3x4):\n", Z)
print("Shape of Z:", Z.shape)
