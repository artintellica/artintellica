import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import normalize, matrix_multiply, sigmoid

Z: NDArray[np.floating] = np.array([[-1.5, 2.0], [0.0, -0.5], [3.0, -2.0]], dtype=np.float64)
Z_operated: NDArray[np.floating] = np.maximum(0, Z)
print("Input matrix Z (3x2):\n", Z)
print("ReLU-like operation (max(0, Z)):\n", Z_operated)
