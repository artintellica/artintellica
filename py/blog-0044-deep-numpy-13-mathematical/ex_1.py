import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import normalize, matrix_multiply, sigmoid

Z: NDArray[np.floating] = np.array([[-1, 0, 1], [-2, 2, 0.5]], dtype=np.float64)
exp_Z: NDArray[np.floating] = np.exp(Z)
print("Input matrix Z (2x3):\n", Z)
print("Exponential of Z (e^Z):\n", exp_Z)
