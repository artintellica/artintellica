import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import normalize, matrix_multiply, mse_loss, sigmoid

Z: NDArray[np.floating] = np.array([[1.0], [-1.0], [2.0]])
A: NDArray[np.floating] = sigmoid(Z)
