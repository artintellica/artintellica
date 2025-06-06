import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import normalize, matrix_multiply

X: NDArray[np.floating] = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
X_transpose: NDArray[np.floating] = np.transpose(X)
print("Original matrix X (3x2):\n", X)
print("Shape of X:", X.shape)
print("Transposed matrix X^T (2x3):\n", X_transpose)
print("Shape of X^T:", X_transpose.shape)
