import numpy as np
from numpy.typing import NDArray
from typing import Union

# Compute the transpose of a matrix
X: NDArray[np.floating] = np.array([[1, 2, 3], [4, 5, 6]])
print("Original matrix X (2x3):\n", X)
X_transpose: NDArray[np.floating] = np.transpose(X)
print("Transposed matrix X^T (3x2):\n", X_transpose)
