import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import normalize, matrix_multiply

A: NDArray[np.floating] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
A_transpose: NDArray[np.floating] = np.transpose(A)
multiplied: NDArray[np.floating] = matrix_multiply(A, A_transpose)
print("Original matrix A (2x3):\n", A)
print("Shape of A:", A.shape)
print("Transposed matrix A^T (3x2):\n", A_transpose)
print("Shape of A^T:", A_transpose.shape)
print("Matrix multiplication A @ A^T (2x2):\n", multiplied)
print("Shape of A @ A^T:", multiplied.shape)
