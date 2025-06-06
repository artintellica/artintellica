import numpy as np
from numpy.typing import NDArray
from typing import Union


def matrix_multiply(
    X: NDArray[np.floating], W: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Perform matrix multiplication between two arrays.
    Args:
        X: First input array/matrix of shape (m, n) with floating-point values
        W: Second input array/matrix of shape (n, p) with floating-point values
    Returns:
        Result of matrix multiplication, shape (m, p) with floating-point values
    """
    return np.matmul(X, W)


# Example usage with smaller matrices to verify
X_small = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
W_small = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float64)
print("Input matrix X_small (2x3):\n", X_small)
print("Weight matrix W_small (3x2):\n", W_small)
Z_small = matrix_multiply(X_small, W_small)
print("Output matrix Z_small (2x2):\n", Z_small)
