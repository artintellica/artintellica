import numpy as np
from numpy.typing import NDArray
from typing import Union


def sigmoid(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the sigmoid activation function element-wise.
    Args:
        Z: Input array of any shape with floating-point values
    Returns:
        Array of the same shape with sigmoid applied element-wise, values in [0, 1]
    """
    return 1 / (1 + np.exp(-Z))


# Example: Apply sigmoid to a 3x2 matrix
Z = np.array([[0.0, 1.0], [-1.0, 2.0], [-2.0, 3.0]])
print("Input matrix Z (3x2):\n", Z)
A = sigmoid(Z)
print("Sigmoid output A (3x2):\n", A)
