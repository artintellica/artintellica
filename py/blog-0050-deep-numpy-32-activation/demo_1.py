import numpy as np
from numpy.typing import NDArray
from typing import Union


def relu(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the ReLU activation function element-wise.
    Args:
        Z: Input array of any shape with floating-point values
    Returns:
        Array of the same shape with ReLU applied element-wise, max(0, Z)
    """
    return np.maximum(0, Z)


def softmax(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the softmax activation function row-wise.
    Args:
        Z: Input array of shape (n_samples, n_classes) with floating-point values
    Returns:
        Array of the same shape with softmax applied row-wise, probabilities summing to 1 per row
    """
    # Subtract the max for numerical stability (avoid overflow in exp)
    Z_max = np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z - Z_max)
    sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
    return exp_Z / sum_exp_Z


# Example 1: Applying ReLU to a 3x2 matrix
Z_relu = np.array([[-1.0, 2.0], [0.0, -3.0], [4.0, -0.5]])
A_relu = relu(Z_relu)
print("Input for ReLU (3x2):\n", Z_relu)
print("Output after ReLU (3x2):\n", A_relu)

# Example 2: Applying Softmax to a 4x10 matrix (simulating MNIST outputs)
Z_softmax = np.random.randn(4, 10)  # Random scores for 4 samples, 10 classes
A_softmax = softmax(Z_softmax)
print("\nInput for Softmax (4x10, first few columns):\n", Z_softmax[:, :3])
print("Output after Softmax (4x10, first few columns):\n", A_softmax[:, :3])
print("Sum of probabilities per sample (should be ~1):\n", np.sum(A_softmax, axis=1))
