import numpy as np
from numpy.typing import NDArray
from typing import Union


def binary_cross_entropy(
    A: NDArray[np.floating], y: NDArray[np.floating]
) -> np.floating:
    """
    Compute the Binary Cross-Entropy loss between predicted probabilities and true labels.
    Args:
        A: Predicted probabilities (after sigmoid), array of shape (n,) or (n,1), values in [0, 1]
        y: True binary labels, array of shape (n,) or (n,1), values in {0, 1}
    Returns:
        Binary cross-entropy loss as a single float
    """
    # Add small epsilon to avoid log(0) issues
    epsilon = 1e-15
    return -np.mean(y * np.log(A + epsilon) + (1 - y) * np.log(1 - A + epsilon))


# Example: Synthetic binary classification data
# Raw outputs (logits) before sigmoid
Z = np.array([[2.0], [-1.0], [3.0], [-2.0]])
# Predicted probabilities after sigmoid
A = 1 / (1 + np.exp(-Z))
# True binary labels (0 or 1)
y_true = np.array([[1.0], [0.0], [1.0], [0.0]])

print("Raw outputs Z (4x1):\n", Z)
print("Predicted probabilities A (4x1):\n", A)
print("True labels y_true (4x1):\n", y_true)
loss_bce = binary_cross_entropy(A, y_true)
print("Binary Cross-Entropy Loss:", loss_bce)
