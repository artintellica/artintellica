import numpy as np
from numpy.typing import NDArray
from typing import Union, List, cast


def l2_regularization(
    weights: List[NDArray[np.floating]], lambda_: float
) -> tuple[float, List[NDArray[np.floating]]]:
    """
    Compute L2 regularization penalty and gradients for a list of weight matrices.
    Args:
        weights: List of weight matrices (e.g., [W1, W2, W3])
        lambda_: Regularization strength (e.g., 0.01)
    Returns:
        Tuple of (l2_penalty, l2_grads):
        - l2_penalty: Scalar penalty term to add to loss (lambda * sum of squared weights)
        - l2_grads: List of gradients for each weight matrix (2 * lambda * W)
    """
    l2_penalty = 0.0
    l2_grads = []
    for W in weights:
        l2_penalty += np.sum(W**2)
        l2_grads.append(2 * lambda_ * W)
    l2_penalty *= lambda_
    l2_penalty = cast(float, l2_penalty)  # Ensure penalty is a scalar
    return l2_penalty, l2_grads
