import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, List


def momentum_update(
    velocity: NDArray[np.floating], gradient: NDArray[np.floating], mu: float, lr: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Perform a momentum-based update on the velocity and return the parameter update.
    Args:
        velocity: Current velocity array, same shape as gradient (initialized to 0)
        gradient: Current gradient of the loss with respect to the parameter
        mu: Momentum coefficient (e.g., 0.9)
        lr: Learning rate (e.g., 0.1)
    Returns:
        Tuple of (updated_velocity, parameter_update):
        - updated_velocity: New velocity after momentum update
        - parameter_update: Update to apply to the parameter (e.g., W += parameter_update)
    """
    updated_velocity = mu * velocity - lr * gradient
    parameter_update = updated_velocity
    return updated_velocity, parameter_update


def accuracy(y_pred: NDArray[np.floating], y_true: NDArray[np.floating]) -> float:
    """
    Compute classification accuracy for multi-class predictions.
    Args:
        y_pred: Predicted probabilities or logits, shape (n_samples, n_classes)
        y_true: True labels, one-hot encoded or class indices, shape (n_samples, n_classes) or (n_samples,)
    Returns:
        Accuracy as a float (fraction of correct predictions)
    """
    if y_true.ndim == 2:  # One-hot encoded
        true_labels = np.argmax(y_true, axis=1)
    else:  # Class indices
        true_labels = y_true
    pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(pred_labels == true_labels)
