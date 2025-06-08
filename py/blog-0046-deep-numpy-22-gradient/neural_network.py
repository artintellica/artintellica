import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable


def normalize(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Normalize the input array X by subtracting the mean and dividing by the standard deviation.

    Parameters:
        X (NDArray[np.floating]): Input array to normalize. Should be a numerical array
            (float or compatible type).

    Returns:
        NDArray[np.floating]: Normalized array with mean approximately 0 and standard
            deviation approximately 1 along each axis.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Handle division by zero by using np.where to avoid warnings
    normalized_X = np.where(std != 0, (X - mean) / std, X - mean)
    return normalized_X


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


def sigmoid(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the sigmoid activation function element-wise.
    Args:
        Z: Input array of any shape with floating-point values
    Returns:
        Array of the same shape with sigmoid applied element-wise, values in [0, 1]
    """
    return 1 / (1 + np.exp(-Z))


def mse_loss(y_pred: NDArray[np.floating], y: NDArray[np.floating]) -> np.floating:
    """
    Compute the Mean Squared Error loss between predicted and true values.
    Args:
        y_pred: Predicted values, array of shape (n,) or (n,1) with floating-point values
        y: True values, array of shape (n,) or (n,1) with floating-point values
    Returns:
        Mean squared error as a single float
    """
    return np.mean((y_pred - y) ** 2)


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

def gradient_descent(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    W: NDArray[np.floating],
    b: NDArray[np.floating],
    lr: float,
    num_iterations: int,
    loss_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], list[float]]:
    """
    Perform gradient descent to minimize loss for linear regression.
    Args:
        X: Input data, shape (n_samples, n_features)
        y: True values, shape (n_samples, 1)
        W: Initial weights, shape (n_features, 1)
        b: Initial bias, shape (1,) or (1,1)
        lr: Learning rate, step size for updates
        num_iterations: Number of iterations to run gradient descent
        loss_fn: Loss function to compute error, e.g., mse_loss
    Returns:
        Tuple of (updated W, updated b, list of loss values over iterations)
    """
    n = X.shape[0]
    loss_history = []

    for _ in range(num_iterations):
        # Forward pass: Compute predictions
        y_pred = X @ W + b
        # Compute loss
        loss = loss_fn(y_pred, y)
        loss_history.append(loss)
        # Compute gradients for W and b (for MSE loss)
        grad_W = (X.T @ (y_pred - y)) / n
        grad_b = np.mean(y_pred - y)
        # Update parameters
        W = W - lr * grad_W
        b = b - lr * grad_b

    return W, b, loss_history
