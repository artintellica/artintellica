import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable


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


# Example: Synthetic data (y = 2x + 1)
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])  # Input (5 samples, 1 feature)
y = np.array([[3.0], [5.0], [7.0], [9.0], [11.0]])  # True values (y = 2x + 1)
W_init = np.array([[0.0]])  # Initial weight (start far from true value 2.0)
b_init = np.array([[0.0]])  # Initial bias (start far from true value 1.0)
lr = 0.1  # Learning rate
num_iterations = 100  # Number of iterations


# Use mse_loss from our library
def mse_loss(y_pred: NDArray[np.floating], y: NDArray[np.floating]) -> np.floating:
    return np.mean((y_pred - y) ** 2)


# Run gradient descent
W_final, b_final, losses = gradient_descent(
    X, y, W_init, b_init, lr, num_iterations, mse_loss
)

print("Initial weight W:", W_init)
print("Initial bias b:", b_init)
print("Final weight W after training:", W_final)
print("Final bias b after training:", b_final)
print("Final loss:", losses[-1])
print("First few losses:", losses[:5])
