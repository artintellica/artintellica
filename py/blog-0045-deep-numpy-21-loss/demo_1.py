import numpy as np
from numpy.typing import NDArray
from typing import Union

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

# Example: Synthetic regression data (y = 2x + 1)
X = np.array([[1.0], [2.0], [3.0], [4.0]])  # Input (4 samples, 1 feature)
W = np.array([[2.0]])  # Weight (approximating the true slope)
b = 1.0  # Bias (approximating the true intercept)
y_pred = X @ W + b  # Predicted values
y_true = np.array([[3.0], [5.0], [7.0], [9.0]])  # True values (y = 2x + 1)

print("Input X (4x1):\n", X)
print("Predicted y_pred (4x1):\n", y_pred)
print("True y_true (4x1):\n", y_true)
loss_mse = mse_loss(y_pred, y_true)
print("MSE Loss:", loss_mse)

