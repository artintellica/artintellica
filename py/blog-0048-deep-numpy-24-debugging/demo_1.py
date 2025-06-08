import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable, Dict
from neural_network import mse_loss


def numerical_gradient(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    params: Dict[str, NDArray[np.floating]],
    loss_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], np.floating],
    forward_fn: Callable[
        [NDArray[np.floating], Dict[str, NDArray[np.floating]]], NDArray[np.floating]
    ],
    h: float = 1e-4,
) -> Dict[str, NDArray[np.floating]]:
    """
    Compute numerical gradients for parameters using central difference approximation.
    Args:
        X: Input data, shape (n_samples, n_features)
        y: True values, shape (n_samples, 1)
        params: Dictionary of parameters (e.g., {'W': ..., 'b': ...})
        loss_fn: Loss function to compute error, e.g., mse_loss
        forward_fn: Function to compute predictions from X and params
        h: Step size for finite difference approximation (default: 1e-4)
    Returns:
        Dictionary of numerical gradients for each parameter
    """
    num_grads = {}

    for param_name, param_value in params.items():
        num_grad = np.zeros_like(param_value)
        it = np.nditer(param_value, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            original_value = param_value[idx]

            # Compute loss at W + h
            param_value[idx] = original_value + h
            y_pred_plus = forward_fn(X, params)
            loss_plus = loss_fn(y_pred_plus, y)

            # Compute loss at W - h
            param_value[idx] = original_value - h
            y_pred_minus = forward_fn(X, params)
            loss_minus = loss_fn(y_pred_minus, y)

            # Central difference approximation
            num_grad[idx] = (loss_plus - loss_minus) / (2 * h)

            # Restore original value
            param_value[idx] = original_value
            it.iternext()

        num_grads[param_name] = num_grad

    return num_grads

# Define forward function for linear regression
def linear_forward(X: NDArray[np.floating], params: Dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
    return X @ params['W'] + params['b']

# Example: Synthetic data for linear regression (y = 2x + 1)
X = np.array([[1.0], [2.0], [3.0], [4.0]])  # Input (4 samples, 1 feature)
y = np.array([[3.0], [5.0], [7.0], [9.0]])  # True values (y = 2x + 1)
n = X.shape[0]

# Initialize parameters
params = {
    'W': np.array([[1.0]]),  # Initial weight (not the true value)
    'b': np.array([[0.5]])   # Initial bias (not the true value)
}

# Compute analytical gradients
y_pred = linear_forward(X, params)
error = y_pred - y
analytical_grad_W = (X.T @ error) / n
analytical_grad_b = np.mean(error)

# Compute numerical gradients
numerical_grads = numerical_gradient(X, y, params, mse_loss, linear_forward, h=1e-4)

# Compare analytical and numerical gradients
print("Analytical Gradient for W:", analytical_grad_W)
print("Numerical Gradient for W:", numerical_grads['W'])
print("Difference for W:", np.abs(analytical_grad_W - numerical_grads['W']))
print("Analytical Gradient for b:", analytical_grad_b)
print("Numerical Gradient for b:", numerical_grads['b'])
print("Difference for b:", np.abs(analytical_grad_b - numerical_grads['b']))
