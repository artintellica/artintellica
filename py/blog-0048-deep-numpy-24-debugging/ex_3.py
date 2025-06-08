import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable, Dict
from neural_network import (
    mse_loss,
    numerical_gradient,
    gradient_descent,
    sigmoid,
    binary_cross_entropy,
)


def logistic_forward(X, params):
    return sigmoid(X @ params["W"] + params["b"])


X = np.array([[0.5], [1.5], [2.5]])
y = np.array([[0.0], [0.0], [1.0]])
n = X.shape[0]
params = {"W": np.array([[0.0]]), "b": np.array([[0.0]])}
y_pred = logistic_forward(X, params)
error = y_pred - y
analytical_grad_W = (X.T @ error) / n
analytical_grad_b = np.mean(error)
numerical_grads = numerical_gradient(
    X, y, params, binary_cross_entropy, logistic_forward, h=1e-4
)
print("Analytical Gradient for W:", analytical_grad_W)
print("Numerical Gradient for W:", numerical_grads["W"])
print("Difference for W:", np.abs(analytical_grad_W - numerical_grads["W"]))
print("Analytical Gradient for b:", analytical_grad_b)
print("Numerical Gradient for b:", numerical_grads["b"])
print("Difference for b:", np.abs(analytical_grad_b - numerical_grads["b"]))
