import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable, Dict
from neural_network import mse_loss, numerical_gradient, gradient_descent


def linear_forward(X, params):
    return X @ params["W"] + params["b"]


X = np.array([[1.0], [2.0]])
y = np.array([[3.0], [5.0]])
n = X.shape[0]
params = {"W": np.array([[1.5]]), "b": np.array([[0.5]])}
y_pred = linear_forward(X, params)
error = y_pred - y
analytical_grad_W = 2 * (X.T @ error) / n
analytical_grad_b = 2 * np.mean(error)
numerical_grads = numerical_gradient(X, y, params, mse_loss, linear_forward, h=1e-6)
print("Analytical Gradient for W:", analytical_grad_W)
print("Numerical Gradient for W:", numerical_grads["W"])
print("Difference for W:", np.abs(analytical_grad_W - numerical_grads["W"]))
print("Analytical Gradient for b:", analytical_grad_b)
print("Numerical Gradient for b:", numerical_grads["b"])
print("Difference for b:", np.abs(analytical_grad_b - numerical_grads["b"]))
