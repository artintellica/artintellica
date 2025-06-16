import torch
import matplotlib.pyplot as plt
from typing import Tuple
import itertools


def simple_linear_regression_data(n: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    # y = 2x + 3 + noise
    X = torch.linspace(-1, 1, n).unsqueeze(1)
    y = 2 * X + 3 + 0.1 * torch.randn(n, 1)
    return X, y


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return ((y_pred - y_true) ** 2).mean()


def model(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return X @ w + b


def numerical_gradient(
    func, params: Tuple[torch.Tensor, ...], param_idx: int, h: float = 1e-5
) -> torch.Tensor:
    grads = torch.zeros_like(params[param_idx])
    param_shape = params[param_idx].shape
    for idx in itertools.product(*(range(s) for s in param_shape)):
        orig = params[param_idx][idx].item()
        params_plus = [p.clone() for p in params]
        params_minus = [p.clone() for p in params]
        params_plus[param_idx][idx] = orig + h
        params_minus[param_idx][idx] = orig - h
        f_plus = func(*params_plus).item()
        f_minus = func(*params_minus).item()
        grad = (f_plus - f_minus) / (2 * h)
        grads[idx] = grad
    return grads


# Generate data
X, y = simple_linear_regression_data(20)
# Initialize parameters with grad enabled
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)


def loss_fn(w_: torch.Tensor, b_: torch.Tensor) -> torch.Tensor:
    y_pred = model(X, w_, b_)
    return mse_loss(y_pred, y)


# Compute analytical gradients
loss = loss_fn(w, b)
loss.backward()  # This fills in w.grad and b.grad

# Compute numerical gradients
num_grad_w = numerical_gradient(loss_fn, (w, b), param_idx=0)
num_grad_b = numerical_gradient(loss_fn, (w, b), param_idx=1)

# print("Analytical grad (w):", w.grad)
# print("Numerical grad (w): ", num_grad_w)
# print("Analytical grad (b):", b.grad)
# print("Numerical grad (b): ", num_grad_b)

h_list = [1e-3, 1e-5, 1e-7]
for h in h_list:
    num_grad_w = numerical_gradient(loss_fn, (w, b), param_idx=0, h=h)
    print(f"Numerical grad (w) with h={h:.0e}:", num_grad_w)
