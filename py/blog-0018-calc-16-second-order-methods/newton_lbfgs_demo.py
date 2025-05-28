# calc-16-hessian-vector/newton_lbfgs_demo.py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import LBFGS, SGD

# Function: f(x) = 0.5 * x^T A x + b^T x
A = np.array([[5, 1], [1, 3]])
b = np.array([-4, 2])


def f_np(x):
    return 0.5 * x @ A @ x + b @ x


# Plot function surface
xg, yg = np.meshgrid(np.linspace(-3, 3, 80), np.linspace(-3, 3, 80))
zg = np.array([f_np(np.array([x, y])) for x, y in zip(xg.ravel(), yg.ravel())]).reshape(
    xg.shape
)
plt.figure(figsize=(6, 5))
plt.contourf(xg, yg, zg, levels=30, cmap="Spectral")
plt.colorbar(label="f(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Convex Quadratic")
plt.tight_layout()
plt.show()

# PyTorch version
A_t = torch.tensor(A, dtype=torch.float64)
b_t = torch.tensor(b, dtype=torch.float64)


def f_torch(x):
    return 0.5 * x @ A_t @ x + b_t @ x


# Initial point
x0 = torch.tensor([2.5, -2.0], dtype=torch.float64, requires_grad=True)

# --------- SGD ---------
x_sgd = x0.clone().detach().requires_grad_(True)
opt_sgd = SGD([x_sgd], lr=0.1)
sgd_path = [x_sgd.detach().numpy().copy()]
for _ in range(40):
    opt_sgd.zero_grad()
    loss = f_torch(x_sgd)
    loss.backward()
    opt_sgd.step()
    sgd_path.append(x_sgd.detach().numpy().copy())

# --------- L‑BFGS ---------
x_lbfgs = x0.clone().detach().requires_grad_(True)
opt_lbfgs = LBFGS(
    [x_lbfgs], lr=1, max_iter=40, history_size=10, line_search_fn="strong_wolfe"
)
lbfgs_path = [x_lbfgs.detach().numpy().copy()]


def closure():
    opt_lbfgs.zero_grad()
    loss = f_torch(x_lbfgs)
    loss.backward()
    lbfgs_path.append(x_lbfgs.detach().numpy().copy())
    return loss


opt_lbfgs.step(closure)

# Plot optimization paths
plt.figure(figsize=(6, 5))
plt.contour(xg, yg, zg, levels=30, cmap="Spectral")
sgd_path = np.array(sgd_path)
lbfgs_path = np.array(lbfgs_path)
plt.plot(sgd_path[:, 0], sgd_path[:, 1], "o-", label="SGD", alpha=0.7)
plt.plot(lbfgs_path[:, 0], lbfgs_path[:, 1], "s-", label="L‑BFGS", alpha=0.7)
plt.scatter(*(-np.linalg.solve(A, b)), color="k", marker="*", s=140, label="Optimum")
plt.xlabel("x")
plt.ylabel("y")
plt.title("SGD vs. L‑BFGS optimization paths")
plt.legend()
plt.tight_layout()
plt.show()
