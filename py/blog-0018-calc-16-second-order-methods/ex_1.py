import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import SGD, LBFGS


# Rosenbrock function: f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
def rosen_np(xy):
    x, y = xy
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def rosen_torch(xy):
    x, y = xy[0], xy[1]
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


# Grid for contour plot
xg, yg = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-1, 3, 200))
zg = np.array([rosen_np([x, y]) for x, y in zip(xg.ravel(), yg.ravel())]).reshape(
    xg.shape
)

# Initial point (away from optimum)
x0 = torch.tensor([-1.2, 1.0], dtype=torch.float64, requires_grad=True)

# --- SGD path ---
sgd_path = [x0.detach().numpy().copy()]
x_sgd = x0.clone().detach().requires_grad_(True)
opt_sgd = SGD([x_sgd], lr=1e-3, momentum=0.0)
for _ in range(5000):
    opt_sgd.zero_grad()
    loss = rosen_torch(x_sgd)
    loss.backward()
    opt_sgd.step()
    if _ % 100 == 0:
        sgd_path.append(x_sgd.detach().numpy().copy())

# --- L-BFGS path ---
x_lbfgs = x0.clone().detach().requires_grad_(True)
opt_lbfgs = LBFGS(
    [x_lbfgs], lr=1.0, max_iter=120, history_size=10, line_search_fn="strong_wolfe"
)
lbfgs_path = [x_lbfgs.detach().numpy().copy()]


def closure():
    opt_lbfgs.zero_grad()
    loss = rosen_torch(x_lbfgs)
    loss.backward()
    lbfgs_path.append(x_lbfgs.detach().numpy().copy())
    return loss


opt_lbfgs.step(closure)

# --- Plot paths ---
plt.figure(figsize=(8, 5))
plt.contour(xg, yg, zg, levels=np.logspace(-1, 3, 20), cmap="Spectral")
sgd_path = np.array(sgd_path)
lbfgs_path = np.array(lbfgs_path)
plt.plot(sgd_path[:, 0], sgd_path[:, 1], "o-", label="SGD", alpha=0.7)
plt.plot(lbfgs_path[:, 0], lbfgs_path[:, 1], "s-", label="L‑BFGS", alpha=0.7)
plt.scatter(1, 1, color="k", marker="*", s=140, label="Global min")
plt.xlabel("x")
plt.ylabel("y")
plt.title("SGD vs. L‑BFGS on Rosenbrock function")
plt.legend()
plt.tight_layout()
plt.show()
