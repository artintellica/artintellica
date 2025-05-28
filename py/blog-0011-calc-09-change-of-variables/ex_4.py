"""
exercise_4_flow_likelihood_grid.py
-------------------------------------------------
Contour plot of log p_x(x) for a 2D affine flow model over [-3,3]^2.
Where is the flow most/least likely to generate points?
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# --- Affine flow parameters
A = torch.tensor([[2.0, 0.3], [0.1, 1.5]], dtype=torch.float64)
b = torch.tensor([1.0, -2.0], dtype=torch.float64)


def f_inv(x):
    # x: (..., 2) torch tensor
    return torch.linalg.solve(A, (x - b).T).T


def log_det_jacobian():
    return torch.logdet(A)


def log_prob_x(x):
    # x: (..., 2) torch tensor
    z = f_inv(x)
    # Standard 2D normal log-prob
    logpz = -0.5 * (z**2).sum(-1) - torch.log(torch.tensor(2 * np.pi))
    logdet = log_det_jacobian()
    return logpz - logdet


# --- Make grid over [-3,3]^2
N = 120
xv = np.linspace(-3, 3, N)
yv = np.linspace(-3, 3, N)
X, Y = np.meshgrid(xv, yv)
grid = np.stack([X.ravel(), Y.ravel()], axis=1)
grid_t = torch.tensor(grid, dtype=torch.float64)

# --- Compute log-likelihood at each grid point
with torch.no_grad():
    logp = log_prob_x(grid_t).numpy()
logp_grid = logp.reshape(N, N)

# --- Plot contour
plt.figure(figsize=(6, 6))
cont = plt.contourf(X, Y, logp_grid, levels=30, cmap="viridis")
plt.colorbar(cont, label=r"$\log p_x(x)$")
plt.title(r"Flow log-likelihood $\log p_x(x)$ over $[-3,3]^2$")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.tight_layout()
plt.show()
