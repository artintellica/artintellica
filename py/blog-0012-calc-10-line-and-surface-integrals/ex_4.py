"""
exercise_4_rl_gradient_trajectory.py
-------------------------------------------------
For reward landscape
    f(x, y) = -((x-1)^2 + (y+1)^2)
plot the gradient field and show the optimal gradient ascent
trajectory starting from (0, 0).
"""

import numpy as np
import matplotlib.pyplot as plt


# Reward function and gradient
def f(x, y):
    return -((x - 1) ** 2 + (y + 1) ** 2)


def gradf(x, y):
    # Gradient ascent (not descent!)
    dfdx = -2 * (x - 1)
    dfdy = -2 * (y + 1)
    return dfdx, dfdy


# Grid for gradient field
xv = np.linspace(-2.5, 2.5, 28)
yv = np.linspace(-3.5, 0.5, 28)
X, Y = np.meshgrid(xv, yv)
U, V = gradf(X, Y)

# Simulate optimal trajectory (gradient ascent)
n_steps = 30
eta = 0.15
traj = np.zeros((n_steps + 1, 2))
traj[0] = (0.0, 0.0)
for k in range(n_steps):
    x, y = traj[k]
    dx, dy = gradf(x, y)
    traj[k + 1] = (x + eta * dx, y + eta * dy)

# Contours of reward landscape
Z = f(X, Y)

plt.figure(figsize=(7, 5))
plt.contourf(X, Y, Z, levels=30, cmap="Greens", alpha=0.7)
plt.colorbar(label="Reward $f(x, y)$")
plt.streamplot(X, Y, U, V, color="k", density=1.1, linewidth=0.7, arrowsize=1)
plt.plot(traj[:, 0], traj[:, 1], "o-r", label="Optimal trajectory")
plt.scatter([1], [-1], color="gold", s=90, edgecolor="k", label="Reward maximum")
plt.scatter([0], [0], color="blue", s=50, edgecolor="k", label="Start")
plt.xlabel("x")
plt.ylabel("y")
plt.title("RL: Gradient Ascent in Reward Landscape")
plt.legend()
plt.tight_layout()
plt.show()
