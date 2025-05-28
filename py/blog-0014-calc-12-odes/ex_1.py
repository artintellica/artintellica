#!/usr/bin/env python3
"""
exercise_1_different_ode.py
-------------------------------------------------
Change alpha, beta in
    dx/dt = alpha * x - beta * y
    dy/dt = beta * x + alpha * y
and plot the resulting trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Try several different (alpha, beta) pairs
params = [
    (-0.3, 1.1),  # standard spiral in
    (0.4, 1.0),  # spiral out (alpha > 0)
    (0.0, 1.0),  # pure rotation (circle)
    (-0.6, 0.3),  # tight spiral in
    (0.2, 0.0),  # exponential growth/decay along x/y axes
]

h0 = [2, 0.5]
t_span = [0, 8]
t_eval = np.linspace(*t_span, 600)

plt.figure(figsize=(8, 6))

for idx, (alpha, beta) in enumerate(params):

    def f(t, h):
        x, y = h
        dx = alpha * x - beta * y
        dy = beta * x + alpha * y
        return [dx, dy]

    sol = solve_ivp(f, t_span, h0, t_eval=t_eval)
    plt.plot(sol.y[0], sol.y[1], label=f"$\\alpha$={alpha}, $\\beta$={beta}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectories for Various $(\\alpha, \\beta)$ in Spiral ODE")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()
