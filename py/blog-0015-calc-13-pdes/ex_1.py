"""
exercise_1_wave_gaussian_bump.py
-------------------------------------------------
Simulate the 1D wave equation, but set the initial u to a Gaussian bump.
Observe how the propagation differs from a square bump.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters
N = 180
dx = 1.0 / N
c = 1.0
dt = 0.8 * dx / c
steps = 240

# --- Initial state: Gaussian bump in the middle
x = np.linspace(0, 1, N)
mu = 0.5
sigma = 0.06
u = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
u_prev = u.copy()  # zero initial velocity

C2 = (c * dt / dx) ** 2

# --- Storage for frames
frames = []

for n in range(steps):
    u_next = np.zeros_like(u)
    u_next[1:-1] = 2 * u[1:-1] - u_prev[1:-1] + C2 * (u[2:] - 2 * u[1:-1] + u[:-2])
    # Fixed boundary conditions
    u_next[0] = u_next[-1] = 0.0
    frames.append(u_next.copy())
    u_prev, u = u, u_next

# --- Animation
fig, ax = plt.subplots(figsize=(7, 3))
(line,) = ax.plot(x, frames[0])
ax.set_ylim(-1.2, 1.2)
ax.set_title("1D Wave Equation: Gaussian Initial Bump")


def animate(i):
    line.set_ydata(frames[i])
    ax.set_xlabel(f"Step {i}")
    return (line,)


ani = animation.FuncAnimation(
    fig, animate, frames=len(frames), interval=35, blit=True, repeat=True
)
plt.show()
