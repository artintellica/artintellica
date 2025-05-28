"""
exercise_3_wave_high_courant.py
-------------------------------------------------
Demonstrate effect of increasing Courant number C (via dt)
in 1D wave equation. Show instability when C > 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters
N = 180
dx = 1.0 / N
c = 1.0
steps = 120

# --- Initial state: Gaussian bump
x = np.linspace(0, 1, N)
mu = 0.5
sigma = 0.06
u0 = np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# --- Try several Courant numbers
C_list = [0.8, 1.0, 1.2]
dt_list = [C * dx / c for C in C_list]
titles = [f"C = {C}" for C in C_list]
frames_list = []

for dt, title in zip(dt_list, titles):
    C2 = (c * dt / dx) ** 2
    u = u0.copy()
    u_prev = u0.copy()
    frames = []
    for n in range(steps):
        u_next = np.zeros_like(u)
        u_next[1:-1] = 2 * u[1:-1] - u_prev[1:-1] + C2 * (u[2:] - 2 * u[1:-1] + u[:-2])
        u_next[0] = u_next[-1] = 0.0  # reflecting boundaries
        frames.append(u_next.copy())
        u_prev, u = u, u_next
    frames_list.append(frames)

# --- Animate all cases side by side
fig, axs = plt.subplots(1, 3, figsize=(14, 3.5), sharey=True)
lines = [ax.plot(x, frames[0])[0] for ax, frames in zip(axs, frames_list)]
for ax, title in zip(axs, titles):
    ax.set_ylim(-1.2, 1.2)
    ax.set_title(title)


def animate(i):
    for k, frames in enumerate(frames_list):
        lines[k].set_ydata(frames[i])
        axs[k].set_xlabel(f"Step {i}")
    return lines


ani = animation.FuncAnimation(
    fig, animate, frames=steps, interval=35, blit=True, repeat=True
)
plt.tight_layout()
plt.show()

print(
    """
Observation:
- For C = 0.8 (C < 1): Stable, wave propagates smoothly.
- For C = 1.0 (C = 1): At the edge of stability.
- For C = 1.2 (C > 1): Solution quickly explodes ("blows up").
Reason: For the explicit finite-difference wave equation, stability requires C ≤ 1 (the Courant–Friedrichs–Lewy, or CFL, condition).
"""
)
