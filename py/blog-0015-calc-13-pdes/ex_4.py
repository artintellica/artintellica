"""
exercise_4_wave_2d_extension.py
-------------------------------------------------
Simulate the 2D wave equation on a 32x32 grid with a hot spot in the center.
Animate the wave propagation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters
N = 32
dx = 1.0 / N
c = 1.0
dt = 0.6 * dx / c  # C < 1 for stability
steps = 100

# --- Initial state: hot spot in the center
u = np.zeros((N, N))
cx, cy = N // 2, N // 2
u[cx, cy] = 1.0

# Optional: make a larger hot spot (uncomment to try)
# u[cx-1:cx+2, cy-1:cy+2] = 1.0

u_prev = u.copy()  # zero initial velocity
C2 = (c * dt / dx) ** 2

# --- Storage for frames
frames = []

for n in range(steps):
    u_next = np.zeros_like(u)
    # Update internal grid (finite difference Laplacian)
    u_next[1:-1, 1:-1] = (
        2 * u[1:-1, 1:-1]
        - u_prev[1:-1, 1:-1]
        + C2
        * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1])
    )
    # Fixed boundary: u=0 at the edges
    u_next[0, :] = u_next[-1, :] = u_next[:, 0] = u_next[:, -1] = 0.0
    frames.append(u_next.copy())
    u_prev, u = u, u_next

# --- Animation
fig, ax = plt.subplots(figsize=(5.3, 5))
im = ax.imshow(frames[0], vmin=-1.0, vmax=1.0, cmap="seismic", origin="lower")
ax.set_title("2D Wave Equation: Hot Spot in Center")


def animate(i):
    im.set_array(frames[i])
    ax.set_xlabel(f"Step {i}")
    return (im,)


ani = animation.FuncAnimation(
    fig, animate, frames=len(frames), interval=60, blit=True, repeat=True
)
plt.tight_layout()
plt.show()
