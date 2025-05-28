"""
exercise_3_heat_multiple_hotspots.py
-------------------------------------------------
Simulate the 2D heat equation with three initial "hot spots"
and visualize how they merge over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

N = 64
D = 0.15  # diffusion constant
steps = 80

# Laplacian kernel for 2D grid
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# --- Initial state: three hot spots
u = np.zeros((N, N))
u[N // 3, N // 3] = 10.0
u[2 * N // 3, N // 2] = 10.0
u[N // 2, 3 * N // 4] = 10.0

# For plotting
fig, axs = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
plot_steps = np.linspace(0, steps - 1, 8, dtype=int)

for s in range(steps):
    lap = convolve(u, kernel, mode="constant")
    u += D * lap
    if s in plot_steps:
        ax = axs.flat[list(plot_steps).index(s)]
        im = ax.imshow(u, vmin=0, vmax=10, cmap="hot")
        ax.set_title(f"step {s}")

plt.colorbar(im, ax=axs.ravel().tolist(), shrink=0.85)
plt.suptitle("2D Heat Equation: Three Hot Spots Merging")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
