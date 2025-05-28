"""
exercise_2_wave_reflecting_absorbing.py
-------------------------------------------------
Simulate the 1D wave equation with both reflecting (fixed)
and absorbing (open) boundaries. Compare their effect.
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

# --- Initial state: Gaussian bump
x = np.linspace(0, 1, N)
mu = 0.5
sigma = 0.06
u0 = np.exp(-0.5 * ((x - mu) / sigma) ** 2)

C2 = (c * dt / dx) ** 2


# --- Simulation function
def simulate_wave(u_init, boundary="reflecting"):
    u = u_init.copy()
    u_prev = u_init.copy()
    frames = []
    for n in range(steps):
        u_next = np.zeros_like(u)
        u_next[1:-1] = 2 * u[1:-1] - u_prev[1:-1] + C2 * (u[2:] - 2 * u[1:-1] + u[:-2])
        # Boundaries
        if boundary == "reflecting":
            u_next[0] = u_next[-1] = 0.0
        elif boundary == "absorbing":
            # Open boundary: "let wave out" by copying previous values
            u_next[0] = u[1]
            u_next[-1] = u[-2]
        frames.append(u_next.copy())
        u_prev, u = u, u_next
    return frames


frames_ref = simulate_wave(u0, boundary="reflecting")
frames_abs = simulate_wave(u0, boundary="absorbing")

# --- Animation side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.3), sharey=True)
(line1,) = ax1.plot(x, frames_ref[0], label="Reflecting (fixed)")
(line2,) = ax2.plot(x, frames_abs[0], label="Absorbing (open)", color="orange")
ax1.set_title("Reflecting Boundaries")
ax2.set_title("Absorbing Boundaries")
ax1.set_ylim(-1.2, 1.2)
ax2.set_ylim(-1.2, 1.2)


def animate(i):
    line1.set_ydata(frames_ref[i])
    line2.set_ydata(frames_abs[i])
    ax1.set_xlabel(f"Step {i}")
    ax2.set_xlabel(f"Step {i}")
    return line1, line2


ani = animation.FuncAnimation(
    fig, animate, frames=len(frames_ref), interval=35, blit=True, repeat=True
)
plt.tight_layout()
plt.show()
