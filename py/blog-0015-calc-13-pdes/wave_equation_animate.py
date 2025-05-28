# calc-13-pde/wave_equation_animate.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters
N = 180
dx = 1.0 / N
c = 1.0  # wave speed
dt = 0.8 * dx / c
steps = 240

# --- Initial state: bump in the middle
u = np.zeros(N)
u[N // 3 : N // 3 * 2] = 1.0
u_prev = u.copy()  # Assume zero velocity (u_prev = u at t=0)

C2 = (c * dt / dx) ** 2

# --- Storage for frames
frames = []

for n in range(steps):
    # Compute new u using finite-difference
    u_next = np.zeros_like(u)
    u_next[1:-1] = 2 * u[1:-1] - u_prev[1:-1] + C2 * (u[2:] - 2 * u[1:-1] + u[:-2])
    # Boundary conditions: fixed ends (u=0)
    u_next[0] = u_next[-1] = 0.0
    frames.append(u_next.copy())
    # Advance
    u_prev, u = u, u_next

# --- Animation
fig, ax = plt.subplots(figsize=(7, 3))
(line,) = ax.plot(frames[0])
ax.set_ylim(-1.2, 1.2)
ax.set_title("1D Wave Equation")


def animate(i):
    line.set_ydata(frames[i])
    ax.set_xlabel(f"Step {i}")
    return (line,)


ani = animation.FuncAnimation(
    fig, animate, frames=len(frames), interval=35, blit=True, repeat=True
)
plt.show()
