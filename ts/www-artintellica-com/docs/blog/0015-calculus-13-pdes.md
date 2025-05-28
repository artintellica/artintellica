+++
title = "Calculus 13: Partial Differential Equations (PDEs) — Simulating the Wave Equation"
date  = "2025‑05‑27"
author = "Ryan X. Charles"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0015-calc-13-pdes"
+++

> _“PDEs let us model how information, energy, or probability spreads in space
> and time.”_

---

## 1 · What Are PDEs?

A **partial differential equation** involves unknown functions of several
variables and their partial derivatives.

### Example: **The 1D Wave Equation**

$$
\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}
$$

where $u(x, t)$ describes the wave (e.g., string displacement), and $c$ is the
speed.

- **Initial condition:** shape of the string at $t=0$.
- **Boundary conditions:** what happens at the ends.

---

## 2 · Why ML Engineers Care

- **Physics-Informed Neural Networks (PINNs):** Learn solutions to PDEs from
  data (used for modeling physical systems).
- **Diffusion, Vision, and Graphs:** Many image and diffusion operations are
  PDEs.
- **Generative models:** E.g., denoising diffusion models are essentially
  solving PDEs in data space.

---

## 3 · Finite-Difference Simulation: 1D Wave Equation

We discretize both space and time.

- $x$ grid: $i = 0, 1, ..., N-1$
- $t$ steps: $n = 0, 1, ...$

The finite-difference form is:

$$
u_i^{n+1} = 2u_i^n - u_i^{n-1} + C^2 (u_{i+1}^n - 2u_i^n + u_{i-1}^n)
$$

where $C = c \frac{\Delta t}{\Delta x}$ (Courant number).

---

## 4 · Python Demo: Animated Wave Equation

```python
# calc-13-pde/wave_equation_animate.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters
N = 180
dx = 1.0 / N
c = 1.0        # wave speed
dt = 0.8 * dx / c
steps = 240

# --- Initial state: bump in the middle
u = np.zeros(N)
u[N // 3:N // 3 * 2] = 1.0
u_prev = u.copy()  # Assume zero velocity (u_prev = u at t=0)

C2 = (c * dt / dx) ** 2

# --- Storage for frames
frames = []

for n in range(steps):
    # Compute new u using finite-difference
    u_next = np.zeros_like(u)
    u_next[1:-1] = (
        2 * u[1:-1] - u_prev[1:-1]
        + C2 * (u[2:] - 2 * u[1:-1] + u[:-2])
    )
    # Boundary conditions: fixed ends (u=0)
    u_next[0] = u_next[-1] = 0.0
    frames.append(u_next.copy())
    # Advance
    u_prev, u = u, u_next

# --- Animation
fig, ax = plt.subplots(figsize=(7, 3))
line, = ax.plot(frames[0])
ax.set_ylim(-1.2, 1.2)
ax.set_title("1D Wave Equation")

def animate(i):
    line.set_ydata(frames[i])
    ax.set_xlabel(f"Step {i}")
    return line,

ani = animation.FuncAnimation(
    fig, animate, frames=len(frames), interval=35, blit=True, repeat=True
)
plt.show()
```

---

## 5 · Exercises

1. **Initial Shape:** Change the initial $u$ to a Gaussian bump. How does the
   propagation differ?
2. **Reflecting vs. Absorbing Boundaries:** Implement open (absorbing)
   boundaries (no reflection) and compare to the fixed (reflecting) boundaries.
3. **Higher Courant Number:** Try increasing $C$ (i.e., `dt`). What happens when
   $C > 1$? Why does the solution become unstable?
4. **2D Extension:** Extend the simulation to a $32\times32$ grid, with a hot
   spot in the center. Animate the 2D wave propagation.

Put solutions in `calc-13-pde/` and tag `v0.1`.

---

**Next:** _Calculus 14 — From Diffusion to Deep Generative Models: Bridging PDEs
and Data._
