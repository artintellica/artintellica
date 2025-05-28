+++
title = "Calculus 11: Divergence, Curl, and Laplacian — Diffusion, Heat, and Curvature"
date  = "2025‑05‑28"
author = "Artintellica"
+++

> _“The Laplacian captures the ‘flow of flow’ — the ultimate second-derivative.
> In physics: heat. In ML: smoothness, diffusion, and curvature.”_

---

## 1 · Key Operators: Divergence, Curl, Laplacian

**For a 2D vector field $\mathbf{F}(x, y) = (F_x, F_y)$:**

- **Divergence:**

  $$
  \nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y}
  $$

  (How much “outflow” from a point.)

- **Curl:**

  $$
  \nabla \times \mathbf{F} = \frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y}
  $$

  (Local “rotation” of the field.)

- **Laplacian:** For scalar field $f(x, y)$:

  $$
  \Delta f = \nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}
  $$

  (Sum of 2nd derivatives; a “curvature” operator.)

---

## 2 · Why ML Engineers Care

| Operator       | ML Example                                                                             |
| -------------- | -------------------------------------------------------------------------------------- |
| **Laplacian**  | Smoothing, diffusion models, graph Laplacian, loss regularization, spectral clustering |
| **Divergence** | Generative flows, conservation laws                                                    |
| **Curl**       | Detecting cycles/rotation in flows, e.g. GAN training dynamics                         |

---

## 3 · Demo ① — The Heat Equation on a Grid

The **heat equation** in 2D:

$$
\frac{\partial u}{\partial t} = D\, \Delta u
$$

where $D$ is the diffusion constant, and $\Delta u$ is the Laplacian.

**Discretized:** We can approximate the Laplacian with a convolution kernel:

$$
\Delta u_{i,j} \approx u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}
$$

### Python Demo: Simulate Heat Diffusion

```python
# calc-11-div-curl-laplace/heat_equation_demo.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

N = 64
D = 0.15   # diffusion constant
steps = 80

# Laplacian kernel for 2D grid
kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])

# Initial "hot spot" in center
u = np.zeros((N, N))
u[N//2, N//2] = 10.0

# For plotting
fig, axs = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
plot_steps = np.linspace(0, steps-1, 8, dtype=int)

for s in range(steps):
    lap = convolve(u, kernel, mode="constant")
    u += D * lap
    if s in plot_steps:
        ax = axs.flat[list(plot_steps).index(s)]
        im = ax.imshow(u, vmin=0, vmax=10, cmap="hot")
        ax.set_title(f"step {s}")

plt.colorbar(im, ax=axs.ravel().tolist(), shrink=0.85)
plt.suptitle("2D Heat Equation Evolution (Diffusion of Hot Spot)")
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()
```

---

## 4 · Connection to ML

- **Diffusion models:** Recent generative models (DDPMs, etc.) _literally_
  simulate heat/diffusion in data space.
- **Graph Laplacian:** On graphs, the Laplacian encodes “smoothness” and is used
  for clustering and semi-supervised learning.
- **Regularization:** Penalizing $\Delta f$ encourages smooth functions.

---

## 5 · Exercises

1. **Divergence and Curl on a Grid:** For $\mathbf{F}(x, y) = [y, -x]$, compute
   divergence and curl at each grid point in $[-2,2]^2$, and plot them as
   images.
2. **Laplacian of a Gaussian:** Compute and plot $\Delta f$ for
   $f(x, y) = \exp(-x^2 - y^2)$. Where is it most negative?
3. **Multiple Hot Spots:** Initialize $u$ with three “hot spots.” Simulate the
   heat equation and show how the blobs merge.
4. **Graph Laplacian:** Construct a 6-node ring graph Laplacian. Simulate
   diffusion of an initial delta on one node for 10 steps and plot the value on
   each node at each step.

Put solutions in `calc-11-div-curl-laplace/` and tag `v0.1`.

---

**Next:** _Calculus 12 — ODEs, Gradient Flows, and Neural Dynamics._
