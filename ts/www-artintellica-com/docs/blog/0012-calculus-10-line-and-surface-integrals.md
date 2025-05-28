+++
title = "Calculus 10: Line & Surface Integrals — Work, Flux, and Streamplots in Machine Learning"
date  = "2025‑05‑28"
author = "Artintellica"
+++

> _“A line integral adds up little bits of work as you move along a path in a
> vector field.”_

---

## 1 · What Are Line and Surface Integrals?

- **Line Integral (Vector Field):** For a path $\mathbf{r}(t)$ in a vector field
  $\mathbf{F}$:

  $$
  W = \int_C \mathbf{F} \cdot d\mathbf{r} = \int_a^b \mathbf{F}(\mathbf{r}(t)) \cdot \mathbf{r}'(t)\;dt
  $$

  Physically: **work** done by the field along the path.

- **Surface Integral (brief):** For a vector field over a surface $S$,

  $$
  \iint_S \mathbf{F} \cdot d\mathbf{S}
  $$

  measures **flux** through the surface (think: how much “stuff” passes
  through).

---

## 2 · Why ML Engineers Care

| Concept                | ML Example                                              |
| ---------------------- | ------------------------------------------------------- |
| **Work/Path Integral** | Energy cost along optimization or RL trajectory         |
| **Surface/Flux**       | Policy flow, divergence in density estimation           |
| **Visualization**      | Streamlines in policy-gradient methods, or GAN dynamics |

---

## 3 · Demo ① — Work Along a Path in a Vector Field

Let’s use the field:

$$
\mathbf{F}(x, y) = [-y, x] \qquad \text{(a simple “rotation” field)}
$$

and the path: a unit circle,
$\mathbf{r}(t) = [\cos t, \sin t],\; t \in [0, 2\pi]$.

The analytic result for the work is $2\pi$.

```python
# calc-10-line-surface/work_circle.py
import numpy as np
import matplotlib.pyplot as plt

# --- Vector field F(x, y) = [-y, x]
def F(x, y):
    return np.array([-y, x])

# --- Path: unit circle
N = 400
t = np.linspace(0, 2 * np.pi, N)
r = np.stack([np.cos(t), np.sin(t)], axis=1)
drdt = np.stack([-np.sin(t), np.cos(t)], axis=1)
Fs = np.stack([F(x, y) for x, y in r])

# Compute dot(F, dr/dt) at each t
dots = np.sum(Fs * drdt, axis=1)
work = np.trapz(dots, t)
print(f"Work along circle: {work:.5f} (analytic = {2 * np.pi:.5f})")

# --- Plot field, path, streamplot
xv, yv = np.meshgrid(np.linspace(-1.3, 1.3, 24), np.linspace(-1.3, 1.3, 24))
U, V = F(xv, yv)
plt.figure(figsize=(6, 6))
plt.streamplot(xv, yv, U, V, color="gray", density=1.1, linewidth=0.7, arrowsize=1)
plt.plot(r[:, 0], r[:, 1], "r", label="Path (unit circle)")
plt.scatter([0], [0], color="k", s=35, label="Origin")
plt.title("Vector field $[-y, x]$ and circular path")
plt.xlabel("x"); plt.ylabel("y")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
```

---

## 4 · Demo ② — Streamplot of a Gradient Field

Let’s look at the field $\mathbf{F}(x, y) = -\nabla f$, where
$f(x, y) = x^2 + y^2$:

$$
\mathbf{F}(x, y) = [-2x, -2y]
$$

This is a classic “downhill to the origin” field.

```python
# calc-10-line-surface/streamplot_grad.py
import numpy as np
import matplotlib.pyplot as plt

def gradF(x, y):
    return -2 * x, -2 * y

xv, yv = np.meshgrid(np.linspace(-2, 2, 28), np.linspace(-2, 2, 28))
U, V = gradF(xv, yv)
plt.figure(figsize=(6, 6))
plt.streamplot(xv, yv, U, V, color="blue", density=1.4, linewidth=1)
plt.title(r"Streamplot: $-\nabla f$, $f(x,y)=x^2+y^2$")
plt.xlabel("x"); plt.ylabel("y")
plt.scatter([0], [0], color="k", s=30, label="Minimum")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
```

---

## 5 · Why Streamplots?

Streamplots show how “agents” or “probability mass” flow under a vector field.

- In RL, this is the _policy flow_.
- In gradient descent, it’s the path loss would take if following steepest
  descent.
- For GANs, dynamics are often viewed as flows in parameter space.

---

## 6 · Exercises

1. **Elliptical Path:** Compute the work done by $\mathbf{F}(x, y) = [-y, x]$
   along the ellipse $x=2\cos t, y=\sin t$. Compare to the analytic result.
2. **Radial Field:** Compute the work along the unit circle for
   $\mathbf{F}(x, y) = [x, y]$. What should the work be? Why?
3. **Custom Field Streamplot:** Visualize the streamplot for
   $\mathbf{F}(x, y) = [\sin y, \cos x]$ over $[-2, 2]^2$.
4. **RL Connection:** For a simple reward landscape
   $f(x, y) = -((x-1)^2 + (y+1)^2)$, plot the gradient field, and show the
   optimal trajectory starting from $(0, 0)$.

Put solutions in `calc-10-line-surface/` and tag `v0.1`.

---

**Next:** _Calculus 11 — Divergence, Curl, and the Geometry of Probability
Flows._
