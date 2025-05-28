+++
title = "Calculus 12: Ordinary Differential Equations (ODEs) — Neural ODEs in Action"
date  = "2025‑05‑28"
author = "Artintellica"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0014-calc-12-odes"
+++

> _“In a Neural ODE, the network defines the velocity. Learning is learning the
> flow.”_

---

## 1 · What Are ODEs?

An **ordinary differential equation (ODE)** specifies the time evolution of a
system via:

$$
\frac{d\mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t)
$$

where $\mathbf{h}(t)$ is the state at time $t$, and $f$ is a vector field.

---

## 2 · Why ML Engineers Care

- **Neural ODEs:** Let a neural network define $f$; solve for $h(t)$ with an ODE
  solver, backpropagate through the solution.
- **Continuous normalizing flows:** Transform distributions smoothly over time.
- **Latent SDEs:** Generative models in continuous time.
- **Control & RL:** Continuous dynamics for agents.

---

## 3 · Demo ① — Numerically Solving a Simple ODE

Let’s solve the “spiral” ODE:

$$
\frac{d}{dt}\begin{bmatrix}x \\ y\end{bmatrix} =
\begin{bmatrix}
\alpha x - \beta y \\
\beta x + \alpha y
\end{bmatrix}
$$

for $\alpha < 0$, $\beta > 0$, which produces spirals to the origin.

```python
# calc-12-ode/spiral_ode_numpy.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

alpha = -0.3
beta = 1.1

def f(t, h):
    x, y = h
    dx = alpha * x - beta * y
    dy = beta * x + alpha * y
    return [dx, dy]

t_span = [0, 8]
h0 = [2, 0.5]
sol = solve_ivp(f, t_span, h0, t_eval=np.linspace(*t_span, 400))

plt.plot(sol.y[0], sol.y[1], label="trajectory")
plt.title("Spiral ODE trajectory")
plt.xlabel("x"); plt.ylabel("y")
plt.grid()
plt.axis("equal")
plt.show()
```

---

## 4 · Demo ② — Training a Neural ODE on Spirals (`torchdiffeq`)

We’ll train a **Neural ODE** to fit spiral trajectories. You need `torchdiffeq`:
`pip install torchdiffeq`

```python
# calc-12-ode/neural_ode_spiral.py
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

torch.manual_seed(42)
device = torch.device("cpu")

# --- Spiral data generation
alpha, beta = -0.4, 1.2

def true_field(t, h):
    x, y = h[..., 0], h[..., 1]
    dx = alpha * x - beta * y
    dy = beta * x + alpha * y
    return torch.stack([dx, dy], -1)

t = torch.linspace(0, 7, 160)
h0 = torch.tensor([2.0, 0.7])

with torch.no_grad():
    true_traj = odeint(true_field, h0, t)

# --- Neural ODE model
class ODEFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 2),
        )

    def forward(self, t, h):
        return self.net(h)

odefunc = ODEFunc()
optimizer = torch.optim.Adam(odefunc.parameters(), lr=0.01)

# --- Training loop
for epoch in range(250):
    pred_traj = odeint(odefunc, h0, t)
    loss = ((pred_traj - true_traj) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 40 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.5f}")

# --- Plot
plt.plot(true_traj[:, 0], true_traj[:, 1], label="True", lw=3)
plt.plot(pred_traj.detach()[:, 0], pred_traj.detach()[:, 1], "--", label="Neural ODE", lw=2)
plt.legend()
plt.xlabel("x"); plt.ylabel("y")
plt.title("Neural ODE fits Spiral Trajectory")
plt.axis("equal")
plt.tight_layout()
plt.show()
```

---

## 5 · How Neural ODEs Work

- The ODE solver integrates the neural net’s output as a velocity field.
- **Gradients** are backpropagated through the ODE solution (adjoint method).
- Enables **continuous-depth** models, generative flows, and SDEs.

---

## 6 · Exercises

1. **Different ODE:** Change $\alpha, \beta$ and see what kind of trajectories
   arise (e.g. make $\alpha > 0$).
2. **Multiple Initial States:** Plot spiral ODE solutions for 4 different $h_0$,
   overlaying the true and Neural ODE fits.
3. **Time-Varying Field:** Modify the neural net to take $t$ as an input as
   well: `torch.cat([h, t*torch.ones_like(h[..., :1])], dim=-1)`.
4. **Noise Robustness:** Add Gaussian noise to the spiral data and train the
   Neural ODE. Plot how well it can still fit the noisy trajectories.

Put solutions in `calc-12-ode/` and tag `v0.1`.

---

**Next:** _Calculus 13 — Partial Differential Equations, SDEs, and Diffusion
Models in Machine Learning._
