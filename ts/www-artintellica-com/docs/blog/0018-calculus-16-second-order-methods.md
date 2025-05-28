+++
title = "Calculus 16: Hessian-Vector Products & Newton-like Steps — Second-Order Optimization in Practice"
date = "2025‑05‑28"
author = "Artintellica"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0018-calc-16-second-order-methods"
+++

> _“First-order methods feel the slope; second-order methods see the
> curvature.”_

---

## 1 · Why Care About Second-Order Methods?

- **First-order methods** (like SGD/Adam) use only the gradient (slope).
- **Second-order methods** (Newton, quasi-Newton, L‑BFGS) use curvature (the
  Hessian or an approximation) for **faster convergence** on convex problems.
- **Hessian-vector products** allow Newton-like steps efficiently **without
  forming the full Hessian matrix**, which is crucial for ML-scale models.

---

## 2 · Mathematics: Hessian, Newton’s Method, and L‑BFGS

### **Gradient:**

$$
\mathbf{g} = \nabla f(\mathbf{x})
$$

### **Hessian:**

$$
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

### **Newton step:**

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - H^{-1} \mathbf{g}
$$

### **Hessian-vector product:**

Given a vector $v$, compute $Hv$ _without forming $H$_.

#### **Pearlmutter’s trick (autodiff magic):**

$$
Hv = \frac{d}{d\epsilon}\Big|_{\epsilon=0} \nabla f(\mathbf{x} + \epsilon v)
$$

---

## 3 · Python Demo: SGD vs. L‑BFGS on a Quadratic

We'll compare both on a convex quadratic, where Newton’s method converges in one
step.

```python
# calc-16-hessian-vector/newton_lbfgs_demo.py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import LBFGS, SGD

# Function: f(x) = 0.5 * x^T A x + b^T x
A = np.array([[5, 1], [1, 3]])
b = np.array([-4, 2])

def f_np(x):
    return 0.5 * x @ A @ x + b @ x

# Plot function surface
xg, yg = np.meshgrid(np.linspace(-3, 3, 80), np.linspace(-3, 3, 80))
zg = np.array([f_np(np.array([x, y])) for x, y in zip(xg.ravel(), yg.ravel())]).reshape(xg.shape)
plt.figure(figsize=(6, 5))
plt.contourf(xg, yg, zg, levels=30, cmap="Spectral")
plt.colorbar(label="f(x)")
plt.xlabel("x"); plt.ylabel("y"); plt.title("Convex Quadratic")
plt.tight_layout(); plt.show()

# PyTorch version
A_t = torch.tensor(A, dtype=torch.float64)
b_t = torch.tensor(b, dtype=torch.float64)

def f_torch(x):
    return 0.5 * x @ A_t @ x + b_t @ x

# Initial point
x0 = torch.tensor([2.5, -2.0], dtype=torch.float64, requires_grad=True)

# --------- SGD ---------
x_sgd = x0.clone().detach().requires_grad_(True)
opt_sgd = SGD([x_sgd], lr=0.1)
sgd_path = [x_sgd.detach().numpy().copy()]
for _ in range(40):
    opt_sgd.zero_grad()
    loss = f_torch(x_sgd)
    loss.backward()
    opt_sgd.step()
    sgd_path.append(x_sgd.detach().numpy().copy())

# --------- L‑BFGS ---------
x_lbfgs = x0.clone().detach().requires_grad_(True)
opt_lbfgs = LBFGS([x_lbfgs], lr=1, max_iter=40, history_size=10, line_search_fn="strong_wolfe")
lbfgs_path = [x_lbfgs.detach().numpy().copy()]
def closure():
    opt_lbfgs.zero_grad()
    loss = f_torch(x_lbfgs)
    loss.backward()
    lbfgs_path.append(x_lbfgs.detach().numpy().copy())
    return loss
opt_lbfgs.step(closure)

# Plot optimization paths
plt.figure(figsize=(6, 5))
plt.contour(xg, yg, zg, levels=30, cmap="Spectral")
sgd_path = np.array(sgd_path)
lbfgs_path = np.array(lbfgs_path)
plt.plot(sgd_path[:, 0], sgd_path[:, 1], "o-", label="SGD", alpha=0.7)
plt.plot(lbfgs_path[:, 0], lbfgs_path[:, 1], "s-", label="L‑BFGS", alpha=0.7)
plt.scatter(*(-np.linalg.solve(A, b)), color="k", marker="*", s=140, label="Optimum")
plt.xlabel("x"); plt.ylabel("y")
plt.title("SGD vs. L‑BFGS optimization paths")
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 4 · Python Demo: Hessian-vector Product via Autograd

```python
# calc-16-hessian-vector/hvp_demo.py
import torch

A = torch.tensor([[5.0, 1.0], [1.0, 3.0]], requires_grad=False)
b = torch.tensor([-4.0, 2.0], requires_grad=False)
x = torch.tensor([1.5, -1.2], requires_grad=True)

def f(x):
    return 0.5 * x @ A @ x + b @ x

fval = f(x)
grad = torch.autograd.grad(fval, x, create_graph=True)[0]
v = torch.tensor([0.6, -1.0])
# Compute Hessian-vector product
hvp = torch.autograd.grad(grad @ v, x)[0]
print("Hessian-vector product Hv:", hvp.numpy())
# Check against explicit H @ v
H = A.numpy()
print("Direct H @ v:", H @ v.numpy())
```

---

## 5 · Exercises

1. **Try Non-Quadratic:** Run SGD and L‑BFGS on a non-quadratic function (e.g.
   Rosenbrock). How do the paths and convergence differ?
2. **Newton Step by Hand:** For the quadratic above, compute the Newton step
   explicitly (using numpy.linalg.solve) and show it lands at the optimum in one
   update.
3. **Hessian-Vector in High D:** Generate a random 20×20 positive-definite $A$,
   pick random $v$, and use autograd to compute $Hv$. Time it versus full
   Hessian computation.
4. **Quasi-Newton Intuition:** Implement a simple BFGS update for a 2D quadratic
   (see Wikipedia) and plot its approximation to the Hessian at each step.

Put solutions in `calc-16-hessian-vector/` and tag `v0.1`.

---

**Next:** _Calculus 17 — Higher-order Gradients, Meta-Learning, and Beyond._
