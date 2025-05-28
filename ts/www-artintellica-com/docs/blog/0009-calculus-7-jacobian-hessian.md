+++
title = "Calculus 7: Jacobian & Hessian — Second-Order Structure for Smarter Learning"
date  = "2025‑05‑28"
author = "Artintellica"
+++

> _“The gradient tells you how to step. The Hessian tells you how to
> **curve**.”_

---

## 1 · What Are Jacobians and Hessians?

Let $f : \mathbb{R}^n \to \mathbb{R}^m$.

- The **Jacobian** $J$ at $\mathbf{x}$ is the $m \times n$ matrix whose $(i, j)$
  entry is $\frac{\partial f_i}{\partial x_j}$.

If $f : \mathbb{R}^n \to \mathbb{R}$, the **Hessian** $H$ is the $n \times n$
matrix whose $(i, j)$ entry is $\frac{\partial^2 f}{\partial x_i \partial x_j}$.

---

## 2 · Why ML Engineers Care

|   Concept    | Why it matters                                                                                                       |
| :----------: | -------------------------------------------------------------------------------------------------------------------- |
| **Jacobian** | How a layer/output responds to changes in all its inputs. Used in sensitivity, invertibility, and normalizing flows. |
| **Hessian**  | Curvature of loss: used in Newton’s method, second‑order optimization, understanding saddle points.                  |
| **Spectrum** | Eigenvalues reveal flatness, sharpness, or ill‑conditioning — affecting convergence speed.                           |

---

## 3 · Demo ① — Batch Jacobian of a Tiny MLP

Suppose our “neural network” is:

$$
f(\mathbf{x}) = \sigma(W\mathbf{x} + \mathbf{b})
$$

with $\sigma(z) = \tanh(z)$, $W$ shape $2 \times 2$, $\mathbf{b}$ shape $2$.

Let’s compute the batch Jacobian with PyTorch.

```python
# calc-07-jacobian-hessian/mlp_jacobian.py
import torch

torch.set_default_dtype(torch.float64)

def mlp(x, W, b):
    return torch.tanh(W @ x + b)     # shape [2]

# random weights, biases, batch of points
W = torch.tensor([[1.0, -0.7],
                  [0.4,  0.9]], requires_grad=True)
b = torch.tensor([0.5, -0.3], requires_grad=True)
xs = [torch.tensor([1.2, -0.8], requires_grad=True),
      torch.tensor([0.3, 2.0], requires_grad=True)]

for i, x in enumerate(xs):
    y = mlp(x, W, b)
    J = torch.zeros((2, 2))
    for j in range(2):           # output dim
        grad = torch.autograd.grad(y[j], x, retain_graph=True, create_graph=True)[0]
        J[j] = grad
    print(f"\nJacobian at input {i}:")
    print(J.detach().numpy())
```

---

## 4 · Demo ② — Hessian Spectrum of a Small MLP Loss

Let’s make a toy 2-layer MLP with scalar output, and plot the eigenvalues of its
Hessian at a random input. (You’ll see how “curved” the loss is in different
directions.)

```python
# calc-07-jacobian-hessian/mlp_hessian_spectrum.py
import torch
import numpy as np

torch.set_default_dtype(torch.float64)

def net(x, params):
    W1, b1, W2, b2 = params
    h = torch.tanh(W1 @ x + b1)
    out = W2 @ h + b2
    return out

# Small network (2‑2‑1)
W1 = torch.randn(2, 2, requires_grad=True)
b1 = torch.randn(2, requires_grad=True)
W2 = torch.randn(1, 2, requires_grad=True)
b2 = torch.randn(1, requires_grad=True)
params = [W1, b1, W2, b2]
x = torch.randn(2, requires_grad=True)

def flat_params(params):
    return torch.cat([p.reshape(-1) for p in params] + [x.reshape(-1)])

y = net(x, params)
all_vars = [W1, b1, W2, b2, x]
y.backward(create_graph=True)

flat = flat_params(params)
n = flat.numel()
hessian = torch.zeros(n, n)
for i in range(n):
    grad_i = torch.autograd.grad(y, flat, retain_graph=True, create_graph=True, allow_unused=True)[0][i]
    row = torch.autograd.grad(grad_i, flat, retain_graph=True, allow_unused=True)[0]
    hessian[i] = row

# Eigen spectrum
eigvals = np.linalg.eigvalsh(hessian.detach().numpy())
print("Hessian eigenvalues:", np.round(eigvals, 5))

import matplotlib.pyplot as plt
plt.stem(eigvals)
plt.title("Hessian eigenvalues (spectrum)")
plt.xlabel("direction")
plt.ylabel("curvature")
plt.tight_layout()
plt.show()
```

_You may see positive, negative, and near‑zero eigenvalues — corresponding to
convex, concave, and flat directions (“saddles”)._

---

## 5 · Interpretation

- **Wide range of eigenvalues** → ill‑conditioning, slow training, sharp minima.
- **Many zero eigenvalues** → flat valleys, over‑parameterization, or symmetry.
- **All positive** → locally convex (easy to optimize).
- **Mixed signs** → saddles and potential slowdowns.

---

## 6 · Exercises

1. **Chain Rule** For $h(x,y) = \tanh(ax + by)$ with arbitrary $a, b$, compute
   the Jacobian analytically and confirm numerically at several points.
2. **Hessian of Quadratic** For $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ with
   a random symmetric $2\times2$ matrix $A$, compute its Hessian analytically
   and check with autograd.
3. **Saddle Plot** For $f(x,y)=x^3-3xy^2$, compute the Hessian at (0,0) and plot
   its eigenvalues at grid points; color-code the sign (saddle/convex/concave).
4. **Fastest Direction** In the MLP Hessian spectrum demo above, pick the
   largest-magnitude eigenvector and show the function’s value along that
   direction. Plot how “steep” or “flat” the landscape is in that direction.

Put solutions in `calc-07-jacobian-hessian/` and tag `v0.1`.

---

**Next:** _Calculus 8 — Chain Rule & Backpropagation: Generalizing Gradients in
Neural Nets._
