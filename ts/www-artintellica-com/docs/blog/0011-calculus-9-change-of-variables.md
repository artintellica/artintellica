+++
title = "Calculus 9: Change of Variables, Jacobians, and Normalizing Flows"
date  = "2025‑05‑28"
author = "Artintellica"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0011-calc-09-change-of-variables"
+++

> _“To model new distributions, we transform simple ones. The math: Jacobians
> and their determinants.”_

---

## 1 · Change of Variables in Probability

Suppose we transform a random vector $\mathbf{z} \in \mathbb{R}^n$ via a
**differentiable, invertible** map $\mathbf{x} = f(\mathbf{z})$.

The probability density changes as:

$$
p_{\mathbf{x}}(\mathbf{x}) = p_{\mathbf{z}}(f^{-1}(\mathbf{x})) \cdot
\left| \det \frac{\partial f^{-1}}{\partial \mathbf{x}} \right|
$$

or (when going forward):

$$
p_{\mathbf{z}}(\mathbf{z}) = p_{\mathbf{x}}(f(\mathbf{z})) \cdot
\left| \det \frac{\partial f}{\partial \mathbf{z}} \right|
$$

In 1D, this is just the absolute value of the derivative.

---

## 2 · The Jacobian Matrix and Its Determinant

Given $f : \mathbb{R}^n \to \mathbb{R}^n$:

$$
J_f(\mathbf{z}) = \frac{\partial f}{\partial \mathbf{z}}
= \left[ \frac{\partial f_i}{\partial z_j} \right]_{i,j=1}^{n}
$$

The **determinant** $|\det J_f|$ measures local “stretch” or “compression” of
volume.

- If $|\det J_f| > 1$: region expands
- If $< 1$: region shrinks
- If $= 0$: transformation is singular (bad for flows)

---

## 3 · Why ML Engineers Care: Flow-Based Models

**Normalizing flows** = trainable, invertible transformations mapping simple
base distributions (e.g. standard Gaussian) into complex ones.

The log-likelihood under a flow is:

$$
\log p_\mathbf{x}(\mathbf{x}) = \log p_\mathbf{z}(f^{-1}(\mathbf{x})) - \log \left| \det J_f(f^{-1}(\mathbf{x})) \right|
$$

Efficient calculation of $\log |\det J|$ is key!

---

## 4 · Demo — 2D Affine Flow and Log-Determinant in PyTorch

Let’s build a 2-D affine transformation:

$$
f(\mathbf{z}) = A \mathbf{z} + \mathbf{b}
$$

where $A$ is invertible.

```python
# calc-09-change-of-vars/affine_flow.py
import torch

torch.set_default_dtype(torch.float64)

# --- Parameters
A = torch.tensor([[2.0, 0.3],
                  [0.1, 1.5]], requires_grad=True)
b = torch.tensor([1.0, -2.0], requires_grad=True)

# --- Inverse affine
def f(z):
    return A @ z + b

def f_inv(x):
    return torch.linalg.solve(A, x - b)

# --- Jacobian determinant (analytic, since affine)
def log_det_jacobian():
    return torch.logdet(A)

# --- Log-likelihood of x under the flow
def log_prob_x(x):
    z = f_inv(x)
    logpz = -0.5 * torch.dot(z, z) - torch.log(torch.tensor(2 * torch.pi))  # standard 2D normal
    logdet = log_det_jacobian()
    return logpz - logdet

# --- Test
x = torch.tensor([2.0, 0.0], requires_grad=True)
print("x:", x.detach().numpy())
print("f⁻¹(x):", f_inv(x).detach().numpy())
print("log|det Jacobian|:", log_det_jacobian().item())
print("log-prob(x):", log_prob_x(x).item())
```

**To compute the Jacobian via autograd:**

```python
def autograd_logdet(x):
    x = x.clone().detach().requires_grad_(True)
    z = f_inv(x)
    J = torch.autograd.functional.jacobian(f_inv, x)
    return torch.log(torch.abs(torch.det(J)))

print("Autograd log|det J|:", autograd_logdet(x).item())
```

For affine, this matches the analytic result.

---

## 5 · Exercises

1. **Jacobian Check** For the flow $f(z) = Az + b$ above, use autograd to
   compute the full Jacobian at several random $z$ and confirm it matches $A$.
2. **Nonlinear Flow** Let $f(z) = \text{tanh}(Az + b)$. For a random point, use
   autograd to compute the Jacobian and its log-determinant. Try with different
   values of $A$.
3. **Volume Transformation** For a square region in $z$, sample 10,000 points,
   transform to $x = f(z)$, and visualize the spread of the samples before and
   after.
4. **Flow Likelihood Grid** Make a contour plot of $\log p_x(x)$ over a grid of
   $x$ covering $[−3,3]^2$. Where is the flow most/least likely to generate
   points?

Put solutions in `calc-09-change-of-vars/` and tag `v0.1`.

---

**Next:** _Calculus 10 — Divergence, Curl, and the Geometry of Probability
Flows._
