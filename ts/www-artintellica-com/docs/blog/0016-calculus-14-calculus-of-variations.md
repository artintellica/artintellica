+++
title = "Calculus 14: Functional Derivatives — The Gradient of Regularization"
date  = "2025‑05‑28"
author = "Artintellica"
+++

> _“Ordinary derivatives find optimal values; functional derivatives find
> optimal functions.”_

---

## 1 · What Are Functional Derivatives?

In classical calculus, we take derivatives of functions to find optimal points.
In the **calculus of variations**, we deal with **functionals**: mappings from
functions to numbers, like

$$
J[f] = \int L(x, f(x), f'(x)) \, dx
$$

We seek the function $f(x)$ that extremizes (minimizes or maximizes) $J[f]$.

---

## 2 · Why ML Engineers Care

- **Weight decay/L2 regularization**: Encourages smooth or small weights.
- **Sobolev (smoothness) regularization**: Penalizes roughness by adding terms
  like $\int (f'(x))^2 dx$.
- **Physics-informed neural nets (PINNs)**: Loss is an integral over how well a
  function solves a PDE.

---

## 3 · A Classic Example: Smoothness Regularizer

Consider the functional:

$$
J[f] = \int_a^b \left( f'(x) \right)^2 dx
$$

This measures the “roughness” of $f$. **Goal:** Find the functional derivative
$\frac{\delta J}{\delta f(x)}$, which gives the gradient with respect to $f(x)$.

### **Derivation (Euler-Lagrange Equation)**

The Euler-Lagrange equation for $J[f]$ is:

$$
\frac{\partial L}{\partial f} - \frac{d}{dx}\left( \frac{\partial L}{\partial f'} \right) = 0
$$

Here, $L = (f'(x))^2$, so $\frac{\partial L}{\partial f} = 0$, and
$\frac{\partial L}{\partial f'} = 2f'(x)$.

Plug in:

$$
0 - \frac{d}{dx}[2f'(x)] = -2f''(x)
$$

So the **functional gradient** is:

$$
\frac{\delta J}{\delta f(x)} = -2f''(x)
$$

This means the “steepest descent” direction is given by the **negative
Laplacian** (second derivative) of $f$.

---

## 4 · Python Demo: Compute the Functional Gradient Numerically

Let’s use finite-differences to compute this gradient for a discretized $f$.

```python
# calc-14-functional-deriv/functional_grad_demo.py
import numpy as np
import matplotlib.pyplot as plt

# Discretize domain
N = 120
x = np.linspace(0, 1, N)
dx = x[1] - x[0]

# A sample function
f = np.sin(3 * np.pi * x) + 0.4 * np.cos(5 * np.pi * x)

# Compute first derivative (central difference)
dfdx = np.zeros_like(f)
dfdx[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
dfdx[0] = (f[1] - f[0]) / dx        # forward difference at left
dfdx[-1] = (f[-1] - f[-2]) / dx     # backward at right

# Compute second derivative (Laplacian)
d2fdx2 = np.zeros_like(f)
d2fdx2[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / dx**2
d2fdx2[0] = d2fdx2[-1] = 0.0  # Dirichlet boundary (could also use one-sided diff)

# Compute functional, and its gradient
J = np.sum(dfdx**2) * dx
func_grad = -2 * d2fdx2

plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.plot(x, f, label="f(x)")
plt.plot(x, dfdx, "--", label="f'(x)")
plt.plot(x, d2fdx2, ":", label="f''(x)")
plt.legend()
plt.title("Function, Derivatives, and Functional Gradient")
plt.subplot(2,1,2)
plt.plot(x, func_grad, "r", label=r"$-2f''(x)$ (functional grad)")
plt.axhline(0, color='k', lw=0.7)
plt.legend()
plt.xlabel("x")
plt.tight_layout()
plt.show()

print(f"Value of regularizer J = ∫ (f')² dx: {J:.5f}")
```

---

## 5 · Exercises

1. **Vary the Function:** Try $f(x) = \sin(k\pi x)$ for different $k$. How does
   the regularizer $J$ change with frequency?
2. **Gradient Descent:** Starting from a noisy $f(x)$, perform one or more
   gradient descent steps to “smooth” the function using the functional
   gradient.
3. **Connection to Weight Decay:** Show that if $f(x)$ is just a constant,
   $J = 0$, and the gradient is zero—matching ordinary L2/weight decay.
4. **Physics-Informed Loss:** For $L = (f'(x) - g(x))^2$, compute the functional
   gradient for the modified loss and code it.

Put solutions in `calc-14-functional-deriv/` and tag `v0.1`.

---

**Next:** _Calculus 15 — The Euler-Lagrange Equation and Deep Learning._
