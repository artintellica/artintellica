+++
title = "Calculus 2: Derivatives & Their Role in Gradient‑Based Learning"
date  = "2025‑05‑27"
author = "Artintellica"
+++

> _“A derivative is a microscopic zoom on how a function changes; in machine
> learning we zoom **millions** of times per second.”_

---

## 1 · What a Derivative Really Measures

For a real‑valued function $f$, the **derivative at $x$** is

$$
f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}.
$$

Geometrically it’s the slope of the tangent line. In optimization, that slope
tells us **which way to tweak parameters** to shrink loss.

---

## 2 · Derivative Rules in One Line

- **Sum**: $(f+g)' = f' + g'$
- **Product**: $(fg)' = f'g + fg'$
- **Chain** (composition): $(g\circ f)' = (g'\!\circ f)\,f'$

Every back‑prop step is the chain rule applied repeatedly through a network’s
layers.

---

## 3 · Why ML Cares

- **SGD & Adam**: require $∇L(θ)$.
- **Exploding/Vanishing Gradients**: symptoms of large/small derivatives.
- **Smooth Activations**: chosen so $f'$ exists everywhere.

---

## 4 · Python Demo ①

### 4·1 Finite Difference vs. Autograd on `sin x`

```python
# calc-02-derivatives/compare_finite_vs_autograd.py
import torch, math, matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# sample many x for a smooth curve
x = torch.linspace(-math.pi, math.pi, 400, requires_grad=True)
y = torch.sin(x)

# autograd derivative (should be cos(x))
grads_auto, = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))

# finite difference derivative
h = 1e-4
with torch.no_grad():
    y_plus  = torch.sin(x + h)
    y_minus = torch.sin(x - h)
grads_fd = (y_plus - y_minus) / (2*h)

# ground truth
grads_true = torch.cos(x)

# --- Plot ----------------------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(x, grads_true,     label="cos(x)  (true)", linewidth=2)
plt.plot(x, grads_auto, "--", label="autograd",     linewidth=1)
plt.plot(x, grads_fd,  ":",  label="finite diff",   linewidth=1)
plt.legend()
plt.xlabel("x")
plt.ylabel("derivative")
plt.title("sin'(x) via three methods")
plt.tight_layout()
plt.show()
```

You’ll see all three curves overlap; discrepancies appear only near machine
precision.

---

## 5 · Python Demo ②

### 5·1 Visualizing a “Slope Field” of a Loss Surface

A slope field is a vector plot of $\nabla f$. Even in 1‑D we can render arrows
indicating slope direction.

```python
# calc-02-derivatives/slope_field.py
import numpy as np, matplotlib.pyplot as plt

def loss(x):        # toy non‑convex loss
    return 0.3*np.sin(3*x) + 0.5*(x**2)

xs = np.linspace(-3, 3, 41)
ys = loss(xs)
grads = np.gradient(ys, xs)        # finite diff for visualization

plt.figure(figsize=(6,3))
plt.plot(xs, ys, color="black")
plt.quiver(xs, ys, np.ones_like(grads), grads, angles='xy',
           scale_units='xy', scale=10, width=0.004, alpha=0.7)
plt.title("Loss curve with slope arrows (direction of steepest ascent)")
plt.xlabel("x"); plt.ylabel("L(x)")
plt.tight_layout(); plt.show()
```

Flip the sign of the arrows to picture **gradient descent** steps.

---

## 6 · From 1‑D to Many D

For $f:\mathbb{R}^n\!\to\mathbb{R}$, the gradient is

$$
\nabla f(x)=\Bigl[\frac{\partial f}{\partial x_1},\dots,\frac{\partial f}{\partial x_n}\Bigr].
$$

`torch.autograd.grad` generalizes seamlessly; PyTorch stores $\nabla f$ in each
tensor’s `.grad` field during `backward()`.

---

## 7 · Exercises

1. **Accuracy Experiment**: vary `h` in Demo ① (`1e‑2` … `1e‑8`) and plot max
   error vs `h`. Explain floating‑point round‑off vs. truncation error.
2. **Custom Function**: pick $f(x)=\tanh(x)$. Compute autograd vs. finite diff
   derivatives and verify against analytic $1-\tanh^2x$.
3. **2‑D Gradient Plot**: extend Demo ② to $f(x,y)=x^2 + 0.5y^2$ and use
   `plt.quiver` on a grid to visualize the vector field.

Push your code to `calc-02-derivatives/` and tag `v0.1`.

---

**Up next:** _Calculus 3 – Fundamental Theorem & Numerical Integration_ — see
you soon!
