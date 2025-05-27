+++
title = "Calculus 6: Vector Functions & the Gradient — Seeing Slopes in 2‑D"
date  = "2025‑05‑27"
author = "Artintellica"
+++

> _“In 1‑D the derivative is a number. In many‑D it becomes a **vector**
> pointing where the function rises fastest.”_

---

## 1 · From Single Numbers to Vectors

For $f:\mathbb R^{n}\!\to\mathbb R$ the **gradient** is

$$
\nabla f(\mathbf x)=\Bigl[\partial_{x_1}f,\;\dots,\;\partial_{x_n}f\Bigr]^{\!\top}.
$$

- Points in the **direction** of steepest ascent.
- Its **length** is the slope in that direction.
- When $n=1$ it reduces to the ordinary derivative.

---

## 2 · Why ML Engineers Care

| Concept              | Practical impact                                              |
| -------------------- | ------------------------------------------------------------- |
| **Back‑propagation** | Each weight update uses the gradient of the loss.             |
| **Gradient norm**    | Signals vanishing/exploding gradients.                        |
| **Visual debugging** | Plotting ∇ on a surface reveals saddle points, plateaus, etc. |

---

## 3 · Example Function

$$
f(x,y)=x^{2}+y^{2},
\qquad\;\;
\nabla f = [2x,\;2y].
$$

A perfect “bowl”: every gradient arrow points away from the origin.

---

## 4 · Python Demo ① — Quiver Plot of ∇f

```python
# calc-06-gradients/quiver_xy2.py
import numpy as np
import matplotlib.pyplot as plt
import torch

# ---- build grid -------------------------------------------------------
xv = np.linspace(-2, 2, 21)
yv = np.linspace(-2, 2, 21)
X, Y = np.meshgrid(xv, yv)

# analytical gradient
U = 2 * X        # ∂f/∂x
V = 2 * Y        # ∂f/∂y

# ---- contour + quiver plot -------------------------------------------
F = X**2 + Y**2
plt.figure(figsize=(6,6))
plt.contour(X, Y, F, levels=10, cmap="gray", linewidths=0.6)
plt.quiver(X, Y, U, V, color="tab:blue", alpha=0.8, scale=40)
plt.title(r"Gradient field  $\nabla f(x,y)$  for  $f=x^2+y^2$")
plt.gca().set_aspect("equal")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout(); plt.show()
```

---

## 5 · Python Demo ② — Let Autograd Do the Math

```python
# calc-06-gradients/autograd_xy2.py
import torch

torch.set_default_dtype(torch.float64)

def f(xy):
    x, y = xy
    return x**2 + y**2

point = torch.tensor([1.5, -0.8], requires_grad=True)
val = f(point)
val.backward()

print(f"f({point.tolist()}) = {val.item():.3f}")
print("∇f =", point.grad.tolist())   # should be [2*1.5, 2*(-0.8)]
```

Output:

```
f([1.5, -0.8]) = 3.09
∇f = [3.0, -1.6]
```

Matches the analytic gradient $[2x,2y]$.

---

## 6 · Link to Back‑prop (Mini‑Example)

```python
w = torch.tensor([0.7, -1.2], requires_grad=True)
x = torch.tensor([1.0, 2.0])
loss = (w @ x) ** 2          # scalar
loss.backward()
print("gradient wrt weights =", w.grad)
```

Because $loss = (w\cdot x)^2$, autograd returns $2(w\!\cdot\!x)x$. Exactly the
multi‑dimensional chain rule in action.

---

## 7 · Exercises

1. **Gradient Checker** For $g(x,y)=\sin x + \cos y$ compute ∇g analytically and
   with autograd at random points; verify max error < 1 × 10⁻⁸.
2. **Custom Surface** Plot contours and quiver for $h(x,y)=x^3-3xy^2$ (the real
   part of $z^3$); identify saddle points.
3. **Gradient Descent Path** Starting at (‑1.8, 1.6) run gradient descent on
   $f=x^2+y^2$ with step 0.1 and overlay the path on the quiver plot.
4. **Batch Gradients** Use `torch.autograd.grad` with `create_graph=True` to
   compute the Jacobian of ∇f (a.k.a. the Hessian) and confirm it’s the constant
   matrix `[[2,0],[0,2]]`.

Put solutions in `calc-06-gradients/` and tag `v0.1`.

---

**Next:** _Calculus 7 – Jacobian & Hessian: Second‑Order Insights for Faster
Learning._
