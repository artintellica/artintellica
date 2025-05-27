+++
title = "Calculus 1: Limits and Continuity"
author = "Artintellica"
date = "2025-05-27"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0003-calc-01-limits"
+++

> _“Calculus doesn’t begin with derivatives or integrals—it begins with asking
> what happens **as we get arbitrarily close** to something.”_

---

## 1 · Why ML Engineers Should Care

Training a neural network is nothing more than **finding limits**—the loss
approaches its minimum as parameter updates approach zero. If the loss function
were _not_ continuous, gradient‑based methods would fail spectacularly. So
before we race to back‑propagation, we need rock‑solid intuition for limits and
continuity.

---

## 2 · Limits in Plain English  

For a function $f(x)$ we say

$$
\lim_{x \to c} f(x) = L
$$

means: for every tolerance $\varepsilon>0$ you pick around $L$, I can pick a
distance $\delta>0$ around $c$ such that whenever $0<|x-c|<\delta$, we have
$|f(x)-L|<\varepsilon$.

This **ε‑δ definition** is the formal backbone of all later calculus. Continuous
functions are those whose limits agree with their function values:

$$
\lim_{x \to c} f(x) = f(c).
$$

---

## 3 · Classic Example: $\displaystyle\frac{\sin x}{x}$

At $x=0$ the formula “$\sin x / x$” looks undefined, yet the limit exists and
equals 1. Graphing a _zoom_ makes the idea visceral.

### 3·1 Python Demo 🖥️

```python
# demo_limits.py
import numpy as np
import matplotlib.pyplot as plt

# 1. sample points around 0
x = np.linspace(-1e-1, 1e-1, 2001)
y = np.where(x != 0, np.sin(x)/x, 1.0)  # define f(0)=1 by continuity

# 2. plot
plt.figure(figsize=(5,4))
plt.plot(x, y, label=r'$\,\sin x / x\,$')
plt.scatter([0], [1], color='black', zorder=3)  # the limiting point
plt.axhline(1, linestyle='--', linewidth=0.7)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Zoom on  sin(x)/x  near 0")
plt.tight_layout()
plt.show()
```

Run it:

```bash
python demo_limits.py
```

You’ll see the curve flatten toward $(0,1)$. Try reducing the range to `1e-3`,
`1e-4`, … to watch the limit converge numerically.

### 3·2 An ε‑δ “Verifier”

```python
def limit_tester(f, c, L, eps=1e-4):
    # naive sweep for a suitable δ
    for delta_exp in range(-1, -8, -1):          # 0.1, 0.01, …, 1e‑7
        δ = 10.0 ** delta_exp
        xs = np.linspace(c - δ, c + δ, 1001)
        xs = xs[xs != c]
        if np.all(np.abs(f(xs) - L) < eps):
            return δ
    return None

f = lambda x: np.sin(x)/x
δ_found = limit_tester(f, 0.0, 1.0, eps=1e-5)
print(f"Found δ = {δ_found} for ε = 1e-5")
```

Use it to _prove experimentally_ that the limit truly is 1.

---

## 4 · Continuity Checklist

A function is continuous at $c$ if:

1. $f(c)$ is defined.
2. $\displaystyle\lim_{x\to c}f(x)$ exists.
3. Those two numbers are equal.

For ML, common activations (ReLU, GELU, sigmoid, …) all satisfy these three,
which is why their gradients behave nicely.

---

## 5 · From Limits to Gradients (Sneak Peek)

If $f$ is continuous and differentiable, then

$$
\frac{d}{dx}f(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}.
$$

Everything we do in back‑prop is disguised limit‑taking! In the next post we’ll
zoom from “approaching $x$” to “slope at $x$.”

---

## 6 · Exercises (Try & Commit)

1. Replace $\sin x / x$ with $|x|/x$. Does the limit at 0 exist? Why not?
2. Write a function that fails condition (2) but passes (1) and (3).
3. Using PyTorch, compute `torch.autograd.grad` of `torch.sin` at several points
   and verify numerically with finite differences (hint:
   `torch.autograd.functional.jacobian`).

Commit your notebooks to `calc-01-limits/` and tag it `v0.1`.

---

**Up next:** _Calculus 2 – Derivatives & Gradient Descent From Scratch_ — see
you there!
