+++
title = "Calculus 5: Taylor & Maclaurin Series—Polynomials That Think They’re Exponential"
date  = "2025‑05‑27"
author = "Artintellica"
+++

> _“If derivatives tell you the slope **here**, Taylor series tell you the whole
> function **near here**.”_

---

## 1 · From Derivatives to Polynomials

For a function $f$ with enough derivatives at $a$,

$$
f(x) \;=\; \sum_{k=0}^{\infty}\frac{f^{(k)}(a)}{k!}\,(x-a)^{k}
\qquad\text{(Taylor series at \(a\)).}
$$

Setting $a=0$ gives the **Maclaurin** series. Example:

$$
e^{x}=1+x+\frac{x^{2}}{2!}+\frac{x^{3}}{3!}+\cdots
$$

A **degree‑$n$ truncation**

$$
T_n(x)=\sum_{k=0}^{n}\frac{f^{(k)}(a)}{k!}(x-a)^{k}
$$

approximates $f$ near $a$ with remainder $R_{n+1}=O((x-a)^{n+1})$.

---

## 2 · Why ML Engineers Care

| Use‑case                        | Where Taylor pops up                                                               |
| ------------------------------- | ---------------------------------------------------------------------------------- |
| **Activation function kernels** | Approximating `exp`, `tanh`, `GELU` on edge devices.                               |
| **Transformers**                | Rotary / ALiBi positional encodings derive from low‑order series of $e^{i\theta}$. |
| **Gradient checkpoints**        | Cheap polynomial surrogates during backward pass to save memory.                   |

---

## 3 · Python Demo ① — Static Error Plot for $e^{x}$

```python
# calc-05-taylor/taylor_error_exp.py
import numpy as np, math, matplotlib.pyplot as plt
from math import factorial

def maclaurin_exp(x, n):
    """Return T_n(x) for e^x."""
    return sum((x**k)/factorial(k) for k in range(n+1))

xs = np.linspace(-3, 3, 400)
true = np.exp(xs)

plt.figure(figsize=(6,4))
for n in [1, 2, 4, 6, 8]:
    approx = maclaurin_exp(xs, n)
    err = np.abs(approx - true)
    plt.plot(xs, err, label=f"n={n}")

plt.yscale("log")
plt.xlabel("x"); plt.ylabel("|e^x - T_n(x)| (log)")
plt.title("Absolute error of Maclaurin truncations of e^x")
plt.legend(); plt.tight_layout(); plt.show()
```

**What to look for:** each +2 degree roughly squares the accuracy radius
around 0.

---

## 4 · Python Demo ② — Animation of Truncation Growth

```python
# calc-05-taylor/animate_exp.py
import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import factorial

# domain & true curve
xs = np.linspace(-3, 3, 400)
true = np.exp(xs)

fig, ax = plt.subplots(figsize=(6,4))
line, = ax.plot([], [], lw=2)
ax.plot(xs, true, 'k--', label="e^x")
ax.set_xlim(-3, 3); ax.set_ylim(-1, 20)
ax.set_title("Building e^x via Maclaurin truncations")
ax.legend()

def taylor_poly(x, n):
    return sum((x**k)/factorial(k) for k in range(n+1))

def init():
    line.set_data([], [])
    return line,

def update(frame):
    y = taylor_poly(xs, frame)
    line.set_data(xs, y)
    line.set_label(f"T_{frame}(x)")
    ax.legend()
    return line,

anim = FuncAnimation(fig, update, frames=range(0, 11), init_func=init,
                     interval=800, blit=True)
anim.save("exp_taylor.gif", writer="pillow")
```

The saved GIF shows the polynomial growing degree‑by‑degree until it hugs
$e^{x}$ on a wider interval.

---

## 5 · Connecting to Neural Nets

- **Softmax stability** — approximating `exp` for logits > 7 avoids overflow.
- **Fourier‑feature networks** — $e^{i\omega x}$ expansion links to sinusoidal
  positional encodings.
- **Hardware kernels** — TPUs/GPU tensor‑cores often use LUT + low‑degree series
  internally for transcendental ops.

---

## 6 · Exercises

1. **Better Radius with Centering** Derive the Taylor series of $e^{x}$ at
   $a=1$. Plot error for $T_4^{(a=0)}$ vs $T_4^{(a=1)}$ on $[-3,3]$. Which
   interval is better approximated?
2. **Activation Approx** Write a function that returns a degree‑5 Maclaurin for
   $\tanh x$. Plot both function and approx on $[-2,2]$; mark max absolute
   error.
3. **PosEnc Link** Show that truncating $e^{i\theta}$ to first two non‑zero
   terms yields $1+i\theta$. Relate to the original transformer’s sine/cosine
   encoding by separating real/imag parts.
4. **Adaptive Degree** Implement a routine that chooses the smallest $n$ such
   that $|e^{x}-T_n(x)|<10^{-4}$ for a given $x$. Test on $x=0.5,1,2,5$. How
   does $n$ grow?

Push solutions to `calc-05-taylor/` and tag `v0.1`.

---

**Next episode:** _Calculus 6 – Gradient, Jacobian, Hessian: Stepping into
Multiple Dimensions._
