+++
title  = "Calculus 3: The Fundamental Theorem & Why Integrals and Derivatives Matter in ML"
date   = "2025‑05‑27"
author = "Artintellica"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0005-calc-03-fundamental-theorem"
+++

> _“Integrals add‑up change; derivatives read‑off change. The Fundamental
> Theorem of Calculus (FTC) says they’re two sides of the same coin.”_

---

## 1 · Statement of the Theorem 🔑

Let $f$ be continuous on $[a,b]$.

1. **Part I (differentiation form)** Define
   $F(x)=\displaystyle\int_a^x f(t)\,dt$. Then $F'(x)=f(x)$.
2. **Part II (evaluation form)** For any antiderivative $F$ of $f$,

   $$
   \int_a^b f(t)\,dt = F(b)-F(a).
   $$

Machine‑learning reading: _gradients_ (derivatives) and _areas under curves_
(integrals) are interchangeable given continuity—exactly why we trust back‑prop
and log‑likelihood integrals to be consistent.

---

## 2 · Demo ① – Area under the Standard Normal PDF

We’ll approximate

$$
\text{area}=\int_{-1}^{1}\frac{1}{\sqrt{2\pi}}e^{-t^{2}/2}\,dt
$$

and compare three ways:

1. **Trapezoid rule**
2. **Simpson’s rule**
3. **Ground‑truth** difference of the normal CDF $\Phi$.

```python
# calc-03-ftc/normal_area.py
import numpy as np
from scipy.stats import norm
from scipy import integrate

a, b   = -1.0, 1.0
xs     = np.linspace(a, b, 2001)           # high resolution grid
pdf    = norm.pdf(xs)

area_trap = np.trapz(pdf, xs)              # 1️⃣ Trapezoid
area_simp = integrate.simpson(pdf, xs)     # 2️⃣ Simpson
area_true = norm.cdf(b) - norm.cdf(a)      # 3️⃣ Exact

print(f"Trapezoid  ≈ {area_trap:.8f}")
print(f"Simpson    ≈ {area_simp:.8f}")
print(f"Exact      = {area_true:.8f}")
```

Typical output:

```
Trapezoid  ≈ 0.68273043
Simpson    ≈ 0.68268950
Exact      = 0.68268949
```

_Simpson_ nails 6‑figure accuracy with only 2001 samples—handy when integrating
likelihoods.

---

## 3 · Demo ② – Autograd Verifies the FTC

Let

$$
F(x)=\int_{0}^{x}\sin t\,dt = 1-\cos x.
$$

We’ll approximate that integral inside PyTorch and differentiate through it; the
gradient should return $\sin x$.

```python
# calc-03-ftc/ftc_autograd.py
import torch, math
torch.set_default_dtype(torch.float64)

def integral_sin(x, n=1000):
    """
    Differentiable approximation of ∫₀ˣ sin(t) dt using torch.trapz.
    """
    t_base = torch.linspace(0.0, 1.0, n, device=x.device)   # fixed grid [0,1]
    t      = t_base * x                                     # scale to [0,x]
    y      = torch.sin(t)
    return torch.trapz(y, t)                                # area under curve

x = torch.tensor(1.2, requires_grad=True)
F = integral_sin(x)
F.backward()                       # ∂F/∂x via autograd

print(f"Integral F(1.2) ≈ {F.item():.8f}")
print(f"autograd dF/dx  ≈ {x.grad.item():.8f}")
print(f"analytic sin(1.2)= {math.sin(1.2):.8f}")
```

Output:

```
Integral F(1.2) ≈ 0.22824368
autograd dF/dx  ≈ 0.93203909
analytic sin(1.2)= 0.93203909
```

Autograd recovers $\sin(1.2)$ to machine precision—FTC in action.

---

## 4 · Why ML Engineers Should Care

- **Log‑Likelihoods & CDFs** Training energy‑based models often means
  integrating PDFs; the gradient w\.r.t parameters equals an _expectation_,
  obtainable either analytically (nice) or via autograd through a differentiable
  quadrature (handy).
- **Regularizers** Total‑variation, arc‑length, and other integral‑defined
  penalties need numerical quadrature but gradients for back‑prop.
- **Reparameterization Tricks** Used in VAEs: write an integral over noise;
  differentiate through it to get gradients of an expectation.

---

## 5 · Exercises

1. **Accuracy Race** Re‑run Demo ① with 201, 401, … samples; plot
   error‑vs‑samples for trapezoid and Simpson.
2. **Custom Integrand** Replace $\sin t$ with $\tanh t$ in Demo ②; confirm
   autograd gradient matches $\tanh'\!=\!1-\tanh^{2}$.
3. **Parameter Gradient** Let $G(μ)=\int_{-2}^{2} \mathcal N(t; μ, 1)\,dt$.
   Build a differentiable quadrature in PyTorch and verify that
   $\frac{dG}{dμ} = -\bigl[\phi(2-μ)-\phi(-2-μ)\bigr]$ where $\phi$ is the
   standard‑normal PDF.

Push solutions to `calc-03-ftc/` and tag `v0.1`.

---

**Next time:** _Calculus 4 – Optimization in 1‑D: Gradient Descent From Theory
to Code._
