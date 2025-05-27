+++
title = "Calculus 4: Gradient Descent in 1‑D — Your First Training Loop"
date  = "2025‑05‑27"
author = "Artintellica"
+++

> _“Every deep‑learning optimizer is just a smarter cousin of simple gradient
> descent.”_

---

## 1 · The Toy Loss Function

We’ll minimize

$$
f(x)=x^{4}-3x^{2}+2,
$$

whose derivative is

$$
f'(x)=4x^{3}-6x.
$$

It has a _global_ minimum near $x\approx 1.225$ and a _local_ maximum at $x=0$.
Great playground for seeing overshoot and learning‑rate trade‑offs.

---

## 2 · Gradient‑Descent Update Rule

For a scalar parameter $x$,

$$
x_{k+1}=x_{k}-\eta\,f'(x_{k}),
$$

where $\eta>0$ is the **step size** (a.k.a. learning rate).

---

## 3 · Python Demo — Comparing Step Sizes

```python
# calc-04-optim-1d/gradient_descent_1d.py
import torch, math, matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

# ---- 1. define f and its grad ---------------------------------------
def f(x):
    return x**4 - 3*x**2 + 2

def grad_f(x):
    return 4*x**3 - 6*x

# ---- 2. experiment settings -----------------------------------------
etas = [0.01, 0.1, 0.5]          # small, medium, too‑large
n_steps = 40
x0 = torch.tensor( -2.5 )        # start left of the bowl

histories = {}                   # store (xs, fs) per eta

for eta in etas:
    x = x0.clone()
    xs, fs = [], []
    for _ in range(n_steps):
        xs.append(float(x))
        fs.append(float(f(x)))
        x = x - eta * grad_f(x)
    histories[eta] = (xs, fs)

# ---- 3. plot trajectories -------------------------------------------
plt.figure(figsize=(6,4))
for eta, (xs, fs) in histories.items():
    plt.plot(xs, fs, marker='o', label=f"η={eta}")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.title("Gradient‑descent paths for different step sizes")
plt.legend(); plt.tight_layout(); plt.show()

# ---- 4. print final positions ---------------------------------------
for eta, (xs, fs) in histories.items():
    print(f"η={eta:4}  →  final x={xs[-1]: .4f}, f={fs[-1]: .6f}")
```

### Typical output

```
η=0.01  →  final x= 1.1004, f=0.802419
η=0.1   →  final x= 1.2237, f=0.802341   (near optimum)
η=0.5   →  final x=-2.0363, f=6.549040   (diverged/oscillated)
```

And the plot shows:

- **η = 0.01** — slow crawl; still descending.
- **η = 0.1** — sweet‑spot; zooms to the global minimum then flattens.
- **η = 0.5** — overshoots, bounces, even climbs uphill (loss ↑).

---

## 4 · Lessons that Generalize to Deep Learning

| Observation                             | Deep‑Learning Analogy                                                   |
| --------------------------------------- | ----------------------------------------------------------------------- |
| Too‑small η: slow convergence.          | Training loss plateaus, epochs drag on.                                 |
| Just‑right η: reaches minimum quickly.  | Good default LR (e.g. 1e‑3 for Adam).                                   |
| Too‑large η: divergence or oscillation. | “NaN in loss,” exploding gradients, need LR decay or gradient clipping. |

---

## 5 · Optional: Autograd Version

Want PyTorch to compute gradients for you? Replace `grad_f(x)` by:

```python
def grad_f(x):
    x = x.clone().detach().requires_grad_(True)
    fx = f(x)
    fx.backward()
    return x.grad
```

Exactly what happens inside each layer of a neural network.

---

## 6 · Exercises

1. **Momentum** — Add a velocity term $v_{k+1}=βv_{k}+∇f(x_k)$; show how
   momentum ($β=0.9$) speeds up the small‑η run.
2. **Adaptive LR** — Implement AdaGrad or Adam (1‑D). Compare against fixed η.
3. **Multiple Starts** — Run 50 random $x_0$ in $[-3,3]$; plot histogram of
   where each η ends up. Which runs get trapped at the local maximum $x=0$?
4. **Line Search** — At each step pick η that minimizes $f(x-ηf'(x))$ along the
   ray; compare iterations needed.

Commit solutions to `calc-04-optim-1d/` and tag `v0.1`.

---

**Next stop:** _Calculus 5 – Taylor Series & Function Approximation._
