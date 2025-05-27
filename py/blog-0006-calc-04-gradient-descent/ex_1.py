# calc-04-optim-1d/momentum_vs_plain.py
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# ---- loss and gradient ------------------------------------------------
def f(x):
    return x**4 - 3 * x**2 + 2


def grad_f(x):
    return 4 * x**3 - 6 * x


# ---- hyper‑params -----------------------------------------------------
eta = 0.01  # small learning rate
beta = 0.9  # momentum coefficient
n_steps = 200  # enough to see convergence
x0 = torch.tensor(-2.5)

# ---- plain GD ---------------------------------------------------------
x_plain = x0.clone()
xs_plain, fs_plain = [], []
for _ in range(n_steps):
    xs_plain.append(float(x_plain))
    fs_plain.append(float(f(x_plain)))
    x_plain -= eta * grad_f(x_plain)

# ---- GD + momentum ----------------------------------------------------
x_mom = x0.clone()
v = torch.tensor(0.0)  # initial velocity
xs_mom, fs_mom = [], []
for _ in range(n_steps):
    xs_mom.append(float(x_mom))
    fs_mom.append(float(f(x_mom)))
    v = beta * v + grad_f(x_mom)
    x_mom -= eta * v

# ---- plot -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(xs_plain, fs_plain, "o-", label="plain GD (η=0.01)")
plt.plot(xs_mom, fs_mom, "s-", label="momentum GD (η=0.01, β=0.9)")
plt.axhline(0.80234, color="gray", linewidth=0.7, linestyle="--", label="true minimum")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Momentum accelerates convergence")
plt.legend()
plt.tight_layout()
plt.show()


# ---- quick numeric summary -------------------------------------------
def where_close(fs, tol=1e-3):
    for i, v in enumerate(fs):
        if abs(v - 0.802341) < tol:
            return i
    return None


i_plain = where_close(fs_plain)
i_mom = where_close(fs_mom)
print("Steps to reach f(x)≈0.803:")
print(f"  plain      : {i_plain} steps")
print(f"  momentum   : {i_mom} steps")
