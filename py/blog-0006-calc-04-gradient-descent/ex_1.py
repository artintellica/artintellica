import torch
import math
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# --- f and its gradient ------------------------------------------------
def f(x):
    return x**4 - 3 * x**2 + 2


def grad_f(x):
    return 4 * x**3 - 6 * x


x_star = math.sqrt(1.5)  # ≈ 1.224744871
f_min = f(x_star)  # = -0.25

# --- hyper‑params ------------------------------------------------------
eta = 0.01
beta = 0.9
steps = 120
x0 = torch.tensor(-1.0)  # starting point where momentum helps


# --- helpers -----------------------------------------------------------
def run_gd(x0, eta, steps):
    x = x0.clone()
    xs, fs = [], []
    for _ in range(steps):
        xs.append(float(x))
        fs.append(float(f(x)))
        x -= eta * grad_f(x)
    return xs, fs


def run_momentum(x0, eta, beta, steps):
    x = x0.clone()
    v = torch.tensor(0.0)
    xs, fs = [], []
    for _ in range(steps):
        xs.append(float(x))
        fs.append(float(f(x)))
        v = beta * v + grad_f(x)
        x -= eta * v
    return xs, fs


# --- run both optimizers ----------------------------------------------
xs_gd, fs_gd = run_gd(x0, eta, steps)
xs_mo, fs_mo = run_momentum(x0, eta, beta, steps)


# --- measure convergence ----------------------------------------------
def first_close(fs, target, tol=1e-3):
    for i, v in enumerate(fs):
        if abs(v - target) < tol:
            return i
    return None


i_gd = first_close(fs_gd, f_min)
i_mo = first_close(fs_mo, f_min)

print(f"f_min = {f_min:.2f}")
print("steps to |f - f_min| < 1e-3")
print(f"  plain GD    : {i_gd} steps")
print(f"  momentum GD : {i_mo} steps")

# --- plot --------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(fs_gd, label="plain GD (η=0.01)")
plt.plot(fs_mo, label="momentum (η=0.01, β=0.9)")
plt.axhline(f_min, color="gray", linestyle="--", linewidth=0.7, label="true minimum")
plt.ylim(f_min - 0.1, 2.5)
plt.xlabel("iteration")
plt.ylabel("f(x)")
plt.title("Momentum accelerates convergence")
plt.legend()
plt.tight_layout()
plt.show()
