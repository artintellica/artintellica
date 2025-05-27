# calc-04-optim-1d/adaptive_lr.py
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# ---- loss and analytic gradient ---------------------------------------
def f(x):
    return x**4 - 3 * x**2 + 2


def grad_f(x):
    return 4 * x**3 - 6 * x


f_min = -0.25
x0 = torch.tensor(-2.0)  # starting point for all runs
steps = 200  # generous upper bound


# -----------------------------------------------------------------------
def run_gd(x0, eta):
    x, fs = x0.clone(), []
    for _ in range(steps):
        fs.append(float(f(x)))
        x -= eta * grad_f(x)
    return fs


def run_adagrad(x0, eta, eps=1e-8):
    x, fs = x0.clone(), []
    g2_sum = torch.tensor(0.0)
    for _ in range(steps):
        fs.append(float(f(x)))
        g = grad_f(x)
        g2_sum += g**2
        x -= (eta / (g2_sum.sqrt() + eps)) * g
    return fs


def run_adam(x0, eta, beta1=0.9, beta2=0.999, eps=1e-8):
    x, fs = x0.clone(), []
    m = v = torch.tensor(0.0)
    for t in range(1, steps + 1):
        fs.append(float(f(x)))
        g = grad_f(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= eta * m_hat / (v_hat.sqrt() + eps)
    return fs


# ---- hyper‑params ------------------------------------------------------
eta_gd = 0.05  # tuned so GD converges but not too fast
eta_adagrad = 0.2
eta_adam = 0.1

fs_gd = run_gd(x0, eta_gd)
fs_adagrad = run_adagrad(x0, eta_adagrad)
fs_adam = run_adam(x0, eta_adam)


# ---- helper: first iteration within 1e-3 of optimum -------------------
def first_close(fs, target, tol=1e-3):
    for i, v in enumerate(fs):
        if abs(v - target) < tol:
            return i
    return None


print("iterations to reach |f - f_min| < 1e-3:")
print(f"  Gradient Descent : {first_close(fs_gd,      f_min)}")
print(f"  AdaGrad          : {first_close(fs_adagrad, f_min)}")
print(f"  Adam             : {first_close(fs_adam,    f_min)}")

# ---- plot --------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.semilogy(fs_gd, label=f"GD  (η={eta_gd})")
plt.semilogy(fs_adagrad, label=f"AdaGrad  (η={eta_adagrad})")
plt.semilogy(fs_adam, label=f"Adam  (η={eta_adam})")
plt.axhline(f_min, color="gray", linestyle="--", linewidth=0.7, label="true minimum")
plt.xlabel("iteration")
plt.ylabel("f(x)  (log scale)")
plt.title("Adaptive vs. fixed learning rate")
plt.legend()
plt.tight_layout()
plt.show()
