# calc-04-optim-1d/adaptive_lr_same_eta.py
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


def f(x):
    return x**4 - 3 * x**2 + 2


def grad_f(x):
    return 4 * x**3 - 6 * x


f_min = -0.25
x0 = torch.tensor(-2.0)
steps = 600
eta = 0.01  # identical "base LR" for everyone


def first_close(fs, tol=1e-3):
    for i, v in enumerate(fs):
        if abs(v - f_min) < tol:
            return i
    return None


# ---------- fixed GD ----------
x = x0.clone()
fs_gd = []
for _ in range(steps):
    fs_gd.append(float(f(x)))
    x -= eta * grad_f(x)

# ---------- AdaGrad ----------
x = x0.clone()
g2_sum = torch.tensor(0.0)
fs_ag = []
eps = 1e-8
for _ in range(steps):
    fs_ag.append(float(f(x)))
    g = grad_f(x)
    g2_sum += g**2
    x -= eta * g / (g2_sum.sqrt() + eps)

# ---------- Adam ----------
x = x0.clone()
m = v = torch.tensor(0.0)
fs_ad = []
beta1, beta2 = 0.9, 0.999
for t in range(1, steps + 1):
    fs_ad.append(float(f(x)))
    g = grad_f(x)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    x -= eta * m_hat / (v_hat.sqrt() + eps)

print("iters to |f - f_min| < 1e-3 :")
print("  GD      :", first_close(fs_gd))
print("  AdaGrad :", first_close(fs_ag))
print("  Adam    :", first_close(fs_ad))

plt.figure(figsize=(6, 4))
plt.semilogy(fs_gd, label="GD (η=0.01)")
plt.semilogy(fs_ag, label="AdaGrad (η=0.01)")
plt.semilogy(fs_ad, label="Adam (η=0.01)")
plt.axhline(f_min, color="gray", ls="--", lw=0.7)
plt.xlabel("iteration")
plt.ylabel("f(x)  (log scale)")
plt.title("Adaptive LR vs. fixed LR (same base η)")
plt.legend()
plt.tight_layout()
plt.show()
