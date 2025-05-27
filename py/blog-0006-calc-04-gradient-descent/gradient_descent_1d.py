# calc-04-optim-1d/gradient_descent_1d.py
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# ---- 1. define f and its grad ---------------------------------------
def f(x):
    return x**4 - 3 * x**2 + 2


def grad_f(x):
    return 4 * x**3 - 6 * x


# ---- 2. experiment settings -----------------------------------------
etas = [0.01, 0.1, 0.5]  # small, medium, too‑large
n_steps = 40
x0 = torch.tensor(-2.5)  # start left of the bowl

histories = {}  # store (xs, fs) per eta

for eta in etas:
    x = x0.clone()
    xs, fs = [], []
    for _ in range(n_steps):
        xs.append(float(x))
        fs.append(float(f(x)))
        x = x - eta * grad_f(x)
    histories[eta] = (xs, fs)

# ---- 3. plot trajectories -------------------------------------------
plt.figure(figsize=(6, 4))
for eta, (xs, fs) in histories.items():
    plt.plot(xs, fs, marker="o", label=f"η={eta}")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gradient‑descent paths for different step sizes")
plt.legend()
plt.tight_layout()
plt.show()

# ---- 4. print final positions ---------------------------------------
for eta, (xs, fs) in histories.items():
    print(f"η={eta:4}  →  final x={xs[-1]: .4f}, f={fs[-1]: .6f}")
