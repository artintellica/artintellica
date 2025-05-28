"""
exercise_4_simple_BFGS_visual.py
-------------------------------------------------
A hand‑rolled BFGS optimizer on a 2‑D quadratic

    f(x) = ½ xᵀ A x + bᵀ x

• starts from x0
• keeps an *inverse*‑Hessian estimate  H⁻¹_k  (2×2)
• updates via the standard BFGS formula
• after each step plots the implied Hessian approximation
  B_k = (H⁻¹_k)⁻¹  against the true Hessian A.
"""

import numpy as np
import matplotlib.pyplot as plt

# -------- quadratic definition --------------------------
A = np.array([[5.0, 1.0], [1.0, 3.0]])  # positive–definite Hessian
b = np.array([-4.0, 2.0])


def f(x):
    return 0.5 * x @ A @ x + b @ x


def grad_f(x):
    return A @ x + b


x_star = -np.linalg.solve(A, b)

# -------- BFGS parameters -------------------------------
x = np.array([2.5, -2.0])  # start
Hinv = np.eye(2)  # initial inverse‑H
max_iter = 12

B_hist = [np.linalg.inv(Hinv)]  # store approximated Hessian
loss_hist = [f(x)]

for k in range(max_iter):
    g = grad_f(x)

    # Search direction  p = -Hinv g
    p = -Hinv @ g

    # For a quadratic, the exact minimiser along p is obtainable:
    alpha = -(g @ p) / (p @ A @ p)
    x_new = x + alpha * p

    # Compute curvature vectors
    s = x_new - x
    y = grad_f(x_new) - g
    ys = y @ s

    if ys > 1e-12:  # skip update if division small
        rho = 1.0 / ys
        MyI = np.eye(2)
        # BFGS update of *inverse* Hessian
        Hinv = (MyI - rho * np.outer(s, y)) @ Hinv @ (
            MyI - rho * np.outer(y, s)
        ) + rho * np.outer(s, s)

    x = x_new
    B_hist.append(np.linalg.inv(Hinv))
    loss_hist.append(f(x))

# -------- plot Hessian approximation elements ------------
B_hist = np.array(B_hist)  # shape (iter+1, 2, 2)
iters = range(len(B_hist))

plt.figure(figsize=(8, 4))
plt.plot(iters, B_hist[:, 0, 0], "o-", label=r"$B_{11}$")
plt.plot(iters, B_hist[:, 0, 1], "s-", label=r"$B_{12}=B_{21}$")
plt.plot(iters, B_hist[:, 1, 1], "^-", label=r"$B_{22}$")
plt.hlines(
    [A[0, 0], A[0, 1], A[1, 1]],
    0,
    max_iter,
    colors=["C0", "C1", "C2"],
    linestyles="dashed",
    label="true $A$",
)
plt.xlabel("iteration k")
plt.ylabel("Hessian element value")
plt.title("BFGS approximation of Hessian elements")
plt.legend()
plt.tight_layout()
plt.show()

# -------- print summary ----------------------------------
print(f"Optimum x*        : {x_star}")
print(f"Last iterate x_k  : {x}")
print(f"Final loss        : {loss_hist[-1]:.4e}")
print("Final Hessian approximation B_k:\n", B_hist[-1])
print("True Hessian A:\n", A)
