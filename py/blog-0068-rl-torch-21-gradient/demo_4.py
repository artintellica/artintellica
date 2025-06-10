import torch
import numpy as np
import matplotlib.pyplot as plt


def grad_fx(x: float) -> float:
    # Derivative of f(x) = x^2 is 2x
    return 2 * x


def fx(x):
    return x**2


x = 5.0  # Start far from zero
eta = 0.1  # Learning rate

trajectory = [x]
for step in range(20):
    x = x - eta * grad_fx(x)
    trajectory.append(x)
print("Final x:", x)

init_x = 5.0
learning_rates = [0.05, 0.2, 0.8, 1.01]
colors = ["b", "g", "r", "orange"]

plt.figure()
for lr, col in zip(learning_rates, colors):
    x = init_x
    hist = [x]
    for _ in range(12):
        x = x - lr * grad_fx(x)
        hist.append(x)
    plt.plot(hist, fx(np.array(hist)), "o-", color=col, label=f"LR={lr}")
plt.plot(0, 0, "kx", markersize=12)
plt.title("Gradient Descent Paths for Different Learning Rates")
plt.xlabel("x value")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
