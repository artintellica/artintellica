import numpy as np
import matplotlib.pyplot as plt


# The function and its minimum
def fx(x):
    return x**2


def grad_fx(x: float) -> float:
    # Derivative of f(x) = x^2 is 2x
    return 2 * x


x = 5.0  # Start far from zero
eta = 0.1  # Learning rate

trajectory = [x]
for step in range(200):
    x = x - eta * grad_fx(x)
    trajectory.append(x)

# Use trajectory from previous demo
steps = np.array(trajectory)
plt.plot(steps, fx(steps), "o-", label="Optimization Path")
plt.plot(0, 0, "rx", markersize=12, label="Minimum")
plt.xlabel("x value")
plt.ylabel("f(x)")
plt.title("Gradient Descent for $f(x) = x^2$")
plt.legend()
plt.grid(True)
plt.show()
