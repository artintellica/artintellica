import torch
import numpy as np
import matplotlib.pyplot as plt


# ### **Exercise 1:** Implement Scalar Gradient Descent for $f(x) = (x-3)^2$


# - Write a function that starts from $x_0 = -7$ and uses gradient descent for 20
#   steps.
# - Print $x$ after each step.
def grad_fx(x: float) -> float:
    # Derivative of f(x) = (x - 3)^2 is 2 * (x - 3)
    return 2 * (x - 3)


x = -7.0  # Start at -7
eta = 0.1  # Learning rate
trajectory = [x]
for step in range(20):
    x = x - eta * grad_fx(x)
    trajectory.append(x)
    print(f"Step {step + 1}: x = {x}")
# Print final value
print("Final x:", x)

# ### **Exercise 2:** Visualize the Optimization Path on a 2D Plot

# - Plot $f(x)$ and overlay the path of $x$ values as you optimize.
x_arr = np.array(trajectory)
plt.plot(x_arr, (x_arr - 3) ** 2, "o-")
plt.plot(3, 0, "rx", label="Minimum")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Optimization Path: Scalar")
plt.legend()
plt.grid(True)
plt.show()
