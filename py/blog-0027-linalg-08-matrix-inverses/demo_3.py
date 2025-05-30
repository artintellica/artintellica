import numpy as np
import matplotlib.pyplot as plt

# Define a 2x2 matrix
A = np.array([[2, 1], [1, 3]])

# Compute inverse
A_inv = np.linalg.inv(A)

# Define b
b = np.array([5, 4])

# Solve using np.linalg.solve
x = np.linalg.solve(A, b)


# Define lines: 2x + y = 5, x + 3y = 4
x_vals = np.linspace(-1, 4, 100)
y1 = 5 - 2 * x_vals  # From 2x + y = 5
y2 = (4 - x_vals) / 3  # From x + 3y = 4

# Plot
plt.figure(figsize=(6, 6))
plt.plot(x_vals, y1, label="2x + y = 5", color="blue")
plt.plot(x_vals, y2, label="x + 3y = 4", color="red")
plt.scatter(x[0], x[1], color="green", s=100, label="Solution")
plt.text(x[0], x[1], f"({x[0]:.1f}, {x[1]:.1f})", color="green", fontsize=12)
plt.grid(True)
plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solution to Linear System")
plt.legend()
plt.show()
