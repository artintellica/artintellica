import numpy as np
import matplotlib.pyplot as plt

# Define a 2x2 matrix A and vector b for the system Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([8, 7])

# Solve the system using np.linalg.solve
try:
    x = np.linalg.solve(A, b)

    # Print results
    print("Matrix A:\n", A)
    print("Vector b:", b)
    print("Solution x:", x)

    # Define lines for plotting: 3x + y = 8, x + 2y = 7
    x_vals = np.linspace(-1, 5, 100)
    y1 = 8 - 3 * x_vals  # From 3x + y = 8
    y2 = (7 - x_vals) / 2  # From x + 2y = 7

    # Plot lines and solution
    plt.figure(figsize=(6, 6))
    plt.plot(x_vals, y1, label="3x + y = 8", color="blue")
    plt.plot(x_vals, y2, label="x + 2y = 7", color="red")
    plt.scatter(x[0], x[1], color="green", s=100, label="Solution")
    plt.text(x[0], x[1], f"({x[0]:.1f}, {x[1]:.1f})", color="green", fontsize=12)
    plt.grid(True)
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Solution to Linear System")
    plt.legend()
    plt.show()
except np.linalg.LinAlgError:
    print("Matrix A is not invertible (singular).")
