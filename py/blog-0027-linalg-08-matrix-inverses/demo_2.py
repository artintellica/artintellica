import numpy as np

# Define a 2x2 matrix
A = np.array([[2, 1], [1, 3]])

# Compute inverse
A_inv = np.linalg.inv(A)

# Define b
b = np.array([5, 4])

# Solve using np.linalg.solve
x = np.linalg.solve(A, b)

# Solve using inverse (for comparison)
x_inv = A_inv @ b

# Print results
print("Matrix A:\n", A)
print("Vector b:", b)
print("Solution x (np.linalg.solve):", x)
print("Solution x (A^-1 @ b):", x_inv)
print("Solutions match?", np.allclose(x, x_inv))
