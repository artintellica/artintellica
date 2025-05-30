import numpy as np

# Define a 3x3 matrix A
A = np.array([[2, 1, -1], [1, 3, 2], [0, 1, 4]])

# Define a 3D vector b
b = np.array([8, 7, 3])

# Solve using np.linalg.solve
x_solve = np.linalg.solve(A, b)

# Solve using inverse method
try:
    A_inv = np.linalg.inv(A)
    x_inv = A_inv @ b

    # Verify solutions match
    solutions_match = np.allclose(x_solve, x_inv)

    # Print results
    print("Matrix A:\n", A)
    print("\nVector b:", b)
    print("\nSolution x (np.linalg.solve):", x_solve)
    print("\nSolution x (A^-1 @ b):", x_inv)
    print("\nSolutions match?", solutions_match)
except np.linalg.LinAlgError:
    print("Matrix A is not invertible (singular).")
