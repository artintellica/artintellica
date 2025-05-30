import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a 4x2 matrix A
A = np.random.randint(low=-5, high=6, size=(4, 2))

# Create a 4D vector b
b = np.array([4, 2, -1, 3])

# Solve the over-determined system Ax = b using np.linalg.lstsq
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

# Compute Ax to check if b is in the column space of A
Ax = A @ x

# Verify if b is in the column space (Ax ≈ b within numerical tolerance)
is_in_col_space = np.allclose(Ax, b, atol=1e-10)

# Compute residual norm to confirm
residual_norm = np.linalg.norm(b - Ax)

# Print results
print("Matrix A (4x2):\n", A)
print("\nVector b:", b)
print("\nSolution x (np.linalg.lstsq):", x)
print("\nRank of A:", rank)
print("\nSingular values:", s)
print("\nComputed Ax:", Ax)
print("\nIs b in the column space of A?", is_in_col_space)
print("\nResidual norm (||b - Ax||):", residual_norm)
