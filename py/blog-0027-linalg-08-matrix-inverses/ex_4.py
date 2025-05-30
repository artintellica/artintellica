import numpy as np

# Create a 3x3 matrix with linearly dependent columns
# Example: third column is 2 * first column
A = np.array([
    [1, 2, 2],   # Third column = 2 * [1, 0, 1]
    [0, 3, 0],
    [1, 4, 2]
])

# Attempt to compute inverse
try:
    A_inv = np.linalg.inv(A)
    print("Matrix A:\n", A)
    print("\nInverse A^-1:\n", A_inv)
except np.linalg.LinAlgError as e:
    print("Matrix A:\n", A)
    print("\nError: Matrix is not invertible (singular).")
    print("Exception message:", str(e))

# Verify linear dependence by computing determinant
det_A = np.linalg.det(A)
print("\nDeterminant of A:", det_A)
print("Is determinant zero?", np.isclose(det_A, 0, atol=1e-10))
