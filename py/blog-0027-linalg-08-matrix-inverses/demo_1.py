import numpy as np

# Define a 2x2 matrix
A = np.array([[2, 1], [1, 3]])

# Compute inverse
A_inv = np.linalg.inv(A)

# Verify A * A_inv = I
identity = A @ A_inv

# Print results
print("Matrix A:\n", A)
print("\nInverse A^-1:\n", A_inv)
print("\nA @ A^-1 (should be identity):\n", identity)
print("Is identity?", np.allclose(identity, np.eye(2)))
