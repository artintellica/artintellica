import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a 2x2 matrix with random integers between 1 and 5
A = np.random.randint(low=1, high=6, size=(2, 2))

# Compute inverse
try:
    A_inv = np.linalg.inv(A)

    # Verify A * A_inv = I
    identity = A @ A_inv

    # Print results
    print("Matrix A:\n", A)
    print("\nInverse A^-1:\n", A_inv)
    print("\nA @ A^-1 (should be identity):\n", identity)
    print("\nIs identity?", np.allclose(identity, np.eye(2)))
except np.linalg.LinAlgError:
    print("Matrix A is not invertible (singular).")
