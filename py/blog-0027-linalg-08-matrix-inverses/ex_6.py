import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a 3x3 matrix with random integers between -5 and 5
A = np.random.randint(low=-5, high=6, size=(3, 3))

# Compute determinant
det_A = np.linalg.det(A)

# Check if invertible (determinant non-zero)
is_invertible = not np.isclose(det_A, 0, atol=1e-10)

# Compute inverse if invertible
if is_invertible:
    try:
        A_inv = np.linalg.inv(A)
        # Verify A * A_inv = I
        identity = A @ A_inv
        is_identity = np.allclose(identity, np.eye(3))
    except np.linalg.LinAlgError:
        A_inv = None
        is_identity = False
else:
    A_inv = None
    is_identity = False

# Print results
print("Matrix A (3x3):\n", A)
print("\nDeterminant of A:", det_A)
print("Is A invertible?", is_invertible)
if is_invertible:
    print("\nInverse A^-1:\n", A_inv)
    print("\nA @ A^-1 (should be identity):\n", identity)
    print("Is identity?", is_identity)
else:
    print("\nMatrix is not invertible (singular).")
