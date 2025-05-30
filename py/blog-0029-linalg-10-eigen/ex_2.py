import numpy as np

# Set random seed for reproducibility (same as Exercise 1)
np.random.seed(42)

# Create the same 3x3 matrix from Exercise 1
A = np.random.randint(low=-5, high=6, size=(3, 3))

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Select the first eigenvalue-eigenvector pair for verification
index = 0
lambda_ = eigenvalues[index]
v = eigenvectors[:, index]

# Verify A v = λ v
Av = A @ v
lambda_v = lambda_ * v

# Check if they match
match = np.allclose(Av, lambda_v, atol=1e-10)

# Print results
print("Matrix A (3x3):\n", A)
print("\nEigenvalues:", eigenvalues)
print("\nEigenvectors (columns):\n", eigenvectors)
print(f"\nSelected eigenvalue (λ_{index+1}):", lambda_)
print(f"Selected eigenvector (v_{index+1}):", v)
print("\nA @ v:", Av)
print("λ * v:", lambda_v)
print("\nDoes A v = λ v hold?", match)
