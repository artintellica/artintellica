import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a 2x3 matrix A
A = np.random.randint(low=-5, high=6, size=(2, 3))

# Create a 2D vector b
b = np.array([4, 2])

# Solve the under-determined system Ax = b using np.linalg.lstsq
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

# Compute nullspace basis using SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)
tol = 1e-10
nullspace_basis = Vt.T[:, S < tol]

# Check nullity (dimension of nullspace)
nullity = A.shape[1] - rank

# Check if multiple solutions exist (non-trivial nullspace)
multiple_solutions = nullity > 0

# Generate another solution if nullspace is non-trivial
if multiple_solutions:
    # Add a random combination of nullspace basis vectors to x
    nullspace_coeff = np.random.uniform(-1, 1, size=nullity)
    x_alt = x + (nullspace_basis @ nullspace_coeff)
    # Verify alternative solution satisfies Ax = b
    Ax_alt = A @ x_alt
    is_valid_solution = np.allclose(Ax_alt, b)
else:
    x_alt = None
    is_valid_solution = False

# Print results
print("Matrix A (2x3):\n", A)
print("\nVector b:", b)
print("\nSolution x (np.linalg.lstsq):", x)
print("\nRank of A:", rank)
print("\nSingular values:", s)
print("\nNullspace basis:\n", nullspace_basis if nullspace_basis.size > 0 else "Empty")
print("\nNullity (dimension of nullspace):", nullity)
print("\nMultiple solutions exist?", multiple_solutions)
if multiple_solutions:
    print("\nAlternative solution x_alt:", x_alt)
    print("Does x_alt satisfy Ax = b?", is_valid_solution)
else:
    print("\nNo alternative solution (unique solution or no solution).")
