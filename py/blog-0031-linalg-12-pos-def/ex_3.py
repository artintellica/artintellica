import numpy as np


def compute_quadratic_form(matrix, vectors):
    """
    Compute the quadratic form x^T A x for a set of vectors x and matrix A.

    Args:
        matrix (np.ndarray): A square matrix A.
        vectors (np.ndarray): An array of vectors x, shape (n_vectors, dim).

    Returns:
        np.ndarray: Array of quadratic form results for each vector.
    """
    results = []
    for x in vectors:
        # Compute x^T A x
        quad_form = x.T @ matrix @ x
        results.append(quad_form)
    return np.array(results)


# Define the matrix A
A = np.array([[4, 1], [1, 3]])

# Generate 10 random 2D vectors
np.random.seed(42)  # For reproducibility
n_vectors = 10
vectors = np.random.randn(n_vectors, 2)

# Compute quadratic forms
print("Matrix A:")
print(A)
print("\nRandom vectors x:")
print(vectors)
quad_forms = compute_quadratic_form(A, vectors)
print("\nQuadratic form results (x^T A x) for each vector:")
for i, qf in enumerate(quad_forms):
    print(f"Vector {i+1}: {qf:.4f}")

# Check if all results are positive
all_positive = np.all(quad_forms > 0)
print("\nAre all quadratic form results positive?", all_positive)
if all_positive:
    print("This supports that A may be positive definite (though not a complete test).")
else:
    print("Not all results are positive, so A is not positive definite.")
