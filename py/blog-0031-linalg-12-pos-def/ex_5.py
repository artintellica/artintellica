import numpy as np


def check_positive_definite(matrix):
    """
    Check if a matrix is positive definite by examining its eigenvalues.

    Args:
        matrix (np.ndarray): A square matrix to be tested.

    Returns:
        bool: True if the matrix is positive definite, False otherwise.
    """
    if not np.allclose(matrix, matrix.T):
        print("Matrix is not symmetric.")
        return False

    eigenvalues = np.linalg.eigvals(matrix)
    if np.all(eigenvalues > 0):
        print("All eigenvalues are positive:", eigenvalues)
        return True
    else:
        print("Not all eigenvalues are positive:", eigenvalues)
        return False


# Set random seed for reproducibility
np.random.seed(42)

# Generate a small dataset: 5 points in 2D
data = np.random.randn(5, 2)
print("Generated dataset (5 points in 2D):")
print(data)

# Compute the covariance matrix
cov_matrix = np.cov(data.T, bias=True)
print("\nCovariance matrix:")
print(cov_matrix)

# Check if covariance matrix is positive definite
print("\nChecking if covariance matrix is positive definite:")
is_pd = check_positive_definite(cov_matrix)
if is_pd:
    print("Covariance matrix is positive definite.")
else:
    print(
        "Covariance matrix is not positive definite. Adding small positive value to diagonal."
    )
    # Add a small positive value to the diagonal to ensure positive definiteness
    cov_matrix_modified = cov_matrix + np.eye(cov_matrix.shape[0]) * 0.01
    print("\nModified covariance matrix (with added diagonal value):")
    print(cov_matrix_modified)
    print("\nChecking modified matrix for positive definiteness:")
    is_pd_modified = check_positive_definite(cov_matrix_modified)
    if is_pd_modified:
        print("Modified covariance matrix is now positive definite.")
    else:
        print("Modified covariance matrix is still not positive definite.")
