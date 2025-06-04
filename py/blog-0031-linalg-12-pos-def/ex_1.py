import numpy as np


def is_positive_definite(matrix):
    """
    Check if a matrix is positive definite.
    A matrix is positive definite if it is symmetric and all its eigenvalues are positive.

    Args:
        matrix (np.ndarray): A square matrix to be tested.

    Returns:
        bool: True if the matrix is positive definite, False otherwise.
    """
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        print("Matrix is not square.")
        return False

    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        print("Matrix is not symmetric.")
        return False

    # Compute eigenvalues and check if all are positive
    eigenvalues = np.linalg.eigvals(matrix)
    if np.all(eigenvalues > 0):
        print("All eigenvalues are positive:", eigenvalues)
        return True
    else:
        print("Not all eigenvalues are positive:", eigenvalues)
        return False


# Test the function on matrix B
B = np.array([[1, 2], [2, 1]])
print("Testing matrix B:")
print(B)
result = is_positive_definite(B)
print("Is matrix B positive definite?", result)

# Text on a matrix known to be positive definite
C = np.array([[4, 1], [1, 3]])
print("\nTesting matrix C:")
print(C)
result = is_positive_definite(C)
print("Is matrix C positive definite?", result)
