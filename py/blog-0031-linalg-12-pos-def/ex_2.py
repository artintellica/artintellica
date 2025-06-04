import numpy as np


def attempt_cholesky(matrix):
    """
    Attempt to perform Cholesky decomposition on a given matrix.
    If it fails, the matrix is not positive definite.

    Args:
        matrix (np.ndarray): A square matrix to be decomposed.

    Returns:
        tuple: (bool, np.ndarray or None) - A boolean indicating if decomposition succeeded,
               and the lower triangular matrix L if successful, otherwise None.
    """
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        print("Matrix is not square.")
        return False, None

    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        print("Matrix is not symmetric.")
        return False, None

    # Attempt Cholesky decomposition
    try:
        L = np.linalg.cholesky(matrix)
        print("Cholesky decomposition succeeded. Lower triangular matrix L:")
        print(L)
        print("Reconstructed matrix from L L^T:")
        print(L @ L.T)
        return True, L
    except np.linalg.LinAlgError:
        print("Cholesky decomposition failed. Matrix is not positive definite.")
        return False, None


# Test the function on matrix B
B = np.array([[1, 2], [2, 1]])
print("Testing matrix B:")
print(B)
success, L = attempt_cholesky(B)
print("Was Cholesky decomposition successful?", success)
if not success:
    print("Matrix B is not positive definite.")

# Test on a matrix known to be positive definite
C = np.array([[4, 1], [1, 3]])
print("\nTesting matrix C:")
print(C)
success, L = attempt_cholesky(C)
print("Was Cholesky decomposition successful?", success)
if success:
    print("Lower triangular matrix L for matrix C:")
    print(L)
    print("Matrix C is positive definite.")
else:
    print("Matrix C is not positive definite.")
