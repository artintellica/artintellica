import numpy as np


def normalize(X):
    """
    Normalize an array to have mean=0 and std=1.
    Args:
        X: NumPy array of any shape
    Returns:
        Normalized array of the same shape
    """
    mean = np.mean(X)
    std = np.std(X)
    if std == 0:  # Avoid division by zero
        return X - mean
    return (X - mean) / std


# Example: Normalize a random 3x2 matrix
random_matrix = np.random.randn(3, 2)
print("Original matrix:\n", random_matrix)
normalized_matrix = normalize(random_matrix)
print("Normalized matrix (mean≈0, std≈1):\n", normalized_matrix)
print("Mean after normalization:", np.mean(normalized_matrix))
print("Std after normalization:", np.std(normalized_matrix))
