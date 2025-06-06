import numpy as np
from numpy.typing import NDArray
from typing import Union

def normalize(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Normalize the input array X by subtracting the mean and dividing by the standard deviation.
    
    Parameters:
        X (NDArray[np.floating]): Input array to normalize. Should be a numerical array
            (float or compatible type).
    
    Returns:
        NDArray[np.floating]: Normalized array with mean approximately 0 and standard
            deviation approximately 1 along each axis.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Handle division by zero by using np.where to avoid warnings
    normalized_X = np.where(std != 0, (X - mean) / std, X - mean)
    return normalized_X

# Example usage
X: NDArray[np.floating] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
normalized_X: NDArray[np.floating] = normalize(X)
print("Original array:\n", X)
print("Normalized array:\n", normalized_X)

Y: NDArray[np.floating] = np.random.randn(5, 2)
normalizedY: NDArray[np.floating] = normalize(Y)
mean_Y = np.mean(Y)
mean_normalizedY = np.mean(normalizedY)
print("Original Y array:\n", Y)
print("Mean of Y array:\n", mean_Y)
print("Normalized Y array:\n", normalizedY)
print("Mean of normalized Y array:\n", mean_normalizedY)
