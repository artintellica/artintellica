import numpy as np

def normalize(X):
    """
    Normalize the input array X by subtracting the mean and dividing by the standard deviation.
    
    Parameters:
    X (np.ndarray): Input array to normalize.
    
    Returns:
    np.ndarray: Normalized array.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    normalized_X = (X - mean) / std
    return normalized_X
