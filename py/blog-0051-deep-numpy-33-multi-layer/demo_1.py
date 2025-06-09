import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import relu, softmax


def forward_mlp(
    X: NDArray[np.floating],
    W1: NDArray[np.floating],
    b1: NDArray[np.floating],
    W2: NDArray[np.floating],
    b2: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute the forward pass of a 2-layer MLP.
    Args:
        X: Input data, shape (n_samples, n_features, e.g., 784 for MNIST)
        W1: Weights for first layer, shape (n_features, n_hidden, e.g., 784x256)
        b1: Bias for first layer, shape (1, n_hidden)
        W2: Weights for second layer, shape (n_hidden, n_classes, e.g., 256x10)
        b2: Bias for second layer, shape (1, n_classes)
    Returns:
        Tuple of (A1, A2):
        - A1: Hidden layer output after ReLU, shape (n_samples, n_hidden)
        - A2: Output layer output after softmax, shape (n_samples, n_classes)
    """
    Z1 = X @ W1 + b1  # First layer linear combination
    A1 = relu(Z1)  # ReLU activation for hidden layer
    Z2 = A1 @ W2 + b2  # Second layer linear combination
    A2 = softmax(Z2)  # Softmax activation for output layer
    return A1, A2
