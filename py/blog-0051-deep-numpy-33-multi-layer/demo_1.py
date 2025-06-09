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


def cross_entropy(A: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """
    Compute categorical cross-entropy loss for multi-class classification.
    Args:
        A: Predicted probabilities after softmax, shape (n_samples, n_classes)
        y: True labels, one-hot encoded, shape (n_samples, n_classes)
    Returns:
        Cross-entropy loss as a single float
    """
    epsilon = 1e-15  # Small value to prevent log(0)
    return -np.mean(np.sum(y * np.log(A + epsilon), axis=1))

# Simulate MNIST data (4 samples for simplicity)
n_samples = 4
n_features = 784  # MNIST image size (28x28)
n_hidden = 256    # Hidden layer size
n_classes = 10    # MNIST digits (0-9)

# Random input data (simulating normalized MNIST images)
X = np.random.randn(n_samples, n_features)

# Initialize parameters with small random values
W1 = np.random.randn(n_features, n_hidden) * 0.01  # Shape (784, 256)
b1 = np.zeros((1, n_hidden))                       # Shape (1, 256)
W2 = np.random.randn(n_hidden, n_classes) * 0.01   # Shape (256, 10)
b2 = np.zeros((1, n_classes))                      # Shape (1, 10)

# Compute forward pass
A1, A2 = forward_mlp(X, W1, b1, W2, b2)

# Simulate one-hot encoded labels for 4 samples (e.g., digits 3, 7, 1, 9)
y = np.zeros((n_samples, n_classes))
y[0, 3] = 1  # Sample 1: digit 3
y[1, 7] = 1  # Sample 2: digit 7
y[2, 1] = 1  # Sample 3: digit 1
y[3, 9] = 1  # Sample 4: digit 9

# Compute cross-entropy loss
loss = cross_entropy(A2, y)

print("Hidden Layer Output A1 shape (after ReLU):", A1.shape)
print("Output Layer Output A2 shape (after softmax):", A2.shape)
print("Output Probabilities A2 (first few columns):\n", A2[:, :3])
print("Sum of probabilities per sample (should be ~1):\n", np.sum(A2, axis=1))
print("True Labels y (one-hot, first few columns):\n", y[:, :3])
print("Cross-Entropy Loss:", loss)
