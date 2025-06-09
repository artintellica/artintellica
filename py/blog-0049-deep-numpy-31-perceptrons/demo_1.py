import numpy as np
from numpy.typing import NDArray
from typing import Union, Dict
from neural_network import gradient_descent, binary_cross_entropy


def forward_perceptron(
    X: NDArray[np.floating], W: NDArray[np.floating], b: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Compute the forward pass of a single-layer perceptron.
    Args:
        X: Input data, shape (n_samples, n_features)
        W: Weights, shape (n_features, 1)
        b: Bias, shape (1, 1) or (1,)
    Returns:
        Output after sigmoid activation, shape (n_samples, 1)
    """
    Z = X @ W + b  # Linear combination
    A = sigmoid(Z)  # Sigmoid activation
    return A


def sigmoid(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the sigmoid activation function element-wise.
    Args:
        Z: Input array of any shape with floating-point values
    Returns:
        Array of the same shape with sigmoid applied element-wise, values in [0, 1]
    """
    return 1 / (1 + np.exp(-Z))


# XOR data: inputs and outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input (4 samples, 2 features)
y = np.array([[0], [1], [1], [0]])  # Output (XOR: 1 if inputs differ, 0 otherwise)

# Initialize parameters
n_features = X.shape[1]
n_samples = X.shape[0]
W_init = np.zeros((n_features, 1))  # Initial weights
b_init = np.zeros((1, 1))  # Initial bias
lr = 0.1  # Learning rate
num_epochs = 1000  # Number of epochs (high to attempt convergence)
batch_size = 4  # Full batch since dataset is small

# Train perceptron using gradient_descent with sigmoid activation
W_final, b_final, losses = gradient_descent(
    X,
    y,
    W_init,
    b_init,
    lr,
    num_epochs,
    batch_size,
    loss_fn=binary_cross_entropy,
    activation_fn=sigmoid,
)

# Evaluate the model
A = forward_perceptron(X, W_final, b_final)
predictions = (A > 0.5).astype(int)
accuracy = np.mean(predictions == y)
print("Final Predictions (probabilities):\n", A)
print("Final Predictions (binary):\n", predictions)
print("True Labels:\n", y)
print("Accuracy:", accuracy)
print("Final Loss:", losses[-1])
print("Loss History (first 5 and last 5):", losses[:5] + losses[-5:])
