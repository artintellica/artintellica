import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, Dict, cast
from neural_network import relu, softmax
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_network import normalize, relu, softmax, cross_entropy


def forward_mlp_3layer(
    X: NDArray[np.floating],
    W1: NDArray[np.floating],
    b1: NDArray[np.floating],
    W2: NDArray[np.floating],
    b2: NDArray[np.floating],
    W3: NDArray[np.floating],
    b3: NDArray[np.floating],
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute the forward pass of a 3-layer MLP.
    Args:
        X: Input data, shape (n_samples, n_features, e.g., 784 for MNIST)
        W1: Weights for first layer, shape (n_features, n_hidden1, e.g., 784x256)
        b1: Bias for first layer, shape (1, n_hidden1)
        W2: Weights for second layer, shape (n_hidden1, n_hidden2, e.g., 256x128)
        b2: Bias for second layer, shape (1, n_hidden2)
        W3: Weights for third layer, shape (n_hidden2, n_classes, e.g., 128x10)
        b3: Bias for third layer, shape (1, n_classes)
    Returns:
        Tuple of (A1, A2, A3):
        - A1: First hidden layer output after ReLU, shape (n_samples, n_hidden1)
        - A2: Second hidden layer output after ReLU, shape (n_samples, n_hidden2)
        - A3: Output layer output after softmax, shape (n_samples, n_classes)
    """
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)
    return A1, A2, A3


def backward_mlp_3layer(
    X: NDArray[np.floating],
    A1: NDArray[np.floating],
    A2: NDArray[np.floating],
    A3: NDArray[np.floating],
    y: NDArray[np.floating],
    W1: NDArray[np.floating],
    W2: NDArray[np.floating],
    W3: NDArray[np.floating],
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
) -> Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Compute gradients for a 3-layer MLP using backpropagation.
    Args:
        X: Input data, shape (n_samples, n_features)
        A1: First hidden layer output after ReLU, shape (n_samples, n_hidden1)
        A2: Second hidden layer output after ReLU, shape (n_samples, n_hidden2)
        A3: Output layer output after softmax, shape (n_samples, n_classes)
        y: True labels, one-hot encoded, shape (n_samples, n_classes)
        W1: Weights for first layer, shape (n_features, n_hidden1)
        W2: Weights for second layer, shape (n_hidden1, n_hidden2)
        W3: Weights for third layer, shape (n_hidden2, n_classes)
        Z1: Pre-activation values for first hidden layer, shape (n_samples, n_hidden1)
        Z2: Pre-activation values for second hidden layer, shape (n_samples, n_hidden2)
    Returns:
        Tuple of gradients (grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3)
    """
    n = X.shape[0]

    # Output layer error (delta3)
    delta3 = A3 - y  # Shape (n_samples, n_classes)

    # Gradients for output layer (W3, b3)
    grad_W3 = (A2.T @ delta3) / n  # Shape (n_hidden2, n_classes)
    grad_b3 = np.mean(delta3, axis=0, keepdims=True)  # Shape (1, n_classes)

    # Second hidden layer error (delta2)
    delta2 = (delta3 @ W3.T) * (Z2 > 0)  # ReLU derivative: 1 if Z2 > 0, 0 otherwise
    # Shape (n_samples, n_hidden2)

    # Gradients for second hidden layer (W2, b2)
    grad_W2 = (A1.T @ delta2) / n  # Shape (n_hidden1, n_hidden2)
    grad_b2 = np.mean(delta2, axis=0, keepdims=True)  # Shape (1, n_hidden2)

    # First hidden layer error (delta1)
    delta1 = (delta2 @ W2.T) * (Z1 > 0)  # ReLU derivative: 1 if Z1 > 0, 0 otherwise
    # Shape (n_samples, n_hidden1)

    # Gradients for first hidden layer (W1, b1)
    grad_W1 = (X.T @ delta1) / n  # Shape (n_features, n_hidden1)
    grad_b1 = np.mean(delta1, axis=0, keepdims=True)  # Shape (1, n_hidden1)

    return grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3


# Load MNIST data (subset for faster training on CPU)
print("Loading MNIST data...")
X_full, y_full = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X_full = X_full.astype(float)
y_full = y_full.astype(int)

# Limit to 5000 samples for faster training
X = X_full[:5000]
y = y_full[:5000]

# Convert labels to one-hot encoding
n_classes = 10
y_one_hot = np.zeros((y.shape[0], n_classes))
y_one_hot[np.arange(y.shape[0]), y] = 1

# Normalize input data
X = normalize(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=0.2, random_state=42
)
X_train = cast(NDArray[np.floating], X_train)
X_test = cast(NDArray[np.floating], X_test)
y_train = cast(NDArray[np.floating], y_train)
y_test = cast(NDArray[np.floating], y_test)

# Initialize parameters for 3-layer MLP (784 -> 256 -> 128 -> 10)
n_features = X_train.shape[1]  # 784 for MNIST
n_hidden1 = 256
n_hidden2 = 128
W1 = np.random.randn(n_features, n_hidden1) * 0.01
b1 = np.zeros((1, n_hidden1))
W2 = np.random.randn(n_hidden1, n_hidden2) * 0.01
b2 = np.zeros((1, n_hidden2))
W3 = np.random.randn(n_hidden2, n_classes) * 0.01
b3 = np.zeros((1, n_classes))

# Training loop
lr = 0.1
num_epochs = 20
batch_size = 64
n_samples = X_train.shape[0]
loss_history = []
accuracy_history = []

print("Starting training...")
for epoch in range(num_epochs):
    # Shuffle data
    indices = np.random.permutation(n_samples)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    # Mini-batch processing
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        batch_size_actual = X_batch.shape[0]

        # Forward pass
        A1, A2, A3 = forward_mlp_3layer(X_batch, W1, b1, W2, b2, W3, b3)

        # Compute gradients via backpropagation
        Z1 = X_batch @ W1 + b1
        Z2 = A1 @ W2 + b2
        grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = backward_mlp_3layer(
            X_batch, A1, A2, A3, y_batch, W1, W2, W3, Z1, Z2
        )

        # Update parameters
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        W2 -= lr * grad_W2
        b2 -= lr * grad_b2
        W3 -= lr * grad_W3
        b3 -= lr * grad_b3

    # Compute loss on full training set at end of epoch
    _, _, A3_full = forward_mlp_3layer(X_train, W1, b1, W2, b2, W3, b3)
    loss = cross_entropy(A3_full, y_train)
    loss_history.append(loss)

    # Compute accuracy on test set
    _, _, A3_test = forward_mlp_3layer(X_test, W1, b1, W2, b2, W3, b3)
    predictions = np.argmax(A3_test, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == true_labels)
    accuracy_history.append(accuracy)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}"
    )

# Plot loss and accuracy history
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.plot(range(1, num_epochs + 1), loss_history, label="Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title("Training Loss Over Epochs")
ax1.legend()
ax1.grid(True)

ax2.plot(range(1, num_epochs + 1), accuracy_history, label="Test Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Test Accuracy Over Epochs")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Final evaluation on test set
_, _, A3_test = forward_mlp_3layer(X_test, W1, b1, W2, b2, W3, b3)
predictions = np.argmax(A3_test, axis=1)
true_labels = np.argmax(y_test, axis=1)
final_accuracy = np.mean(predictions == true_labels)
print("Final Test Accuracy:", final_accuracy)
