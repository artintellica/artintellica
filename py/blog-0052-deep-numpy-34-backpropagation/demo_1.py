import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, Dict, cast
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_network import normalize, relu, softmax, cross_entropy, forward_mlp


def backward_mlp(
    X: NDArray[np.floating],
    A1: NDArray[np.floating],
    A2: NDArray[np.floating],
    y: NDArray[np.floating],
    W1: NDArray[np.floating],
    W2: NDArray[np.floating],
    Z1: NDArray[np.floating],
) -> Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Compute gradients for a 2-layer MLP using backpropagation.
    Args:
        X: Input data, shape (n_samples, n_features)
        A1: Hidden layer output after ReLU, shape (n_samples, n_hidden)
        A2: Output layer output after softmax, shape (n_samples, n_classes)
        y: True labels, one-hot encoded, shape (n_samples, n_classes)
        W1: Weights for first layer, shape (n_features, n_hidden)
        W2: Weights for second layer, shape (n_hidden, n_classes)
        Z1: Pre-activation values for hidden layer, shape (n_samples, n_hidden)
    Returns:
        Tuple of gradients (grad_W1, grad_b1, grad_W2, grad_b2)
    """
    n = X.shape[0]

    # Output layer error (delta2)
    delta2 = A2 - y  # Shape (n_samples, n_classes)

    # Gradients for output layer (W2, b2)
    grad_W2 = (A1.T @ delta2) / n  # Shape (n_hidden, n_classes)
    grad_b2 = np.mean(delta2, axis=0, keepdims=True)  # Shape (1, n_classes)

    # Hidden layer error (delta1)
    delta1 = (delta2 @ W2.T) * (Z1 > 0)  # ReLU derivative: 1 if Z1 > 0, 0 otherwise
    # Shape (n_samples, n_hidden)

    # Gradients for hidden layer (W1, b1)
    grad_W1 = (X.T @ delta1) / n  # Shape (n_features, n_hidden)
    grad_b1 = np.mean(delta1, axis=0, keepdims=True)  # Shape (1, n_hidden)

    return grad_W1, grad_b1, grad_W2, grad_b2


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

# Initialize parameters for 2-layer MLP (784 -> 256 -> 10)
n_features = X_train.shape[1]  # 784 for MNIST
n_hidden = 256
W1 = np.random.randn(n_features, n_hidden) * 0.01
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_classes) * 0.01
b2 = np.zeros((1, n_classes))

# Training loop
lr = 0.1
num_epochs = 20
batch_size = 64
n_samples = X_train.shape[0]
loss_history = []

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
        A1, A2 = forward_mlp(X_batch, W1, b1, W2, b2)

        # Compute gradients via backpropagation
        Z1 = X_batch @ W1 + b1  # Pre-activation for ReLU
        grad_W1, grad_b1, grad_W2, grad_b2 = backward_mlp(
            X_batch, A1, A2, y_batch, W1, W2, Z1
        )

        # Update parameters
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        W2 -= lr * grad_W2
        b2 -= lr * grad_b2

    # Compute loss on full training set at end of epoch
    _, A2_full = forward_mlp(X_train, W1, b1, W2, b2)
    loss = cross_entropy(A2_full, y_train)
    loss_history.append(loss)

    # Compute accuracy on test set
    _, A2_test = forward_mlp(X_test, W1, b1, W2, b2)
    predictions = np.argmax(A2_test, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == true_labels)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}"
    )

# Plot loss history
plt.plot(range(1, num_epochs + 1), loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Final evaluation on test set
_, A2_test = forward_mlp(X_test, W1, b1, W2, b2)
predictions = np.argmax(A2_test, axis=1)
true_labels = np.argmax(y_test, axis=1)
final_accuracy = np.mean(predictions == true_labels)
print("Final Test Accuracy:", final_accuracy)
