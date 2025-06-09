import numpy as np
from numpy.typing import NDArray
from typing import Union, List, cast
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_network import (
    normalize,
    relu,
    softmax,
    cross_entropy,
    forward_mlp_3layer,
    backward_mlp_3layer,
)
import matplotlib.pyplot as plt


def l2_regularization(
    weights: List[NDArray[np.floating]], lambda_: float
) -> tuple[float, List[NDArray[np.floating]]]:
    """
    Compute L2 regularization penalty and gradients for a list of weight matrices.
    Args:
        weights: List of weight matrices (e.g., [W1, W2, W3])
        lambda_: Regularization strength (e.g., 0.01)
    Returns:
        Tuple of (l2_penalty, l2_grads):
        - l2_penalty: Scalar penalty term to add to loss (lambda * sum of squared weights)
        - l2_grads: List of gradients for each weight matrix (2 * lambda * W)
    """
    l2_penalty = 0.0
    l2_grads = []
    for W in weights:
        l2_penalty += np.sum(W**2)
        l2_grads.append(2 * lambda_ * W)
    l2_penalty *= lambda_
    l2_penalty = cast(float, l2_penalty)  # Ensure penalty is a scalar
    return l2_penalty, l2_grads


def dropout(
    A: NDArray[np.floating], p: float, training: bool = True
) -> NDArray[np.floating]:
    """
    Apply dropout to an activation matrix by randomly setting elements to 0.
    Args:
        A: Activation matrix, shape (any)
        p: Keep probability (e.g., 0.8 to keep 80% of neurons)
        training: Boolean, apply dropout only during training (default: True)
    Returns:
        Activation matrix after dropout (same shape as input)
    """
    if training:
        mask = np.random.binomial(1, p, size=A.shape)
        return A * mask
    return A


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

# Training loop with L2 regularization and dropout
lr = 0.1
num_epochs = 20
batch_size = 64
lambda_l2 = 0.01  # L2 regularization strength
dropout_p = 0.8  # Keep probability for dropout (80% keep, 20% drop)
n_samples = X_train.shape[0]
loss_history_with_reg = []
accuracy_history_with_reg = []
loss_history_no_reg = []
accuracy_history_no_reg = []


# Function to train with or without regularization
def train_mlp(with_reg: bool, W1, b1, W2, b2, W3, b3, X_train, X_test, y_train, y_test):
    W1_copy = W1.copy()
    b1_copy = b1.copy()
    W2_copy = W2.copy()
    b2_copy = b2.copy()
    W3_copy = W3.copy()
    b3_copy = b3.copy()
    loss_history = []
    accuracy_history = []
    for epoch in range(num_epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        total_loss = 0.0
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            # Forward pass with dropout if enabled
            A1, A2, A3 = forward_mlp_3layer(
                X_batch, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy
            )
            if with_reg:
                A1 = dropout(A1, dropout_p, training=True)
                A2 = dropout(A2, dropout_p, training=True)
            # Compute loss
            data_loss = cross_entropy(A3, y_batch)
            l2_penalty = 0.0
            l2_grads = []
            if with_reg:
                l2_penalty, l2_grads = l2_regularization(
                    [W1_copy, W2_copy, W3_copy], lambda_l2
                )
            total_loss += data_loss + l2_penalty
            # Backpropagation
            Z1 = X_batch @ W1_copy + b1_copy
            Z2 = A1 @ W2_copy + b2_copy
            grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = backward_mlp_3layer(
                X_batch, A1, A2, A3, y_batch, W1_copy, W2_copy, W3_copy, Z1, Z2
            )
            if with_reg:
                grad_W1 += l2_grads[0]
                grad_W2 += l2_grads[1]
                grad_W3 += l2_grads[2]
            # Update parameters
            W1_copy -= lr * grad_W1
            b1_copy -= lr * grad_b1
            W2_copy -= lr * grad_W2
            b2_copy -= lr * grad_b2
            W3_copy -= lr * grad_W3
            b3_copy -= lr * grad_b3
        # Compute loss on full training set (without dropout)
        _, _, A3_full = forward_mlp_3layer(
            X_train, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy
        )
        loss = cross_entropy(A3_full, y_train)
        loss_history.append(loss)
        # Compute accuracy on test set (without dropout)
        _, _, A3_test = forward_mlp_3layer(
            X_test, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy
        )
        predictions = np.argmax(A3_test, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == true_labels)
        accuracy_history.append(accuracy)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f} (With Reg: {with_reg})"
        )
    return loss_history, accuracy_history


# Train with regularization
print("Training with L2 Regularization and Dropout...")
loss_history_with_reg, accuracy_history_with_reg = train_mlp(
    True,
    W1.copy(),
    b1.copy(),
    W2.copy(),
    b2.copy(),
    W3.copy(),
    b3.copy(),
    X_train,
    X_test,
    y_train,
    y_test,
)

# Train without regularization
print("Training without Regularization...")
loss_history_no_reg, accuracy_history_no_reg = train_mlp(
    False,
    W1.copy(),
    b1.copy(),
    W2.copy(),
    b2.copy(),
    W3.copy(),
    b3.copy(),
    X_train,
    X_test,
    y_train,
    y_test,
)

# Plot loss and accuracy history for comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.plot(
    range(1, num_epochs + 1), loss_history_with_reg, label="Training Loss (With Reg)"
)
ax1.plot(range(1, num_epochs + 1), loss_history_no_reg, label="Training Loss (No Reg)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title("Training Loss Over Epochs")
ax1.legend()
ax1.grid(True)

ax2.plot(
    range(1, num_epochs + 1),
    accuracy_history_with_reg,
    label="Test Accuracy (With Reg)",
)
ax2.plot(
    range(1, num_epochs + 1), accuracy_history_no_reg, label="Test Accuracy (No Reg)"
)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Test Accuracy Over Epochs")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
