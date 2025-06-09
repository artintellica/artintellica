import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, List, cast
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_network import (
    normalize,
    relu,
    softmax,
    cross_entropy,
    forward_mlp_3layer,
    backward_mlp_3layer,
    l2_regularization,
    dropout,
    accuracy,
)
import matplotlib.pyplot as plt


def momentum_update(
    velocity: NDArray[np.floating], gradient: NDArray[np.floating], mu: float, lr: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Perform a momentum-based update on the velocity and return the parameter update.
    Args:
        velocity: Current velocity array, same shape as gradient (initialized to 0)
        gradient: Current gradient of the loss with respect to the parameter
        mu: Momentum coefficient (e.g., 0.9)
        lr: Learning rate (e.g., 0.1)
    Returns:
        Tuple of (updated_velocity, parameter_update):
        - updated_velocity: New velocity after momentum update
        - parameter_update: Update to apply to the parameter (e.g., W += parameter_update)
    """
    updated_velocity = mu * velocity - lr * gradient
    parameter_update = updated_velocity
    return updated_velocity, parameter_update


def accuracy(y_pred: NDArray[np.floating], y_true: NDArray[np.floating]) -> float:
    """
    Compute classification accuracy for multi-class predictions.
    Args:
        y_pred: Predicted probabilities or logits, shape (n_samples, n_classes)
        y_true: True labels, one-hot encoded or class indices, shape (n_samples, n_classes) or (n_samples,)
    Returns:
        Accuracy as a float (fraction of correct predictions)
    """
    if y_true.ndim == 2:  # One-hot encoded
        true_labels = np.argmax(y_true, axis=1)
    else:  # Class indices
        true_labels = y_true
    pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(pred_labels == true_labels)


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

# Initialize velocities for momentum
v_W1 = np.zeros_like(W1)
v_b1 = np.zeros_like(b1)
v_W2 = np.zeros_like(W2)
v_b2 = np.zeros_like(b2)
v_W3 = np.zeros_like(W3)
v_b3 = np.zeros_like(b3)

# Training loop with momentum, L2 regularization, and dropout
lr = 0.1
mu = 0.9  # Momentum coefficient
num_epochs = 20
batch_size = 64
lambda_l2 = 0.01  # L2 regularization strength
dropout_p = 0.8  # Keep probability for dropout
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

        # Forward pass with dropout
        A1, A2, A3 = forward_mlp_3layer(X_batch, W1, b1, W2, b2, W3, b3)
        A1_drop = dropout(A1, dropout_p, training=True)
        A2_drop = dropout(A2, dropout_p, training=True)

        # Compute gradients via backpropagation
        Z1 = X_batch @ W1 + b1
        Z2 = A1_drop @ W2 + b2
        grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = backward_mlp_3layer(
            X_batch, A1_drop, A2_drop, A3, y_batch, W1, W2, W3, Z1, Z2
        )

        # Add L2 regularization gradients
        l2_penalty, l2_grads = l2_regularization([W1, W2, W3], lambda_l2)
        grad_W1 += l2_grads[0]
        grad_W2 += l2_grads[1]
        grad_W3 += l2_grads[2]

        # Update parameters with momentum
        v_W1, update_W1 = momentum_update(v_W1, grad_W1, mu, lr)
        v_b1, update_b1 = momentum_update(v_b1, grad_b1, mu, lr)
        v_W2, update_W2 = momentum_update(v_W2, grad_W2, mu, lr)
        v_b2, update_b2 = momentum_update(v_b2, grad_b2, mu, lr)
        v_W3, update_W3 = momentum_update(v_W3, grad_W3, mu, lr)
        v_b3, update_b3 = momentum_update(v_b3, grad_b3, mu, lr)
        W1 += update_W1
        b1 += update_b1
        W2 += update_W2
        b2 += update_b2
        W3 += update_W3
        b3 += update_b3

    # Compute loss on full training set (without dropout)
    _, _, A3_full = forward_mlp_3layer(X_train, W1, b1, W2, b2, W3, b3)
    loss = cross_entropy(A3_full, y_train)
    loss_history.append(loss)

    # Compute accuracy on test set (without dropout)
    _, _, A3_test = forward_mlp_3layer(X_test, W1, b1, W2, b2, W3, b3)
    test_accuracy = accuracy(A3_test, y_test)
    accuracy_history.append(test_accuracy)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
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
final_accuracy = accuracy(A3_test, y_test)
print("Final Test Accuracy:", final_accuracy)
