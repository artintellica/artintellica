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
    l2_regularization,
    dropout,
)
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X_full, y_full = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X_full = X_full.astype(float)
y_full = y_full.astype(int)
X = X_full[:1000]
y = y_full[:1000]
n_classes = 10
y_one_hot = np.zeros((y.shape[0], n_classes))
y_one_hot[np.arange(y.shape[0]), y] = 1
X = normalize(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=0.2, random_state=42
)
X_train = cast(NDArray[np.floating], X_train)
X_test = cast(NDArray[np.floating], X_test)
y_train = cast(NDArray[np.floating], y_train)
y_test = cast(NDArray[np.floating], y_test)
W1 = np.random.randn(784, 256) * 0.01
b1 = np.zeros((1, 256))
W2 = np.random.randn(256, 128) * 0.01
b2 = np.zeros((1, 128))
W3 = np.random.randn(128, 10) * 0.01
b3 = np.zeros((1, 10))
lr = 0.1
num_epochs = 10
batch_size = 32
dropout_p = 0.8
n_samples = X_train.shape[0]
loss_history_with_dropout = []
accuracy_history_with_dropout = []
loss_history_no_dropout = []
accuracy_history_no_dropout = []


def train_mlp_with_dropout(
    use_dropout: bool, W1, b1, W2, b2, W3, b3, X_train, X_test, y_train, y_test
):
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
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            A1, A2, A3 = forward_mlp_3layer(
                X_batch, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy
            )
            if use_dropout:
                A1 = dropout(A1, dropout_p, training=True)
                A2 = dropout(A2, dropout_p, training=True)
            Z1 = X_batch @ W1_copy + b1_copy
            Z2 = A1 @ W2_copy + b2_copy
            grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = backward_mlp_3layer(
                X_batch, A1, A2, A3, y_batch, W1_copy, W2_copy, W3_copy, Z1, Z2
            )
            W1_copy -= lr * grad_W1
            b1_copy -= lr * grad_b1
            W2_copy -= lr * grad_W2
            b2_copy -= lr * grad_b2
            W3_copy -= lr * grad_W3
            b3_copy -= lr * grad_b3
        _, _, A3_full = forward_mlp_3layer(
            X_train, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy
        )
        loss = cross_entropy(A3_full, y_train)
        loss_history.append(loss)
        _, _, A3_test = forward_mlp_3layer(
            X_test, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy
        )
        predictions = np.argmax(A3_test, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == true_labels)
        accuracy_history.append(accuracy)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f} (Dropout: {use_dropout})"
        )
    return loss_history, accuracy_history


print("Training with Dropout...")
loss_history_with_dropout, accuracy_history_with_dropout = train_mlp_with_dropout(
    True, W1, b1, W2, b2, W3, b3, X_train, X_test, y_train, y_test
)

print("Training without Dropout...")
loss_history_no_dropout, accuracy_history_no_dropout = train_mlp_with_dropout(
    False, W1, b1, W2, b2, W3, b3, X_train, X_test, y_train, y_test
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.plot(
    range(1, num_epochs + 1),
    loss_history_with_dropout,
    label="Training Loss (With Dropout)",
)
ax1.plot(
    range(1, num_epochs + 1),
    loss_history_no_dropout,
    label="Training Loss (No Dropout)",
)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title("Training Loss Over Epochs")
ax1.legend()
ax1.grid(True)

ax2.plot(
    range(1, num_epochs + 1),
    accuracy_history_with_dropout,
    label="Test Accuracy (With Dropout)",
)
ax2.plot(
    range(1, num_epochs + 1),
    accuracy_history_no_dropout,
    label="Test Accuracy (No Dropout)",
)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Test Accuracy Over Epochs")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
