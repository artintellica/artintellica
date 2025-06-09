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
    momentum_update,
    accuracy,
)
import matplotlib.pyplot as plt

# Your code here
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X_full, y_full = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X_full = X_full.astype(float)
y_full = y_full.astype(int)
X = X_full[:2000]
y = y_full[:2000]
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
v_W1 = np.zeros_like(W1)
v_b1 = np.zeros_like(b1)
v_W2 = np.zeros_like(W2)
v_b2 = np.zeros_like(b2)
v_W3 = np.zeros_like(W3)
v_b3 = np.zeros_like(b3)
lr = 0.1
mu = 0.9
num_epochs = 15
batch_size = 64
lambda_l2 = 0.01
dropout_p = 0.8
n_samples = X_train.shape[0]
loss_history_with_momentum = []
accuracy_history_with_momentum = []
loss_history_no_momentum = []
accuracy_history_no_momentum = []


def train_mlp_with_momentum(
    use_momentum: bool,
    W1: NDArray[np.floating],
    b1: NDArray[np.floating],
    W2: NDArray[np.floating],
    b2: NDArray[np.floating],
    W3: NDArray[np.floating],
    b3: NDArray[np.floating],
    v_W1: NDArray[np.floating],
    v_b1: NDArray[np.floating],
    v_W2: NDArray[np.floating],
    v_b2: NDArray[np.floating],
    v_W3: NDArray[np.floating],
    v_b3: NDArray[np.floating],
    X_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    y_train: NDArray[np.floating],
    y_test: NDArray[np.floating],
):
    W1_copy = W1.copy()
    b1_copy = b1.copy()
    W2_copy = W2.copy()
    b2_copy = b2.copy()
    W3_copy = W3.copy()
    b3_copy = b3.copy()
    v_W1_copy = v_W1.copy()
    v_b1_copy = v_b1.copy()
    v_W2_copy = v_W2.copy()
    v_b2_copy = v_b2.copy()
    v_W3_copy = v_W3.copy()
    v_b3_copy = v_b3.copy()
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
            A1_drop = dropout(A1, dropout_p, training=True)
            A2_drop = dropout(A2, dropout_p, training=True)
            Z1 = X_batch @ W1_copy + b1_copy
            Z2 = A1_drop @ W2_copy + b2_copy
            grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = backward_mlp_3layer(
                X_batch,
                A1_drop,
                A2_drop,
                A3,
                y_batch,
                W1_copy,
                W2_copy,
                W3_copy,
                Z1,
                Z2,
            )
            l2_penalty, l2_grads = l2_regularization(
                [W1_copy, W2_copy, W3_copy], lambda_l2
            )
            grad_W1 += l2_grads[0]
            grad_W2 += l2_grads[1]
            grad_W3 += l2_grads[2]
            if use_momentum:
                v_W1_copy, update_W1 = momentum_update(v_W1_copy, grad_W1, mu, lr)
                v_b1_copy, update_b1 = momentum_update(v_b1_copy, grad_b1, mu, lr)
                v_W2_copy, update_W2 = momentum_update(v_W2_copy, grad_W2, mu, lr)
                v_b2_copy, update_b2 = momentum_update(v_b2_copy, grad_b2, mu, lr)
                v_W3_copy, update_W3 = momentum_update(v_W3_copy, grad_W3, mu, lr)
                v_b3_copy, update_b3 = momentum_update(v_b3_copy, grad_b3, mu, lr)
            else:
                update_W1 = -lr * grad_W1
                update_b1 = -lr * grad_b1
                update_W2 = -lr * grad_W2
                update_b2 = -lr * grad_b2
                update_W3 = -lr * grad_W3
                update_b3 = -lr * grad_b3
            W1_copy += update_W1
            b1_copy += update_b1
            W2_copy += update_W2
            b2_copy += update_b2
            W3_copy += update_W3
            b3_copy += update_b3
        _, _, A3_full = forward_mlp_3layer(
            X_train, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy
        )
        loss = cross_entropy(A3_full, y_train)
        loss_history.append(loss)
        _, _, A3_test = forward_mlp_3layer(
            X_test, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy
        )
        test_accuracy = accuracy(A3_test, y_test)
        accuracy_history.append(test_accuracy)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Accuracy: {test_accuracy:.4f} (Momentum: {use_momentum})"
        )
    return loss_history, accuracy_history


print("Training with Momentum...")
loss_history_with_momentum, accuracy_history_with_momentum = train_mlp_with_momentum(
    True,
    W1,
    b1,
    W2,
    b2,
    W3,
    b3,
    v_W1,
    v_b1,
    v_W2,
    v_b2,
    v_W3,
    v_b3,
    X_train,
    X_test,
    y_train,
    y_test,
)

print("Training without Momentum...")
loss_history_no_momentum, accuracy_history_no_momentum = train_mlp_with_momentum(
    False,
    W1,
    b1,
    W2,
    b2,
    W3,
    b3,
    v_W1,
    v_b1,
    v_W2,
    v_b2,
    v_W3,
    v_b3,
    X_train,
    X_test,
    y_train,
    y_test,
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.plot(
    range(1, num_epochs + 1),
    loss_history_with_momentum,
    label="Training Loss (With Momentum)",
)
ax1.plot(
    range(1, num_epochs + 1),
    loss_history_no_momentum,
    label="Training Loss (No Momentum)",
)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title("Training Loss Over Epochs")
ax1.legend()
ax1.grid(True)

ax2.plot(
    range(1, num_epochs + 1),
    accuracy_history_with_momentum,
    label="Test Accuracy (With Momentum)",
)
ax2.plot(
    range(1, num_epochs + 1),
    accuracy_history_no_momentum,
    label="Test Accuracy (No Momentum)",
)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Test Accuracy Over Epochs")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
