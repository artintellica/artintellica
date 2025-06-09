from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, Dict, cast
from neural_network import forward_mlp, backward_mlp, cross_entropy, normalize

X_full, y_full = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X_full = X_full.astype(float)
y_full = y_full.astype(int)
mask = (y_full == 0) | (y_full == 1)
X = X_full[mask][:500]
y = y_full[mask][:500]
y_one_hot = np.zeros((y.shape[0], 2))
y_one_hot[y == 0, 0] = 1
y_one_hot[y == 1, 1] = 1
X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=0.2, random_state=42
)
X_train = cast(NDArray[np.floating], X_train)
X_test = cast(NDArray[np.floating], X_test)
y_train = cast(NDArray[np.floating], y_train)
y_test = cast(NDArray[np.floating], y_test)
W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros((1, 128))
W2 = np.random.randn(128, 2) * 0.01
b2 = np.zeros((1, 2))
lr = 0.1
num_epochs = 10
batch_size = 32
n_samples = X_train.shape[0]
loss_history = []
for epoch in range(num_epochs):
    indices = np.random.permutation(n_samples)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        A1, A2 = forward_mlp(X_batch, W1, b1, W2, b2)
        Z1 = X_batch @ W1 + b1
        grad_W1, grad_b1, grad_W2, grad_b2 = backward_mlp(
            X_batch, A1, A2, y_batch, W1, W2, Z1
        )
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        W2 -= lr * grad_W2
        b2 -= lr * grad_b2
    _, A2_full = forward_mlp(X_train, W1, b1, W2, b2)
    loss = cross_entropy(A2_full, y_train)
    loss_history.append(loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
_, A2_test = forward_mlp(X_test, W1, b1, W2, b2)
predictions = np.argmax(A2_test, axis=1)
true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == true_labels)
print("Test Accuracy:", accuracy)
