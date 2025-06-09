import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, Dict, cast
from neural_network import forward_mlp_3layer, relu, softmax
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_network import (
    normalize,
    relu,
    softmax,
    cross_entropy,
    forward_mlp,
    backward_mlp,
    forward_mlp_3layer,
    backward_mlp_3layer,
)

# Your code here
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X_full = X_full.astype(float)
y_full = y_full.astype(int)
X = X_full[:1000]
y = y_full[:1000]
n_classes = 10
y_one_hot = np.zeros((y.shape[0], n_classes))
y_one_hot[np.arange(y.shape[0]), y] = 1
X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
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
num_epochs = 100
batch_size = 32
n_samples = X_train.shape[0]
loss_history = []
for epoch in range(num_epochs):
   indices = np.random.permutation(n_samples)
   X_shuffled = X_train[indices]
   y_shuffled = y_train[indices]
   if epoch == 0:  # Only print gradients for first epoch
       print("Gradient magnitudes in first epoch:")
   for start_idx in range(0, n_samples, batch_size):
       end_idx = min(start_idx + batch_size, n_samples)
       X_batch = X_shuffled[start_idx:end_idx]
       y_batch = y_shuffled[start_idx:end_idx]
       A1, A2, A3 = forward_mlp_3layer(X_batch, W1, b1, W2, b2, W3, b3)
       Z1 = X_batch @ W1 + b1
       Z2 = A1 @ W2 + b2
       grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = backward_mlp_3layer(
           X_batch, A1, A2, A3, y_batch, W1, W2, W3, Z1, Z2)
       if epoch == 0 and start_idx == 0:  # Print for first batch of first epoch
           print(f"Avg |grad_W1| (first layer): {np.mean(np.abs(grad_W1)):.6f}")
           print(f"Avg |grad_W2| (second layer): {np.mean(np.abs(grad_W2)):.6f}")
           print(f"Avg |grad_W3| (third layer): {np.mean(np.abs(grad_W3)):.6f}")
       W1 -= lr * grad_W1
       b1 -= lr * grad_b1
       W2 -= lr * grad_W2
       b2 -= lr * grad_b2
       W3 -= lr * grad_W3
       b3 -= lr * grad_b3
   _, _, A3_full = forward_mlp_3layer(X_train, W1, b1, W2, b2, W3, b3)
   loss = cross_entropy(A3_full, y_train)
   loss_history.append(loss)
   print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
_, _, A3_test = forward_mlp_3layer(X_test, W1, b1, W2, b2, W3, b3)
predictions = np.argmax(A3_test, axis=1)
true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == true_labels)
print("Test Accuracy:", accuracy)
