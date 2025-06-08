import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable, Tuple, List, cast
from neural_network import (
    gradient_descent,
    binary_cross_entropy,
    sigmoid,
    normalize,
    mse_loss,
)


def gradient_descent_no_shuffle(
    X, y, W, b, lr, num_epochs, batch_size, loss_fn, activation_fn=lambda x: x
):
    n_samples = X.shape[0]
    loss_history = []
    for epoch in range(num_epochs):
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            batch_size_actual = X_batch.shape[0]
            Z_batch = X_batch @ W + b
            y_pred_batch = activation_fn(Z_batch)
            error = y_pred_batch - y_batch
            grad_W = (X_batch.T @ error) / batch_size_actual
            grad_b = np.mean(error)
            W = W - lr * grad_W
            b = b - lr * grad_b
        y_pred_full = activation_fn(X @ W + b)
        loss = loss_fn(y_pred_full, y)
        loss_history.append(loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    return W, b, loss_history


X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
y = np.array([[2.0], [4.0], [6.0], [8.0], [10.0], [12.0], [14.0], [16.0]])
W_init = np.array([[0.0]])
b_init = np.array([[0.0]])
lr = 0.01
num_epochs = 10
batch_size = 2
W_final, b_final, losses = gradient_descent_no_shuffle(
    X, y, W_init, b_init, lr, num_epochs, batch_size, mse_loss
)
print("Initial weight W:", W_init)
print("Initial bias b:", b_init)
print("Final weight W:", W_final)
print("Final bias b:", b_final)
print("Loss history:", losses)
