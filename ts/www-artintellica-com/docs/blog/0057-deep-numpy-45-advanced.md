+++
title = "Learn Deep Learning with NumPy, Part 4.5: Advanced Optimization and Capstone"
author = "Artintellica"
date = "2025-06-09"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0057-deep-numpy-45-advanced"
+++

## Introduction

Welcome back to our blog series, _"Learn Deep Learning with NumPy"_! In Part
4.4, we tackled overfitting by implementing L2 regularization and dropout,
enhancing the generalization of our deep models on MNIST. Now, in Part 4.5, we
reach a pivotal moment in Module 4 as we introduce _advanced optimization_ with
momentum-based gradient descent and present a capstone project. This post
culminates our deep learning journey by training a final model—either a 3-layer
MLP or a simple CNN—on MNIST, aiming for ~90% accuracy, while visualizing
performance and learned features.

By the end of this post, you’ll understand the mathematics of momentum
optimization, implement `momentum_update()` and `accuracy()` in NumPy, and train
a polished deep learning model as a capstone, showcasing everything we’ve built.
We’ll also visualize filters (for CNNs) or performance metrics using
`matplotlib`. Let’s dive into the math and code for advanced optimization and
our capstone project!

---

## Why Advanced Optimization Matters in Deep Learning

Standard gradient descent, as used in previous posts, updates parameters based
solely on the current gradient, which can lead to slow convergence or
oscillations, especially in deep networks with complex loss landscapes.
_Momentum-based gradient descent_ addresses this by incorporating a "velocity"
term that accumulates past gradients, smoothing updates and accelerating
convergence along relevant directions while dampening oscillations.

In deep learning, advanced optimization is crucial because:

- It speeds up training by navigating the loss surface more efficiently.
- It stabilizes updates, helping avoid local minima or saddle points.
- It improves performance on datasets like MNIST, where fine-tuning can push
  accuracy higher.

In this post, we’ll implement momentum optimization and apply it to train a
final model—either a 3-layer MLP or a simple CNN—on MNIST as our capstone
project, targeting ~90% test accuracy. We’ll also compute accuracy explicitly
and visualize results. Let’s explore the math behind momentum and our capstone
setup.

---

## Mathematical Foundations: Momentum-Based Gradient Descent and Accuracy

### Momentum-Based Gradient Descent

Momentum optimization enhances gradient descent by maintaining a moving average
of past gradients, called velocity, to guide parameter updates. For a parameter
$W$, gradient $\nabla L$, learning rate $\eta$, and momentum coefficient $\mu$
(e.g., 0.9), the update rules are:

$$
v = \mu v - \eta \nabla L
$$

$$
W \leftarrow W + v
$$

Where:

- $v$ is the velocity (initialized to 0), accumulating past gradients scaled by
  $\mu$ (momentum, typically 0.9 for 90% past influence).
- $-\eta \nabla L$ is the current gradient step, scaled by the learning rate.
- The update $W + v$ incorporates both past and current gradient information,
  smoothing the optimization path.

This method accelerates convergence along consistent gradient directions (like a
ball rolling downhill with momentum) and reduces oscillations in conflicting
directions, often leading to faster and more stable training than vanilla
gradient descent.

### Accuracy Metric

To evaluate model performance, we compute classification accuracy as the
fraction of correct predictions:

$$
\text{acc} = \frac{\text{number of correct predictions}}{\text{total predictions}}
$$

For multi-class tasks like MNIST (10 digits), we predict the class with the
highest probability from the softmax output and compare it to the true label,
aggregating correct matches over the dataset. Now, let’s implement momentum
optimization and accuracy computation in NumPy and apply them to our capstone
project.

---

## Implementing Advanced Optimization and Capstone with NumPy

We’ll create `momentum_update()` to apply momentum-based gradient descent and
`accuracy()` to evaluate model performance. As our capstone, we’ll train a
3-layer MLP (784 → 256 → 128 → 10) on MNIST, incorporating momentum, L2
regularization, and dropout from previous posts, targeting ~90% test accuracy.
We’ll visualize loss and accuracy over epochs with `matplotlib`.

### Momentum Update Implementation

Here’s the implementation of momentum-based updates for gradient descent:

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, List

def momentum_update(velocity: NDArray[np.floating], gradient: NDArray[np.floating], mu: float, lr: float) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
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
```

### Accuracy Metric Implementation

Here’s the implementation of accuracy computation for multi-class
classification:

```python
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
```

### Capstone Example: Training a 3-Layer MLP on MNIST with Momentum

As our capstone project, let’s train a 3-layer MLP (784 → 256 → 128 → 10) on
MNIST using momentum-based gradient descent, L2 regularization, and dropout.
We’ll visualize loss and accuracy over epochs using `matplotlib`, targeting ~90%
test accuracy.

**Note**: Ensure you have `sklearn` (`pip install scikit-learn`) for loading
MNIST and `matplotlib` (`pip install matplotlib`) for plotting. We’ll use a
subset of MNIST for CPU efficiency.

```python
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_network import normalize, relu, softmax, cross_entropy, forward_mlp_3layer, backward_mlp_3layer, l2_regularization, dropout, accuracy

# Load MNIST data (subset for faster training on CPU)
print("Loading MNIST data...")
X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
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
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

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
dropout_p = 0.8   # Keep probability for dropout
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
            X_batch, A1_drop, A2_drop, A3, y_batch, W1, W2, W3, Z1, Z2)

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
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plot loss and accuracy history
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.plot(range(1, num_epochs + 1), loss_history, label='Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.set_title('Training Loss Over Epochs')
ax1.legend()
ax1.grid(True)

ax2.plot(range(1, num_epochs + 1), accuracy_history, label='Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Test Accuracy Over Epochs')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Final evaluation on test set
_, _, A3_test = forward_mlp_3layer(X_test, W1, b1, W2, b2, W3, b3)
final_accuracy = accuracy(A3_test, y_test)
print("Final Test Accuracy:", final_accuracy)
```

**Output** (approximate, values will vary due to randomness and subset size):

```
Loading MNIST data...
Starting training...
Epoch 1/20, Loss: 2.3012, Test Accuracy: 0.1150
Epoch 2/20, Loss: 2.2876, Test Accuracy: 0.1380
...
Epoch 10/20, Loss: 0.6013, Test Accuracy: 0.8520
...
Epoch 20/20, Loss: 0.2985, Test Accuracy: 0.9040
Final Test Accuracy: 0.9040
```

**Loss and Accuracy Plots**: (Two `matplotlib` plots will display, showing
training loss decreasing over epochs from ~2.3 to below 0.3, and test accuracy
increasing to ~90%, demonstrating successful learning with momentum
optimization.)

In this capstone example, we train a 3-layer MLP (784 → 256 → 128 → 10) on a
subset of MNIST (5000 samples for CPU efficiency) over 20 epochs with a batch
size of 64. Using momentum-based gradient descent (`mu=0.9`), L2 regularization
(`lambda_=0.01`), and dropout (`p=0.8`), we achieve ~90% test accuracy, a
significant milestone. Momentum often accelerates convergence compared to
standard gradient descent, as seen in smoother loss reduction. Training time
remains manageable (~3-5 minutes on CPU).

---

## Organizing Our Growing Library

Let’s update our `neural_network.py` file to include the `momentum_update()` and
`accuracy()` functions alongside our previous implementations. These complete
our toolkit for advanced optimization and evaluation of deep learning models.

```python
# neural_network.py
import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable, Tuple, List, Dict
from scipy import signal

def normalize(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Normalize the input array X by subtracting the mean and dividing by the standard deviation.
    Parameters:
        X (NDArray[np.floating]): Input array to normalize. Should be a numerical array
            (float or compatible type).
    Returns:
        NDArray[np.floating]: Normalized array with mean approximately 0 and standard
            deviation approximately 1 along each axis.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    normalized_X = X - mean  # Start with mean subtraction
    mask = std != 0  # Create a mask for non-zero std
    if np.any(mask):
        normalized_X[:, mask] = normalized_X[:, mask] / std[mask]
    return normalized_X

def matrix_multiply(X: NDArray[np.floating], W: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Perform matrix multiplication between two arrays.
    Args:
        X: First input array/matrix of shape (m, n) with floating-point values
        W: Second input array/matrix of shape (n, p) with floating-point values
    Returns:
        Result of matrix multiplication, shape (m, p) with floating-point values
    """
    return np.matmul(X, W)

def sigmoid(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the sigmoid activation function element-wise.
    Args:
        Z: Input array of any shape with floating-point values
    Returns:
        Array of the same shape with sigmoid applied element-wise, values in [0, 1]
    """
    return 1 / (1 + np.exp(-Z))

def relu(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the ReLU activation function element-wise.
    Args:
        Z: Input array of any shape with floating-point values
    Returns:
        Array of the same shape with ReLU applied element-wise, max(0, Z)
    """
    return np.maximum(0, Z)

def softmax(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the softmax activation function row-wise.
    Args:
        Z: Input array of shape (n_samples, n_classes) with floating-point values
    Returns:
        Array of the same shape with softmax applied row-wise, probabilities summing to 1 per row
    """
    Z_max = np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z - Z_max)
    sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
    return exp_Z / sum_exp_Z

def mse_loss(y_pred: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """
    Compute the Mean Squared Error loss between predicted and true values.
    Args:
        y_pred: Predicted values, array of shape (n,) or (n,1) with floating-point values
        y: True values, array of shape (n,) or (n,1) with floating-point values
    Returns:
        Mean squared error as a single float
    """
    return np.mean((y_pred - y) ** 2)

def binary_cross_entropy(A: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """
    Compute the Binary Cross-Entropy loss between predicted probabilities and true labels.
    Args:
        A: Predicted probabilities (after sigmoid), array of shape (n,) or (n,1), values in [0, 1]
        y: True binary labels, array of shape (n,) or (n,1), values in {0, 1}
    Returns:
        Binary cross-entropy loss as a single float
    """
    epsilon = 1e-15
    return -np.mean(y * np.log(A + epsilon) + (1 - y) * np.log(1 - A + epsilon))

def cross_entropy(A: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """
    Compute categorical cross-entropy loss for multi-class classification.
    Args:
        A: Predicted probabilities after softmax, shape (n_samples, n_classes)
        y: True labels, one-hot encoded, shape (n_samples, n_classes)
    Returns:
        Cross-entropy loss as a single float
    """
    epsilon = 1e-15  # Small value to prevent log(0)
    return -np.mean(np.sum(y * np.log(A + epsilon), axis=1))

def gradient_descent(X: NDArray[np.floating], y: NDArray[np.floating], W: NDArray[np.floating],
                     b: NDArray[np.floating], lr: float, num_epochs: int, batch_size: int,
                     loss_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], float],
                     activation_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]] = lambda x: x) -> Tuple[NDArray[np.floating], NDArray[np.floating], List[float]]:
    """
    Perform mini-batch gradient descent to minimize loss.
    Args:
        X: Input data, shape (n_samples, n_features)
        y: True values, shape (n_samples, 1)
        W: Initial weights, shape (n_features, 1)
        b: Initial bias, shape (1,) or (1,1)
        lr: Learning rate, step size for updates
        num_epochs: Number of full passes through the dataset
        batch_size: Size of each mini-batch
        loss_fn: Loss function to compute error, e.g., mse_loss or binary_cross_entropy
        activation_fn: Activation function to apply to linear output (default: identity)
    Returns:
        Tuple of (updated W, updated b, list of loss values over epochs)
    """
    n_samples = X.shape[0]
    loss_history = []

    for epoch in range(num_epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
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

def numerical_gradient(X: NDArray[np.floating], y: NDArray[np.floating], params: Dict[str, NDArray[np.floating]],
                      loss_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], float],
                      forward_fn: Callable[[NDArray[np.floating], Dict[str, NDArray[np.floating]]], NDArray[np.floating]],
                      h: float = 1e-4) -> Dict[str, NDArray[np.floating]]:
    """
    Compute numerical gradients for parameters using central difference approximation.
    Args:
        X: Input data, shape (n_samples, n_features)
        y: True values, shape (n_samples, 1)
        params: Dictionary of parameters (e.g., {'W': ..., 'b': ...})
        loss_fn: Loss function to compute error, e.g., mse_loss
        forward_fn: Function to compute predictions from X and params
        h: Step size for finite difference approximation (default: 1e-4)
    Returns:
        Dictionary of numerical gradients for each parameter
    """
    num_grads = {}

    for param_name, param_value in params.items():
        num_grad = np.zeros_like(param_value)
        it = np.nditer(param_value, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            original_value = param_value[idx]

            param_value[idx] = original_value + h
            y_pred_plus = forward_fn(X, params)
            loss_plus = loss_fn(y_pred_plus, y)

            param_value[idx] = original_value - h
            y_pred_minus = forward_fn(X, params)
            loss_minus = loss_fn(y_pred_minus, y)

            num_grad[idx] = (loss_plus - loss_minus) / (2 * h)

            param_value[idx] = original_value
            it.iternext()

        num_grads[param_name] = num_grad

    return num_grads

def forward_perceptron(X: NDArray[np.floating], W: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.floating]:
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

def forward_mlp(X: NDArray[np.floating], W1: NDArray[np.floating], b1: NDArray[np.floating],
                W2: NDArray[np.floating], b2: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute the forward pass of a 2-layer MLP.
    Args:
        X: Input data, shape (n_samples, n_features, e.g., 784 for MNIST)
        W1: Weights for first layer, shape (n_features, n_hidden, e.g., 784x256)
        b1: Bias for first layer, shape (1, n_hidden)
        W2: Weights for second layer, shape (n_hidden, n_classes, e.g., 256x10)
        b2: Bias for second layer, shape (1, n_classes)
    Returns:
        Tuple of (A1, A2):
        - A1: Hidden layer output after ReLU, shape (n_samples, n_hidden)
        - A2: Output layer output after softmax, shape (n_samples, n_classes)
    """
    Z1 = X @ W1 + b1  # First layer linear combination
    A1 = relu(Z1)      # ReLU activation for hidden layer
    Z2 = A1 @ W2 + b2  # Second layer linear combination
    A2 = softmax(Z2)   # Softmax activation for output layer
    return A1, A2

def backward_mlp(X: NDArray[np.floating], A1: NDArray[np.floating], A2: NDArray[np.floating],
                y: NDArray[np.floating], W1: NDArray[np.floating], W2: NDArray[np.floating],
                Z1: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
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

def forward_mlp_3layer(X: NDArray[np.floating], W1: NDArray[np.floating], b1: NDArray[np.floating],
                       W2: NDArray[np.floating], b2: NDArray[np.floating],
                       W3: NDArray[np.floating], b3: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
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

def backward_mlp_3layer(X: NDArray[np.floating], A1: NDArray[np.floating], A2: NDArray[np.floating],
                        A3: NDArray[np.floating], y: NDArray[np.floating],
                        W1: NDArray[np.floating], W2: NDArray[np.floating], W3: NDArray[np.floating],
                        Z1: NDArray[np.floating], Z2: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
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

def conv2d(image: NDArray[np.floating], filter_kernel: NDArray[np.floating], stride: int = 1) -> NDArray[np.floating]:
    """
    Perform 2D convolution on an image using a filter kernel.
    Args:
        image: Input image, shape (height, width)
        filter_kernel: Convolution filter, shape (filter_height, filter_width)
        stride: Stride of the convolution operation (default: 1)
    Returns:
        Output feature map after convolution, shape depends on input, filter size, and stride
    """
    output = signal.convolve2d(image, filter_kernel, mode='valid', boundary='fill', fillvalue=0)
    if stride > 1:
        output = output[::stride, ::stride]
    return output

def max_pool(X: NDArray[np.floating], size: int = 2, stride: int = 2) -> NDArray[np.floating]:
    """
    Perform 2D max pooling on an input feature map.
    Args:
        X: Input feature map, shape (height, width) or (height, width, channels)
        size: Size of the pooling window (default: 2 for 2x2 pooling)
        stride: Stride of the pooling operation (default: 2)
    Returns:
        Output after max pooling, shape depends on input, size, and stride
    """
    if len(X.shape) == 2:
        height, width = X.shape
        channels = 1
        X = X.reshape(height, width, 1)
    else:
        height, width, channels = X.shape

    out_height = (height - size) // stride + 1
    out_width = (width - size) // stride + 1
    output = np.zeros((out_height, out_width, channels))

    for i in range(out_height):
        for j in range(out_width):
            x_start = i * stride
            x_end = x_start + size
            y_start = j * stride
            y_end = y_start + size
            region = X[x_start:x_end, y_start:y_end, :]
            output[i, j, :] = np.max(region, axis=(0, 1))

    if channels == 1:
        output = output[:, :, 0]
    return output

def l2_regularization(weights: List[NDArray[np.floating]], lambda_: float) -> Tuple[float, List[NDArray[np.floating]]]:
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
        l2_penalty += np.sum(W ** 2)
        l2_grads.append(2 * lambda_ * W)
    l2_penalty *= lambda_
    return l2_penalty, l2_grads

def dropout(A: NDArray[np.floating], p: float, training: bool = True) -> NDArray[np.floating]:
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

def momentum_update(velocity: NDArray[np.floating], gradient: NDArray[np.floating], mu: float, lr: float) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
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
```

You can now import these new functions using
`from neural_network import momentum_update, accuracy`. They complete our
toolkit for advanced optimization and evaluation, enhancing the training of deep
learning models like MLPs and CNNs.

---

## Exercises: Practice with Advanced Optimization and Capstone

To reinforce your understanding of momentum-based gradient descent and to
practice building a capstone deep learning model, try these Python-focused
coding exercises. They’ll help you build intuition for advanced optimization and
evaluate model performance. Run the code and compare outputs to verify your
solutions.

1. **Momentum Update on Synthetic Gradients**  
   Create a synthetic gradient `gradient = np.array([[1.0, 2.0], [3.0, 4.0]])`
   and initialize a velocity `velocity = np.zeros((2, 2))`. Apply
   `momentum_update()` with `mu=0.9` and `lr=0.1` for two consecutive updates
   (simulating two steps). Print the velocity and parameter update after each
   step to observe how momentum accumulates past gradients.

   ```python
   # Your code here
   gradient1 = np.array([[1.0, 2.0], [3.0, 4.0]])
   gradient2 = np.array([[0.5, 1.5], [2.5, 3.5]])
   velocity = np.zeros((2, 2))
   mu = 0.9
   lr = 0.1
   # First update
   velocity, update = momentum_update(velocity, gradient1, mu, lr)
   print("First Update - Velocity:\n", velocity)
   print("First Update - Parameter Update:\n", update)
   # Second update
   velocity, update = momentum_update(velocity, gradient2, mu, lr)
   print("Second Update - Velocity:\n", velocity)
   print("Second Update - Parameter Update:\n", update)
   ```

2. **Accuracy Calculation on Synthetic Predictions**  
   Create synthetic predictions
   `y_pred = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])` (3 samples, 2
   classes) and true labels `y_true = np.array([[1, 0], [0, 1], [1, 0]])`
   (one-hot encoded). Compute accuracy using `accuracy()` and verify it matches
   the fraction of correct predictions (should be 1.0 as all predictions match
   true labels). Then, test with class indices `y_true = np.array([0, 1, 0])` to
   confirm identical results.

   ```python
   # Your code here
   y_pred = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
   y_true_one_hot = np.array([[1, 0], [0, 1], [1, 0]])
   y_true_indices = np.array([0, 1, 0])
   acc_one_hot = accuracy(y_pred, y_true_one_hot)
   acc_indices = accuracy(y_pred, y_true_indices)
   print("Accuracy with one-hot labels:", acc_one_hot)
   print("Accuracy with index labels:", acc_indices)
   ```

3. **Momentum Update in Single Epoch Training**  
   Use a small synthetic dataset `X = np.random.randn(5, 3)` (5 samples, 3
   features) and one-hot labels
   `y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]])` (2 classes).
   Initialize a 3 → 4 → 2 MLP with small random weights and velocities as zeros.
   Perform one epoch of training with `forward_mlp()` and `backward_mlp()`,
   using `momentum_update()` (`mu=0.9`, `lr=0.1`). Print loss before and after
   the update to observe improvement.

   ```python
   # Your code here
   X = np.random.randn(5, 3)
   y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]])
   W1 = np.random.randn(3, 4) * 0.1
   b1 = np.zeros((1, 4))
   W2 = np.random.randn(4, 2) * 0.1
   b2 = np.zeros((1, 2))
   v_W1 = np.zeros_like(W1)
   v_b1 = np.zeros_like(b1)
   v_W2 = np.zeros_like(W2)
   v_b2 = np.zeros_like(b2)
   mu = 0.9
   lr = 0.1
   # Initial forward pass and loss
   Z1 = X @ W1 + b1
   A1 = relu(Z1)
   Z2 = A1 @ W2 + b2
   A2 = softmax(Z2)
   initial_loss = cross_entropy(A2, y)
   print("Initial Loss:", initial_loss)
   # Backpropagation
   delta2 = A2 - y
   grad_W2 = (A1.T @ delta2) / X.shape[0]
   grad_b2 = np.mean(delta2, axis=0, keepdims=True)
   delta1 = (delta2 @ W2.T) * (Z1 > 0)
   grad_W1 = (X.T @ delta1) / X.shape[0]
   grad_b1 = np.mean(delta1, axis=0, keepdims=True)
   # Momentum updates
   v_W1, update_W1 = momentum_update(v_W1, grad_W1, mu, lr)
   v_b1, update_b1 = momentum_update(v_b1, grad_b1, mu, lr)
   v_W2, update_W2 = momentum_update(v_W2, grad_W2, mu, lr)
   v_b2, update_b2 = momentum_update(v_b2, grad_b2, mu, lr)
   W1 += update_W1
   b1 += update_b1
   W2 += update_W2
   b2 += update_b2
   # Final forward pass and loss
   Z1 = X @ W1 + b1
   A1 = relu(Z1)
   Z2 = A1 @ W2 + b2
   A2 = softmax(Z2)
   final_loss = cross_entropy(A2, y)
   print("Final Loss after one update:", final_loss)
   ```

4. **Capstone Training with Momentum on MNIST Subset**  
   Load a subset of MNIST (e.g., 2000 samples) using
   `fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)`.
   Train a 3-layer MLP (784 → 256 → 128 → 10) for 15 epochs with `lr=0.1`,
   `mu=0.9`, `batch_size=64`, incorporating momentum, L2 regularization
   (`lambda_=0.01`), and dropout (`p=0.8`). Compare test accuracy and loss
   history to a model without momentum (standard gradient descent). Plot both
   loss and accuracy curves for comparison.

   ```python
   # Your code here
   from sklearn.datasets import fetch_openml
   from sklearn.model_selection import train_test_split
   import matplotlib.pyplot as plt
   X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
   X_full = X_full.astype(float)
   y_full = y_full.astype(int)
   X = X_full[:2000]
   y = y_full[:2000]
   n_classes = 10
   y_one_hot = np.zeros((y.shape[0], n_classes))
   y_one_hot[np.arange(y.shape[0]), y] = 1
   X = normalize(X)
   X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
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

   def train_mlp_with_momentum(use_momentum: bool, W1, b1, W2, b2, W3, b3, v_W1, v_b1, v_W2, v_b2, v_W3, v_b3):
       W1_copy = W1.copy()
       b1_copy = b1.copy()
       W2_copy = W2.copy()
       b2_copy = b2.copy()
       W3_copy = W3.copy()
       b3_copy = b3.copy()
       v_W1_copy = v_W1.copy() if use_momentum else None
       v_b1_copy = v_b1.copy() if use_momentum else None
       v_W2_copy = v_W2.copy() if use_momentum else None
       v_b2_copy = v_b2.copy() if use_momentum else None
       v_W3_copy = v_W3.copy() if use_momentum else None
       v_b3_copy = v_b3.copy() if use_momentum else None
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
               A1, A2, A3 = forward_mlp_3layer(X_batch, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy)
               A1_drop = dropout(A1, dropout_p, training=True)
               A2_drop = dropout(A2, dropout_p, training=True)
               Z1 = X_batch @ W1_copy + b1_copy
               Z2 = A1_drop @ W2_copy + b2_copy
               grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = backward_mlp_3layer(
                   X_batch, A1_drop, A2_drop, A3, y_batch, W1_copy, W2_copy, W3_copy, Z1, Z2)
               l2_penalty, l2_grads = l2_regularization([W1_copy, W2_copy, W3_copy], lambda_l2)
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
           _, _, A3_full = forward_mlp_3layer(X_train, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy)
           loss = cross_entropy(A3_full, y_train)
           loss_history.append(loss)
           _, _, A3_test = forward_mlp_3layer(X_test, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy)
           test_accuracy = accuracy(A3_test, y_test)
           accuracy_history.append(test_accuracy)
           print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Accuracy: {test_accuracy:.4f} (Momentum: {use_momentum})")
       return loss_history, accuracy_history

   print("Training with Momentum...")
   loss_history_with_momentum, accuracy_history_with_momentum = train_mlp_with_momentum(
       True, W1, b1, W2, b2, W3, b3, v_W1, v_b1, v_W2, v_b2, v_W3, v_b3)

   print("Training without Momentum...")
   loss_history_no_momentum, accuracy_history_no_momentum = train_mlp_with_momentum(
       False, W1, b1, W2, b2, W3, b3, v_W1, v_b1, v_W2, v_b2, v_W3, v_b3)

   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
   ax1.plot(range(1, num_epochs + 1), loss_history_with_momentum, label='Training Loss (With Momentum)')
   ax1.plot(range(1, num_epochs + 1), loss_history_no_momentum, label='Training Loss (No Momentum)')
   ax1.set_xlabel('Epoch')
   ax1.set_ylabel('Cross-Entropy Loss')
   ax1.set_title('Training Loss Over Epochs')
   ax1.legend()
   ax1.grid(True)

   ax2.plot(range(1, num_epochs + 1), accuracy_history_with_momentum, label='Test Accuracy (With Momentum)')
   ax2.plot(range(1, num_epochs + 1), accuracy_history_no_momentum, label='Test Accuracy (No Momentum)')
   ax2.set_xlabel('Epoch')
   ax2.set_ylabel('Accuracy')
   ax2.set_title('Test Accuracy Over Epochs')
   ax2.legend()
   ax2.grid(True)

   plt.tight_layout()
   plt.show()
   ```

These exercises will help you build intuition for implementing momentum-based
gradient descent, evaluating model performance with accuracy, and observing the
benefits of advanced optimization in training deep models like MLPs on datasets
such as MNIST.

---

## Closing Thoughts

Congratulations on reaching the capstone of our deep learning journey with
advanced optimization! In this post, we’ve explored the mathematics of
momentum-based gradient descent, implemented `momentum_update()` and
`accuracy()` in NumPy, and trained a 3-layer MLP on MNIST as our capstone
project, achieving ~90% test accuracy with the aid of momentum, L2
regularization, and dropout. We’ve visualized loss and accuracy over epochs,
showcasing the power of our complete toolkit.

This marks the culmination of our technical exploration in Module 4. In the next
and final post of the series (Part 4.6: _Series Conclusion and Future
Directions_), we’ll reflect on our journey, summarize key learnings, and discuss
potential future paths for expanding your deep learning skills beyond NumPy.

Until then, experiment with the code and exercises above. If you have questions
or want to share your solutions, drop a comment below—I’m excited to hear from
you. Let’s wrap up our deep learning toolkit together!

**Next Up**: Series Conclusion and Future Directions
