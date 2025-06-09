+++
title = "Learn Deep Learning with NumPy, Part 4.3: Pooling and CNN Architecture"
author = "Artintellica"
date = "2025-06-09"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0055-deep-numpy-43-pooling"
+++

## Introduction

Welcome back to our blog series, _"Learn Deep Learning with NumPy"_! In Part
4.2, we implemented convolutional layers with `conv2d()`, enabling spatial
feature extraction from MNIST images using filters. Now, in Part 4.3, we’ll
build on this foundation by introducing _pooling layers_ and combining them with
convolutions and dense layers to construct a simple _Convolutional Neural
Network (CNN)_. This architecture will be more efficient and effective for image
classification than fully connected MLPs.

By the end of this post, you’ll understand the role of pooling in reducing
spatial dimensions, implement `max_pool()` for 2x2 max pooling, and build a
basic CNN with one convolutional layer (8 filters, 3x3), one max pooling layer,
and one dense layer to classify MNIST digits. We’ll reuse `conv2d()` and other
toolkit functions, completing a full CNN structure. Let’s dive into the math and
code for pooling and CNN architecture!

---

## Why Pooling and CNN Architecture Matter in Deep Learning

Convolutional layers, as seen in Part 4.2, extract local features like edges or
textures by applying filters across an image. However, the resulting feature
maps retain high spatial dimensions, leading to computational overhead and risk
of overfitting due to excessive detail. _Pooling layers_ address this by
downsampling feature maps, reducing their size while preserving important
information. This makes the network more efficient and helps it generalize
better by focusing on dominant features.

A typical _CNN architecture_ combines:

- **Convolutional Layers**: Extract spatial features.
- **Pooling Layers**: Reduce spatial dimensions, maintaining key patterns.
- **Dense Layers**: Perform final classification based on extracted features.

In deep learning, this structure is powerful because:

- It reduces the number of parameters and computations compared to fully
  connected layers.
- It maintains spatial hierarchy, learning low-level features (e.g., edges) in
  early layers and high-level features (e.g., shapes) in later layers.
- It achieves state-of-the-art performance on image tasks like MNIST
  classification.

In this post, we’ll implement max pooling and build a simple CNN for MNIST with
one convolutional layer (8 filters, 3x3), one max pooling layer (2x2), and one
dense layer for classification. Let’s explore the math behind pooling and CNNs.

---

## Mathematical Foundations: Max Pooling and CNN Structure

### Max Pooling

Max pooling is a downsampling technique that slides a window (e.g., 2x2) over
the input feature map and outputs the maximum value in each region. For an input
feature map $X$ of shape $(H, W)$ and a pooling window of size $s \times s$
(e.g., 2x2) with stride $s$, the output at position $(i, j)$ is:

$$
\text{out}[i, j] = \max_{m=0}^{s-1} \max_{n=0}^{s-1} X[i \cdot s + m, j \cdot s + n]
$$

Where:

- $i$ and $j$ index the output grid, scaled by stride $s$.
- $m$ and $n$ iterate over the pooling window.
- **Output Size**: For input height $H$ and pooling size $s$ with stride $s$,
  the output height is:
  $$
  H_{\text{out}} = \lfloor \frac{H}{s} \rfloor
  $$
  Similarly for width, assuming no padding.

Max pooling retains the strongest activations (e.g., brightest edges) in each
region, reducing spatial dimensions while preserving dominant features.

### CNN Structure

A simple CNN for MNIST might include:

1. **Convolutional Layer**: Apply multiple filters (e.g., 8 filters of 3x3) to
   the input image (28x28), producing feature maps (e.g., 26x26x8 with valid
   convolution).
2. **Pooling Layer**: Apply max pooling (e.g., 2x2 with stride 2) to each
   feature map, reducing size (e.g., 13x13x8).
3. **Dense Layer**: Flatten the pooled feature maps into a vector and pass
   through a fully connected layer with softmax for classification (e.g., 10
   classes for MNIST digits).

This structure reduces parameters compared to MLPs and leverages spatial
locality, making it ideal for images. Now, let’s implement max pooling and build
a simple CNN in NumPy.

---

## Implementing Pooling and CNN Architecture with NumPy

We’ll create a `max_pool()` function for 2x2 max pooling and build a simple CNN
architecture with one convolutional layer (using `conv2d()` from Part 4.2), one
max pooling layer, and one dense layer for MNIST classification. We’ll focus on
forward propagation for now, with full training to follow in later posts.

### Max Pooling Implementation

Here’s the implementation of `max_pool()` for 2D max pooling with a specified
window size and stride:

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union

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
```

### Example: Building a Simple CNN for MNIST

Let’s build a simple CNN with one convolutional layer (8 filters, 3x3), one max
pooling layer (2x2, stride=2), and one dense layer for MNIST classification.
We’ll apply it to a small batch of images, focusing on forward propagation (full
training with backpropagation will be covered in a later post).

**Note**: Ensure you have `sklearn` (`pip install scikit-learn`) for loading
MNIST and `matplotlib` (`pip install matplotlib`) for visualization. We’ll use a
small batch for CPU efficiency.

```python
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from neural_network import conv2d, softmax, normalize

# Load MNIST data (small batch for simplicity)
print("Loading MNIST data...")
X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X_full = X_full.astype(float)
y_full = y_full.astype(int)

# Use a small batch of 4 images
batch_size = 4
X_batch = X_full[:batch_size].reshape(batch_size, 28, 28)  # Shape (4, 28, 28)
labels = y_full[:batch_size]

# Define 8 filters of size 3x3 (random for demo)
n_filters = 8
filters = [np.random.randn(3, 3) * 0.1 for _ in range(n_filters)]

# Step 1: Convolutional Layer
feature_maps = np.zeros((batch_size, 26, 26, n_filters))  # Output shape after 3x3 valid conv
for i in range(batch_size):
    for f in range(n_filters):
        feature_maps[i, :, :, f] = conv2d(X_batch[i], filters[f], stride=1)

# Step 2: Max Pooling Layer (2x2, stride=2)
pooled_maps = np.zeros((batch_size, 13, 13, n_filters))  # Output shape after 2x2 pooling
for i in range(batch_size):
    for f in range(n_filters):
        pooled_maps[i, :, :, f] = max_pool(feature_maps[i, :, :, f], size=2, stride=2)

# Step 3: Flatten and Dense Layer (for demo, random weights)
n_flattened = 13 * 13 * n_filters  # 13x13x8 = 2197
flattened = pooled_maps.reshape(batch_size, n_flattened)
W_dense = np.random.randn(n_flattened, 10) * 0.01  # 10 classes for MNIST
b_dense = np.zeros((1, 10))
logits = flattened @ W_dense + b_dense
probs = softmax(logits)

# Visualize one input image and one feature map after pooling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(X_batch[0], cmap='gray')
ax1.set_title(f"Input Image (Digit: {labels[0]})")
ax1.axis('off')

ax2.imshow(pooled_maps[0, :, :, 0], cmap='gray')
ax2.set_title("Pooled Feature Map (Filter 1)")
ax2.axis('off')

plt.tight_layout()
plt.show()

print("Input Batch Shape:", X_batch.shape)
print("Feature Maps Shape (after conv):", feature_maps.shape)
print("Pooled Maps Shape (after pooling):", pooled_maps.shape)
print("Flattened Shape:", flattened.shape)
print("Output Probabilities Shape:", probs.shape)
print("Output Probabilities (first sample, first few classes):\n", probs[0, :3])
```

**Output** (approximate, shapes are exact):

```
Loading MNIST data...
Input Batch Shape: (4, 28, 28)
Feature Maps Shape (after conv): (4, 26, 26, 8)
Pooled Maps Shape (after pooling): (4, 13, 13, 8)
Flattened Shape: (4, 2197)
Output Probabilities Shape: (4, 10)
Output Probabilities (first sample, first few classes):
 [0.099 0.101 0.098]
```

**Visualization**: (Two `matplotlib` plots will display: the first input MNIST
digit image and the corresponding pooled feature map from the first filter after
2x2 max pooling. The pooled map shows reduced spatial dimensions while retaining
strong activations.)

In this example, we process a batch of 4 MNIST images (28x28) through a simple
CNN: a convolutional layer with 8 filters (3x3) produces feature maps (26x26x8),
max pooling (2x2, stride=2) reduces them to (13x13x8), and a dense layer maps
the flattened output (2197 features) to 10 class probabilities. This
demonstrates the CNN structure, though parameters are random (not trained).
Training time remains minimal for this small batch.

---

## Organizing Our Growing Library

Let’s update our `neural_network.py` file to include the `max_pool()` function
alongside our previous implementations. This function will be a key component
for building full CNNs in future posts.

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
```

You can now import this new function using
`from neural_network import max_pool`. Combined with `conv2d()`, it forms a
critical part of building Convolutional Neural Networks (CNNs) for image
processing tasks.

---

## Exercises: Practice with Pooling and CNN Architecture

To reinforce your understanding of pooling layers and CNN architecture, try
these Python-focused coding exercises. They’ll help you build intuition for
downsampling with max pooling and constructing a complete CNN pipeline. Run the
code and compare outputs to verify your solutions.

1. **Max Pooling on Synthetic Feature Map**  
   Create a synthetic feature map
   `X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])`
   (4x4). Apply `max_pool()` with `size=2` and `stride=2`. Print the input and
   output feature maps. Verify that the output size is (2x2) and contains the
   maximum values from each 2x2 region.

   ```python
   # Your code here
   X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
   pooled = max_pool(X, size=2, stride=2)
   print("Input Feature Map (4x4):\n", X)
   print("Output after Max Pooling (2x2):\n", pooled)
   print("Output Shape:", pooled.shape)
   ```

2. **Effect of Stride on Max Pooling Output**  
   Using the same feature map from Exercise 1, apply `max_pool()` with `size=2`
   but `stride=1`. Print the input and output feature maps. Verify that the
   output size is larger than with `stride=2` and note how overlapping windows
   affect the result.

   ```python
   # Your code here
   X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
   pooled = max_pool(X, size=2, stride=1)
   print("Input Feature Map (4x4):\n", X)
   print("Output after Max Pooling (stride=1):\n", pooled)
   print("Output Shape:", pooled.shape)
   ```

3. **Convolution and Pooling on MNIST Image**  
   Load a single MNIST image using
   `fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)`.
   Reshape it to 28x28. Apply `conv2d()` with a 3x3 edge detection filter
   `np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])`, then apply `max_pool()`
   with `size=2` and `stride=2`. Visualize the input image, convolved feature
   map, and pooled feature map using `matplotlib`. Observe the size reduction
   and feature retention.

   ```python
   # Your code here
   from sklearn.datasets import fetch_openml
   import matplotlib.pyplot as plt
   X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
   X_full = X_full.astype(float)
   image = X_full[0].reshape(28, 28)
   label = y_full[0]
   filter_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
   conv_map = conv2d(image, filter_kernel, stride=1)
   pooled_map = max_pool(conv_map, size=2, stride=2)
   fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
   ax1.imshow(image, cmap='gray')
   ax1.set_title(f"Input Image (Digit: {label})")
   ax1.axis('off')
   ax2.imshow(conv_map, cmap='gray')
   ax2.set_title("Convolved Feature Map")
   ax2.axis('off')
   ax3.imshow(pooled_map, cmap='gray')
   ax3.set_title("Pooled Feature Map")
   ax3.axis('off')
   plt.tight_layout()
   plt.show()
   print("Input Image Shape:", image.shape)
   print("Convolved Map Shape:", conv_map.shape)
   print("Pooled Map Shape:", pooled_map.shape)
   ```

4. **CNN Forward Pass on Small Batch with Multiple Filters**  
   Load a batch of 2 MNIST images, reshape to (2, 28, 28). Define 4 random 3x3
   filters. Apply `conv2d()` to each image with each filter (resulting in shape
   (2, 26, 26, 4)), then apply `max_pool()` with `size=2`, `stride=2` (resulting
   in shape (2, 13, 13, 4)). Flatten to (2, 13*13*4) and pass through a dense
   layer with random weights to 10 classes. Print shapes at each step to verify
   the CNN forward pass.

   ```python
   # Your code here
   from sklearn.datasets import fetch_openml
   X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
   X_full = X_full.astype(float)
   batch_size = 2
   X_batch = X_full[:batch_size].reshape(batch_size, 28, 28)
   labels = y_full[:batch_size]
   n_filters = 4
   filters = [np.random.randn(3, 3) * 0.1 for _ in range(n_filters)]
   feature_maps = np.zeros((batch_size, 26, 26, n_filters))
   for i in range(batch_size):
       for f in range(n_filters):
           feature_maps[i, :, :, f] = conv2d(X_batch[i], filters[f], stride=1)
   pooled_maps = np.zeros((batch_size, 13, 13, n_filters))
   for i in range(batch_size):
       for f in range(n_filters):
           pooled_maps[i, :, :, f] = max_pool(feature_maps[i, :, :, f], size=2, stride=2)
   n_flattened = 13 * 13 * n_filters
   flattened = pooled_maps.reshape(batch_size, n_flattened)
   W_dense = np.random.randn(n_flattened, 10) * 0.01
   b_dense = np.zeros((1, 10))
   logits = flattened @ W_dense + b_dense
   probs = softmax(logits)
   print("Input Batch Shape:", X_batch.shape)
   print("Feature Maps Shape (after conv):", feature_maps.shape)
   print("Pooled Maps Shape (after pooling):", pooled_maps.shape)
   print("Flattened Shape:", flattened.shape)
   print("Output Probabilities Shape:", probs.shape)
   ```

These exercises will help you build intuition for how pooling layers downsample
feature maps, how they integrate with convolutional layers in a CNN pipeline,
and their impact on spatial dimensions and feature retention.

---

## Closing Thoughts

Congratulations on implementing max pooling and constructing a simple CNN
architecture! In this post, we’ve explored the mathematics of max pooling, built
`max_pool()` for 2x2 downsampling, and combined it with `conv2d()` and a dense
layer to form a basic CNN for MNIST. This architecture leverages spatial
hierarchies, making it more efficient and effective for image data than MLPs.

In the next chapter (Part 4.4: _Training CNNs with Backpropagation_), we’ll
implement backpropagation for CNNs, including gradients for convolutional and
pooling layers, and train our simple CNN on MNIST to achieve high accuracy,
completing our foundational deep learning journey.

Until then, experiment with the code and exercises above. If you have questions
or want to share your solutions, drop a comment below—I’m excited to hear from
you. Let’s keep building our deep learning toolkit together!

**Next Up**: Part 4.4 – Training CNNs with Backpropagation
