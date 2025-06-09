+++
title = "Learn Deep Learning with NumPy, Part 4.2: Convolutional Layers for CNNs"
author = "Artintellica"
date = "2025-06-09"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0054-deep-numpy-42-convolutional"
+++

## Introduction

Welcome back to our blog series, _"Learn Deep Learning with NumPy"_! In Part
4.1, we extended our Multi-Layer Perceptron (MLP) to a deeper 3-layer
architecture and explored challenges like vanishing gradients in training deep
networks. Now, in Part 4.2, we’re shifting focus to a transformative component
of modern deep learning: _Convolutional Neural Networks (CNNs)_. We’ll start by
implementing convolutional layers, the core building block of CNNs, which are
particularly powerful for processing image data like MNIST.

By the end of this post, you’ll understand the mathematics of convolution,
implement a `conv2d()` function using `scipy.signal.convolve2d`, and apply it to
MNIST images to extract feature maps with a 3x3 filter. This marks our first
step into CNNs, enabling spatial feature extraction that outperforms fully
connected layers for visual data. Let’s dive into the math and code for
convolutional layers!

---

## Why Convolutional Layers Matter in Deep Learning

Fully connected layers, as in MLPs, treat every input feature (e.g., pixel)
independently, ignoring spatial relationships in data like images. This leads to
a massive number of parameters for large inputs and a loss of local pattern
information. _Convolutional layers_, the foundation of CNNs, address this by
applying small, shared filters (or kernels) across the input, capturing local
patterns like edges or textures while drastically reducing parameter count
through weight sharing.

In deep learning, convolutional layers are crucial because:

- They preserve spatial structure by operating on local regions of the input.
- They reduce parameters via shared weights, making training more efficient.
- They learn hierarchical features (e.g., edges in early layers, complex shapes
  in deeper layers), excelling at tasks like image classification.

In this post, we’ll implement a basic convolutional layer for 2D inputs, apply
it to MNIST images (28x28), and generate feature maps using a 3x3 filter. This
sets the stage for full CNNs in future posts. Let’s explore the math behind
convolution.

---

## Mathematical Foundations: Convolution in 2D

Convolution is a mathematical operation that slides a small filter (or kernel)
over an input image, computing a weighted sum of local regions to produce a
feature map. For a 2D input image $I$ (shape $H \times W$) and a filter $F$
(shape $f_h \times f_w$, e.g., 3x3), the output at position $(i, j)$ of the
feature map is:

$$
\text{out}[i, j] = \sum_{m=0}^{f_h-1} \sum_{n=0}^{f_w-1} I[i+m, j+n] \cdot F[m, n]
$$

Where:

- $i$ and $j$ are the top-left coordinates of the current filter position on the
  image.
- $m$ and $n$ iterate over the filter’s dimensions.
- The output size depends on the input size, filter size, stride (step size of
  filter sliding), and padding (optional border zeros).

### Stride and Output Size

- **Stride ($s$)**: The number of pixels the filter shifts each step. A stride
  of 1 moves one pixel at a time; a stride of 2 skips every other pixel.
- **Output Size**: For input height $H$, filter height $f_h$, and stride $s$,
  the output height is:
  $$
  H_{\text{out}} = \lfloor \frac{H - f_h}{s} \rfloor + 1
  $$
  Similarly for width. No padding is assumed here (valid convolution); padding
  will be covered in later posts.

### Intuition

Convolution acts like a sliding window, applying the same filter weights to
every local patch of the image, detecting patterns (e.g., edges if the filter is
an edge detector). Multiple filters produce multiple feature maps, each
highlighting different aspects of the input. Now, let’s implement a 2D
convolution operation in NumPy using `scipy.signal.convolve2d` for efficiency.

---

## Implementing Convolutional Layers with NumPy

We’ll create a `conv2d()` function to perform 2D convolution on image data,
leveraging `scipy.signal.convolve2d` for optimized computation. We’ll apply it
to MNIST images (28x28) using a 3x3 filter to generate feature maps,
demonstrating spatial feature extraction.

### 2D Convolution Implementation

Here’s the implementation of `conv2d()` for a single filter, supporting stride
and valid padding (no border zeros):

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union
from scipy import signal

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
    # Use scipy.signal.convolve2d with 'valid' mode (no padding)
    # 'valid' mode means output size is reduced based on filter size
    output = signal.convolve2d(image, filter_kernel, mode='valid', boundary='fill', fillvalue=0)

    # Apply stride by downsampling the output
    if stride > 1:
        output = output[::stride, ::stride]

    return output
```

### Example: Applying Convolution to MNIST Images

Let’s apply our `conv2d()` function to MNIST images (28x28) using a 3x3 filter
to detect simple features. We’ll visualize the input image and output feature
map using `matplotlib`.

**Note**: Ensure you have `sklearn` (`pip install scikit-learn`) for loading
MNIST and `matplotlib` (`pip install matplotlib`) for visualization. We’ll use a
single image for simplicity.

```python
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load MNIST data (single image for simplicity)
print("Loading MNIST data...")
X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X_full = X_full.astype(float)
image = X_full[0].reshape(28, 28)  # First image, shape (28, 28)
label = y_full[0]

# Define a simple 3x3 filter (e.g., edge detection)
filter_kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])  # Sobel-like filter for edges

# Apply convolution with stride=1
feature_map = conv2d(image, filter_kernel, stride=1)

# Visualize input image and output feature map
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title(f"Input Image (Digit: {label})")
ax1.axis('off')

ax2.imshow(feature_map, cmap='gray')
ax2.set_title("Feature Map (Edge Detection)")
ax2.axis('off')

plt.tight_layout()
plt.show()

print("Input Image Shape:", image.shape)
print("Feature Map Shape:", feature_map.shape)
```

**Output** (approximate, shapes are exact):

```
Loading MNIST data...
Input Image Shape: (28, 28)
Feature Map Shape: (26, 26)
```

**Visualization**: (Two `matplotlib` plots will display: the input MNIST digit
image and the output feature map after convolution with a 3x3 edge-detection
filter. The feature map highlights areas of intensity change, like digit
boundaries.)

In this example, we apply a 3x3 Sobel-like filter to a single MNIST image
(28x28), producing a feature map of shape (26, 26) due to 'valid' convolution
(no padding) with stride=1. The output size is reduced because the filter can’t
fully cover the edges without padding. The visualization shows the filter
detecting edges or sharp changes in pixel intensity, a basic form of feature
extraction crucial for CNNs.

---

## Organizing Our Growing Library

Let’s update our `neural_network.py` file to include the `conv2d()` function
alongside our previous implementations. This function will be a building block
for constructing full CNNs in future posts.

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
```

You can now import this new function using `from neural_network import conv2d`.
It serves as a foundational component for building Convolutional Neural Networks
(CNNs) in future posts, enabling spatial feature extraction from image data.

---

## Exercises: Practice with Convolutional Layers

To reinforce your understanding of convolutional layers and their application to
image data, try these Python-focused coding exercises. They’ll help you build
intuition for how convolution extracts features and the impact of parameters
like stride. Run the code and compare outputs to verify your solutions.

1. **Convolution with a Simple Filter on Synthetic Image**  
   Create a small synthetic image
   `image = np.array([[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]])`
   (4x4). Define a 2x2 averaging filter
   `filter_kernel = np.array([[0.25, 0.25], [0.25, 0.25]])`. Apply `conv2d()`
   with `stride=1` and print the input image and output feature map. Verify the
   output size and that it smooths the input values.

   ```python
   # Your code here
   image = np.array([[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]])
   filter_kernel = np.array([[0.25, 0.25], [0.25, 0.25]])
   feature_map = conv2d(image, filter_kernel, stride=1)
   print("Input Image (4x4):\n", image)
   print("Output Feature Map:\n", feature_map)
   print("Output Shape:", feature_map.shape)
   ```

2. **Effect of Stride on Output Size**  
   Using the same image and filter from Exercise 1, apply `conv2d()` with
   `stride=2`. Print the input image and output feature map. Verify that the
   output size is reduced compared to `stride=1` and note how stride affects the
   result.

   ```python
   # Your code here
   image = np.array([[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]])
   filter_kernel = np.array([[0.25, 0.25], [0.25, 0.25]])
   feature_map = conv2d(image, filter_kernel, stride=2)
   print("Input Image (4x4):\n", image)
   print("Output Feature Map (stride=2):\n", feature_map)
   print("Output Shape:", feature_map.shape)
   ```

3. **Convolution on MNIST with Edge Detection Filter**  
   Load a single MNIST image (e.g., first image) using
   `fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)`.
   Reshape it to 28x28. Define a 3x3 horizontal edge detection filter
   `filter_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])`. Apply
   `conv2d()` with `stride=1` and visualize the input image and output feature
   map using `matplotlib`. Observe where the filter highlights horizontal edges.

   ```python
   # Your code here
   from sklearn.datasets import fetch_openml
   import matplotlib.pyplot as plt
   X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
   X_full = X_full.astype(float)
   image = X_full[0].reshape(28, 28)
   label = y_full[0]
   filter_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
   feature_map = conv2d(image, filter_kernel, stride=1)
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
   ax1.imshow(image, cmap='gray')
   ax1.set_title(f"Input Image (Digit: {label})")
   ax1.axis('off')
   ax2.imshow(feature_map, cmap='gray')
   ax2.set_title("Feature Map (Horizontal Edges)")
   ax2.axis('off')
   plt.tight_layout()
   plt.show()
   print("Input Image Shape:", image.shape)
   print("Feature Map Shape:", feature_map.shape)
   ```

4. **Multiple Filters on a Single MNIST Image**  
   Load a single MNIST image as in Exercise 3. Define two 3x3 filters: one for
   horizontal edges `[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]` and one for vertical
   edges `[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]`. Apply `conv2d()` with each
   filter (stride=1) and visualize the input image and both feature maps.
   Observe how different filters extract different spatial features.

   ```python
   # Your code here
   from sklearn.datasets import fetch_openml
   import matplotlib.pyplot as plt
   X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
   X_full = X_full.astype(float)
   image = X_full[0].reshape(28, 28)
   label = y_full[0]
   horizontal_filter = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
   vertical_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
   horizontal_map = conv2d(image, horizontal_filter, stride=1)
   vertical_map = conv2d(image, vertical_filter, stride=1)
   fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
   ax1.imshow(image, cmap='gray')
   ax1.set_title(f"Input Image (Digit: {label})")
   ax1.axis('off')
   ax2.imshow(horizontal_map, cmap='gray')
   ax2.set_title("Feature Map (Horizontal Edges)")
   ax2.axis('off')
   ax3.imshow(vertical_map, cmap='gray')
   ax3.set_title("Feature Map (Vertical Edges)")
   ax3.axis('off')
   plt.tight_layout()
   plt.show()
   print("Input Image Shape:", image.shape)
   print("Horizontal Feature Map Shape:", horizontal_map.shape)
   print("Vertical Feature Map Shape:", vertical_map.shape)
   ```

These exercises will help you build intuition for how convolutional layers work,
the effect of filter design and stride on output feature maps, and their role in
extracting spatial features from images.

---

## Closing Thoughts

Congratulations on implementing your first convolutional layer with `conv2d()`!
In this post, we’ve explored the mathematics of 2D convolution, built a reusable
function using `scipy.signal.convolve2d`, and applied it to MNIST images to
extract feature maps with a 3x3 filter. This marks a pivotal shift from fully
connected MLPs to spatially aware CNNs, setting the stage for more powerful
image processing models.

In the next chapter (Part 4.3: _Pooling Layers and CNN Architecture_), we’ll
complement convolutional layers with pooling layers to reduce spatial dimensions
and build a basic CNN architecture for MNIST classification, further improving
efficiency and performance.

Until then, experiment with the code and exercises above. If you have questions
or want to share your solutions, drop a comment below—I’m excited to hear from
you. Let’s keep building our deep learning toolkit together!

**Next Up**: Part 4.3 – Pooling Layers and CNN Architecture
