+++
title = "Learn Deep Learning with NumPy, Part 3.2: Activation Functions for Neural Networks"
author = "Artintellica"
date = "2025-06-09"
+++

## Introduction

Welcome back to our blog series, _"Learn Deep Learning with NumPy"_! In Part
3.1, we introduced neural networks with a single-layer perceptron, building on
logistic regression to classify data using the sigmoid activation function.
While sigmoid is useful for binary classification, it’s just one of many
activation functions that power neural networks. In Part 3.2, we’ll dive deeper
into activation functions, implementing two crucial ones—ReLU (Rectified Linear
Unit) and softmax—that enable neural networks to learn complex, non-linear
patterns.

By the end of this post, you’ll understand why activation functions are
essential for introducing non-linearity, implement `relu()` and `softmax()` in
NumPy, and see them applied to synthetic data and MNIST-like outputs. These
functions will be key components of our toolkit, reused in multi-layer
perceptrons (MLPs) and beyond. Let’s explore the math and code behind activation
functions!

---

## Why Activation Functions Matter in Neural Networks

Activation functions are the heart of neural networks, transforming linear
combinations of inputs and weights into non-linear outputs. Without non-linear
activation functions, stacking multiple linear layers (like `X @ W + b`) would
still result in a linear model, incapable of solving complex problems like image
classification or natural language processing. Non-linearity allows neural
networks to learn intricate patterns and decision boundaries.

In this post, we’ll focus on two widely used activation functions:

- **ReLU (Rectified Linear Unit)**: Commonly used in hidden layers, it
  introduces sparsity and helps mitigate vanishing gradient issues during
  training.
- **Softmax**: Used in the output layer for multi-class classification, it
  converts raw scores into probabilities that sum to 1.

We’ll also briefly touch on their derivatives, which are critical for computing
gradients during backpropagation (covered in Part 3.4). Let’s dive into the
mathematics of these functions.

---

## Mathematical Foundations: ReLU and Softmax

### ReLU (Rectified Linear Unit)

ReLU is a simple yet powerful activation function that outputs the input if it’s
positive and 0 otherwise. For a scalar input $z$, it is defined as:

$$
f(z) = \max(0, z)
$$

Applied element-wise to a matrix or vector $Z$, it sets all negative values to 0
while preserving positive values. ReLU is popular in hidden layers because:

- It introduces non-linearity.
- It’s computationally efficient (just a threshold operation).
- It helps prevent vanishing gradients by not squashing positive values (unlike
  sigmoid, which compresses outputs to [0, 1]).

The derivative of ReLU, used in backpropagation, is also straightforward:

$$
\frac{\partial \text{ReLU}}{\partial z} = \begin{cases}
1 & \text{if } z > 0 \\
0 & \text{otherwise}
\end{cases}
$$

### Softmax

Softmax is used in the output layer for multi-class classification, converting
raw scores (logits) into probabilities that sum to 1. For a vector $z$ of length
$k$ (representing scores for $k$ classes), the softmax output for the $i$-th
element is:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^k e^{z_j}}
$$

Applied to a matrix $Z$ of shape $(n, k)$ (e.g., $n$ samples, $k$ classes),
softmax operates row-wise, ensuring each row’s outputs sum to 1. This makes it
ideal for interpreting outputs as class probabilities (e.g., for MNIST with 10
digits). The derivative of softmax is more complex due to its dependency on all
elements, but we’ll cover it in detail during backpropagation.

Now, let’s implement these activation functions in NumPy and see them in action
with examples.

---

## Implementing Activation Functions with NumPy

We’ll create reusable `relu()` and `softmax()` functions, adding them to our
deep learning toolkit. These will be essential for building multi-layer neural
networks in upcoming posts.

### ReLU Implementation

Here’s the implementation of ReLU using NumPy’s `maximum()` function for
efficiency:

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union

def relu(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the ReLU activation function element-wise.
    Args:
        Z: Input array of any shape with floating-point values
    Returns:
        Array of the same shape with ReLU applied element-wise, max(0, Z)
    """
    return np.maximum(0, Z)
```

### Softmax Implementation

Here’s the implementation of softmax, ensuring numerical stability by
subtracting the maximum value to prevent overflow in the exponential operation:

```python
def softmax(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the softmax activation function row-wise.
    Args:
        Z: Input array of shape (n_samples, n_classes) with floating-point values
    Returns:
        Array of the same shape with softmax applied row-wise, probabilities summing to 1 per row
    """
    # Subtract the max for numerical stability (avoid overflow in exp)
    Z_max = np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z - Z_max)
    sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
    return exp_Z / sum_exp_Z
```

### Example: Applying ReLU and Softmax

Let’s test these activation functions on synthetic data. We’ll apply ReLU to a
small matrix and softmax to a matrix simulating MNIST classification outputs (4
samples, 10 classes).

```python
# Example 1: Applying ReLU to a 3x2 matrix
Z_relu = np.array([[-1.0, 2.0],
                   [0.0, -3.0],
                   [4.0, -0.5]])
A_relu = relu(Z_relu)
print("Input for ReLU (3x2):\n", Z_relu)
print("Output after ReLU (3x2):\n", A_relu)

# Example 2: Applying Softmax to a 4x10 matrix (simulating MNIST outputs)
Z_softmax = np.random.randn(4, 10)  # Random scores for 4 samples, 10 classes
A_softmax = softmax(Z_softmax)
print("\nInput for Softmax (4x10, first few columns):\n", Z_softmax[:, :3])
print("Output after Softmax (4x10, first few columns):\n", A_softmax[:, :3])
print("Sum of probabilities per sample (should be ~1):\n", np.sum(A_softmax, axis=1))
```

**Output** (approximate, softmax values will vary due to randomness):

```
Input for ReLU (3x2):
 [[-1.   2. ]
  [ 0.  -3. ]
  [ 4.  -0.5]]
Output after ReLU (3x2):
 [[0. 2.]
  [0. 0.]
  [4. 0.]]

Input for Softmax (4x10, first few columns):
 [[ 0.123  -0.456  0.789]
  [-1.234  0.567  -0.321]
  [ 0.987  -0.654  1.234]
  [-0.111  0.222  -0.333]]
Output after Softmax (4x10, first few columns):
 [[0.134  0.075  0.262]
  [0.032  0.195  0.080]
  [0.247  0.048  0.316]
  [0.098  0.137  0.079]]
Sum of probabilities per sample (should be ~1):
 [1. 1. 1. 1.]
```

In the ReLU example, negative values are set to 0, while positive values remain
unchanged, demonstrating its thresholding behavior. In the softmax example, raw
scores are converted to probabilities summing to 1 for each sample, suitable for
multi-class classification like MNIST (10 digits). The subtraction of the
maximum value (`Z_max`) in `softmax()` ensures numerical stability by preventing
overflow in `np.exp()`.

---

## Organizing Our Growing Library

Let’s update our `neural_network.py` file to include the `relu()` and
`softmax()` functions alongside our previous implementations. These activation
functions will be critical for building multi-layer neural networks in upcoming
posts.

```python
# neural_network.py
import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable, Tuple, List, Dict

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
```

You can now import these new functions using
`from neural_network import relu, softmax`. They will be essential for building
and training multi-layer neural networks in the upcoming posts.

---

## Exercises: Practice with Activation Functions

To reinforce your understanding of activation functions, try these
Python-focused coding exercises. They’ll prepare you for using non-linear
activations in neural networks. Run the code and compare outputs to verify your
solutions.

1. **Applying ReLU to Synthetic Data**  
   Create a 4x3 matrix
   `Z = np.array([[-1.5, 2.0, 0.0], [-0.5, -2.0, 1.5], [3.0, -1.0, 0.5], [-3.0, 4.0, -0.2]])`.
   Apply `relu()` to it and print the input and output. Verify that all negative
   values are set to 0 while positive values remain unchanged.

   ```python
   # Your code here
   Z = np.array([[-1.5, 2.0, 0.0], [-0.5, -2.0, 1.5], [3.0, -1.0, 0.5], [-3.0, 4.0, -0.2]])
   A_relu = relu(Z)
   print("Input Z (4x3):\n", Z)
   print("Output after ReLU (4x3):\n", A_relu)
   ```

2. **Applying Softmax to Synthetic Classification Data**  
   Create a 3x5 matrix `Z = np.random.randn(3, 5)` to simulate scores for 3
   samples across 5 classes. Apply `softmax()` to it and print the input,
   output, and the sum of probabilities per sample (should be ~1). Verify that
   the output values are between 0 and 1 and sum to 1 per row.

   ```python
   # Your code here
   Z = np.random.randn(3, 5)
   A_softmax = softmax(Z)
   sums = np.sum(A_softmax, axis=1)
   print("Input Z (3x5):\n", Z)
   print("Output after Softmax (3x5):\n", A_softmax)
   print("Sum of probabilities per sample (should be ~1):\n", sums)
   ```

3. **ReLU on Larger Matrix with Mixed Values**  
   Create a 5x4 matrix `Z` with a mix of positive, negative, and zero values
   (e.g., using `np.random.uniform(-5, 5, (5, 4))`). Apply `relu()` and count
   the number of elements set to 0 (negative or zero inputs). Print the input,
   output, and count.

   ```python
   # Your code here
   Z = np.random.uniform(-5, 5, (5, 4))
   A_relu = relu(Z)
   zero_count = np.sum(A_relu == 0)
   print("Input Z (5x4):\n", Z)
   print("Output after ReLU (5x4):\n", A_relu)
   print("Number of elements set to 0:", zero_count)
   ```

4. **Softmax Numerical Stability Test**  
   Create a 2x3 matrix
   `Z = np.array([[1000, 1000, 1000], [-1000, -1000, -1000]])` with extreme
   values that could cause overflow in `np.exp()`. Apply `softmax()` and print
   the output. Observe how subtracting the maximum value in our implementation
   prevents overflow and still produces valid probabilities summing to 1.

   ```python
   # Your code here
   Z = np.array([[1000, 1000, 1000], [-1000, -1000, -1000]])
   A_softmax = softmax(Z)
   sums = np.sum(A_softmax, axis=1)
   print("Input Z (2x3, extreme values):\n", Z)
   print("Output after Softmax (2x3):\n", A_softmax)
   print("Sum of probabilities per sample (should be ~1):\n", sums)
   ```

These exercises will help you build intuition for how ReLU and softmax
activation functions work, their behavior with different input ranges, and their
role in introducing non-linearity or probabilistic outputs in neural networks.

---

## Closing Thoughts

Congratulations on mastering ReLU and softmax activation functions, essential
tools for building powerful neural networks! In this post, we’ve explored why
non-linear activations are critical, implemented `relu()` and `softmax()` in
NumPy, and applied them to synthetic data simulating real-world scenarios like
MNIST classification. These functions expand our toolkit, preparing us for
deeper architectures.

In the next chapter (Part 3.3: _Multi-Layer Perceptrons and Forward
Propagation_), we’ll combine multiple layers with activations like ReLU to build
MLPs capable of solving non-linear problems, moving beyond the limitations of
single-layer perceptrons. We’ll apply this to MNIST for digit classification.

Until then, experiment with the code and exercises above. If you have questions
or want to share your solutions, drop a comment below—I’m excited to hear from
you. Let’s keep building our deep learning toolkit together!

**Next Up**: Part 3.3 – Multi-Layer Perceptrons and Forward Propagation
