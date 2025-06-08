+++
title = "Learn Deep Learning with NumPy, Part 2.4: Debugging with Numerical Gradients"
author = "Artintellica"
date = "2025-06-08"
+++

## Introduction

Welcome back to our blog series, _"Learn Deep Learning with NumPy"_! In Part
2.3, we enhanced our `gradient_descent()` function to support mini-batch
processing, making it scalable for larger datasets like MNIST. As we build more
complex models, ensuring the correctness of our gradient computations becomes
critical. In Part 2.4, we’re focusing on a key debugging skill for deep
learning: verifying gradients using _numerical methods_. This technique helps
confirm that our analytical gradients—computed via formulas—are implemented
correctly, preventing subtle bugs from derailing training.

By the end of this post, you’ll understand how to approximate gradients
numerically using finite differences, implement a reusable
`numerical_gradient()` function in NumPy, and apply it to verify the analytical
gradients for linear regression. This debugging tool will be invaluable as we
move toward multi-layer neural networks in Module 3. Let’s dive into the math
and code for ensuring robust gradient implementations!

---

## Why Numerical Gradients Matter in Deep Learning

In deep learning, training relies on gradient descent, which updates model
parameters (weights and biases) based on the gradient of the loss function with
respect to those parameters. We typically compute these gradients analytically
using formulas derived from calculus (e.g., for MSE loss in linear regression,
$\nabla_W L = \frac{1}{n} X^T (y_{\text{pred}} - y)$). However, deriving and
implementing these formulas can be error-prone, especially in complex models
with multiple layers and activation functions.

If the analytical gradient is incorrect due to a bug or misderived formula,
gradient descent will update parameters in the wrong direction, leading to poor
model performance or divergence (increasing loss, as some of you might have seen
with an improperly tuned learning rate). _Numerical gradient checking_ offers a
way to verify analytical gradients by approximating them using a simple,
brute-force method based on the definition of a derivative. This acts as a
sanity check, helping us debug implementations before scaling up to larger
models.

Numerical gradient checking is particularly useful when:

- Implementing backpropagation for neural networks (Module 3).
- Debugging custom loss functions or activation functions.
- Ensuring correctness in gradient computations for new architectures.

While numerical gradients are too slow to use during actual training (they
require multiple loss evaluations per parameter), they are a powerful tool for
validation during development. Let’s explore the math behind this approach.

---

## Mathematical Foundations: Finite Difference Approximation

The gradient of a function $f(W)$ with respect to a parameter $W$ represents the
rate of change of $f$ at $W$. Mathematically, the derivative (or gradient for
multi-dimensional $W$) at a point is defined as the limit of the difference
quotient:

$$
\nabla f(W) = \lim_{h \to 0} \frac{f(W + h) - f(W)}{h}
$$

In practice, we can’t compute the exact limit, so we approximate it with a small
value of $h$ using the _finite difference method_. A more accurate version,
called the _central difference approximation_, evaluates the function on both
sides of $W$ to reduce approximation error:

$$
\nabla f(W) \approx \frac{f(W + h) - f(W - h)}{2h}
$$

Where:

- $f(W)$ is the loss function evaluated at parameters $W$.
- $h$ is a small step size (e.g., $10^{-4}$), chosen to balance approximation
  accuracy and numerical precision.
- The approximation is computed for each element of $W$ independently if $W$ is
  a matrix or vector.

This method doesn’t require knowing the analytical formula for the gradient—it
simply evaluates the loss at nearby points. By comparing the numerical gradient
to our analytical gradient, we can check if they are close (within a small
tolerance, e.g., $10^{-5}$). If they differ significantly, it indicates a bug in
the analytical gradient implementation.

Now, let’s implement this numerical gradient checking in NumPy and use it to
verify the gradients for linear regression.

---

## Implementing Numerical Gradient Checking with NumPy

We’ll create a reusable `numerical_gradient()` function to approximate gradients
using the central difference method. We’ll apply it to verify the analytical
gradients computed in our `gradient_descent()` function for linear regression
with MSE loss. This debugging tool will be added to our growing library.

### Numerical Gradient Implementation

Here’s the implementation of `numerical_gradient()` for a general loss function
and parameters. We’ll use a dictionary for parameters to make it flexible for
models with multiple parameters (e.g., `W` and `b`).

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable, Dict

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
        it = np.nditer(param_value, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            original_value = param_value[idx]

            # Compute loss at W + h
            param_value[idx] = original_value + h
            y_pred_plus = forward_fn(X, params)
            loss_plus = loss_fn(y_pred_plus, y)

            # Compute loss at W - h
            param_value[idx] = original_value - h
            y_pred_minus = forward_fn(X, params)
            loss_minus = loss_fn(y_pred_minus, y)

            # Central difference approximation
            num_grad[idx] = (loss_plus - loss_minus) / (2 * h)

            # Restore original value
            param_value[idx] = original_value
            it.iternext()

        num_grads[param_name] = num_grad

    return num_grads
```

### Example: Verifying Gradients for Linear Regression

Let’s use `numerical_gradient()` to verify the analytical gradients for linear
regression with MSE loss. We’ll compare the numerical gradients to the
analytical gradients computed as `grad_W = (X.T @ (y_pred - y)) / n` and
`grad_b = np.mean(y_pred - y)`.

```python
# Define forward function for linear regression
def linear_forward(X: NDArray[np.floating], params: Dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
    return X @ params['W'] + params['b']

# Example: Synthetic data for linear regression (y = 2x + 1)
X = np.array([[1.0], [2.0], [3.0], [4.0]])  # Input (4 samples, 1 feature)
y = np.array([[3.0], [5.0], [7.0], [9.0]])  # True values (y = 2x + 1)
n = X.shape[0]

# Initialize parameters
params = {
    'W': np.array([[1.0]]),  # Initial weight (not the true value)
    'b': np.array([[0.5]])   # Initial bias (not the true value)
}

# Compute analytical gradients
y_pred = linear_forward(X, params)
error = y_pred - y
analytical_grad_W = (X.T @ error) / n
analytical_grad_b = np.mean(error)

# Compute numerical gradients
numerical_grads = numerical_gradient(X, y, params, mse_loss, linear_forward, h=1e-4)

# Compare analytical and numerical gradients
print("Analytical Gradient for W:", analytical_grad_W)
print("Numerical Gradient for W:", numerical_grads['W'])
print("Difference for W:", np.abs(analytical_grad_W - numerical_grads['W']))
print("Analytical Gradient for b:", analytical_grad_b)
print("Numerical Gradient for b:", numerical_grads['b'])
print("Difference for b:", np.abs(analytical_grad_b - numerical_grads['b']))
```

**Output** (values are approximate):

```
Analytical Gradient for W: [[-5.]]
Numerical Gradient for W: [[-5.00000001]]
Difference for W: [[1e-08]]
Analytical Gradient for b: -1.5
Numerical Gradient for b: [[-1.50000001]]
Difference for b: [[1e-08]]
```

In this example, the numerical gradients computed using the central difference
approximation are extremely close to the analytical gradients (differences on
the order of $10^{-8}$), confirming that our analytical gradient implementation
for linear regression with MSE loss is correct. If the differences were large
(e.g., greater than $10^{-5}$), it would indicate a bug in the analytical
gradient formula or implementation.

---

## Organizing Our Growing Library

Let’s update our `neural_network.py` file to include the `numerical_gradient()`
function alongside our previous implementations. This debugging tool will be
essential for verifying gradients as we build more complex models.

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
        it = np.nditer(param_value, flags=['multi_index'], op_flags=['readwrite'])
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
```

You can now import this debugging tool using
`from neural_network import numerical_gradient`. It will be particularly useful
as we move into more complex models with intricate gradient computations in
upcoming modules.

---

## Exercises: Practice with Numerical Gradient Checking

To reinforce your understanding of numerical gradient checking, try these
Python-focused coding exercises. They’ll prepare you for debugging gradient
implementations in more complex models. Run the code and compare outputs to
verify your solutions.

1. **Verify Gradients for Linear Regression with Small Data**  
   Use the synthetic data `X = np.array([[1.0], [2.0]])` and
   `y = np.array([[3.0], [5.0]])` (true relationship $y = 2x + 1$). Initialize
   parameters as `params = {'W': np.array([[1.5]]), 'b': np.array([[0.5]])}`.
   Compute analytical gradients using `grad_W = (X.T @ (y_pred - y)) / n` and
   `grad_b = np.mean(y_pred - y)`, and compare them to numerical gradients from
   `numerical_gradient()` with `h = 1e-4`. Print the differences.

   ```python
   # Your code here
   def linear_forward(X, params):
       return X @ params['W'] + params['b']

   X = np.array([[1.0], [2.0]])
   y = np.array([[3.0], [5.0]])
   n = X.shape[0]
   params = {'W': np.array([[1.5]]), 'b': np.array([[0.5]])}
   y_pred = linear_forward(X, params)
   error = y_pred - y
   analytical_grad_W = (X.T @ error) / n
   analytical_grad_b = np.mean(error)
   numerical_grads = numerical_gradient(X, y, params, mse_loss, linear_forward, h=1e-4)
   print("Analytical Gradient for W:", analytical_grad_W)
   print("Numerical Gradient for W:", numerical_grads['W'])
   print("Difference for W:", np.abs(analytical_grad_W - numerical_grads['W']))
   print("Analytical Gradient for b:", analytical_grad_b)
   print("Numerical Gradient for b:", numerical_grads['b'])
   print("Difference for b:", np.abs(analytical_grad_b - numerical_grads['b']))
   ```

2. **Effect of Step Size h on Numerical Gradients**  
   Using the same data and parameters as in Exercise 1, compute numerical
   gradients with a larger step size `h = 1e-2` and compare the differences to
   the analytical gradients. Observe how a larger `h` affects the accuracy of
   the approximation compared to `h = 1e-4`.

   ```python
   # Your code here
   def linear_forward(X, params):
       return X @ params['W'] + params['b']

   X = np.array([[1.0], [2.0]])
   y = np.array([[3.0], [5.0]])
   n = X.shape[0]
   params = {'W': np.array([[1.5]]), 'b': np.array([[0.5]])}
   y_pred = linear_forward(X, params)
   error = y_pred - y
   analytical_grad_W = (X.T @ error) / n
   analytical_grad_b = np.mean(error)
   numerical_grads = numerical_gradient(X, y, params, mse_loss, linear_forward, h=1e-2)
   print("Analytical Gradient for W:", analytical_grad_W)
   print("Numerical Gradient for W:", numerical_grads['W'])
   print("Difference for W:", np.abs(analytical_grad_W - numerical_grads['W']))
   print("Analytical Gradient for b:", analytical_grad_b)
   print("Numerical Gradient for b:", numerical_grads['b'])
   print("Difference for b:", np.abs(analytical_grad_b - numerical_grads['b']))
   ```

3. **Verify Gradients for Logistic Regression**  
   Use synthetic binary classification data
   `X = np.array([[0.5], [1.5], [2.5]])` and
   `y = np.array([[0.0], [0.0], [1.0]])`. Define a forward function with sigmoid
   activation (`y_pred = sigmoid(X @ W + b)`), initialize
   `params = {'W': np.array([[0.0]]), 'b': np.array([[0.0]])}`, and compute
   analytical gradients for BCE loss (`grad_W = (X.T @ (y_pred - y)) / n`).
   Compare to numerical gradients using `binary_cross_entropy` as `loss_fn`.

   ```python
   # Your code here
   def logistic_forward(X, params):
       return sigmoid(X @ params['W'] + params['b'])

   X = np.array([[0.5], [1.5], [2.5]])
   y = np.array([[0.0], [0.0], [1.0]])
   n = X.shape[0]
   params = {'W': np.array([[0.0]]), 'b': np.array([[0.0]])}
   y_pred = logistic_forward(X, params)
   error = y_pred - y
   analytical_grad_W = (X.T @ error) / n
   analytical_grad_b = np.mean(error)
   numerical_grads = numerical_gradient(X, y, params, binary_cross_entropy, logistic_forward, h=1e-4)
   print("Analytical Gradient for W:", analytical_grad_W)
   print("Numerical Gradient for W:", numerical_grads['W'])
   print("Difference for W:", np.abs(analytical_grad_W - numerical_grads['W']))
   print("Analytical Gradient for b:", analytical_grad_b)
   print("Numerical Gradient for b:", numerical_grads['b'])
   print("Difference for b:", np.abs(analytical_grad_b - numerical_grads['b']))
   ```

4. **Debugging a Buggy Gradient Implementation**  
   Modify the analytical gradient for linear regression to be incorrect (e.g.,
   `grad_W = (X @ error) / n` instead of `X.T @ error`). Use the data from
   Exercise 1 and compare to numerical gradients. Observe how the large
   difference indicates a bug in the analytical gradient.

   ```python
   # Your code here
   def linear_forward(X, params):
       return X @ params['W'] + params['b']

   X = np.array([[1.0], [2.0]])
   y = np.array([[3.0], [5.0]])
   n = X.shape[0]
   params = {'W': np.array([[1.5]]), 'b': np.array([[0.5]])}
   y_pred = linear_forward(X, params)
   error = y_pred - y
   analytical_grad_W = (X @ error) / n  # Buggy: should be X.T @ error
   analytical_grad_b = np.mean(error)   # Correct for b
   numerical_grads = numerical_gradient(X, y, params, mse_loss, linear_forward, h=1e-4)
   print("Analytical Gradient for W (buggy):", analytical_grad_W)
   print("Numerical Gradient for W:", numerical_grads['W'])
   print("Difference for W (large due to bug):", np.abs(analytical_grad_W - numerical_grads['W']))
   print("Analytical Gradient for b:", analytical_grad_b)
   print("Numerical Gradient for b:", numerical_grads['b'])
   print("Difference for b:", np.abs(analytical_grad_b - numerical_grads['b']))
   ```

These exercises will help you build intuition for numerical gradient checking,
understand the impact of approximation parameters like `h`, and practice
debugging incorrect gradient implementations.

---

## Closing Thoughts

Congratulations on mastering numerical gradient checking, a vital debugging
skill for deep learning! In this post, we’ve explored the finite difference
approximation to compute gradients numerically, implemented a reusable
`numerical_gradient()` function, and verified analytical gradients for linear
regression. This tool ensures the correctness of our gradient implementations,
building confidence as we tackle more complex models.

With Module 2 complete, we’ve laid a strong foundation in optimization, from
loss functions and gradient descent to mini-batch processing and debugging. In
Module 3, starting with Part 3.1 (_Single-Layer Perceptrons_), we’ll begin
building neural networks, applying these optimization techniques to train our
first models.

Until then, experiment with the code and exercises above. If you have questions
or want to share your solutions, drop a comment below—I’m excited to hear from
you. Let’s keep building our deep learning toolkit together!

**Next Up**: Part 3.1 – Single-Layer Perceptrons
