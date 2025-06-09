+++
title = "Learn Deep Learning with NumPy, Part 3.1: Single-Layer Perceptrons"
author = "Artintellica"
date = "2025-06-05"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0049-deep-numpy-31-perceptrons"
+++

## Introduction

Welcome back to our blog series, _"Learn Deep Learning with NumPy"_! Having
completed Module 2, where we mastered optimization techniques like gradient
descent, mini-batch processing, and gradient debugging, we’re now stepping into
Module 3: _Basic Neural Networks_. In Part 3.1, we’ll introduce the concept of
neural networks with the simplest form—a _single-layer perceptron_. This model
serves as a bridge from logistic regression (which we explored in Part 2.3) to
more complex neural architectures.

By the end of this post, you’ll understand the structure and mathematics of a
single-layer perceptron, implement one using NumPy for a classic problem (the
XOR gate), and reuse our existing toolkit functions like `sigmoid()` and
`gradient_descent()`. This marks our first step into neural networks, setting
the stage for multi-layer models in upcoming chapters. Let’s dive into the math
and code to build our first neural network!

---

## Why Single-Layer Perceptrons Matter in Deep Learning

A single-layer perceptron is one of the earliest neural network models, dating
back to the 1950s, and it forms the foundation for understanding more complex
architectures. It extends logistic regression by introducing the concept of a
_neuron_—a computational unit that processes inputs through weights, a bias, and
an activation function to produce an output. While limited in capability (it can
only solve linearly separable problems), the perceptron introduces key ideas
like forward propagation and parameter updates via gradients, which are central
to all neural networks.

In deep learning, perceptrons are building blocks:

- They mimic biological neurons, taking multiple inputs, weighting them, summing
  them with a bias, and applying a non-linear activation.
- They can be trained with gradient descent, just like logistic regression, to
  classify data.
- Stacking multiple perceptrons or layers overcomes their limitations, leading
  to powerful models (which we’ll explore in later parts).

In this post, we’ll build a single-layer perceptron for binary classification,
focusing on the XOR problem—a classic task that, while not linearly separable,
helps illustrate the model’s behavior and limitations. Let’s explore the math
behind this fundamental model.

---

## Mathematical Foundations: Single-Layer Perceptron

A single-layer perceptron processes input data through a simple pipeline to
produce an output. For an input matrix $X$ of shape $(n, d)$ (where $n$ is the
number of samples and $d$ is the number of features), the perceptron computes:

1. **Linear Combination**:

   $$
   Z = XW + b
   $$

   Where:

   - $W$ is the weight matrix of shape $(d, 1)$ (one output neuron).
   - $b$ is the bias term, a scalar or shape $(1, 1)$, broadcasted to match
     $Z$’s shape.
   - $Z$ is the pre-activation output of shape $(n, 1)$.

2. **Activation Function**:

   $$
   A = \sigma(Z)
   $$

   Where $\sigma(Z)$ is typically the sigmoid function,
   $\sigma(z) = \frac{1}{1 + e^{-z}}$, applied element-wise to map $Z$ to
   probabilities between 0 and 1. This makes the output interpretable for binary
   classification.

3. **Loss Function**: For binary classification, we use Binary Cross-Entropy
   (BCE) loss to measure error between predictions $A$ and true labels $y$
   (values 0 or 1):

   $$
   L = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(A_i) + (1 - y_i) \log(1 - A_i) \right]
   $$

4. **Gradients for Training**: The gradients of BCE loss with respect to $W$ and
   $b$ are derived using the chain rule. Since the derivative of sigmoid
   simplifies nicely, the gradients are:
   $$
   \nabla_W L = \frac{1}{n} X^T (A - y)
   $$
   $$
   \nabla_b L = \frac{1}{n} \sum_{i=1}^n (A_i - y_i)
   $$
   These gradients are used in gradient descent to update $W$ and $b$:
   $W \leftarrow W - \eta \nabla_W L$, $b \leftarrow b - \eta \nabla_b L$, where
   $\eta$ is the learning rate.

This structure—linear combination, activation, loss, and gradient
updates—mirrors logistic regression but introduces the neural network
terminology and framework. Now, let’s implement a single-layer perceptron in
NumPy and train it on the XOR problem.

---

## Implementing a Single-Layer Perceptron with NumPy

We’ll create a simple perceptron model for binary classification, reusing our
`sigmoid()` and `gradient_descent()` functions from previous posts. We’ll test
it on the XOR problem, a classic task where inputs are binary pairs (e.g.,
`[0, 0]`, `[0, 1]`, `[1, 0]`, `[1, 1]`) and outputs are 0 or 1 based on whether
the inputs differ (output 1 for `[0, 1]` and `[1, 0]`, 0 otherwise). Note that
XOR is not linearly separable, so a single-layer perceptron will struggle to
solve it perfectly, but this exercise illustrates the model’s behavior.

### Forward Pass for Perceptron

Let’s define a forward pass function for the perceptron, computing the output
from inputs and parameters.

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union, Dict

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

def sigmoid(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the sigmoid activation function element-wise.
    Args:
        Z: Input array of any shape with floating-point values
    Returns:
        Array of the same shape with sigmoid applied element-wise, values in [0, 1]
    """
    return 1 / (1 + np.exp(-Z))
```

### Example: Training a Perceptron on XOR Data

Now, let’s train the perceptron on the XOR problem using our existing
`gradient_descent()` function, adapted for logistic regression with sigmoid
activation and binary cross-entropy loss.

```python
from neural_network import gradient_descent, binary_cross_entropy

# XOR data: inputs and outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input (4 samples, 2 features)
y = np.array([[0], [1], [1], [0]])  # Output (XOR: 1 if inputs differ, 0 otherwise)

# Initialize parameters
n_features = X.shape[1]
W_init = np.zeros((n_features, 1))  # Initial weights
b_init = np.zeros((1, 1))  # Initial bias
lr = 0.1  # Learning rate
num_epochs = 1000  # Number of epochs (high to attempt convergence)
batch_size = 4  # Full batch since dataset is small

# Train perceptron using gradient_descent with sigmoid activation
W_final, b_final, losses = gradient_descent(
    X, y, W_init, b_init, lr, num_epochs, batch_size,
    loss_fn=binary_cross_entropy, activation_fn=sigmoid
)

# Evaluate the model
A = forward_perceptron(X, W_final, b_final)
predictions = (A > 0.5).astype(int)
accuracy = np.mean(predictions == y)
print("Final Predictions (probabilities):\n", A)
print("Final Predictions (binary):\n", predictions)
print("True Labels:\n", y)
print("Accuracy:", accuracy)
print("Final Loss:", losses[-1])
print("Loss History (first 5 and last 5):", losses[:5] + losses[-5:])
```

**Output** (approximate, may vary due to randomness):

```
Final Predictions (probabilities):
 [[0.5       ]
  [0.5       ]
  [0.5       ]
  [0.5       ]]
Final Predictions (binary):
 [[0]
  [0]
  [0]
  [0]]
True Labels:
 [[0]
  [1]
  [1]
  [0]]
Accuracy: 0.5
Final Loss: 0.6931
Loss History (first 5 and last 5): [0.69314718, 0.69314718, 0.69314718, 0.69314718, 0.69314718, 0.69314718, 0.69314718, 0.69314718, 0.69314718, 0.69314718]
```

In this example, we train a single-layer perceptron on the XOR problem. Notice
that the accuracy is around 50%, and predictions hover near 0.5, indicating the
model fails to learn the XOR pattern effectively. This is expected because XOR
is not linearly separable—a single-layer perceptron cannot draw a decision
boundary to separate the classes perfectly. The loss remains near 0.6931 (ln(2),
the BCE loss for random guessing), showing no significant improvement. This
limitation highlights why we need multi-layer neural networks (coming in Part
3.3) to solve non-linear problems like XOR.

---

## Organizing Our Growing Library

Let’s update our `neural_network.py` file to include the `forward_perceptron()`
function alongside our previous implementations. This function will be a
building block for neural network structures.

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

You can now import this new function using
`from neural_network import forward_perceptron`. It reuses `sigmoid()` and
integrates with `gradient_descent()`, forming the basis for neural network
structures we’ll expand in future posts.

---

## Exercises: Practice with Single-Layer Perceptrons

To reinforce your understanding of single-layer perceptrons, try these
Python-focused coding exercises. They’ll prepare you for building more complex
neural networks in future chapters. Run the code and compare outputs to verify
your solutions.

1. **Train Perceptron on Linearly Separable Data (AND Gate)**  
   Use synthetic data for the AND gate:
   `X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])` and
   `y = np.array([[0], [0], [0], [1]])` (output 1 only if both inputs are 1).
   Initialize `W = np.zeros((2, 1))` and `b = np.zeros((1, 1))`. Train with
   `gradient_descent()` using `lr = 0.1`, `num_epochs = 500`, `batch_size = 4`,
   `loss_fn=binary_cross_entropy`, and `activation_fn=sigmoid`. Print final
   predictions and accuracy.

   ```python
   # Your code here
   X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
   y = np.array([[0], [0], [0], [1]])
   W_init = np.zeros((2, 1))
   b_init = np.zeros((1, 1))
   lr = 0.1
   num_epochs = 500
   batch_size = 4
   W_final, b_final, losses = gradient_descent(X, y, W_init, b_init, lr, num_epochs, batch_size, binary_cross_entropy, sigmoid)
   A = forward_perceptron(X, W_final, b_final)
   predictions = (A > 0.5).astype(int)
   accuracy = np.mean(predictions == y)
   print("Final Predictions (probabilities):\n", A)
   print("Final Predictions (binary):\n", predictions)
   print("True Labels:\n", y)
   print("Accuracy:", accuracy)
   print("Final Loss:", losses[-1])
   ```

2. **Train Perceptron on OR Gate Data**  
   Use synthetic data for the OR gate:
   `X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])` and
   `y = np.array([[0], [1], [1], [1]])` (output 1 if at least one input is 1).
   Initialize and train as in Exercise 1. Print final predictions and accuracy.
   Observe if the perceptron can learn this linearly separable pattern.

   ```python
   # Your code here
   X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
   y = np.array([[0], [1], [1], [1]])
   W_init = np.zeros((2, 1))
   b_init = np.zeros((1, 1))
   lr = 0.1
   num_epochs = 500
   batch_size = 4
   W_final, b_final, losses = gradient_descent(X, y, W_init, b_init, lr, num_epochs, batch_size, binary_cross_entropy, sigmoid)
   A = forward_perceptron(X, W_final, b_final)
   predictions = (A > 0.5).astype(int)
   accuracy = np.mean(predictions == y)
   print("Final Predictions (probabilities):\n", A)
   print("Final Predictions (binary):\n", predictions)
   print("True Labels:\n", y)
   print("Accuracy:", accuracy)
   print("Final Loss:", losses[-1])
   ```

3. **Effect of Learning Rate on XOR Training**  
   Use the XOR data from the example
   (`X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])`,
   `y = np.array([[0], [1], [1], [0]])`). Train with a higher learning rate
   `lr = 1.0` (instead of 0.1), keeping `num_epochs = 1000` and
   `batch_size = 4`. Compare the final loss and accuracy to the example. Observe
   if a higher learning rate helps or hinders learning XOR.

   ```python
   # Your code here
   X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
   y = np.array([[0], [1], [1], [0]])
   W_init = np.zeros((2, 1))
   b_init = np.zeros((1, 1))
   lr = 1.0  # Higher learning rate
   num_epochs = 1000
   batch_size = 4
   W_final, b_final, losses = gradient_descent(X, y, W_init, b_init, lr, num_epochs, batch_size, binary_cross_entropy, sigmoid)
   A = forward_perceptron(X, W_final, b_final)
   predictions = (A > 0.5).astype(int)
   accuracy = np.mean(predictions == y)
   print("Final Predictions (probabilities):\n", A)
   print("Final Predictions (binary):\n", predictions)
   print("True Labels:\n", y)
   print("Accuracy:", accuracy)
   print("Final Loss:", losses[-1])
   ```

4. **Perceptron on Larger Synthetic Data**  
   Generate a larger synthetic dataset for a linearly separable problem:
   `X = np.array([[i, j] for i in range(-2, 3) for j in range(-2, 3)])` (25
   points) and `y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)` (output
   1 if sum of inputs > 0). Train a perceptron with `lr = 0.1`,
   `num_epochs = 200`, and `batch_size = 5`. Print final accuracy and observe if
   the perceptron learns this separable pattern.

   ```python
   # Your code here
   X = np.array([[i, j] for i in range(-2, 3) for j in range(-2, 3)])
   y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)
   W_init = np.zeros((2, 1))
   b_init = np.zeros((1, 1))
   lr = 0.1
   num_epochs = 200
   batch_size = 5
   W_final, b_final, losses = gradient_descent(X, y, W_init, b_init, lr, num_epochs, batch_size, binary_cross_entropy, sigmoid)
   A = forward_perceptron(X, W_final, b_final)
   predictions = (A > 0.5).astype(int)
   accuracy = np.mean(predictions == y)
   print("Final Predictions (probabilities):\n", A)
   print("Final Predictions (binary):\n", predictions)
   print("True Labels:\n", y)
   print("Accuracy:", accuracy)
   print("Final Loss:", losses[-1])
   ```

These exercises will help you build intuition for how single-layer perceptrons
work, their ability to solve linearly separable problems, and their limitations
with non-linear patterns like XOR.

---

## Closing Thoughts

Congratulations on building your first neural network—a single-layer perceptron!
In this post, we’ve introduced the fundamental structure of neural networks,
implemented a perceptron using `forward_perceptron()` with `sigmoid()`
activation, and trained it on the XOR problem using `gradient_descent()`. While
the perceptron struggled with XOR due to its linear separability limitation,
this exercise lays the groundwork for understanding neural network concepts like
forward propagation and gradient-based training.

In the next chapter (Part 3.2: _Activation Functions for Neural Networks_),
we’ll explore additional activation functions like ReLU and softmax, which
introduce non-linearity and enable neural networks to tackle complex patterns.
This will prepare us for multi-layer perceptrons in Part 3.3.

Until then, experiment with the code and exercises above. If you have questions
or want to share your solutions, drop a comment below—I’m excited to hear from
you. Let’s keep building our deep learning toolkit together!

**Next Up**: Part 3.2 – Activation Functions for Neural Networks
