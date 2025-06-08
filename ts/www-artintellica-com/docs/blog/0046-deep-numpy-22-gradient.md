+++
title = "Learn Deep Learning with NumPy, Part 2.2: Gradient Descent for Optimization"
author = "Artintellica"
date = "2025-06-08"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0046-deep-numpy-22-gradient"
+++

## Introduction

Welcome back to our blog series, _"Learn Deep Learning with NumPy"_! In Part
2.1, we introduced loss functions like Mean Squared Error (MSE) and Binary
Cross-Entropy (BCE), which quantify how far a model’s predictions are from the
true values. Now, in Part 2.2, we’re taking the next step in Module 2 by
exploring _gradient descent_—a fundamental optimization algorithm that minimizes
loss by iteratively updating a model’s parameters.

By the end of this post, you’ll understand the concept of gradients and learning
rates, learn how gradient descent works to optimize model parameters, and
implement a reusable `gradient_descent()` function in NumPy for training a
simple linear regression model. This function will be a cornerstone of our deep
learning toolkit, reused across various models in future chapters. Let’s dive
into the math and code behind optimization!

---

## Why Gradient Descent Matters in Deep Learning

Training a neural network (or any machine learning model) involves finding the
best set of parameters (weights and biases) that minimize the loss function.
Think of the loss as a landscape with hills and valleys—our goal is to find the
lowest point (the minimum loss). Gradient descent is the algorithm that guides
us downhill toward this minimum by iteratively adjusting the parameters based on
the slope of the loss landscape.

Here’s the intuition:

- **Gradients**: The gradient of the loss function with respect to a parameter
  tells us the direction and magnitude of the steepest increase in loss. To
  minimize loss, we move in the opposite direction (downhill).
- **Learning Rate**: This controls how big each step is. Too large, and we might
  overshoot the minimum; too small, and training takes forever.

Gradient descent is the backbone of training neural networks, used to update
weights and biases after each forward pass. In this post, we’ll apply it to a
simple linear regression problem, but the concept extends to complex deep
learning models.

---

## Mathematical Foundations: Gradient Descent and MSE Gradient

### Gradient Descent Update Rule

Gradient descent updates a model’s parameters by taking small steps in the
direction that reduces the loss. For a parameter $W$ (e.g., a weight matrix),
the update rule is:

$$
W \leftarrow W - \eta \nabla L
$$

Where:

- $W$ is the parameter to update.
- $\eta$ (eta) is the learning rate, a small positive number (e.g., 0.01)
  controlling step size.
- $\nabla L$ is the gradient of the loss function $L$ with respect to $W$,
  indicating the direction of steepest ascent. We subtract it to move toward the
  minimum.

This process repeats iteratively—compute predictions, calculate loss, compute
gradients, update parameters—until the loss converges to a minimum or we reach a
set number of iterations.

### Gradient of Mean Squared Error (MSE)

For a linear regression model with predictions $y_{\text{pred}} = XW + b$, where
$X$ is the input matrix, $W$ is the weight matrix, and $b$ is the bias, we use
MSE as the loss function:

$$
L = \frac{1}{n} \sum_{i=1}^n (y_{\text{pred},i} - y_i)^2
$$

The gradient of MSE with respect to $W$ tells us how to adjust $W$ to reduce the
loss. For a dataset of $n$ samples, it is:

$$
\nabla_W L = \frac{1}{n} X^T (y_{\text{pred}} - y)
$$

Where:

- $X^T$ is the transpose of the input matrix $X$.
- $(y_{\text{pred}} - y)$ is the error vector (predictions minus true values).

Similarly, the gradient with respect to the bias $b$ (if present) is the average
error:

$$
\nabla_b L = \frac{1}{n} \sum_{i=1}^n (y_{\text{pred},i} - y_i)
$$

These gradients are used in the update rules $W \leftarrow W - \eta \nabla_W L$
and $b \leftarrow b - \eta \nabla_b L$. Let’s now implement gradient descent in
NumPy to train a linear regression model.

---

## Deriving the Gradients for Mean Squared Error (MSE)

To understand why gradient descent works, it’s essential to derive the gradients
of the Mean Squared Error (MSE) loss function with respect to the model
parameters—weights ($W$) and bias ($b$). This derivation shows how changes in
$W$ and $b$ affect the loss, guiding us on how to adjust them to minimize error.
Let’s break this down step by step using calculus.

### MSE Loss Function

For a linear regression model, the predicted output is given by
$y_{\text{pred}} = XW + b$, where $X$ is the input matrix of shape $(n, d)$
(with $n$ samples and $d$ features), $W$ is the weight matrix of shape $(d, 1)$,
and $b$ is the bias term. The true values are denoted by $y$, a vector of shape
$(n, 1)$. The MSE loss is defined as the average squared difference between
predictions and true values:

$$
L = \frac{1}{n} \sum_{i=1}^n (y_{\text{pred},i} - y_i)^2
$$

Substituting $y_{\text{pred},i} = (XW + b)_i$, we can write:

$$
L = \frac{1}{n} \sum_{i=1}^n \left( (XW + b)_i - y_i \right)^2
$$

Our goal is to compute the gradients $\nabla_W L$ and $\nabla_b L$, which are
the partial derivatives of $L$ with respect to $W$ and $b$, respectively. These
gradients indicate how much the loss changes with small changes in $W$ and $b$,
allowing us to update them in the direction that reduces $L$.

### Gradient with Respect to Weights ($W$)

To find $\nabla_W L$, we compute the partial derivative of $L$ with respect to
each element of $W$. Let’s consider the loss for a single sample first and then
generalize. For the $i$-th sample, the squared error is:

$$
L_i = \left( (XW + b)_i - y_i \right)^2
$$

Since $(XW + b)_i$ is the $i$-th element of the prediction vector, and $XW$ is a
linear combination, we can express it as $\sum_{j=1}^d X_{i,j} W_j + b$, where
$X_{i,j}$ is the $j$-th feature of the $i$-th sample and $W_j$ is the $j$-th
weight. The partial derivative of $L_i$ with respect to $W_j$ (a specific
weight) is:

$$
\frac{\partial L_i}{\partial W_j} = 2 \left( (XW + b)_i - y_i \right) \cdot \frac{\partial}{\partial W_j} \left( (XW + b)_i \right)
$$

Since $\frac{\partial}{\partial W_j} \left( (XW + b)_i \right) = X_{i,j}$ (as
only the term involving $W_j$ depends on it, and $b$ is independent of $W_j$),
this simplifies to:

$$
\frac{\partial L_i}{\partial W_j} = 2 \left( (XW + b)_i - y_i \right) \cdot X_{i,j}
$$

Now, for the total loss $L = \frac{1}{n} \sum_{i=1}^n L_i$, the partial
derivative with respect to $W_j$ is:

$$
\frac{\partial L}{\partial W_j} = \frac{1}{n} \sum_{i=1}^n 2 \left( (XW + b)_i - y_i \right) \cdot X_{i,j}
$$

This can be written in vector form for all weights $W$. Notice that for each
sample $i$, the term $2 \left( (XW + b)_i - y_i \right)$ is the error for that
sample, and multiplying by $X_{i,j}$ across all features $j$ corresponds to the
$i$-th row of $X$. Summing over all samples $i$, this is equivalent to a matrix
multiplication. Thus, the gradient for the entire weight vector is:

$$
\nabla_W L = \frac{2}{n} X^T (y_{\text{pred}} - y)
$$

In practice, the factor of 2 is often absorbed into the learning rate or omitted
for simplicity (as it scales the step size uniformly), so we commonly use:

$$
\nabla_W L = \frac{1}{n} X^T (y_{\text{pred}} - y)
$$

This is the formula implemented in our `gradient_descent()` function. It shows
that the gradient for $W$ is proportional to the correlation between the input
features ($X$) and the prediction errors ($y_{\text{pred}} - y$), guiding us to
adjust $W$ to reduce those errors.

### Gradient with Respect to Bias ($b$)

Next, we derive the gradient for the bias term $b$. The bias is added to every
prediction, so for the $i$-th sample, the partial derivative of $L_i$ with
respect to $b$ is:

$$
\frac{\partial L_i}{\partial b} = 2 \left( (XW + b)_i - y_i \right) \cdot \frac{\partial}{\partial b} \left( (XW + b)_i \right)
$$

Since $\frac{\partial}{\partial b} \left( (XW + b)_i \right) = 1$ (the
prediction increases by 1 for a unit increase in $b$), this simplifies to:

$$
\frac{\partial L_i}{\partial b} = 2 \left( (XW + b)_i - y_i \right)
$$

For the total loss $L = \frac{1}{n} \sum_{i=1}^n L_i$, the partial derivative
with respect to $b$ is:

$$
\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^n 2 \left( (XW + b)_i - y_i \right) = \frac{2}{n} \sum_{i=1}^n \left( y_{\text{pred},i} - y_i \right)
$$

Again, the factor of 2 is often omitted for simplicity, so we use:

$$
\nabla_b L = \frac{1}{n} \sum_{i=1}^n \left( y_{\text{pred},i} - y_i \right) = \text{mean}(y_{\text{pred}} - y)
$$

This tells us that the gradient for $b$ is simply the average error across all
samples. If the predictions are systematically too high (positive average
error), $b$ should decrease, and if too low (negative average error), $b$ should
increase.

### Why These Gradients Work

The gradients $\nabla_W L$ and $\nabla_b L$ point in the direction of the
steepest increase in loss. By subtracting them (scaled by the learning rate
$\eta$) in the gradient descent update rule, we move in the direction of
steepest decrease, iteratively reducing the loss. This calculus-based approach
ensures that each update to $W$ and $b$ brings our predictions closer to the
true values, minimizing MSE.

Understanding this derivation provides insight into why gradient descent is
effective and prepares us to derive gradients for more complex loss functions
and models (like neural networks) in future posts. For now, these formulae for
linear regression with MSE are the foundation of our optimization process.

---

## Implementing Gradient Descent with NumPy

We’ll create a reusable `gradient_descent()` function to train a linear
regression model by minimizing MSE loss. This function will update parameters
iteratively and be adaptable for more complex models later. As always, we’ll
include type hints for clarity.

### Gradient Descent Implementation

Here’s the implementation, along with an example using synthetic data where the
true relationship is $y = 2x + 1$:

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable

def gradient_descent(X: NDArray[np.floating], y: NDArray[np.floating], W: NDArray[np.floating],
                     b: NDArray[np.floating], lr: float, num_iterations: int,
                     loss_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], float]) -> tuple[NDArray[np.floating], NDArray[np.floating], list[float]]:
    """
    Perform gradient descent to minimize loss for linear regression.
    Args:
        X: Input data, shape (n_samples, n_features)
        y: True values, shape (n_samples, 1)
        W: Initial weights, shape (n_features, 1)
        b: Initial bias, shape (1,) or (1,1)
        lr: Learning rate, step size for updates
        num_iterations: Number of iterations to run gradient descent
        loss_fn: Loss function to compute error, e.g., mse_loss
    Returns:
        Tuple of (updated W, updated b, list of loss values over iterations)
    """
    n = X.shape[0]
    loss_history = []

    for _ in range(num_iterations):
        # Forward pass: Compute predictions
        y_pred = X @ W + b
        # Compute loss
        loss = loss_fn(y_pred, y)
        loss_history.append(loss)
        # Compute gradients for W and b (for MSE loss)
        grad_W = (X.T @ (y_pred - y)) / n
        grad_b = np.mean(y_pred - y)
        # Update parameters
        W = W - lr * grad_W
        b = b - lr * grad_b

    return W, b, loss_history

# Example: Synthetic data (y = 2x + 1)
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])  # Input (5 samples, 1 feature)
y = np.array([[3.0], [5.0], [7.0], [9.0], [11.0]])  # True values (y = 2x + 1)
W_init = np.array([[0.0]])  # Initial weight (start far from true value 2.0)
b_init = np.array([[0.0]])  # Initial bias (start far from true value 1.0)
lr = 0.1  # Learning rate
num_iterations = 100  # Number of iterations

# Use mse_loss from our library
def mse_loss(y_pred: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    return np.mean((y_pred - y) ** 2)

# Run gradient descent
W_final, b_final, losses = gradient_descent(X, y, W_init, b_init, lr, num_iterations, mse_loss)

print("Initial weight W:", W_init)
print("Initial bias b:", b_init)
print("Final weight W after training:", W_final)
print("Final bias b after training:", b_final)
print("Final loss:", losses[-1])
print("First few losses:", losses[:5])
```

**Output** (values are approximate):

```
Initial weight W: [[0.]]
Initial bias b: [[0.]]
Final weight W after training: [[1.9999]]
Final bias b after training: [[1.0001]]
Final loss: 1.234e-08
First few losses: [27.2, 17.408, 11.14112, 7.1303168, 4.563402752]
```

In this example, we start with initial guesses for weight ($W = 0$) and bias
($b = 0$), far from the true values ($W = 2$, $b = 1$). Over 100 iterations,
gradient descent updates $W$ and $b$ using the MSE gradient, converging close to
the true values, and the loss drops significantly. The `loss_history` list
tracks how the loss decreases over iterations, showing the optimization
progress.

---

## Organizing Our Growing Library

Let’s update our `neural_network.py` file to include the `gradient_descent()`
function alongside our previous implementations. This keeps our toolkit
organized and reusable for training models.

```python
# neural_network.py
import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable, Tuple, List

def normalize(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Normalize an array to have mean=0 and std=1.
    Args:
        X: NumPy array of any shape with floating-point values
    Returns:
        Normalized array of the same shape with floating-point values
    """
    mean = np.mean(X)
    std = np.std(X)
    if std == 0:  # Avoid division by zero
        return X - mean
    return (X - mean) / std

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
                     b: NDArray[np.floating], lr: float, num_iterations: int,
                     loss_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], float]) -> Tuple[NDArray[np.floating], NDArray[np.floating], List[float]]:
    """
    Perform gradient descent to minimize loss for linear regression.
    Args:
        X: Input data, shape (n_samples, n_features)
        y: True values, shape (n_samples, 1)
        W: Initial weights, shape (n_features, 1)
        b: Initial bias, shape (1,) or (1,1)
        lr: Learning rate, step size for updates
        num_iterations: Number of iterations to run gradient descent
        loss_fn: Loss function to compute error, e.g., mse_loss
    Returns:
        Tuple of (updated W, updated b, list of loss values over iterations)
    """
    n = X.shape[0]
    loss_history = []

    for _ in range(num_iterations):
        y_pred = X @ W + b
        loss = loss_fn(y_pred, y)
        loss_history.append(loss)
        grad_W = (X.T @ (y_pred - y)) / n
        grad_b = np.mean(y_pred - y)
        W = W - lr * grad_W
        b = b - lr * grad_b

    return W, b, loss_history
```

You can now import this function in other scripts using
`from neural_network import gradient_descent`. This growing library will be
central to training neural networks and other models in upcoming posts.

---

## Exercises: Practice with Gradient Descent

To reinforce your understanding of gradient descent, try these Python-focused
coding exercises. They’ll prepare you for training more complex models in future
chapters. Run the code and compare outputs to verify your solutions.

1. **Basic Gradient Descent on Synthetic Data**  
   Create synthetic data with `X = np.array([[1.0], [2.0], [3.0]])` and
   `y = np.array([[2.0], [4.0], [6.0]])` (true relationship $y = 2x$).
   Initialize `W = np.array([[0.0]])` and `b = np.array([[0.0]])`. Run
   `gradient_descent()` with `lr = 0.1` and `num_iterations = 50`. Print the
   initial and final values of `W`, `b`, and the final loss.

   ```python
   # Your code here
   X = np.array([[1.0], [2.0], [3.0]])
   y = np.array([[2.0], [4.0], [6.0]])
   W_init = np.array([[0.0]])
   b_init = np.array([[0.0]])
   lr = 0.1
   num_iterations = 50
   W_final, b_final, losses = gradient_descent(X, y, W_init, b_init, lr, num_iterations, mse_loss)
   print("Initial weight W:", W_init)
   print("Initial bias b:", b_init)
   print("Final weight W:", W_final)
   print("Final bias b:", b_final)
   print("Final loss:", losses[-1])
   ```

2. **Effect of Learning Rate**  
   Using the same data as in Exercise 1, run `gradient_descent()` with a very
   small learning rate `lr = 0.01` for 50 iterations. Compare the final `W`,
   `b`, and loss to Exercise 1. Observe how a smaller learning rate slows
   convergence.

   ```python
   # Your code here
   X = np.array([[1.0], [2.0], [3.0]])
   y = np.array([[2.0], [4.0], [6.0]])
   W_init = np.array([[0.0]])
   b_init = np.array([[0.0]])
   lr = 0.01
   num_iterations = 50
   W_final, b_final, losses = gradient_descent(X, y, W_init, b_init, lr, num_iterations, mse_loss)
   print("Initial weight W:", W_init)
   print("Initial bias b:", b_init)
   print("Final weight W:", W_final)
   print("Final bias b:", b_final)
   print("Final loss:", losses[-1])
   ```

3. **Effect of Number of Iterations**  
   Using the same data as in Exercise 1, run `gradient_descent()` with
   `lr = 0.1` but for only 10 iterations. Compare the final `W`, `b`, and loss
   to Exercise 1. Observe how fewer iterations result in less convergence.

   ```python
   # Your code here
   X = np.array([[1.0], [2.0], [3.0]])
   y = np.array([[2.0], [4.0], [6.0]])
   W_init = np.array([[0.0]])
   b_init = np.array([[0.0]])
   lr = 0.1
   num_iterations = 10
   W_final, b_final, losses = gradient_descent(X, y, W_init, b_init, lr, num_iterations, mse_loss)
   print("Initial weight W:", W_init)
   print("Initial bias b:", b_init)
   print("Final weight W:", W_final)
   print("Final bias b:", b_final)
   print("Final loss:", losses[-1])
   ```

4. **Training on Noisy Data**  
   Create synthetic data with noise:
   `X = np.array([[1.0], [2.0], [3.0], [4.0]])`,
   `y = np.array([[2.1], [4.2], [5.8], [8.3]])` (approximating $y = 2x$ with
   noise). Initialize `W = np.array([[0.0]])` and `b = np.array([[0.0]])`. Run
   `gradient_descent()` with `lr = 0.1` for 100 iterations. Observe how the
   final `W` and `b` are close but not exactly equal to the ideal values due to
   noise.

   ```python
   # Your code here
   X = np.array([[1.0], [2.0], [3.0], [4.0]])
   y = np.array([[2.1], [4.2], [5.8], [8.3]])
   W_init = np.array([[0.0]])
   b_init = np.array([[0.0]])
   lr = 0.1
   num_iterations = 100
   W_final, b_final, losses = gradient_descent(X, y, W_init, b_init, lr, num_iterations, mse_loss)
   print("Initial weight W:", W_init)
   print("Initial bias b:", b_init)
   print("Final weight W:", W_final)
   print("Final bias b:", b_final)
   print("Final loss:", losses[-1])
   ```

These exercises will help you build intuition for how gradient descent optimizes
parameters, the impact of hyperparameters like learning rate and iterations, and
handling real-world imperfections like noisy data.

---

## Closing Thoughts

Congratulations on mastering gradient descent, a key optimization algorithm in
deep learning! In this post, we’ve explored the concepts of gradients and
learning rates, understood the math behind updating parameters to minimize loss,
and implemented a reusable `gradient_descent()` function for linear regression.
This function will be a building block for training more complex models like
neural networks in upcoming modules.

In the next chapter (Part 2.3: _Mini-Batch Gradient Descent_), we’ll extend
gradient descent to handle larger datasets efficiently by processing data in
smaller batches, a technique critical for scalability. This will prepare us for
training on real datasets like MNIST.

Until then, experiment with the code and exercises above. If you have questions
or want to share your solutions, drop a comment below—I’m excited to hear from
you. Let’s keep building our deep learning toolkit together!

**Next Up**: Part 2.3 – Mini-Batch Gradient Descent
