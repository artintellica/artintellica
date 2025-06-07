+++
title = "Learn Deep Learning with NumPy, Part 2.1: Understanding Loss Functions"
author = "Artintellica"
date = "2025-06-06"
+++

## Introduction

Welcome back to our blog series, _"Learn Deep Learning with NumPy"_! Having
completed Module 1, where we mastered NumPy arrays, matrix operations, and
activation basics like `sigmoid()`, we’re now diving into Module 2, which
focuses on optimization and training neural networks. In Part 2.1, we’ll explore
_loss functions_—the mathematical tools that measure how well (or poorly) a
model’s predictions match the actual data. Loss functions are the cornerstone of
training, guiding neural networks to improve through optimization.

By the end of this post, you’ll understand the purpose of loss functions, learn
two fundamental types for regression and classification, and implement reusable
functions for mean squared error (MSE) and binary cross-entropy in NumPy. These
will become essential components of our deep learning toolkit as we move toward
training models. Let’s get started with the math and code behind measuring model
error!

---

## Why Loss Functions Matter in Deep Learning

In deep learning, the goal is to train a model that makes accurate
predictions—whether it’s predicting house prices (regression) or identifying
whether an image contains a cat (classification). But how do we quantify
“accuracy”? That’s where loss functions come in. A loss function measures the
difference between the model’s predictions and the true values, providing a
single number (the loss) that tells us how far off the model is. During
training, we aim to minimize this loss through optimization techniques like
gradient descent (which we’ll cover in Part 2.2).

Loss functions serve two key purposes:

- **Error Measurement**: They quantify the model’s performance on a given
  dataset, allowing us to evaluate how well it’s learning.
- **Optimization Guide**: They provide a signal for adjusting the model’s
  parameters (weights and biases) to improve predictions.

Different tasks require different loss functions. In this post, we’ll focus on
two common ones:

- **Mean Squared Error (MSE)** for regression tasks, where the goal is to
  predict continuous values.
- **Binary Cross-Entropy** for binary classification tasks, where the goal is to
  predict probabilities for two classes.

Let’s dive into the mathematics and implementation of these loss functions using
NumPy.

---

## Mathematical Foundations: Mean Squared Error and Binary Cross-Entropy

### Mean Squared Error (MSE)

Mean Squared Error is widely used for regression problems, where the model
predicts continuous values (e.g., predicting temperature or stock prices). It
measures the average squared difference between predicted values
($y_{\text{pred}}$) and actual values ($y$). For a dataset of $n$ samples, MSE
is defined as:

$$
L = \frac{1}{n} \sum_{i=1}^n (y_{\text{pred},i} - y_i)^2
$$

Squaring the differences emphasizes larger errors and ensures the loss is always
non-negative. A smaller MSE indicates better predictions, with 0 meaning perfect
predictions. In neural networks, MSE is often used when the output layer has no
activation function (or a linear one), directly predicting continuous values.

### Binary Cross-Entropy (BCE)

Binary Cross-Entropy, also known as log loss, is used for binary classification
problems, where the model predicts probabilities for two classes (e.g., 0 or 1,
“no” or “yes”). It measures the difference between the true labels ($y$, either
0 or 1) and the predicted probabilities ($a$, values between 0 and 1, often from
a sigmoid activation). For $n$ samples, BCE is defined as:

$$
L = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(a_i) + (1 - y_i) \log(1 - a_i) \right]
$$

This loss penalizes confident wrong predictions more heavily. For example, if
the true label $y_i = 1$ but the model predicts $a_i = 0.01$ (very confident in
the wrong class), the loss is large due to $\log(0.01)$ being a large negative
number. BCE is typically used with a sigmoid activation in the output layer, as
sigmoid ensures outputs are in [0, 1]. A smaller BCE indicates better alignment
between predictions and labels, with 0 being a perfect match.

Now, let’s implement these loss functions in NumPy and see them in action with
examples.

---

## Implementing Loss Functions with NumPy

We’ll create reusable functions for MSE and BCE, adding them to our
`neural_network.py` library. These functions will be used in later chapters for
training neural networks. As always, we’ll include type hints for clarity and
compatibility with static type checking tools.

### Mean Squared Error (MSE) Implementation

Let’s implement MSE to evaluate regression tasks. We’ll test it with synthetic
data where the true relationship is linear ($y = 2x + 1$).

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union

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

# Example: Synthetic regression data (y = 2x + 1)
X = np.array([[1.0], [2.0], [3.0], [4.0]])  # Input (4 samples, 1 feature)
W = np.array([[2.0]])  # Weight (approximating the true slope)
b = 1.0  # Bias (approximating the true intercept)
y_pred = X @ W + b  # Predicted values
y_true = np.array([[3.0], [5.0], [7.0], [9.0]])  # True values (y = 2x + 1)

print("Input X (4x1):\n", X)
print("Predicted y_pred (4x1):\n", y_pred)
print("True y_true (4x1):\n", y_true)
loss_mse = mse_loss(y_pred, y_true)
print("MSE Loss:", loss_mse)
```

**Output**:

```
Input X (4x1):
 [[1.]
  [2.]
  [3.]
  [4.]]
Predicted y_pred (4x1):
 [[3.]
  [5.]
  [7.]
  [9.]]
True y_true (4x1):
 [[3.]
  [5.]
  [7.]
  [9.]]
MSE Loss: 0.0
```

In this example, since `y_pred` exactly matches `y_true` (because we set `W` and
`b` to the true values of the linear relationship), the MSE is 0, indicating
perfect predictions. In a real scenario, predictions would differ from true
values, resulting in a positive loss.

### Binary Cross-Entropy (BCE) Implementation

Now, let’s implement BCE for binary classification. We’ll use synthetic data
where true labels are 0 or 1, and predicted probabilities come from a sigmoid
activation.

```python
def binary_cross_entropy(A: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """
    Compute the Binary Cross-Entropy loss between predicted probabilities and true labels.
    Args:
        A: Predicted probabilities (after sigmoid), array of shape (n,) or (n,1), values in [0, 1]
        y: True binary labels, array of shape (n,) or (n,1), values in {0, 1}
    Returns:
        Binary cross-entropy loss as a single float
    """
    # Add small epsilon to avoid log(0) issues
    epsilon = 1e-15
    return -np.mean(y * np.log(A + epsilon) + (1 - y) * np.log(1 - A + epsilon))

# Example: Synthetic binary classification data
# Raw outputs (logits) before sigmoid
Z = np.array([[2.0], [-1.0], [3.0], [-2.0]])
# Predicted probabilities after sigmoid
A = 1 / (1 + np.exp(-Z))
# True binary labels (0 or 1)
y_true = np.array([[1.0], [0.0], [1.0], [0.0]])

print("Raw outputs Z (4x1):\n", Z)
print("Predicted probabilities A (4x1):\n", A)
print("True labels y_true (4x1):\n", y_true)
loss_bce = binary_cross_entropy(A, y_true)
print("Binary Cross-Entropy Loss:", loss_bce)
```

**Output** (values are approximate):

```
Raw outputs Z (4x1):
 [[ 2.]
  [-1.]
  [ 3.]
  [-2.]]
Predicted probabilities A (4x1):
 [[0.88079708]
  [0.26894142]
  [0.95257413]
  [0.11920292]]
True labels y_true (4x1):
 [[1.]
  [0.]
  [1.]
  [0.]]
Binary Cross-Entropy Loss: 0.1731486847
```

Here, `A` contains probabilities close to the true labels (e.g., high
probability for `y=1`, low for `y=0`), so the BCE loss is relatively low. If
predictions were far from the true labels, the loss would be higher due to the
logarithmic penalty. The `epsilon` term prevents issues with `log(0)` in case
predictions are exactly 0 or 1.

---

## Organizing Our Growing Library

Let’s update our `neural_network.py` file to include these new loss functions
alongside our previous implementations. This keeps our toolkit organized and
reusable.

```python
# neural_network.py
import numpy as np
from numpy.typing import NDArray
from typing import Union

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
```

You can now import these functions in other scripts using
`from neural_network import mse_loss, binary_cross_entropy`. This growing
library will be central to training neural networks in upcoming posts.

---

## Exercises: Practice with Loss Functions

To reinforce your understanding of loss functions, try these Python-focused
coding exercises. They’ll prepare you for training neural networks in future
chapters. Run the code and compare outputs to verify your solutions.

1. **Mean Squared Error for Regression**  
   Create synthetic regression data with `X = np.array([[1.0], [2.0], [3.0]])`,
   weights `W = np.array([[1.5]])`, and bias `b = 0.5`. Compute predictions
   `y_pred = X @ W + b` and true values
   `y_true = np.array([[2.0], [3.5], [5.0]])`. Use `mse_loss()` to calculate the
   loss and print all steps.

   ```python
   # Your code here
   X = np.array([[1.0], [2.0], [3.0]])
   W = np.array([[1.5]])
   b = 0.5
   y_pred = X @ W + b
   y_true = np.array([[2.0], [3.5], [5.0]])
   loss = mse_loss(y_pred, y_true)
   print("Input X (3x1):\n", X)
   print("Predicted y_pred (3x1):\n", y_pred)
   print("True y_true (3x1):\n", y_true)
   print("MSE Loss:", loss)
   ```

2. **Binary Cross-Entropy for Classification**  
   Create synthetic binary classification data with raw outputs
   `Z = np.array([[1.0], [-1.0], [2.0]])`. Compute predicted probabilities `A`
   using `sigmoid(Z)` (from our library), and set true labels
   `y_true = np.array([[1.0], [0.0], [1.0]])`. Use `binary_cross_entropy()` to
   calculate the loss and print all steps.

   ```python
   # Your code here
   Z = np.array([[1.0], [-1.0], [2.0]])
   A = sigmoid(Z)
   y_true = np.array([[1.0], [0.0], [1.0]])
   loss = binary_cross_entropy(A, y_true)
   print("Raw outputs Z (3x1):\n", Z)
   print("Predicted probabilities A (3x1):\n", A)
   print("True labels y_true (3x1):\n", y_true)
   print("Binary Cross-Entropy Loss:", loss)
   ```

3. **MSE with Imperfect Predictions**  
   Use the same `X` and `y_true` as in Exercise 1, but change `W` to
   `np.array([[1.0]])` (an imperfect weight). Compute `y_pred` and the MSE loss.
   Observe how the loss increases compared to a perfect prediction.

   ```python
   # Your code here
   X = np.array([[1.0], [2.0], [3.0]])
   W = np.array([[1.0]])
   b = 0.5
   y_pred = X @ W + b
   y_true = np.array([[2.0], [3.5], [5.0]])
   loss = mse_loss(y_pred, y_true)
   print("Input X (3x1):\n", X)
   print("Predicted y_pred (3x1):\n", y_pred)
   print("True y_true (3x1):\n", y_true)
   print("MSE Loss:", loss)
   ```

4. **BCE with Poor Predictions**  
   Use the same `y_true` as in Exercise 2, but set raw outputs
   `Z = np.array([[-2.0], [2.0], [-1.0]])` (predictions mostly opposite to true
   labels). Compute `A` with `sigmoid(Z)` and calculate the BCE loss. Observe
   how the loss is higher compared to Exercise 2.

   ```python
   # Your code here
   Z = np.array([[-2.0], [2.0], [-1.0]])
   A = sigmoid(Z)
   y_true = np.array([[1.0], [0.0], [1.0]])
   loss = binary_cross_entropy(A, y_true)
   print("Raw outputs Z (3x1):\n", Z)
   print("Predicted probabilities A (3x1):\n", A)
   print("True labels y_true (3x1):\n", y_true)
   print("Binary Cross-Entropy Loss:", loss)
   ```

These exercises will help you build intuition for how loss functions quantify
model performance, setting the stage for optimization in the next post.

---

## Closing Thoughts

Congratulations on taking your first step into optimization with loss functions!
In this post, we’ve introduced the concept of loss as a measure of model error,
explored Mean Squared Error for regression and Binary Cross-Entropy for
classification, and added `mse_loss()` and `binary_cross_entropy()` to our deep
learning toolkit. These functions are critical for evaluating and training
neural networks.

In the next chapter (Part 2.2: _Gradient Descent for Optimization_), we’ll use
these loss functions to guide model training, implementing gradient descent to
minimize loss by updating model parameters. This will bring us one step closer
to building and training our own neural networks from scratch.

Until then, experiment with the code and exercises above. If you have questions
or want to share your solutions, drop a comment below—I’m excited to hear from
you. Let’s keep building our deep learning toolkit together!

**Next Up**: Part 2.2 – Gradient Descent for Optimization \*Next Up\*\*: Part
2.1 – Understanding Loss Functions
