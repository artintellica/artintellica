+++
title = "Learn Deep Learning with NumPy, Part 1.3: Mathematical Functions and Activation Basics"
author = "Artintellica"
date = "2025-06-06"
+++

# Learn Deep Learning with NumPy, Part 1.3: Mathematical Functions and Activation Basics

## Introduction

Welcome back to our blog series, _"Learn Deep Learning with NumPy"_! In Parts
1.1 and 1.2, we explored NumPy arrays and matrix operations, building
foundational tools like `normalize()` and `matrix_multiply()`. Today, in Part
1.3, we’re concluding Module 1 by diving into NumPy’s mathematical functions and
introducing the concept of activation functions, which are crucial for neural
networks. Specifically, we’ll focus on the exponential function, the sigmoid
activation, and a preview of ReLU (Rectified Linear Unit) using maximum
operations.

By the end of this post, you’ll understand how NumPy handles element-wise
mathematical operations, why activation functions are essential for neural
networks, and you’ll have implemented a reusable `sigmoid()` function for your
growing deep learning toolkit. Let’s explore the math and code that bring neural
networks to life!

---

## Why Mathematical Functions Matter in Neural Networks

Neural networks rely on mathematical functions to transform data at various
stages. Two key areas where these functions play a critical role are:

- **Activation Functions**: These introduce non-linearity into the network,
  allowing it to learn complex patterns. Without non-linear activations, a
  neural network would simply be a stack of linear transformations (like matrix
  multiplications), incapable of solving problems like image classification or
  natural language processing.
- **Loss Functions and Optimization**: Mathematical functions are used to
  compute errors (losses) and gradients, guiding the network’s learning process
  through optimization techniques like gradient descent.

NumPy provides efficient, vectorized implementations of common mathematical
operations like exponentials, logarithms, and element-wise maximums, which are
building blocks for activations and losses. In this post, we’ll focus on
activation functions, starting with the sigmoid function, and preview ReLU, both
of which we’ll use extensively in later modules when building neural networks.

---

## Mathematical Foundations: Exponential, Sigmoid, and ReLU

Let’s cover the key mathematical functions we’ll implement, focusing on their
roles in neural networks.

### Exponential Function ($e^x$)

The exponential function, $e^x$, is a fundamental operation used in many
activation functions and loss computations. For a vector or matrix $Z$, applying
$e^z$ element-wise produces a new array where each element $z_i$ is transformed
to $e^{z_i}$. In neural networks, this often appears in functions like sigmoid
and softmax.

### Sigmoid Function ($\sigma(z) = \frac{1}{1 + e^{-z}}$)

The sigmoid function is a classic activation function that maps any real number
to a value between 0 and 1. Mathematically, for an input $z$, it is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

In a neural network, sigmoid is often used in binary classification tasks at the
output layer, where it interprets the raw output (logits) as probabilities. For
a matrix $Z$, sigmoid is applied element-wise, transforming each value
independently. The output range [0, 1] makes it intuitive for tasks like
predicting whether an input belongs to a class (e.g., 0 for “no,” 1 for “yes”).

### ReLU (Rectified Linear Unit) via Maximum ($\max(0, z)$)

ReLU is another popular activation function that introduces non-linearity by
outputting the input directly if it’s positive, and 0 otherwise:

$$
\text{ReLU}(z) = \max(0, z)
$$

ReLU is widely used in hidden layers of deep networks because it helps mitigate
issues like vanishing gradients and is computationally efficient. Like sigmoid,
it’s applied element-wise to arrays. We’ll preview it here using NumPy’s
`maximum()` function and implement it fully in later chapters.

Let’s now implement these functions with NumPy and see them in action.

---

## Implementing Mathematical Functions with NumPy

NumPy provides vectorized operations that apply mathematical functions to entire
arrays at once, making them perfect for neural network computations. Let’s
explore how to use `np.exp()` for exponentials, build a `sigmoid()` function,
and preview ReLU with `np.maximum()`.

### Exponential Function with `np.exp()`

Here’s how to apply the exponential function to a vector or matrix using NumPy:

```python
import numpy as np

# Create a small matrix of values
Z = np.array([[0, 1, -1],
              [2, -2, 0.5]])
print("Input matrix Z (2x3):\n", Z)

# Apply exponential element-wise
exp_Z = np.exp(Z)
print("Exponential of Z (e^Z):\n", exp_Z)
```

**Output** (values are approximate):

```
Input matrix Z (2x3):
 [[ 0.   1.  -1. ]
  [ 2.  -2.   0.5]]
Exponential of Z (e^Z):
 [[1.         2.71828183 0.36787944]
  [7.3890561  0.13533528 1.64872127]]
```

Notice how each element is transformed independently. This operation is a
building block for more complex functions like sigmoid.

### Building a Reusable Sigmoid Function

Let’s implement the sigmoid function using `np.exp()`. We’ll make it reusable
with type hints for our deep learning library.

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union

def sigmoid(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the sigmoid activation function element-wise.
    Args:
        Z: Input array of any shape with floating-point values
    Returns:
        Array of the same shape with sigmoid applied element-wise, values in [0, 1]
    """
    return 1 / (1 + np.exp(-Z))

# Example: Apply sigmoid to a 3x2 matrix
Z = np.array([[0.0, 1.0],
              [-1.0, 2.0],
              [-2.0, 3.0]])
print("Input matrix Z (3x2):\n", Z)
A = sigmoid(Z)
print("Sigmoid output A (3x2):\n", A)
```

**Output** (values are approximate):

```
Input matrix Z (3x2):
 [[ 0.  1.]
  [-1.  2.]
  [-2.  3.]]
Sigmoid output A (3x2):
 [[0.5        0.73105858]
  [0.26894142 0.88079708]
  [0.11920292 0.95257413]]
```

As expected, all output values are between 0 and 1. For instance,
$\sigma(0) = 0.5$, $\sigma(1) \approx 0.731$, and $\sigma(-2) \approx 0.119$.
This `sigmoid()` function will be reused in later chapters for neural network
activations, especially in binary classification tasks.

### Previewing ReLU with `np.maximum()`

Let’s also preview the ReLU activation using NumPy’s `maximum()` function, which
applies element-wise maximum between two values or arrays. We’ll implement a
full `relu()` function in Module 3, but here’s a quick look:

```python
# Apply ReLU-like operation using np.maximum
Z = np.array([[-1.0, 0.0, 1.0],
              [-2.0, 3.0, -0.5]])
print("Input matrix Z (2x3):\n", Z)
A_relu = np.maximum(0, Z)
print("ReLU-like output (max(0, Z)):\n", A_relu)
```

**Output**:

```
Input matrix Z (2x3):
 [[-1.   0.   1. ]
  [-2.   3.  -0.5]]
ReLU-like output (max(0, Z)):
 [[0. 0. 1.]
  [0. 3. 0.]]
```

ReLU sets all negative values to 0 while preserving positive values, introducing
sparsity and non-linearity into the network. This simplicity makes it
computationally efficient for deep networks.

---

## Organizing Our Growing Library

As we build our deep learning toolkit, it’s important to keep our code
organized. Let’s add the `sigmoid()` function to the same `neural_network.py`
file where we’ve stored `normalize()` and `matrix_multiply()` from previous
posts. Here’s how the file might look now:

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
```

You can now import these functions in other scripts using
`from neural_network import sigmoid, normalize, matrix_multiply`. Keeping our
functions in a single module makes them easy to reuse as we progress through the
series.

---

## Exercises: Practice with Mathematical Functions

To solidify your understanding of NumPy’s mathematical functions and activation
basics, try these Python-focused coding exercises. They’ll prepare you for
implementing neural network layers in future chapters. Run the code and compare
outputs to verify your solutions.

1. **Exponential Function Application**  
   Create a 2x3 matrix `Z` with values `[[-1, 0, 1], [-2, 2, 0.5]]`. Apply
   `np.exp()` to it and print the result. Observe how each element is
   transformed.

   ```python
   # Your code here
   Z = np.array([[-1, 0, 1], [-2, 2, 0.5]], dtype=np.float64)
   exp_Z = np.exp(Z)
   print("Input matrix Z (2x3):\n", Z)
   print("Exponential of Z (e^Z):\n", exp_Z)
   ```

2. **Sigmoid Function Application**  
   Using the same 2x3 matrix `Z` from Exercise 1, apply the `sigmoid()` function
   we wrote and print the result. Verify that all output values are between 0
   and 1.

   ```python
   # Your code here
   Z = np.array([[-1, 0, 1], [-2, 2, 0.5]], dtype=np.float64)
   sigmoid_Z = sigmoid(Z)
   print("Input matrix Z (2x3):\n", Z)
   print("Sigmoid output (2x3):\n", sigmoid_Z)
   ```

3. **ReLU-like Operation with `np.maximum()`**  
   Create a 3x2 matrix `Z` with values
   `[[-1.5, 2.0], [0.0, -0.5], [3.0, -2.0]]`. Apply a ReLU-like operation using
   `np.maximum(0, Z)` and print the result. Confirm that negative values are set
   to 0.

   ```python
   # Your code here
   Z = np.array([[-1.5, 2.0], [0.0, -0.5], [3.0, -2.0]], dtype=np.float64)
   relu_Z = np.maximum(0, Z)
   print("Input matrix Z (3x2):\n", Z)
   print("ReLU-like output (max(0, Z)):\n", relu_Z)
   ```

4. **Combining Operations**  
   Create a 2x2 matrix `Z` with values `[[1, -1], [2, -2]]`. First, apply
   `np.exp(-Z)`, then use the result to compute sigmoid manually as
   `1 / (1 + exp(-Z))`. Compare your manual sigmoid computation to the output of
   our `sigmoid()` function.

   ```python
   # Your code here
   Z = np.array([[1, -1], [2, -2]], dtype=np.float64)
   exp_neg_Z = np.exp(-Z)
   manual_sigmoid = 1 / (1 + exp_neg_Z)
   func_sigmoid = sigmoid(Z)
   print("Input matrix Z (2x2):\n", Z)
   print("Manual sigmoid computation:\n", manual_sigmoid)
   print("Function sigmoid output:\n", func_sigmoid)
   ```

These exercises will help you build intuition for mathematical transformations,
which are essential for activation functions and loss computations in neural
networks.

---

## Closing Thoughts

Congratulations on completing Module 1 of our deep learning journey with NumPy!
In this post, we’ve explored NumPy’s mathematical functions, implemented a
reusable `sigmoid()` function, and previewed ReLU using `np.maximum()`. These
tools are critical for introducing non-linearity into neural networks, enabling
them to solve complex problems beyond simple linear transformations.

With Module 1 behind us, we’ve built a solid foundation of NumPy basics—arrays,
matrix operations, and mathematical functions. In Module 2, starting with Part
2.1 (_Understanding Loss Functions_), we’ll shift focus to optimization,
introducing loss functions to measure model error and setting the stage for
training neural networks with gradient descent.

Until then, experiment with the code and exercises above. If you have questions
or want to share your solutions, drop a comment below—I’m excited to hear from
you. Let’s keep building our deep learning toolkit together!

**Next Up**: Part 2.1 – Understanding Loss Functions
