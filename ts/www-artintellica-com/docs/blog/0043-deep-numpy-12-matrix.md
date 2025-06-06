+++
title = "Learn Deep Learning with NumPy, Part 1.2: Matrix Operations for Neural Networks"
author = "Artintellica"
date = "2025-06-05"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0043-deep-numpy-12-matrix"
+++

## Introduction

Welcome back to our blog series, _"Learn Deep Learning with NumPy"_! In Part
1.1, we introduced NumPy arrays and built our first reusable function for data
preprocessing. Today, in Part 1.2, we’re diving deeper into the mathematical
foundation of neural networks by exploring matrix operations with NumPy.
Specifically, we’ll focus on matrix multiplication, a core operation that
computes the output of neural network layers.

By the end of this post, you’ll understand how matrix multiplication works, why
it’s essential for neural networks, and how to implement it efficiently using
NumPy. We’ll also create another reusable function, `matrix_multiply()`, to
solidify our growing deep learning toolkit. Let’s jump into the world of linear
algebra and see how it powers neural networks!

---

## Why Matrix Operations Matter in Neural Networks

Neural networks are, at their core, a series of mathematical transformations
applied to input data. One of the most fundamental transformations occurs in
each layer, where input data is combined with learned weights to produce an
output. This process is mathematically represented as matrix multiplication. For
example, if $X$ is a matrix of input data and $W$ is a matrix of weights, the
output $Z$ of a layer (before adding biases or applying activations) is computed
as:

$$
Z = XW
$$

This operation is not just a one-off calculation—it’s repeated for every layer,
every forward pass, and during backpropagation to update weights. NumPy’s
efficient handling of matrix operations makes it ideal for implementing these
computations from scratch. Understanding matrix multiplication, along with
related concepts like transpose and dot products, is crucial for building and
debugging neural networks. Let’s explore these concepts with clear examples.

---

## Matrix Multiplication: The Heart of Neural Network Layers

Matrix multiplication is a way of combining two matrices to produce a new
matrix. For two matrices $X$ of shape $(m, n)$ and $W$ of shape $(n, p)$, the
result $Z = XW$ will have shape $(m, p)$. The element $Z_{i,j}$ is computed as
the dot product of the $i$-th row of $X$ and the $j$-th column of $W$:

$$
Z_{i,j} = \sum_{k=1}^n X_{i,k} \cdot W_{k,j}
$$

In the context of a neural network, think of $X$ as a batch of input data (e.g.,
$m$ samples with $n$ features each) and $W$ as the weights of a layer (mapping
$n$ input features to $p$ output features). The result $Z$ represents the raw
output of the layer for each sample before any activation function is applied.

NumPy provides a convenient operator, `@`, or the function `np.matmul()`, to
perform matrix multiplication efficiently. Let’s see it in action with a simple
example.

### Basic Matrix Multiplication with NumPy

Here’s how to compute $Z = XW$ using NumPy for a small example:

```python
import numpy as np

# Input matrix X of shape (4, 2) - 4 samples, 2 features each
X = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]])
print("Input matrix X (4x2):\n", X)

# Weight matrix W of shape (2, 3) - mapping 2 input features to 3 output features
W = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6]])
print("Weight matrix W (2x3):\n", W)

# Compute Z = X @ W, resulting in shape (4, 3)
Z = X @ W
print("Output matrix Z (4x3):\n", Z)
```

**Output**:

```
Input matrix X (4x2):
 [[1 2]
  [3 4]
  [5 6]
  [7 8]]
Weight matrix W (2x3):
 [[0.1 0.2 0.3]
  [0.4 0.5 0.6]]
Output matrix Z (4x3):
 [[0.9 1.2 1.5]
  [1.9 2.6 3.3]
  [2.9 4.0 5.1]
  [3.9 5.4 6.9]]
```

In this example, each row of $Z$ represents the output for one input sample
across three output features. For instance, the first row of $Z$ is computed as:

- $Z_{0,0} = 1 \cdot 0.1 + 2 \cdot 0.4 = 0.9$
- $Z_{0,1} = 1 \cdot 0.2 + 2 \cdot 0.5 = 1.2$
- $Z_{0,2} = 1 \cdot 0.3 + 2 \cdot 0.6 = 1.5$

This operation scales efficiently with NumPy, even for large matrices, making it
perfect for neural network computations where thousands of samples and features
are common.

---

## Building a Reusable Function: Matrix Multiplication

To make our code modular and reusable, let’s encapsulate matrix multiplication
into a function called `matrix_multiply()`. We’ll use type hints to specify the
expected input and output types for clarity and to support static type checking.
This function will be a building block for forward propagation in neural
networks.

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union

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

# Example usage with smaller matrices to verify
X_small = np.array([[1, 2, 3],
                    [4, 5, 6]], dtype=np.float64)
W_small = np.array([[0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6]], dtype=np.float64)
print("Input matrix X_small (2x3):\n", X_small)
print("Weight matrix W_small (3x2):\n", W_small)
Z_small = matrix_multiply(X_small, W_small)
print("Output matrix Z_small (2x2):\n", Z_small)
```

**Output**:

```
Input matrix X_small (2x3):
 [[1. 2. 3.]
  [4. 5. 6.]]
Weight matrix W_small (3x2):
 [[0.1 0.2]
  [0.3 0.4]
  [0.5 0.6]]
Output matrix Z_small (2x2):
 [[2.2 2.8]
  [4.9 6.2]]
```

This `matrix_multiply()` function uses `np.matmul()` under the hood, which is
optimized for matrix operations. It will be reused in later chapters when we
implement forward propagation for neural network layers. The type hints
(`NDArray[np.floating]`) indicate that we expect NumPy arrays with
floating-point values, aligning with typical neural network data types.

---

## Other Key Matrix Operations: Transpose and Dot Products

Beyond matrix multiplication, two other operations are frequently used in neural
networks: transpose and dot products. Let’s explore them briefly with NumPy.

### Transpose

The transpose of a matrix flips its rows and columns. For a matrix $X$ of shape
$(m, n)$, its transpose $X^T$ has shape $(n, m)$, and $(X^T)_{i,j} = X_{j,i}$.
Transpose is crucial in backpropagation, where we often need to align dimensions
for gradient computations.

```python
# Compute the transpose of a matrix
X = np.array([[1, 2, 3],
              [4, 5, 6]])
print("Original matrix X (2x3):\n", X)
X_transpose = np.transpose(X)
print("Transposed matrix X^T (3x2):\n", X_transpose)
```

**Output**:

```
Original matrix X (2x3):
 [[1 2 3]
  [4 5 6]]
Transposed matrix X^T (3x2):
 [[1 4]
  [2 5]
  [3 6]]
```

### Dot Product

The dot product is a special case of matrix multiplication for vectors. For two
vectors $u$ and $v$ of length $n$, the dot product is a scalar computed as:

$$
u \cdot v = \sum_{i=1}^n u_i \cdot v_i
$$

In neural networks, dot products are used within matrix multiplication (as shown
earlier) and for computing similarities or losses. NumPy’s `np.dot()` can handle
both vector dot products and matrix multiplication, though we’ll stick to `@` or
`np.matmul()` for clarity in matrix contexts.

```python
# Compute dot product of two vectors
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
dot_product = np.dot(u, v)
print("Vector u:", u)
print("Vector v:", v)
print("Dot product u · v:", dot_product)
```

**Output**:

```
Vector u: [1 2 3]
Vector v: [4 5 6]
Dot product u · v: 32
```

This result is computed as
$1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 4 + 10 + 18 = 32$.

---

## Math Behind Matrix Multiplication

Let’s recap the mathematics of matrix multiplication, as it’s central to neural
networks. Given matrices $X$ of shape $(m, n)$ and $W$ of shape $(n, p)$, their
product $Z = XW$ has shape $(m, p)$, with each element defined as:

$$
Z_{i,j} = \sum_{k=1}^n X_{i,k} \cdot W_{k,j}
$$

This formula represents a weighted sum: for each output element $Z_{i,j}$, we’re
summing the contributions of all input features (from $X$) weighted by the
corresponding weights (from $W$). In a neural network layer:

- $X$ might be input data (rows = samples, columns = features).
- $W$ contains the learned weights (rows = input features, columns = output
  features).
- $Z$ is the pre-activation output for each sample across output features.

For transpose, if $X$ is $(m, n)$, $X^T$ is $(n, m)$, swapping rows and columns.
This is often used in gradient descent to align dimensions, such as computing
gradients like $\nabla_W L = X^T \cdot \delta$.

Understanding these operations mathematically ensures we can debug and extend
our implementations as we build more complex neural networks.

---

## Exercises: Practice with Matrix Operations

To reinforce your understanding of matrix operations, try these Python-focused
coding exercises. They’ll prepare you for implementing neural network layers in
future chapters. Run the code and compare outputs to verify your solutions.

1. **Basic Matrix Multiplication**  
   Create a 3x2 matrix `X` with values `[[1, 2], [3, 4], [5, 6]]` and a 2x4
   matrix `W` with values `[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]`. Use
   the `matrix_multiply()` function to compute `Z = X @ W` and print the result.
   Verify the shape of `Z` is (3, 4).

   ```python
   # Your code here
   X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
   W = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float64)
   Z = matrix_multiply(X, W)
   print("Input matrix X (3x2):\n", X)
   print("Weight matrix W (2x4):\n", W)
   print("Output matrix Z (3x4):\n", Z)
   print("Shape of Z:", Z.shape)
   ```

2. **Transpose Operation**  
   Using the matrix `X` from Exercise 1, compute its transpose and print both
   the original and transposed matrices. Verify that the shape changes from
   (3, 2) to (2, 3).

   ```python
   # Your code here
   X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
   X_transpose = np.transpose(X)
   print("Original matrix X (3x2):\n", X)
   print("Shape of X:", X.shape)
   print("Transposed matrix X^T (2x3):\n", X_transpose)
   print("Shape of X^T:", X_transpose.shape)
   ```

3. **Dot Product of Vectors**  
   Create two vectors `u = [1, 2, 3, 4]` and `v = [0.1, 0.2, 0.3, 0.4]`. Compute
   their dot product using `np.dot()` and manually verify the result by
   calculating the sum of element-wise products.

   ```python
   # Your code here
   u = np.array([1, 2, 3, 4], dtype=np.float64)
   v = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
   dot_product = np.dot(u, v)
   print("Vector u:", u)
   print("Vector v:", v)
   print("Dot product u · v:", dot_product)
   ```

4. **Matrix Multiplication with Transpose**  
   Create a 2x3 matrix `A` with values `[[1, 2, 3], [4, 5, 6]]`. Compute its
   transpose `A_transpose`, then use `matrix_multiply()` to calculate
   `A @ A_transpose` (resulting in a 2x2 matrix). Print all steps and verify the
   shape.

   ```python
   # Your code here
   A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
   A_transpose = np.transpose(A)
   result = matrix_multiply(A, A_transpose)
   print("Matrix A (2x3):\n", A)
   print("Transpose A^T (3x2):\n", A_transpose)
   print("Result A @ A^T (2x2):\n", result)
   print("Shape of result:", result.shape)
   ```

These exercises will help you build intuition for matrix operations, which are
essential when we start coding neural network layers in upcoming modules.

---

## Closing Thoughts

Congratulations on mastering matrix operations with NumPy! In this post, we’ve
explored matrix multiplication ($Z = XW$), the cornerstone of neural network
layer computations, and built a reusable `matrix_multiply()` function for
forward propagation. We’ve also covered transpose and dot products, key tools
for aligning dimensions and computing similarities in deep learning.

In the next chapter (Part 1.3: _Mathematical Functions and Activation Basics_),
we’ll introduce NumPy’s mathematical functions and preview activation functions
like sigmoid, which add non-linearity to neural networks. This will complete
Module 1, setting a strong foundation for optimization and model building in
Module 2.

Until then, work through the exercises above to solidify your understanding. If
you have questions or want to share your solutions, leave a comment below—I’m
excited to hear from you. Let’s keep building our deep learning toolkit
together!

**Next Up**: Part 1.3 – Mathematical Functions and Activation Basics
