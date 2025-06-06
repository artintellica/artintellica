+++
title = "Learn Deep Learning with NumPy, Part 1.1: Getting Started with NumPy Arrays"
author = "Artintellica"
date = "2025-06-05"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0042-deep-numpy-11-arrays"
+++

## Introduction

Welcome to the first chapter of our blog series, _"Learn Deep Learning with
NumPy"_! In this post, we’re kicking off Module 1 by diving into the foundation
of numerical computing in Python—NumPy arrays. NumPy is a powerful library that
will serve as our primary tool for building neural networks from scratch. Today,
we’ll explore how to create, manipulate, and perform basic operations with
arrays, and we’ll see why they’re ideal for the vectorized computations at the
heart of deep learning.

By the end of this post, you’ll be comfortable with NumPy basics, understand key
concepts like array shapes and broadcasting, and have written your first
reusable function for data preprocessing. Let’s get started with the building
blocks of our deep learning journey!

---

## Why NumPy for Deep Learning?

Deep learning relies heavily on mathematical operations over large collections
of numbers—think matrices of weights, inputs, or gradients in a neural network.
Performing these operations efficiently is crucial, and that’s where NumPy
shines. NumPy provides a data structure called an _array_ (similar to a list or
matrix) that allows for fast, vectorized computations without the need for slow
Python loops. Instead of processing elements one by one, NumPy operates on
entire arrays at once, which is exactly how neural networks compute layer
outputs or update parameters.

In this series, we’ll use NumPy arrays to represent everything from input data
to model weights. Understanding arrays is the first step toward coding neural
networks, as operations like matrix multiplication ($Z = XW$) or element-wise
activation functions ($A = \text{sigmoid}(Z)$) are naturally expressed with
arrays. Let’s dive into creating and manipulating them.

---

## Getting Started with NumPy Arrays

First, ensure you have NumPy installed. If you haven’t already, run the
following command in your terminal or command prompt:

```bash
pip install numpy
```

Now, let’s import NumPy in Python and explore array creation. Open your
preferred editor or Jupyter Notebook, and follow along with the code.

### Creating Arrays

NumPy arrays can be created in several ways. Here are the most common methods
we’ll use in deep learning:

```python
import numpy as np

# Create an array from a list
X = np.array([[1, 2, 3], [4, 5, 6]])
print("Array from list:\n", X)

# Create an array of zeros (useful for initializing biases or placeholders)
zeros = np.zeros((2, 3))
print("Array of zeros:\n", zeros)

# Create an array of random numbers (useful for initializing weights)
W = np.random.randn(3, 2)  # Standard normal distribution (mean=0, std=1)
print("Random array:\n", W)
```

**Output** (random values will vary):

```
Array from list:
 [[1 2 3]
  [4 5 6]]
Array of zeros:
 [[0. 0. 0.]
  [0. 0. 0.]]
Random array:
 [[ 0.123 -0.456]
  [-0.789  0.321]
  [ 0.654 -0.987]]
```

Notice the shape of each array: `X` is 2x3 (2 rows, 3 columns), `zeros` is 2x3,
and `W` is 3x2. The shape is a fundamental property of arrays, accessed via
`X.shape`, and it dictates how operations like matrix multiplication work in
neural networks.

### Array Shapes and Dimensions

Understanding an array’s shape is critical because neural network operations
often require matching dimensions. For example, to compute a layer’s output with
$Z = XW$, the number of columns in $X$ must equal the number of rows in $W$.
Let’s inspect shapes and dimensions:

```python
# Check the shape of an array
print("Shape of X:", X.shape)  # (2, 3)
print("Number of dimensions of X:", X.ndim)  # 2 (a 2D array/matrix)

# Reshape an array (must maintain total number of elements)
X_reshaped = X.reshape(3, 2)
print("Reshaped X to 3x2:\n", X_reshaped)
```

**Output**:

```
Shape of X: (2, 3)
Number of dimensions of X: 2
Reshaped X to 3x2:
 [[1 2]
  [3 4]
  [5 6]]
```

Reshaping is handy in deep learning, for example, when flattening an image
(e.g., a 28x28 pixel grid) into a 784-element vector for input to a neural
network.

---

## Basic Array Operations

NumPy arrays support a variety of operations that are essential for neural
network computations. Let’s explore element-wise operations and broadcasting,
which allow us to avoid loops and write concise code.

### Element-Wise Operations

Element-wise operations apply a function to each element of an array
independently. These are used in neural networks for tasks like adding biases or
applying activation functions.

```python
# Element-wise addition
X_plus_5 = X + 5
print("X + 5:\n", X_plus_5)

# Element-wise multiplication
X_times_2 = X * 2
print("X * 2:\n", X_times_2)
```

**Output**:

```
X + 5:
 [[ 6  7  8]
  [ 9 10 11]]
X * 2:
 [[ 2  4  6]
  [ 8 10 12]]
```

### Broadcasting

Broadcasting is a powerful feature where NumPy automatically expands a smaller
array (or scalar) to match the shape of a larger array during operations. This
is useful for adding a single bias value to an entire row or column in a neural
network layer.

```python
# Broadcasting a scalar across an array
bias = 10
X_with_bias = X + bias
print("X with broadcasted bias:\n", X_with_bias)

# Broadcasting a 1D array across rows
row_bias = np.array([1, 2, 3])
X_with_row_bias = X + row_bias
print("X with row bias broadcasted:\n", X_with_row_bias)
```

**Output**:

```
X with broadcasted bias:
 [[11 12 13]
  [14 15 16]]
X with row bias broadcasted:
 [[2 4 6]
  [5 7 9]]
```

Broadcasting saves us from writing explicit loops, making our code faster and
cleaner—perfect for neural network operations where we often add biases or scale
entire matrices.

---

## Indexing and Slicing Arrays

In deep learning, we often need to access parts of an array, such as selecting a
subset of data samples or specific features. NumPy provides intuitive indexing
and slicing for this purpose.

```python
# Access a single element
element = X[0, 1]  # Row 0, Column 1
print("Element at (0,1):", element)

# Slice rows or columns
first_row = X[0, :]  # All columns of row 0
first_column = X[:, 0]  # All rows of column 0
print("First row:", first_row)
print("First column:", first_column)

# Select a submatrix
submatrix = X[0:2, 1:3]  # Rows 0-1, Columns 1-2
print("Submatrix:\n", submatrix)
```

**Output**:

```
Element at (0,1): 2
First row: [1 2 3]
First column: [1 4]
Submatrix:
 [[2 3]
  [5 6]]
```

Slicing is particularly useful when working with datasets like MNIST, where you
might need to extract batches of images or specific pixel values.

---

## Building Our First Reusable Function: Normalization

Data preprocessing is a crucial step in deep learning. Neural networks often
perform better when input data is normalized—scaled to have a mean of 0 and a
standard deviation of 1. Let’s write a `normalize()` function to preprocess
arrays, which we’ll reuse in later chapters (e.g., for MNIST images).

Here’s the implementation with type hints for parameters and return values,
along with an example:

```python
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

# Example: Normalize a random 3x2 matrix
random_matrix = np.random.randn(3, 2)
print("Original matrix:\n", random_matrix)
normalized_matrix = normalize(random_matrix)
print("Normalized matrix (mean≈0, std≈1):\n", normalized_matrix)
print("Mean after normalization:", np.mean(normalized_matrix))
print("Std after normalization:", np.std(normalized_matrix))
```

**Output** (values will vary due to randomness):

```
Original matrix:
 [[ 0.123 -0.456]
  [-0.789  0.321]
  [ 0.654 -0.987]]
Normalized matrix (mean≈0, std≈1):
 [[ 0.345 -0.234]
  [-0.987  0.543]
  [ 0.876 -1.234]]
Mean after normalization: 1.1102230246251565e-17  # Approximately 0
Std after normalization: 0.9999999999999999      # Approximately 1
```

Normalization ensures that data across different scales (e.g., pixel values from
0 to 255 in images) is brought to a consistent range, helping neural networks
train faster and more reliably. This `normalize()` function will be a key part
of our growing library. The type hints (`NDArray[np.floating]`) specify that the
input and output are NumPy arrays with floating-point values, improving code
readability and enabling static type checking with tools like `mypy`.

---

## Math Behind Array Operations

Let’s briefly touch on the mathematics of array operations, as they underpin
neural network computations. For an array $X$ of shape $(m, n)$:

- **Element-wise addition**: Adding a scalar $c$ to $X$ results in a new array
  where each element $X_{i,j}$ becomes $X_{i,j} + c$.
- **Element-wise multiplication**: Multiplying $X$ by a scalar $c$ results in
  each element becoming $c \cdot X_{i,j}$.
- **Broadcasting**: If you add a vector $v$ of shape $(n,)$ to $X$, NumPy
  replicates $v$ across all $m$ rows, effectively computing $X_{i,j} + v_j$ for
  each element.

For normalization, if $X$ has elements $x_1, x_2, \dots, x_k$ (where
$k = m \cdot n$), we compute:

$$
\text{mean} = \frac{1}{k} \sum_{i=1}^k x_i, \quad \text{std} = \sqrt{\frac{1}{k} \sum_{i=1}^k (x_i - \text{mean})^2}
$$

Then, the normalized array has elements:

$$
\text{normalized}_i = \frac{x_i - \text{mean}}{\text{std}}
$$

These operations are vectorized in NumPy, meaning they’re computed over entire
arrays at once, which is much faster than element-by-element loops—a critical
advantage for neural networks handling thousands or millions of values.

---

## Exercises: Practice with NumPy Arrays

To solidify your understanding, try these Python-focused coding exercises.
They’re designed to get you comfortable with NumPy arrays and prepare you for
neural network implementations. Solutions can be checked by running the code and
comparing outputs.

1. **Array Creation and Shapes**  
   Create a 4x3 array of ones using `np.ones()`, then reshape it into a 3x4
   array. Print the original and reshaped arrays along with their shapes.

   ```python
   # Your code here
   ones = np.ones((4, 3))
   print("Original 4x3 array of ones:\n", ones)
   print("Shape:", ones.shape)
   reshaped_ones = ones.reshape(3, 4)
   print("Reshaped 3x4 array:\n", reshaped_ones)
   print("New shape:", reshaped_ones.shape)
   ```

2. **Element-Wise Operations and Broadcasting**  
   Create a 2x3 array with values `[[1, 2, 3], [4, 5, 6]]`. Add a row vector
   `[10, 20, 30]` to it using broadcasting, then multiply the result by 2. Print
   each step.

   ```python
   # Your code here
   X = np.array([[1, 2, 3], [4, 5, 6]])
   row_vec = np.array([10, 20, 30])
   X_with_row = X + row_vec
   print("After adding row vector:\n", X_with_row)
   X_scaled = X_with_row * 2
   print("After multiplying by 2:\n", X_scaled)
   ```

3. **Slicing Practice**  
   Using the same 2x3 array from Exercise 2, extract the second row and the
   first two columns as a submatrix. Print the results.

   ```python
   # Your code here
   X = np.array([[1, 2, 3], [4, 5, 6]])
   second_row = X[1, :]
   first_two_cols = X[:, 0:2]
   print("Second row:", second_row)
   print("First two columns:\n", first_two_cols)
   ```

4. **Normalization Application**  
   Generate a 5x2 random matrix using `np.random.randn(5, 2)`, apply the
   `normalize()` function we wrote, and verify that the mean is approximately 0
   and the standard deviation is approximately 1.

   ```python
   # Your code here
   random_data = np.random.randn(5, 2)
   print("Original random matrix:\n", random_data)
   normalized_data = normalize(random_data)
   print("Normalized matrix:\n", normalized_data)
   print("Mean after normalization:", np.mean(normalized_data))
   print("Std after normalization:", np.std(normalized_data))
   ```

These exercises reinforce the core concepts of array manipulation, which we’ll
build upon when implementing neural network layers and data preprocessing
pipelines.

---

## Closing Thoughts

Congratulations on completing your first step into deep learning with NumPy! In
this post, we’ve introduced NumPy arrays, explored their creation and
manipulation, and written a reusable `normalize()` function for data
preprocessing. Arrays are the cornerstone of neural networks because they enable
fast, vectorized operations—whether it’s adding biases, computing layer outputs,
or normalizing inputs.

In the next chapter (Part 1.2: _Matrix Operations for Neural Networks_), we’ll
dive deeper into linear algebra with NumPy, focusing on matrix multiplication
($Z = XW$), a key operation for computing neural network layer outputs. We’ll
build another reusable function and see how these concepts directly apply to
forward propagation.

Until then, experiment with the code and exercises above. If you have questions
or want to share your solutions, drop a comment below—I’d love to hear from you.
Let’s keep building our deep learning toolkit together!

**Next Up**: Part 1.2 – Matrix Operations for Neural Networks
