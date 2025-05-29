+++
title = "Linear Algebra for Machine Learning, Part 3: Matrices as Data & Transformations"
author = "Artintellica"
date = "2025-05-29"
+++

Welcome to the third post in our series on **Linear Algebra for Machine
Learning**! Having covered vectors and matrices as data and transformations, we
now turn to **matrix arithmetic**: addition, scaling, and multiplication. These
operations are the building blocks of many machine learning (ML) algorithms,
enabling linear combinations and weighted sums critical for models like neural
networks. In this post, we’ll explore the mathematics behind these operations,
their ML applications, and how to implement them in Python using **NumPy** and
**PyTorch**. We’ll also include visualizations and Python exercises to reinforce
your understanding.

---

## The Math: Matrix Arithmetic

### Matrix Addition

Matrix addition combines two matrices of the **same dimensions** by adding
corresponding elements. For two $ m \times n $ matrices $ A $ and $ B $:

$$
 C = A + B, \quad \text{where} \quad c_{ij} = a_{ij} + b\_{ij}
$$

For example, if:

$$
 A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 5
& 6 \\ 7 & 8 \end{bmatrix}
$$

then:

$$
 A + B = \begin{bmatrix} 1+5 & 2+6 \\ 3+7 & 4+8 \end{bmatrix} =
\begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}
$$

Addition is **commutative** ($ A + B = B + A $) and **associative** ($ (A + B) +
C = A + (B + C) $).

### Matrix Scaling

Scaling a matrix multiplies every element by a scalar. For a matrix $ A $ and
scalar $ k $:

$$
 B = kA, \quad \text{where} \quad b_{ij} = k \cdot a_{ij}
$$

For example, if $ k = 2 $ and $ A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$:

$$
 2A = \begin{bmatrix} 2 \cdot 1 & 2 \cdot 2 \\ 2 \cdot 3 & 2 \cdot 4
\end{bmatrix} = \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix}
$$

### Matrix Multiplication

Matrix multiplication combines two matrices to produce a new matrix,
representing the composition of linear transformations or weighted sums. For an
$ m \times n $ matrix $ A $ and an $ n \times p $ matrix $ B $, the product $ C
= AB $ is an $ m \times p $ matrix where:

$$
 c_{ij} = \sum_{k=1}^n a_{ik} b_{kj}
$$

Each element $ c\_{ij} $ is the dot product of the $ i
$-th row of $ A $ and the
$ j $-th column of $ B $. For example:

$$
 A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 5
& 6 \\ 7 & 8 \end{bmatrix}
$$

$$
 AB = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot
5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\
43 & 50 \end{bmatrix}
$$

Matrix multiplication is **not commutative** ($ AB \neq BA
$) but is
**associative** ($ (AB)C = A(BC)
$) and **distributive** over addition ($ A(B + C) = AB + AC $).

### Broadcasting

In ML, **broadcasting** allows operations on arrays of different shapes by
automatically expanding dimensions. For example, adding a scalar to a matrix or
a vector to each row/column leverages broadcasting to align shapes.

---

## ML Context: Why Matrix Arithmetic Matters

Matrix arithmetic is fundamental to ML:

- **Linear Combinations**: Matrix multiplication computes weighted sums, as in
  neural network layers ($ \mathbf{y} = W \mathbf{x} + \mathbf{b} $).
- **Data Processing**: Addition and scaling adjust datasets, such as normalizing
  features or combining predictions.
- **Optimization**: Gradient updates involve scaling and adding matrices (e.g.,
  weight updates in gradient descent).
- **Broadcasting**: Enables efficient computation on batches of data without
  explicit looping.

These operations underpin algorithms like linear regression, neural networks,
and principal component analysis (PCA).

---

## Python Code: Matrix Arithmetic

Let’s implement matrix addition, scaling, multiplication, and broadcasting using
**NumPy** and **PyTorch**, with visualizations to illustrate their effects.

### Setup

Install the required libraries if needed:

```bash
pip install numpy torch matplotlib
```

### Matrix Addition

Let’s add two matrices:

```python
import numpy as np
import matplotlib.pyplot as plt

# Define two 2x2 matrices
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# Matrix addition
C = A + B

# Print results
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("A + B:\n", C)

# Visualize matrices as heatmaps
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(A, cmap='viridis')
plt.title('Matrix A')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(B, cmap='viridis')
plt.title('Matrix B')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(C, cmap='viridis')
plt.title('A + B')
plt.colorbar()
plt.tight_layout()
plt.show()
```

**Output:**

```
Matrix A:
 [[1 2]
  [3 4]]
Matrix B:
 [[5 6]
  [7 8]]
A + B:
 [[ 6  8]
  [10 12]]
```

This code adds two $ 2 \times 2 $ matrices and visualizes them as heatmaps,
showing how corresponding elements combine.

### Matrix Scaling

Let’s scale a matrix:

```python
# Scale matrix A by 2
k = 2
scaled_A = k * A

# Print result
print("Scalar k:", k)
print("Scaled matrix (k * A):\n", scaled_A)

# Visualize
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(A, cmap='viridis')
plt.title('Matrix A')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(scaled_A, cmap='viridis')
plt.title('k * A')
plt.colorbar()
plt.tight_layout()
plt.show()
```

**Output:**

```
Scalar k: 2
Scaled matrix (k * A):
 [[2 4]
  [6 8]]
```

This scales matrix $ A $ by 2, doubling each element, and visualizes the result.

### Matrix Multiplication

Let’s multiply matrices:

```python
# Matrix multiplication
AB = A @ B  # or np.matmul(A, B)

# Print result
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("A @ B:\n", AB)
```

**Output:**

```
Matrix A:
 [[1 2]
  [3 4]]
Matrix B:
 [[5 6]
  [7 8]]
A @ B:
 [[19 22]
  [43 50]]
```

This computes $ AB $ using NumPy’s `@` operator, showing the weighted sums.

### Broadcasting

Let’s add a vector to each row of a matrix using broadcasting:

```python
# Define a vector
v = np.array([1, -1])

# Add vector to each row of A
A_plus_v = A + v  # Broadcasting automatically expands v to match A's shape

# Print result
print("Vector v:", v)
print("Matrix A:\n", A)
print("A + v (broadcasted):\n", A_plus_v)
```

**Output:**

```
Vector v: [ 1 -1]
Matrix A:
 [[1 2]
  [3 4]]
A + v (broadcasted):
 [[2 1]
  [4 3]]
```

Broadcasting adds $ v $ to each row of $ A $, equivalent to adding $
\begin{bmatrix} 1 & -1 \\ 1 & -1 \end{bmatrix} $.

### PyTorch: Matrix Operations

Let’s perform multiplication in PyTorch:

```python
import torch

# Convert to PyTorch tensors
A_torch = torch.tensor(A, dtype=torch.float32)
B_torch = torch.tensor(B, dtype=torch.float32)

# Matrix multiplication
AB_torch = A_torch @ B_torch

# Print result
print("PyTorch A @ B:\n", AB_torch.numpy())
```

**Output:**

```
PyTorch A @ B:
 [[19. 22.]
  [43. 50.]]
```

This confirms PyTorch’s results match NumPy’s.

---

## Exercises

Try these Python exercises to deepen your understanding. Solutions will be
discussed in the next post!

1. **Matrix Addition**: Create two $ 3 \times 3 $ matrices with random integers
   between 0 and 9 using NumPy. Compute their sum and visualize all three
   matrices as heatmaps.
2. **Matrix Scaling**: Scale a $ 2 \times 3 $ matrix of your choice by a scalar
   $ k = 1.5 $. Print the original and scaled matrices and visualize them.
3. **Matrix Multiplication**: Define matrices $ A = \begin{bmatrix} 1 & 2 \\ 3 &
   4 \end{bmatrix} $ and $ B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} $.
   Compute $ AB $ and $ BA $ using NumPy and print the results to verify that
   matrix multiplication is not commutative.
4. **Broadcasting**: Create a $ 4 \times 2 $ matrix and a 2D vector. Add the
   vector to each row of the matrix using broadcasting. Print the original
   matrix, vector, and result.
5. **PyTorch Verification**: Convert the matrices from Exercise 3 to PyTorch
   tensors, compute $ AB $, and verify the result matches NumPy’s.
6. **Linear Combination**: Create a $ 3 \times 2 $ matrix and two 2D vectors.
   Compute the linear combination of the matrix’s columns using the vectors as
   weights (i.e., matrix-vector multiplication). Visualize the result alongside
   the original vectors in a 2D plot.

---

## What’s Next?

In the next post, we’ll dive into **dot products and cosine similarity**,
exploring their role in measuring similarity for tasks like word embeddings and
recommendation systems. We’ll provide more Python examples and exercises to keep
building your ML intuition.

Happy learning, and see you in Part 4!
