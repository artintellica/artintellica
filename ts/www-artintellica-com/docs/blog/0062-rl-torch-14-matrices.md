+++
title = "Learn Reinforcement Learning with PyTorch, Part 1.4: Matrices—Construction, Shapes, and Basic Operations"
author = "Artintellica"
date = "2024-06-10"
+++

## Introduction

Welcome back to Artintellica’s open-source Reinforcement Learning course with
PyTorch! After mastering vector operations like addition, scalar multiplication,
and dot products, it’s time to step into the world of **matrices**—2D tensors
that are fundamental to machine learning and reinforcement learning (RL).
Matrices represent linear transformations, weight layers in neural networks, and
transition models in RL environments, making them indispensable.

In this post, you will:

- Learn how to construct matrices as 2D tensors in PyTorch and understand their
  shapes.
- Explore basic matrix operations like transposition, element-wise addition, and
  multiplication.
- Understand broadcasting between matrices and vectors, a powerful feature of
  PyTorch.
- Practice these concepts with hands-on coding exercises.

Let’s build on our vector knowledge and dive into the 2D realm!

---

## Mathematics: Matrices and Basic Operations

A **matrix** is a rectangular array of numbers arranged in rows and columns,
often denoted as $A \in \mathbb{R}^{m \times n}$, where $m$ is the number of
rows and $n$ is the number of columns. For example, a $2 \times 3$ matrix looks
like:

$$
A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23}
\end{bmatrix}
$$

In PyTorch, matrices are represented as 2D tensors with shape `(m, n)`.

### Key Concepts and Operations

1. **Shape**: The dimensions of a matrix, e.g., `(rows, columns)`. Shape
   determines compatibility for operations.
2. **Transpose**: The transpose of a matrix $A$, denoted $A^T$, swaps rows and
   columns, so $A^T_{ij} = A_{ji}$. For the above matrix, $A^T$ is a
   $3 \times 2$ matrix.
3. **Element-wise Addition**: For two matrices $A$ and $B$ of the same shape,
   $C = A + B$ means $c_{ij} = a_{ij} + b_{ij}$.
4. **Element-wise Multiplication**: Similarly, $C = A * B$ means
   $c_{ij} = a_{ij} \cdot b_{ij}$ (also called the Hadamard product).
5. **Broadcasting**: PyTorch can automatically expand smaller tensors to match
   the shape of larger ones during operations, enabling efficient computation
   without explicit looping.

These operations are foundational for neural network layers (where weights are
matrices) and RL algorithms (where matrices might represent state transitions or
policy mappings).

---

## Python Demonstrations

Let’s see how to work with matrices in PyTorch. We’ll construct matrices,
inspect their shapes, and perform basic operations.

### Demo 1: Creating Matrices with Specific Shapes

```python
import torch

# Create a 2x3 matrix (2 rows, 3 columns)
A: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])
print("Matrix A:\n", A)
print("Shape of A:", A.shape)

# Create a 3x2 matrix using arange and reshape
B: torch.Tensor = torch.arange(6, dtype=torch.float32).reshape(3, 2)
print("Matrix B:\n", B)
print("Shape of B:", B.shape)
```

**Expected Output:**

```
Matrix A:
 tensor([[1., 2., 3.],
         [4., 5., 6.]])
Shape of A: torch.Size([2, 3])
Matrix B:
 tensor([[0., 1.],
         [2., 3.],
         [4., 5.]])
Shape of B: torch.Size([3, 2])
```

### Demo 2: Transposing a Matrix

```python
# Transpose matrix A (2x3 -> 3x2)
A_transpose: torch.Tensor = A.T
print("Transpose of A (A^T):\n", A_transpose)
print("Shape of A^T:", A_transpose.shape)
```

**Expected Output:**

```
Transpose of A (A^T):
 tensor([[1., 4.],
         [2., 5.],
         [3., 6.]])
Shape of A^T: torch.Size([3, 2])
```

### Demo 3: Element-wise Addition and Multiplication

```python
# Create two 2x3 matrices for element-wise operations
C: torch.Tensor = torch.tensor([[0.5, 1.5, 2.5],
                               [3.5, 4.5, 5.5]])

# Element-wise addition
D_add: torch.Tensor = A + C
print("Element-wise addition (A + C):\n", D_add)

# Element-wise multiplication (Hadamard product)
D_mul: torch.Tensor = A * C
print("Element-wise multiplication (A * C):\n", D_mul)
```

**Expected Output:**

```
Element-wise addition (A + C):
 tensor([[1.5, 3.5, 5.5],
         [7.5, 9.5, 11.5]])
Element-wise multiplication (A * C):
 tensor([[0.5, 3.0, 7.5],
         [14.0, 22.5, 33.0]])
```

### Demo 4: Broadcasting with a Matrix and a Vector

Broadcasting allows PyTorch to align dimensions automatically. Let’s add a
vector to each row of a matrix.

```python
# 1D vector to broadcast
vec: torch.Tensor = torch.tensor([10.0, 20.0, 30.0])
print("Vector to broadcast:", vec)
print("Shape of vec:", vec.shape)

# Broadcasting: Add vec to each row of A
result_broadcast: torch.Tensor = A + vec
print("A + broadcasted vec:\n", result_broadcast)
```

**Expected Output:**

```
Vector to broadcast: tensor([10., 20., 30.])
Shape of vec: torch.Size([3])
A + broadcasted vec:
 tensor([[11., 22., 33.],
         [14., 25., 36.]])
```

Here, `vec` (shape `(3,)`) is broadcasted to match the shape of `A` (shape
`(2, 3)`), effectively adding it to each row.

---

## Exercises

Let’s apply these concepts with hands-on coding tasks. Use a new Python script
or Jupyter notebook for these exercises.

### **Exercise 1: Create 2D Tensors (Matrices) with Specific Shapes**

- Create a $3 \times 4$ matrix `M1` filled with sequential numbers from 0 to 11
  (use `torch.arange` and `reshape`).
- Create a $2 \times 5$ matrix `M2` filled with random numbers between 0 and 1
  (use `torch.rand`).
- Print both matrices and their shapes.

### **Exercise 2: Transpose a Matrix and Verify with PyTorch**

- Take matrix `M1` from Exercise 1 and compute its transpose `M1_T`.
- Verify the shape of `M1_T` is `(4, 3)`.
- Print both `M1` and `M1_T` to confirm the rows and columns are swapped.

### **Exercise 3: Perform Element-wise Addition and Multiplication on Two Matrices**

- Create two $3 \times 3$ matrices `M3` and `M4` with any float values (e.g.,
  use `torch.tensor` manually or `torch.ones`/`torch.full`).
- Compute their element-wise sum and product.
- Print the original matrices and the results.

### **Exercise 4: Demonstrate Broadcasting with a Matrix and a Vector**

- Create a $4 \times 3$ matrix `M5` with any values.
- Create a 1D vector `v1` of length 3 with any values.
- Use broadcasting to add `v1` to each row of `M5`.
- Print the original matrix, vector, and result to confirm the operation.

---

### **Sample Starter Code for Exercises**

```python
import torch

# EXERCISE 1: Create Matrices with Specific Shapes
M1: torch.Tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
M2: torch.Tensor = torch.rand(2, 5)
print("M1 (3x4):\n", M1)
print("M1 shape:", M1.shape)
print("M2 (2x5):\n", M2)
print("M2 shape:", M2.shape)

# EXERCISE 2: Transpose a Matrix
M1_T: torch.Tensor = M1.T
print("Transpose of M1 (M1^T):\n", M1_T)
print("Shape of M1^T:", M1_T.shape)

# EXERCISE 3: Element-wise Operations
M3: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0],
                                [7.0, 8.0, 9.0]])
M4: torch.Tensor = torch.tensor([[0.5, 1.5, 2.5],
                                [3.5, 4.5, 5.5],
                                [6.5, 7.5, 8.5]])
M3_add_M4: torch.Tensor = M3 + M4
M3_mul_M4: torch.Tensor = M3 * M4
print("M3:\n", M3)
print("M4:\n", M4)
print("Element-wise sum (M3 + M4):\n", M3_add_M4)
print("Element-wise product (M3 * M4):\n", M3_mul_M4)

# EXERCISE 4: Broadcasting with Matrix and Vector
M5: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0],
                                [7.0, 8.0, 9.0],
                                [10.0, 11.0, 12.0]])
v1: torch.Tensor = torch.tensor([10.0, 20.0, 30.0])
result_broadcast: torch.Tensor = M5 + v1
print("M5 (4x3):\n", M5)
print("Vector v1:", v1)
print("M5 + broadcasted v1:\n", result_broadcast)
```

---

## Conclusion

In this post, you’ve expanded your PyTorch toolkit by mastering matrices—2D
tensors that are critical for representing and manipulating data in
reinforcement learning and beyond. You’ve learned:

- How to construct matrices with specific shapes and inspect their dimensions.
- How to transpose matrices, swapping rows and columns.
- How to perform element-wise operations like addition and multiplication.
- How broadcasting enables efficient operations between matrices and vectors.

**Next Up:** In Part 1.5, we’ll dive deeper into **broadcasting and element-wise
operations**, exploring more complex scenarios and pitfalls to avoid. Matrices
are the gateway to neural networks and linear transformations, so keep
practicing these basics—they’ll pay off as we move toward RL algorithms!

_See you in the next post!_
