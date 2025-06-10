+++
title = "Learn Reinforcement Learning with PyTorch, Part 1.6: Matrix Multiplication and Transpose—What, Why, and How"
author = "Artintellica"
date = "2024-06-10"
+++

## Introduction

Welcome back to Artintellica’s RL with PyTorch series! Having mastered
elementwise operations and broadcasting, you’re ready to level up: **matrix
multiplication** and **transpose** are truly foundational topics for neural
networks (layers are matrix multiplies!), data transformations, and even
understanding how RL agents learn.

In this post, you will:

- Grasp the _why_ and _how_ of matrix multiplication and transpose, in both math
  and code.
- Learn to multiply matrices “the PyTorch way” and manually with loops.
- See visually how the transpose reshapes your data and why it matters.
- Troubleshoot and fix common matrix shape errors—a crucial skill for debugging
  neural nets and RL code.

Let's dive in!

---

## Mathematics: Matrix Multiplication and Transpose

### Matrix Multiplication

Given matrices $A$ of shape $(m, n)$ and $B$ of shape $(n, p)$, the product
$C = AB$ is a new matrix of shape $(m, p)$:

$$
C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}
$$

- Each entry $C_{ij}$ is the dot product of the $i$-th row of $A$ and the $j$-th
  column of $B$.
- **Dimensional rule:** The number of _columns_ in $A$ must equal the number of
  _rows_ in $B$.

### Matrix Transpose

The **transpose** of a matrix $A$, denoted as $A^T$, swaps row and column
indices:

$$
(A^T)_{ij} = A_{ji}
$$

So, if $A$ is $m \times n$, then $A^T$ is $n \times m$.

Transposing is fundamental for aligning shapes in matrix operations.

---

## Python Demonstrations

Let's see how to do all this in PyTorch—cleanly, concisely, and reproducibly.

### Demo 1: Multiply Two Matrices Using `@` and `torch.matmul`

```python
import torch

# A: 2x3 matrix
A: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])
# B: 3x2 matrix
B: torch.Tensor = torch.tensor([[7.0, 8.0],
                               [9.0, 10.0],
                               [11.0, 12.0]])

# Method 1: Using "@" operator
C1: torch.Tensor = A @ B
print("A @ B:\n", C1)

# Method 2: Using torch.matmul
C2: torch.Tensor = torch.matmul(A, B)
print("torch.matmul(A, B):\n", C2)
```

**Output:**

```
A @ B:
 tensor([[ 58.,  64.],
         [139., 154.]])
torch.matmul(A, B):
 tensor([[ 58.,  64.],
         [139., 154.]])
```

### Demo 2: Matrix Multiplication “By Hand” Using Loops

Let’s manually implement matrix multiplication and compare the results.

```python
def matmul_manual(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "Matrix dimensions do not match!"
    C = torch.zeros((m, p), dtype=A.dtype)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

C3: torch.Tensor = matmul_manual(A, B)
print("Manual matmul(A, B):\n", C3)
print("Equal to PyTorch matmul?", torch.allclose(C1, C3))
```

### Demo 3: Visualize the Effect of Transposing a Matrix

Let’s see how the data and shape changes when transposing.

```python
import matplotlib.pyplot as plt

# Visualize data of a matrix and its transpose
M: torch.Tensor = torch.tensor([[1, 2, 3],
                                [4, 5, 6]])

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(M, cmap='viridis', aspect='auto')
axs[0].set_title('Original M\nshape={}'.format(M.shape))
axs[0].set_xlabel('Columns')
axs[0].set_ylabel('Rows')
axs[1].imshow(M.T, cmap='viridis', aspect='auto')
axs[1].set_title('Transposed M\nshape={}'.format(M.T.shape))
axs[1].set_xlabel('Columns')
axs[1].set_ylabel('Rows')
plt.tight_layout()
plt.show()
```

### Demo 4: Shape-Mismatch Error and How to Fix

A common scenario: trying to multiply incompatible matrices. Let’s see how to
catch and fix it.

```python
D: torch.Tensor = torch.tensor([[1.0, 2.0]])
E: torch.Tensor = torch.tensor([[3.0, 4.0]])
try:
    bad_result = D @ E
except RuntimeError as err:
    print("Shape mismatch error:", err)

# Fix: Transpose E
fixed_result = D @ E.T
print("Fixed result (D @ E.T):", fixed_result)
```

---

## Exercises

Practice and visualize these concepts with hands-on code!

### **Exercise 1:** Multiply Two Matrices Using `@` and `torch.matmul`

- Create matrix $M_1$ of shape $(2, 4)$ (e.g., fill with numbers from 1 to 8).
- Create matrix $M_2$ of shape $(4, 3)$ (e.g., fill with numbers 9 to 20).
- Multiply using both `@` and `torch.matmul`. Print both results—are they equal?

### **Exercise 2:** Implement Matrix Multiplication “By Hand” Using Loops and Compare

- Implement matrix multiplication manually using nested loops.
- Compare the manual result with PyTorch’s builtin `@`; confirm they are
  identical.

### **Exercise 3:** Visualize the Effect of Transposing a Matrix

- Create any $3 \times 5$ matrix with sequential values.
- Plot the matrix and its transpose side-by-side using `imshow` and print their
  shapes.

### **Exercise 4:** Explain and Fix a Common Shape-Mismatch Error in Matmul

- Intentionally attempt $X @ Y$ where $X$ is $3 \times 2$ and $Y$ is
  $3 \times 2$ (not allowed).
- Print the error.
- Fix the error by transposing $Y$ or $X$ and perform the multiplication
  successfully.

---

### **Sample Starter Code for Exercises**

```python
import torch
import matplotlib.pyplot as plt

# EXERCISE 1
M1: torch.Tensor = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
M2: torch.Tensor = torch.arange(9, 21, dtype=torch.float32).reshape(4, 3)
prod1: torch.Tensor = M1 @ M2
prod2: torch.Tensor = torch.matmul(M1, M2)
print("M1:\n", M1)
print("M2:\n", M2)
print("M1 @ M2:\n", prod1)
print("torch.matmul(M1, M2):\n", prod2)

# EXERCISE 2
def matmul_by_hand(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    m, n = X.shape
    n2, p = Y.shape
    assert n == n2, "Cannot multiply: shapes incompatible!"
    result = torch.zeros((m, p), dtype=X.dtype)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i, j] += X[i, k] * Y[k, j]
    return result

manual_out: torch.Tensor = matmul_by_hand(M1, M2)
print("Manual matmul:\n", manual_out)
print("Manual matches @ operator:", torch.allclose(prod1, manual_out))

# EXERCISE 3
A3: torch.Tensor = torch.arange(15, dtype=torch.float32).reshape(3, 5)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(A3, cmap="plasma", aspect='auto')
plt.title(f"Original (shape={A3.shape})")
plt.subplot(1, 2, 2)
plt.imshow(A3.T, cmap="plasma", aspect='auto')
plt.title(f"Transposed (shape={A3.T.shape})")
plt.tight_layout()
plt.show()
print("Original shape:", A3.shape)
print("Transposed shape:", A3.T.shape)

# EXERCISE 4
X = torch.ones(3, 2)
Y = torch.arange(6).reshape(3, 2)
try:
    wrong = X @ Y
except RuntimeError as e:
    print("Shape mismatch error:", e)
fixed = X @ Y.T  # or Y.T @ X.T
print("Fixed multiplication:\n", fixed)
```

---

## Conclusion

You’ve now mastered **matrix multiplication** and **transpose**—two of the most
common and important operations in both deep learning and reinforcement
learning.

- You know the math, the code, and the _why_.
- You can implement and debug matrix multiplies manually, and can spot and fix
  common shape mistakes.
- You saw how transposes alter shapes and why that matters for data pipelines
  and neural nets.

**Next up:** We'll explore the geometry of tensors, norms, distances, and
projections—a crucial stepping stone to understanding state spaces and rewards
in RL!

_Keep experimenting with shapes and products, and see you in Part 1.7!_
