+++
title = "Learn Reinforcement Learning with PyTorch, Part 1.5: Broadcasting and Elementwise Operations in PyTorch"
author = "Artintellica"
date = "2024-06-10"
+++

## Introduction

Welcome back to Artintellica’s open-source RL journey with PyTorch! In the last post, we learned about matrices, their construction, and basic operations. Today, we’re focusing on a feature that makes PyTorch (and NumPy) extremely expressive and concise: **broadcasting**.

Broadcasting allows you to perform elementwise operations across tensors of different shapes without manual replication, making code shorter, less error-prone, and more efficient. It's a cornerstone for neural networks, RL policy parameters, quick dataset transformations, and more.

In this post, you’ll:

- Build a solid mathematical and mental model for broadcasting.
- Use it to add row/column vectors to matrices in just one line.
- Diagnose and fix common broadcasting shape errors.
- Compare manual element-wise code to PyTorch's broadcasting.

---

## Mathematics: Broadcasting and Elementwise Operations

**Elementwise Operations:**  
Given two tensors $A$ and $B$ of the same shape, an elementwise operation (e.g., addition or multiplication) produces a tensor $C$ where $c_{ij} = f(a_{ij}, b_{ij})$ for some binary function $f$ (like $+$ or $\times$).

**Broadcasting:**  
If shapes are compatible (more below), PyTorch "broadcasts" the smaller tensor across the larger one as needed, implicitly "expanding" its dimensions so elementwise operations are still possible without extra memory cost.

**Broadcasting rules:**  
For each dimension, starting from the end:
- If the dimensions are equal, or
- If one dimension is 1,  
then the operation can proceed by expanding the size-1 dimension as needed.

For example, $A$ of shape $(m, n)$ and $b$ of shape $(n,)$:
- $b$ is broadcast to shape $(m, n)$, so $A + b$ adds $b$ to every row of $A$.

---

## Python Demonstrations

### Demo 1: Add a Row Vector to Each Row of a Matrix (Broadcasting)

```python
import torch

A: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0],
                                [7.0, 8.0, 9.0]])
row: torch.Tensor = torch.tensor([10.0, 20.0, 30.0])  # Shape: (3,)

# Broadcasting: Add row to each row of A
A_plus_row: torch.Tensor = A + row
print("Matrix A:\n", A)
print("Row vector:", row)
print("Result of A + row:\n", A_plus_row)
```

**Explanation:**  
Here, `row` is broadcast to each row of `A`, so each element of the row vector is added to the corresponding column of every row.

---

### Demo 2: Multiply a Matrix by a Column Vector (Broadcasting)

```python
col: torch.Tensor = torch.tensor([[2.0], [3.0], [4.0]])  # Shape: (3, 1)

# Broadcasting: Multiply each row of A by the corresponding element in col
A_times_col: torch.Tensor = A * col
print("Column vector:\n", col)
print("Result of A * col:\n", A_times_col)
```

**Explanation:**  
Here, the column vector (shape `(3,1)`) multiplies each row of `A` by a different scalar (2, 3, 4), thanks to broadcasting along the columns.

---

### Demo 3: Identify and Fix a Broadcasting Error

Some shapes cannot be broadcast due to mismatch. Let's see an example and fix it:

```python
try:
    bad_vec: torch.Tensor = torch.tensor([1.0, 2.0])
    # A has shape (3,3); bad_vec has shape (2,). Not broadcastable!
    res = A + bad_vec
except RuntimeError as e:
    print("Broadcasting error:", e)

# Fix: Use a vector of shape (3,) or reshape for broadcasting compatibility
good_vec: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
fixed_res: torch.Tensor = A + good_vec
print("Fixed result (A + good_vec):\n", fixed_res)
```

---

### Demo 4: Compare Manual Elementwise Operations with PyTorch's Broadcasting

You could implement element-wise addition using loops, but broadcasting is faster and more readable.

```python
# Manual using loops (for small examples)
manual_sum: torch.Tensor = torch.empty_like(A)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        manual_sum[i, j] = A[i, j] + row[j]

print("Manual elementwise sum:\n", manual_sum)

# PyTorch broadcasting (just one line)
broadcast_sum: torch.Tensor = A + row
print("Broadcast sum:\n", broadcast_sum)

# Confirm equality
print("Are they equal?", torch.allclose(manual_sum, broadcast_sum))
```

---

## Exercises

Try these in a Python file or Jupyter notebook:

### **Exercise 1:** Add a Row Vector to Each Row of a Matrix Using Broadcasting

- Create a $3 \times 4$ matrix `M` (e.g., `torch.arange(12).reshape(3,4)`).
- Create a row vector `v` of length 4 (e.g., `[1.0, 10.0, 100.0, 1000.0]`).
- Use broadcasting to add `v` to each row of `M`. Print the result.

### **Exercise 2:** Multiply a Matrix by a Column Vector Using Broadcasting

- Create a $4 \times 2$ matrix `N` (e.g., with `torch.tensor` or `torch.arange`).
- Create a column vector of shape $(4,1)$ (e.g., [[2.], [4.], [6.], [8.]]).
- Use broadcasting to multiply `N` and the column vector. Print the result.

### **Exercise 3:** Identify and Fix a Broadcasting Error in Code

- Intentionally try to broadcast a $(5,3)$ matrix with a vector of shape $(2,)$.
- Observe and print the error message.
- Fix the shapes so that broadcasting works (e.g., use a vector of shape $(3,)$ or $(5,1)$).

### **Exercise 4:** Compare Manual Elementwise Operations with PyTorch’s Operators

- For a $2 \times 3$ matrix `P` and a row vector `w` of length 3,
    - Compute `P + w` using explicit loops and save to `P_manual`.
    - Compute `P + w` using broadcasting (`P + w`) and save as `P_broadcast`.
    - Print both results and confirm they are identical.

---

### **Sample Starter Code for Exercises**

```python
import torch

# EXERCISE 1
M: torch.Tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
v: torch.Tensor = torch.tensor([1.0, 10.0, 100.0, 1000.0])
M_plus_v: torch.Tensor = M + v
print("M:\n", M)
print("v:", v)
print("M + v:\n", M_plus_v)

# EXERCISE 2
N: torch.Tensor = torch.tensor([[1.0, 2.0],
                                [3.0, 4.0],
                                [5.0, 6.0],
                                [7.0, 8.0]])
col_vec: torch.Tensor = torch.tensor([[2.0],
                                      [4.0],
                                      [6.0],
                                      [8.0]])
N_times_col: torch.Tensor = N * col_vec
print("N:\n", N)
print("col_vec:\n", col_vec)
print("N * col_vec:\n", N_times_col)

# EXERCISE 3
try:
    bad_mat: torch.Tensor = torch.ones(5, 3)
    bad_vec: torch.Tensor = torch.arange(2)
    result_bad = bad_mat + bad_vec
except RuntimeError as e:
    print("Broadcasting error:", e)

# Fixed version
good_vec: torch.Tensor = torch.tensor([10.0, 20.0, 30.0])
result_good = bad_mat + good_vec
print("Fixed broadcast:\n", result_good)

# EXERCISE 4
P: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])
w: torch.Tensor = torch.tensor([7.0, 8.0, 9.0])
# Manual addition
P_manual: torch.Tensor = torch.empty_like(P)
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        P_manual[i, j] = P[i, j] + w[j]
print("P_manual:\n", P_manual)
# PyTorch broadcast
P_broadcast: torch.Tensor = P + w
print("P_broadcast:\n", P_broadcast)
# Are they the same?
print("Equal?", torch.allclose(P_manual, P_broadcast))
```

---

## Conclusion

In this post, you learned how broadcasting makes your tensor code more elegant, efficient, and readable. You can now:

- Use broadcasting to add row and column vectors to matrices seamlessly.
- Diagnose and fix shape incompatibilities in elementwise operations.
- Appreciate how PyTorch's built-in operators allow you to avoid explicit Python loops.

**Next Time:** We'll explore **matrix multiplication and the transpose**—the foundation of neural network layers and linear transformations in RL. Make sure to experiment with manual and broadcasted operations, as shape errors are the #1 source of data bugs in deep learning and RL!

*See you in Part 1.6!*
