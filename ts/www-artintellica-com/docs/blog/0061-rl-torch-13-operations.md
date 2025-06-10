+++
title = "Learn Reinforcement Learning with PyTorch, Part 1.3: Basic Vector Operations—Addition, Scalar Multiplication, and Dot Product"
author = "Artintellica"
date = "2024-06-09"
+++

## Introduction

Welcome back to Artintellica’s open-source Reinforcement Learning course with
PyTorch! Having mastered creating, reshaping, and visualizing vectors and
scalars in the previous post, we’re now ready to perform fundamental operations
on them. Today, we’ll explore **vector addition**, **scalar multiplication**,
and the **dot product**—operations that are at the heart of nearly every
computation in machine learning and reinforcement learning (RL).

In this post, you will:

- Understand the mathematical definitions and intuitions behind these
  operations.
- See how PyTorch makes them intuitive and efficient with tensor operations.
- Apply these concepts through hands-on code demonstrations.
- Practice with exercises, including visualizing vector operations in 2D space.

These operations are building blocks for everything from updating agent
parameters in RL to computing rewards or state transitions. Let’s dive in!

---

## Mathematics: Vector Operations

Let’s start with the mathematical foundations of the operations we’ll implement.
We’ll use vectors in $\mathbb{R}^n$, meaning they have $n$ components.

### Vector Addition

Given two vectors $\mathbf{u} = [u_1, u_2, \ldots, u_n]$ and
$\mathbf{v} = [v_1, v_2, \ldots, v_n]$ of the same dimension, their sum is
defined element-wise:

$$
\mathbf{w} = \mathbf{u} + \mathbf{v} = [u_1 + v_1, u_2 + v_2, \ldots, u_n + v_n]
$$

Geometrically, this corresponds to placing the tail of $\mathbf{v}$ at the head
of $\mathbf{u}$ and finding the resulting vector from the tail of $\mathbf{u}$
to the head of $\mathbf{v}$.

### Scalar Multiplication

Multiplying a vector $\mathbf{u}$ by a scalar $c \in \mathbb{R}$ scales each
component:

$$
c \cdot \mathbf{u} = [c \cdot u_1, c \cdot u_2, \ldots, c \cdot u_n]
$$

Geometrically, this stretches (if $|c| > 1$), shrinks (if $|c| < 1$), or
reverses (if $c < 0$) the vector while keeping its direction (or flipping it if
negative).

### Dot Product

The dot product of two vectors $\mathbf{u}$ and $\mathbf{v}$ of the same
dimension is a scalar value computed as:

$$
\mathbf{u} \cdot \mathbf{v} = u_1 \cdot v_1 + u_2 \cdot v_2 + \ldots + u_n \cdot v_n
$$

Geometrically, it measures how much one vector "goes in the direction" of
another and is related to the angle $\theta$ between them via:

$$
\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \cdot \|\mathbf{v}\| \cdot \cos(\theta)
$$

where $\|\mathbf{u}\|$ is the magnitude (Euclidean norm) of $\mathbf{u}$. In RL,
dot products are often used in similarity measures or projections.

---

## Python Demonstrations

Let’s implement these operations using PyTorch tensors. We’ll work with small
vectors for clarity, but these operations scale to any dimension.

### Demo 1: Vector Addition

```python
import torch

# Define two vectors
u: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
v: torch.Tensor = torch.tensor([4.0, 5.0, 6.0])

# Element-wise addition
w: torch.Tensor = u + v
print("u:", u)
print("v:", v)
print("u + v:", w)
```

**Expected Output:**

```
u: tensor([1., 2., 3.])
v: tensor([4., 5., 6.])
u + v: tensor([5., 7., 9.])
```

### Demo 2: Scalar Multiplication

```python
# Scalar multiplication
c: float = 2.0
scaled_u: torch.Tensor = c * u
print(f"u scaled by {c}:", scaled_u)
```

**Expected Output:**

```
u scaled by 2.0: tensor([2., 4., 6.])
```

PyTorch uses **broadcasting** to apply the scalar across all elements, so you
don’t need to loop manually.

### Demo 3: Dot Product (Manual and Built-In)

```python
# Manual dot product using element-wise multiplication and sum
manual_dot: torch.Tensor = (u * v).sum()
print("Manual dot product (u · v):", manual_dot.item())

# Built-in dot product
builtin_dot: torch.Tensor = torch.dot(u, v)
print("Built-in dot product (u · v):", builtin_dot.item())
```

**Expected Output:**

```
Manual dot product (u · v): 32.0
Built-in dot product (u · v): 32.0
```

Both methods yield the same result:
$1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 4 + 10 + 18 = 32$.

### Demo 4: Visualizing Vector Addition and Scalar Multiplication in 2D

Visualization helps build geometric intuition. We’ll use 2D vectors for
simplicity.

```python
import matplotlib.pyplot as plt

# 2D vectors
u_2d: torch.Tensor = torch.tensor([1.0, 2.0])
v_2d: torch.Tensor = torch.tensor([2.0, 1.0])
sum_uv: torch.Tensor = u_2d + v_2d
scaled_u: torch.Tensor = 1.5 * u_2d

# Plotting
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, u_2d[0], u_2d[1], angles='xy', scale_units='xy', scale=1, color='b', label='u')
plt.quiver(0, 0, v_2d[0], v_2d[1], angles='xy', scale_units='xy', scale=1, color='r', label='v')
plt.quiver(0, 0, sum_uv[0], sum_uv[1], angles='xy', scale_units='xy', scale=1, color='g', label='u + v')
plt.quiver(0, 0, scaled_u[0], scaled_u[1], angles='xy', scale_units='xy', scale=1, color='purple', label='1.5 * u')

plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title("Vector Addition and Scalar Multiplication in 2D")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
```

This plots vectors as arrows from the origin, showing how addition forms a
parallelogram resultant and how scaling stretches a vector.

---

## Exercises

Let’s apply what you’ve learned with hands-on coding tasks. Use a new Python
script or Jupyter notebook for these exercises.

### **Exercise 1: Implement Vector Addition Using PyTorch**

- Create two 1D vectors $\mathbf{a} = [2, 3, 4, 5]$ and
  $\mathbf{b} = [1, 1, 1, 1]$ as float tensors.
- Compute their sum $\mathbf{c} = \mathbf{a} + \mathbf{b}$ using PyTorch.
- Print all three vectors.

### **Exercise 2: Scale a Vector by a Scalar (Both Manual and Using Broadcasting)**

- Take vector $\mathbf{a}$ from Exercise 1 and scale it by $c = 3.0$.
- First, do this manually by multiplying each element (using a loop or
  element-wise operation).
- Then, use PyTorch’s broadcasting (direct multiplication).
- Print both results to confirm they match.

### **Exercise 3: Compute the Dot Product of Two Vectors (Manual and Built-In)**

- Using vectors $\mathbf{a}$ and $\mathbf{b}$ from Exercise 1, compute their dot
  product manually (element-wise multiply and sum).
- Compute the dot product using PyTorch’s built-in `torch.dot` function.
- Print both results to confirm they are identical.

### **Exercise 4: Visualize Vector Addition and Scalar Multiplication in 2D with Matplotlib**

- Create two 2D vectors $\mathbf{d} = [1, 3]$ and $\mathbf{e} = [2, 1]$.
- Compute their sum $\mathbf{f} = \mathbf{d} + \mathbf{e}$ and a scaled version
  of $\mathbf{d}$ by $c = 0.5$.
- Plot all four vectors ($\mathbf{d}$, $\mathbf{e}$, $\mathbf{f}$, and scaled
  $\mathbf{d}$) using `plt.quiver` as arrows from the origin, with different
  colors and a legend.

---

### **Sample Starter Code for Exercises**

```python
import torch
import matplotlib.pyplot as plt

# EXERCISE 1: Vector Addition
a: torch.Tensor = torch.tensor([2.0, 3.0, 4.0, 5.0])
b: torch.Tensor = torch.tensor([1.0, 1.0, 1.0, 1.0])
c: torch.Tensor = a + b
print("a:", a)
print("b:", b)
print("c = a + b:", c)

# EXERCISE 2: Scalar Multiplication
c_scalar: float = 3.0
# Manual scaling (using element-wise operation for simplicity)
manual_scaled_a: torch.Tensor = torch.tensor([c_scalar * x for x in a])
# Broadcasting scaling
broadcast_scaled_a: torch.Tensor = c_scalar * a
print("Manually scaled a:", manual_scaled_a)
print("Broadcast scaled a:", broadcast_scaled_a)

# EXERCISE 3: Dot Product
manual_dot: torch.Tensor = (a * b).sum()
builtin_dot: torch.Tensor = torch.dot(a, b)
print("Manual dot product (a · b):", manual_dot.item())
print("Built-in dot product (a · b):", builtin_dot.item())

# EXERCISE 4: Visualization in 2D
d: torch.Tensor = torch.tensor([1.0, 3.0])
e: torch.Tensor = torch.tensor([2.0, 1.0])
f: torch.Tensor = d + e
scaled_d: torch.Tensor = 0.5 * d

plt.figure(figsize=(6, 6))
plt.quiver(0, 0, d[0], d[1], angles='xy', scale_units='xy', scale=1, color='b', label='d')
plt.quiver(0, 0, e[0], e[1], angles='xy', scale_units='xy', scale=1, color='r', label='e')
plt.quiver(0, 0, f[0], f[1], angles='xy', scale_units='xy', scale=1, color='g', label='d + e')
plt.quiver(0, 0, scaled_d[0], scaled_d[1], angles='xy', scale_units='xy', scale=1, color='purple', label='0.5 * d')

plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title("Vector Operations in 2D")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
```

---

## Conclusion

In this post, you’ve taken a significant step forward by mastering basic vector
operations—addition, scalar multiplication, and the dot product—using PyTorch.
These operations are not just mathematical abstractions; they are the foundation
of nearly every computation in reinforcement learning, from updating policies to
computing value functions.

- You’ve learned the mathematical definitions and geometric intuitions behind
  these operations.
- You’ve implemented them efficiently in PyTorch, leveraging broadcasting and
  built-in functions.
- You’ve visualized how vectors combine and scale in 2D space, building
  intuition for higher-dimensional operations.

**Next Up:** In Part 1.4, we’ll extend these concepts to **matrices**—2D tensors
that represent linear transformations and are critical for neural networks and
state transitions in RL. Keep practicing these vector operations, as they’ll be
essential for everything to come!

_See you in the next post!_
