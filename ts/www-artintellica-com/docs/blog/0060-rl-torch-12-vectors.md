+++
title = "Learning Reinforcement Learning with PyTorch, Part 1.2: Vectors and Scalars—Hands-On with PyTorch Tensors"
author = "Artintellica"
date = "2024-06-09"
+++

## Introduction

Welcome back to Artintellica’s open-source RL course! After setting up PyTorch and running your first tensor operations, it’s time to ground ourselves in the basics: **vectors and scalars**. Tensors are the backbone of all computations in PyTorch and reinforcement learning. Understanding how to create, reshape, and manipulate these 1D (and 0D) structures is crucial whether you’re implementing bandits, processing observations, or building neural networks.

In this post, we’ll:

- See how **scalars** (single numbers) and **vectors** (1D arrays) arise in both mathematics and PyTorch.
- Learn how to create and reshape tensors, index and slice them, and compute basic statistics.
- Visualize vectors as sequences—an essential skill for debugging neural networks and RL environments.
- Provide hands-on coding exercises so you have full mastery before moving on to matrices (next time).

Let’s get started!

---

## Mathematics: Scalars and Vectors

Mathematically,

- A **scalar** is a single number, $a \in \mathbb{R}$ (for example, the reward at a particular step).
- A **vector** is a one-dimensional array of numbers, $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$, where $n$ is the number of elements (for example, an agent’s observation or a parameter vector).

In PyTorch:

- **Scalar tensor:** A tensor of zero dimensions (`torch.tensor(3.0)`)
- **Vector tensor:** A tensor with shape `(n,)` for $n$ elements, i.e., `torch.tensor([1.0, 2.0, 3.0])`

PyTorch allows you to reshape, index, slice, and analyze these with almost NumPy-like power—plus GPU acceleration.

---

## Demonstrations: Cooking with Scalars and Vectors

### Creating and Reshaping Scalars & Vectors

```python
import torch

# Scalar (0-dimensional tensor, holds a single value)
a: torch.Tensor = torch.tensor(7.5)
print("Scalar a:", a)
print("Shape (should be torch.Size([])):", a.shape)
print("Dimensions (should be 0):", a.dim())

# 1-D vector tensor
v: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("Vector v:", v)
print("Shape:", v.shape)
print("Dimensions:", v.dim())

# Changing shape: Reshape vector v to a column vector (n, 1)
v_col: torch.Tensor = v.unsqueeze(1)  # adds a new dimension at position 1
print("v as column vector (shape):", v_col.shape)

# Flatten v_col back to 1D
v_flat: torch.Tensor = v_col.squeeze()
print("v_col squeezed to 1D:", v_flat.shape)
```

### Indexing, Slicing, and Reversing

```python
# Indexing: Get the second element (index 1)
second: torch.Tensor = v[1]
print("Second element:", second.item())  # .item() to extract Python float

# Slicing: Get the first three elements
first_three: torch.Tensor = v[:3]
print("First three elements:", first_three)

# Reversing the vector
reversed_v: torch.Tensor = v.flip(0)
print("Reversed v:", reversed_v)
```

### Basic Statistics: Mean, Sum, Std

```python
# Compute statistics
mean_v: torch.Tensor = v.mean()
sum_v: torch.Tensor = v.sum()
std_v: torch.Tensor = v.std(unbiased=False)  # Match numpy's normalization

print(f"Mean: {mean_v.item():.2f}")
print(f"Sum: {sum_v.item():.2f}")
print(f"Std: {std_v.item():.2f}")
```

### Plotting Vectors with Matplotlib

For visualization (so important in RL diagnostics!), let’s plot a vector as a line graph.

```python
import matplotlib.pyplot as plt

# Let's create a vector of sine values
t: torch.Tensor = torch.linspace(0, 2 * torch.pi, 100)
sin_t: torch.Tensor = torch.sin(t)

plt.figure(figsize=(8, 4))
plt.plot(t.numpy(), sin_t.numpy())
plt.title("Sine Wave")
plt.xlabel("t")
plt.ylabel("sin(t)")
plt.grid(True)
plt.show()
```

---

## Exercises

#### **Exercise 1: Create Scalar and 1D Vector Tensors; Change Their Shape**

- Create a scalar tensor with the value $42$.
- Create a vector tensor with values $[3, 1, 4, 1, 5, 9]$ as floats.
- Reshape your vector to a **column** (shape $(6, 1)$) and then back to **row** (shape $(6,)$).

#### **Exercise 2: Index, Slice, and Reverse Vectors**

- Print the first and last element of your vector.
- Slice out every other element (e.g., elements at even indices).
- Reverse your vector using PyTorch.

#### **Exercise 3: Compute Mean, Sum, and Standard Deviation**

- Compute and print the mean, sum, and standard deviation of your vector, with each rounded to two decimals.

#### **Exercise 4: Plot a Vector as a Line Graph Using Matplotlib**

- Create a PyTorch vector containing $100$ linearly spaced points between $0$ and $4\pi$.
- Compute the cosine of each value.
- Use matplotlib to plot the vector as a line graph.

---

### **Sample Starter Code for Exercises**

```python
import torch
import matplotlib.pyplot as plt

# EXERCISE 1
scalar: torch.Tensor = torch.tensor(42.0)
vec: torch.Tensor = torch.tensor([3, 1, 4, 1, 5, 9], dtype=torch.float32)
vec_col: torch.Tensor = vec.unsqueeze(1)
vec_row: torch.Tensor = vec_col.squeeze()
print("Scalar:", scalar)
print("Vector:", vec)
print("As column shape:", vec_col.shape)
print("Back to row shape:", vec_row.shape)

# EXERCISE 2
print("First element:", vec[0].item())
print("Last element:", vec[-1].item())
print("Every other element:", vec[::2])
print("Reversed:", vec.flip(0))

# EXERCISE 3
print("Mean: {:.2f}".format(vec.mean().item()))
print("Sum: {:.2f}".format(vec.sum().item()))
print("Std: {:.2f}".format(vec.std(unbiased=False).item()))

# EXERCISE 4
t: torch.Tensor = torch.linspace(0, 4 * torch.pi, 100)
cos_t: torch.Tensor = torch.cos(t)
plt.plot(t.numpy(), cos_t.numpy())
plt.title(r"$\cos(t)$ from $t=0$ to $4\pi$")
plt.xlabel("t")
plt.ylabel("cos(t)")
plt.grid(True)
plt.show()
```

---

## Conclusion

Today, you’ve gained hands-on experience with the two most fundamental building blocks in PyTorch and RL—scalars and vectors. You learned to create, reshape, index, reverse, compute statistics, and even plot them. Mastery of these basics makes everything to come—matrix math, neural networks, and reinforcement learning algorithms—much easier and more intuitive.

**In the next post**, we will extend these skills to *vector operations*: addition, scaling, and dot products. Being comfortable with tensors, dimensions, and slicing will let you focus on building and debugging RL agents instead of being tripped up by tensor shapes!

*Keep practicing, play with your own vectors, and see you in Part 1.3!*
