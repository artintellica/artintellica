+++
title = "Learn Reinforcement Learning with PyTorch, Part 1.7: Geometry with Tensors—Norms, Distance, Angles, and Projections"
author = "Artintellica"
date = "2024-06-09"
+++

## Introduction

Welcome back to Artintellica’s RL with PyTorch! After conquering matrix multiplication and transpose, you’re ready for the next foundation: **geometry with tensors**. Modern machine learning is deeply geometric. Understanding vector length, distance, angle, and projections lets you reason about similarity, optimization, and even why RL agents make the decisions they do.

In this post, you’ll:

- Compute norms (lengths) and distances in high-dimensional space.
- Quantify similarity using cosine of angles between vectors.
- Project vectors onto each other and visualize the result.
- Practice and build intuition through hands-on, code-first exercises.

Let’s unlock the geometric heart of tensors!

---

## Mathematics: Geometry, Norms, and Projections

### Norm (Length) of a Vector

The **Euclidean norm** (or $L_2$ norm) of vector $\mathbf{v}$ is its length:

$$
\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}
$$

Also written as $\|\mathbf{v}\|_2$.

### Distance Between Two Vectors

The **Euclidean distance** between vectors $\mathbf{a}$ and $\mathbf{b}$ is:

$$
d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\| = \sqrt{\sum_{i=1}^n (a_i - b_i)^2}
$$

### Cosine Similarity (Angle Between Vectors)

The **cosine similarity** between vectors $\mathbf{a}$ and $\mathbf{b}$ is:

$$
\cos \theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}
$$

- $\cos\theta = 1$ means vectors point the same way.
- $\cos\theta = 0$ means they are perpendicular.
- $\cos\theta = -1$ means they point in opposite directions.

Cosine similarity is a key measure in ML and RL for comparing states, actions, or gradients.

### Projection of One Vector onto Another

The **projection** of $\mathbf{a}$ onto $\mathbf{b}$ (think: the shadow of $\mathbf{a}$ on $\mathbf{b}$), denoted $\operatorname{proj}_{\mathbf{b}} \mathbf{a}$, is:

$$
\operatorname{proj}_{\mathbf{b}} \mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|^2} \mathbf{b}
$$

This decomposes $\mathbf{a}$ into a component "along" $\mathbf{b}$ (the projection) and the remainder (orthogonal).

---

## Python Demonstrations

Let’s see each operation in PyTorch and visualize!

### Demo 1: Compute the Euclidean Norm (Length) of a Vector

```python
import torch

v: torch.Tensor = torch.tensor([3.0, 4.0])
norm_v: torch.Tensor = torch.norm(v, p=2)
print("Vector v:", v)
print("Euclidean norm (||v||):", norm_v.item())
```
**Output:**  
Should print $5.0$ since $\sqrt{3^2 + 4^2} = 5$.

---

### Demo 2: Find the Distance Between Two Vectors

```python
a: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
b: torch.Tensor = torch.tensor([4.0, 6.0, 8.0])
dist: torch.Tensor = torch.dist(a, b, p=2)
print("a:", a)
print("b:", b)
print("Euclidean distance between a and b:", dist.item())
# or equivalently...
alt_dist: torch.Tensor = torch.norm(a - b)
print("Alternative distance (torch.norm):", alt_dist.item())
```

---

### Demo 3: Calculate Cosine Similarity Between Two Vectors

```python
# Cosine similarity with a manual formula
a_norm: torch.Tensor = torch.norm(a)
b_norm: torch.Tensor = torch.norm(b)
cos_sim: torch.Tensor = torch.dot(a, b) / (a_norm * b_norm)
print("Cosine similarity between a and b:", cos_sim.item())

# Built-in version for batches
cos_sim_builtin: torch.Tensor = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
print("Cosine similarity (PyTorch builtin, batch):", cos_sim_builtin.item())
```

---

### Demo 4: Project One Vector onto Another and Plot Both

```python
import matplotlib.pyplot as plt

# 2D vectors for visualization
a2d: torch.Tensor = torch.tensor([3.0, 1.0])
b2d: torch.Tensor = torch.tensor([2.0, 0.0])
# Compute projection
proj_length: torch.Tensor = torch.dot(a2d, b2d) / torch.dot(b2d, b2d)
proj_vec: torch.Tensor = proj_length * b2d
print("a2d:", a2d)
print("b2d:", b2d)
print("Projection of a2d onto b2d:", proj_vec)

# Plot
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, a2d[0], a2d[1], angles='xy', scale_units='xy', scale=1, color='b', label='a2d')
plt.quiver(0, 0, b2d[0], b2d[1], angles='xy', scale_units='xy', scale=1, color='r', label='b2d')
plt.quiver(0, 0, proj_vec[0], proj_vec[1], angles='xy', scale_units='xy', scale=1, color='g', label='proj_b(a)')
plt.legend()
plt.xlim(-1, 5)
plt.ylim(-1, 3)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.3)
plt.axvline(0, color='black', linewidth=0.3)
plt.title("Vector Projection in 2D")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

---

## Exercises

Use these to deepen your understanding and intuition!

### **Exercise 1:** Compute the Euclidean Norm (Length) of a Vector

- Create a 1D tensor `v = [6, 8]` (float).
- Compute and print its Euclidean norm (should be $10$).

### **Exercise 2:** Find the Distance Between Two Vectors

- Create tensors `a = [1, 7, 2, 5]` and `b = [5, 1, 2, -1]` (float).
- Compute the Euclidean distance between them.

### **Exercise 3:** Calculate the Cosine Similarity Between Two Vectors

- Use the same `a` and `b` as Exercise 2.
- Calculate the cosine similarity using both the formula and `torch.nn.functional.cosine_similarity`.

### **Exercise 4:** Project One Vector onto Another and Plot Both

- Create $2$D vectors `u = [4, 3]` and `v = [5, 0]`.
- Compute the projection of `u` onto `v`.
- Plot `u`, `v`, and the projection vector from the origin.

---

### **Sample Starter Code for Exercises**

```python
import torch
import matplotlib.pyplot as plt

# EXERCISE 1
v: torch.Tensor = torch.tensor([6.0, 8.0])
norm_v: torch.Tensor = torch.norm(v)
print("Norm of v:", norm_v.item())

# EXERCISE 2
a: torch.Tensor = torch.tensor([1.0, 7.0, 2.0, 5.0])
b: torch.Tensor = torch.tensor([5.0, 1.0, 2.0, -1.0])
dist: torch.Tensor = torch.norm(a - b)
print("Distance between a and b:", dist.item())

# EXERCISE 3
dot: torch.Tensor = torch.dot(a, b)
norm_a: torch.Tensor = torch.norm(a)
norm_b: torch.Tensor = torch.norm(b)
cosine_sim: torch.Tensor = dot / (norm_a * norm_b)
print("Cosine similarity (formula):", cosine_sim.item())
# Using built-in
cosine_sim_builtin: torch.Tensor = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
print("Cosine similarity (builtin):", cosine_sim_builtin.item())

# EXERCISE 4
u: torch.Tensor = torch.tensor([4.0, 3.0])
v: torch.Tensor = torch.tensor([5.0, 0.0])
proj_length: torch.Tensor = torch.dot(u, v) / torch.dot(v, v)
proj_vec: torch.Tensor = proj_length * v
print("Projection of u onto v:", proj_vec)

plt.figure(figsize=(6, 6))
plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='b', label='u')
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='v')
plt.quiver(0, 0, proj_vec[0], proj_vec[1], angles='xy', scale_units='xy', scale=1, color='g', label='proj_v(u)')
plt.legend()
plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.3)
plt.axvline(0, color='black', linewidth=0.3)
plt.title("Projection of u onto v")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

---

## Conclusion

Geometry underpins how we measure, compare, and manipulate data in ML and RL. In this post, you’ve:

- Calculated norms, distances, and angles with PyTorch.
- Used projections to decompose vectors—just like you’ll do with state values and features in RL.
- Built your geometric and coding intuition for higher-level RL structures.

**Next up:** We’ll use these geometric tools for *linear transformations*—rotations, scalings, and the transformations at the heart of data processing and neural networks.

*See you in Part 1.8!*
