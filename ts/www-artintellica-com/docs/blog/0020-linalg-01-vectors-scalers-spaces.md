+++
title = "Linear Algebra 1: Vectors, Scalars, and Spaces: The Language of Machine Learning"
author = "Artintellica"
date = "2025-05-29"
+++

Welcome to the first post in our series on **Linear Algebra for Machine
Learning**! If you're diving into machine learning (ML), you've likely heard
that linear algebra is the backbone of many algorithms. From representing data
to optimizing neural networks, linear algebra provides the language and tools we
need. In this post, we'll start with the basics: **vectors**, **scalars**, and
**vector spaces**. We'll explore their mathematical foundations, their role in
ML, and how to work with them in Python using **NumPy** and **PyTorch**. Plus,
we'll visualize these concepts and provide exercises to solidify your
understanding.

---

## The Math: Vectors, Scalars, and Spaces

### Scalars

A **scalar** is a single number—think of it as a magnitude without direction. In
ML, scalars often represent weights, biases, or loss values. For example, a
learning rate (like 0.01) or a model's accuracy score is a scalar.
Mathematically, scalars belong to a field, typically the real numbers (ℝ).

### Vectors

A **vector** is an ordered list of scalars, representing a point or direction in
space. We denote a vector in ℝⁿ (n-dimensional space) as:

$$ \mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} $$

Each $ v_i $ is a scalar component. Geometrically, a vector in 2D or 3D can be
visualized as an arrow with a direction and magnitude. In ML, vectors are
everywhere:

- A **feature vector** represents a data point (e.g., [age, height, weight]).
- A **weight vector** defines a model's parameters.
- Word embeddings (like Word2Vec) are high-dimensional vectors.

### Vector Spaces

A **vector space** is a collection of vectors that can be added together and
scaled by scalars while staying within the space. Formally, a vector space over
ℝ must satisfy properties like closure under addition and scaling. For ML, we
care about ℝⁿ, the space of all n-dimensional real-valued vectors.

Key idea: Vectors in a vector space can be combined linearly (via addition and
scaling) to reach other points in the space. This is crucial for tasks like
transforming data or optimizing models.

---

## ML Context: Why Vectors Matter

In machine learning, vectors are the workhorses of data representation:

- **Data Points**: Each sample in a dataset (e.g., an image or a customer
  profile) is often a vector. For example, a grayscale image of size 28x28 (like
  in MNIST) can be flattened into a 784-dimensional vector.
- **Model Parameters**: In a neural network, weights and biases are vectors (or
  matrices/tensors, which we'll cover later).
- **Embeddings**: In natural language processing (NLP), words or sentences are
  represented as vectors in high-dimensional spaces, capturing semantic meaning.

Understanding vectors lets you manipulate data efficiently and reason about
algorithms like gradient descent or principal component analysis (PCA).

---

## Python Code: Working with Vectors

Let’s see how to represent and manipulate vectors using **NumPy** and
**PyTorch**. We’ll also visualize vectors in 2D and 3D using **Matplotlib**.

### Setup

First, install the required libraries if you haven’t already:

```bash
pip install numpy torch matplotlib
```

### Creating Vectors

Here’s how to create vectors in NumPy and PyTorch:

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# NumPy vector
v_numpy = np.array([2, 3])
print("NumPy vector:", v_numpy)

# PyTorch tensor
v_torch = torch.tensor([2, 3], dtype=torch.float32)
print("PyTorch vector:", v_torch)
```

Output:

```
NumPy vector: [2 3]
PyTorch vector: tensor([2., 3.])
```

Both NumPy arrays and PyTorch tensors are efficient for vector operations.
PyTorch tensors are particularly useful for ML because they support GPU
acceleration and automatic differentiation.

### Vector Addition and Scaling

Let’s perform basic operations:

```python
# Define two vectors
u = np.array([1, 2])
v = np.array([2, -1])

# Addition
sum_uv = u + v
print("u + v =", sum_uv)

# Scaling
scaled_v = 2 * v
print("2 * v =", scaled_v)
```

Output:

```
u + v = [3 1]
2 * v = [4 -2]
```

### Visualizing Vectors in 2D

Let’s plot vectors to build intuition:

```python
def plot_2d_vectors(vectors, labels, colors):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)  # Origin point [0, 0]

    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(*origin, *vec, color=color, scale=1, scale_units='xy', angles='xy')
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)

    plt.grid(True)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.title("2D Vector Visualization")
    plt.show()

# Plot u, v, and their sum
plot_2d_vectors(
    [u, v, sum_uv],
    ['u', 'v', 'u+v'],
    ['blue', 'red', 'green']
)
```

This code generates a plot showing vectors $ \mathbf{u} = [1, 2] $, $ \mathbf{v}
= [2, -1] $, and their sum as arrows in 2D space.

### Visualizing Vectors in 3D

For a 3D example:

```python
def plot_3d_vectors(vectors, labels, colors):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    origin = np.zeros(3)

    for vec, label, color in zip(vectors, labels, colors):
        ax.quiver(*origin, *vec, color=color)
        ax.text(vec[0], vec[1], vec[2], label, color=color)

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Vector Visualization")
    plt.show()

# 3D vectors
u_3d = np.array([1, 2, 3])
v_3d = np.array([2, -1, 1])
plot_3d_vectors(
    [u_3d, v_3d],
    ['u', 'v'],
    ['blue', 'red']
)
```

This plots two 3D vectors, showing their direction and magnitude in space.

---

## Exercises

Try these exercises to deepen your understanding. Solutions will be discussed in
the next post!

### Math Exercises

1. **Vector Addition**: Given vectors $ \mathbf{a} = [3, -2] $ and $ \mathbf{b}
   = [-1, 4] $, compute $ \mathbf{a} + \mathbf{b} $ and $ 2\mathbf{a} -
   \mathbf{b} $. Sketch the result geometrically.
2. **Vector Space Properties**: Prove that the set of all 2D vectors (ℝ²) is
   closed under vector addition and scalar multiplication.
3. **Dimension of a Vector**: If a dataset has 10 features per sample, what is
   the dimension of each feature vector? What vector space do these vectors live
   in?

### Python Exercises

1. **Vector Operations**: Write a Python function that takes two NumPy arrays
   (vectors) and returns their sum and the scaled version of the first vector by
   a scalar input. Test it with $ \mathbf{u} = [4, 1] $, $ \mathbf{v} = [-2, 3]
   $, and scalar $ c = 3 $.
2. **Visualization**: Modify the 2D plotting code to include a third vector that
   is the scaled version of $ \mathbf{u} $. Adjust the plot limits if needed.
3. **PyTorch Conversion**: Convert a NumPy vector to a PyTorch tensor and
   perform vector addition using PyTorch. Verify the result matches NumPy’s.

### ML Hands-On

1. **Feature Vector**: Load a sample from the MNIST dataset (using
   `torchvision.datasets.MNIST`) and flatten it into a vector. Print its
   dimension and visualize the first 10 elements as a bar plot.
2. **Vector Exploration**: Create a 3D vector representing a synthetic data
   point (e.g., [height, weight, age]). Normalize the vector (divide by its
   maximum value) and plot it in 3D.

---

## What’s Next?

In the next post, we’ll dive into **matrices**—how they represent data (like
images) and act as transformations in ML. We’ll explore reshaping, indexing, and
visualizing matrices, with plenty of Python code and ML examples.

Happy learning, and see you in Part 2!
