+++
title = "Vectors, Scalars, and Spaces: The Language of Machine Learning"
author = "Artintellica"
date = "2025-05-29"
+++

Machine learning is powered by math, and **linear algebra** is the language of
much of that math. Before we can build models, optimize loss functions, or train
neural nets, we must understand how data is represented—as **scalars**,
**vectors**, and in higher dimensions, as **spaces**. In this post, we’ll
explore these foundational concepts, see their connection to machine learning,
and get hands-on with code and exercises.

---

## Scalars, Vectors, and Data Representation

At its core, **machine learning** models transform data into predictions. That
data must be expressed numerically.

- A **scalar** is a single number.  
  Example: The temperature in Austin right now, $x = 96.3$.

- A **vector** is a collection of numbers arranged in order, representing a
  point in space.  
  Example: An RGB color:
  $\vec{c} = \begin{bmatrix} 255 \\ 140 \\ 0 \end{bmatrix}$

- A **space** (specifically a _vector space_) is a collection of all possible
  vectors of a given size/type. For example, all real-valued 3-dimensional
  vectors: $\mathbb{R}^3$.

In machine learning:

- **Features** are represented as vectors: $[x_1, x_2, ..., x_n]$
- **Weights** in models are also vectors: $[w_1, w_2, ..., w_n]$
- Datasets are collections (matrices) of such vectors.

---

## Math: Vectors in $\mathbb{R}^n$

A vector $\vec{x} \in \mathbb{R}^n$ is an ordered list of $n$ real numbers:

$$
\vec{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}
$$

Key properties:

- **Addition:**
  $\vec{x} + \vec{y} = \begin{bmatrix} x_1 + y_1 \\ \dots \\ x_n + y_n \end{bmatrix}$
- **Scalar multiplication:**
  $a \vec{x} = \begin{bmatrix} a x_1 \\ \dots \\ a x_n \end{bmatrix}$
- **Zero vector:** $\vec{0} = \begin{bmatrix} 0 \\ \dots \\ 0 \end{bmatrix}$
- **Dimension:** The number of components (features).

**In ML:**

- Each image (flattened) is a vector of pixel values.
- Each word embedding is a vector in a high-dimensional space.
- Model weights live in the same vector space as the features.

---

## Visualizing Vectors in 2D/3D

Vectors are geometric objects—arrows from the origin.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define two 2D vectors
x = np.array([2, 1])
y = np.array([1, 2])

plt.figure(figsize=(5,5))
plt.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1, color='r', label='x')
plt.quiver(0, 0, y[0], y[1], angles='xy', scale_units='xy', scale=1, color='b', label='y')
plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.legend()
plt.title('Vectors in 2D Space')
plt.show()
```

---

## Python: Creating and Using Vectors

Let’s create vectors using NumPy and PyTorch—these are the basic data types used
for all machine learning work.

```python
import numpy as np
import torch

# Scalars
s = 3.14
t = torch.tensor(3.14)

# Vectors (1D arrays)
v_np = np.array([1.0, 2.0, 3.0])
v_torch = torch.tensor([1.0, 2.0, 3.0])

print("NumPy vector:", v_np)
print("PyTorch vector:", v_torch)
```

**Why two libraries?**

- **NumPy** is for generic scientific computing.
- **PyTorch** is for deep learning (automatic gradients, GPUs).

---

## Why This Matters in Machine Learning

- **Data is always vectorized**: Images, sounds, and texts become vectors.
- **Model parameters are vectors**: All the learnable numbers in your model.
- **Geometric interpretation**: Operations like distance, projection, and
  similarity are all vector operations.
- **Computational efficiency**: Vectorized code runs faster and is easier to
  optimize.

---

## Exercises

**1. Practice: Create and Plot Vectors**

- Using NumPy, create a vector \$\vec{a} = \[4, -2]\$ and plot it as an arrow
  from the origin in 2D.

**2. ML Context: Feature Vectors**

- The classic _Iris_ dataset in sklearn is \$4\$-dimensional. Load the dataset
  and print the shape of the data. How many samples? What is the dimensionality?

**3. Code Challenge: Random Vectors**

- Generate \$1000\$ random 3D vectors using NumPy. Compute the average (mean)
  vector.

**4. PyTorch Tensor Practice**

- Create a PyTorch tensor with values $\[10, 20, 30]\$. Multiply it by \$0.1\$
  and print the result.

---

## Conclusion

Vectors and scalars are the "alphabet" of machine learning. They let us encode
data, define models, and carry out computations efficiently. Every major idea in
ML—optimization, similarity, prediction—starts with vectors. In the next post,
we’ll see how matrices and higher-dimensional arrays build on this foundation,
unlocking the ability to transform and manipulate entire datasets at once.

---

**Further Reading:**

- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [PyTorch Tensors](https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html)
