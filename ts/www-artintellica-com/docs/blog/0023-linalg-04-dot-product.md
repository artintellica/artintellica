+++
title = "Linear Algebra for Machine Learning, Part 4: Dot Product and Cosine Similarity"
author = "Artintellica"
date = "2025-05-29"
+++

Welcome to the fourth post in our series on **Linear Algebra for Machine
Learning**! After exploring matrix arithmetic, we now focus on the **dot
product** and **cosine similarity**, two fundamental concepts for measuring
relationships between vectors. These tools are crucial in machine learning (ML)
for tasks like similarity search, projections, and word embeddings. In this
post, we’ll cover their mathematical foundations, their applications in ML, and
how to implement them in Python using **NumPy** and **PyTorch**. We’ll include
visualizations and Python exercises to solidify your understanding.

---

## The Math: Dot Product and Cosine Similarity

### Dot Product

The **dot product** of two vectors $ \mathbf{u} = [u_1, u_2, \dots, u_n] $ and $
\mathbf{v} = [v_1, v_2, \dots, v_n] $ in $ \mathbb{R}^n $ is a scalar computed
as:

$$
 \mathbf{u} \cdot \mathbf{v} = u*1 v_1 + u_2 v_2 + \dots + u_n v_n =
\sum*{i=1}^n u_i v_i
$$

Geometrically, the dot product relates to the angle $ \theta $ between the
vectors:

$$
 \mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos \theta
$$

where $ \|\mathbf{u}\| = \sqrt{u_1^2 + \dots + u_n^2} $ is the Euclidean norm
(length) of $ \mathbf{u} $. The dot product is:

- Positive if $ \theta < 90^\circ $ (vectors point in similar directions).
- Zero if $ \theta = 90^\circ $ (vectors are orthogonal).
- Negative if $ \theta > 90^\circ $ (vectors point in opposite directions).

The dot product also represents the **projection** of one vector onto another,
scaled by the length of the second vector.

### Cosine Similarity

**Cosine similarity** measures the cosine of the angle between two vectors,
focusing on their directional similarity, regardless of magnitude:

$$
 \text{cosine\_similarity}(\mathbf{u}, \mathbf{v}) = \cos \theta =
\frac{\mathbf{u} \cdot \mathbf{v}}{|\mathbf{u}| |\mathbf{v}|}
$$

The value ranges from:

- $ 1 $: Vectors are identical in direction ($ \theta = 0^\circ $).
- $ 0 $: Vectors are orthogonal ($ \theta = 90^\circ $).
- $ -1 $: Vectors are opposite ($ \theta = 180^\circ $).

Cosine similarity is widely used in ML because it’s robust to vector magnitude,
making it ideal for comparing high-dimensional data like word embeddings.

---

## ML Context: Why Dot Product and Cosine Similarity Matter

In machine learning, these concepts are essential:

- **Similarity Search**: Cosine similarity compares documents, images, or user
  profiles in recommendation systems.
- **Word Embeddings**: In natural language processing (NLP), cosine similarity
  measures semantic similarity between word vectors (e.g., Word2Vec, GloVe).
- **Projections**: The dot product is used in algorithms like PCA to project
  data onto principal components.
- **Neural Networks**: Dot products compute weighted sums in layers, and cosine
  similarity can regularize embeddings.

Understanding these tools helps you quantify relationships in data and build
effective ML models.

---

## Python Code: Dot Product and Cosine Similarity

Let’s implement dot product and cosine similarity using **NumPy** and
**PyTorch**, with visualizations to illustrate their geometric meaning.

### Setup

Install the required libraries if needed:

```bash
pip install numpy torch matplotlib
```

### Dot Product

Let’s compute the dot product of two vectors:

```python
import numpy as np
import matplotlib.pyplot as plt

# Define two 2D vectors
u = np.array([1, 2])
v = np.array([3, 1])

# Compute dot product
dot_product = np.dot(u, v)

# Print results
print("Vector u:", u)
print("Vector v:", v)
print("Dot product u · v:", dot_product)

# Visualize vectors
def plot_2d_vectors(vectors, labels, colors):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)
    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(*origin, *vec, color=color, scale=1, scale_units='xy', angles='xy')
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)
    plt.grid(True)
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Vectors for Dot Product")
    plt.show()

plot_2d_vectors(
    [u, v],
    ['u', 'v'],
    ['blue', 'red']
)
```

**Output:**

```
Vector u: [1 2]
Vector v: [3 1]
Dot product u · v: 5
```

This computes $ \mathbf{u} \cdot \mathbf{v} = 1 \cdot 3 + 2 \cdot 1 = 5 $ and
plots the vectors to show their directional relationship.

### Cosine Similarity with NumPy

Let’s compute cosine similarity:

```python
# Compute norms
norm_u = np.linalg.norm(u)
norm_v = np.linalg.norm(v)

# Compute cosine similarity
cosine_sim = dot_product / (norm_u * norm_v)

# Print results
print("Norm of u:", norm_u)
print("Norm of v:", norm_v)
print("Cosine similarity:", cosine_sim)
```

**Output:**

```
Norm of u: 2.23606797749979
Norm of v: 3.1622776601683795
Cosine similarity: 0.7071067811865475
```

This computes $ \cos \theta \approx 0.707 $, indicating the angle between $
\mathbf{u} $ and $ \mathbf{v} $ is about $ 45^\circ $.

### Cosine Similarity with PyTorch

Let’s use PyTorch’s `cosine_similarity`:

```python
import torch

# Convert to PyTorch tensors
u_torch = torch.tensor(u, dtype=torch.float32)
v_torch = torch.tensor(v, dtype=torch.float32)

# Compute cosine similarity
cosine_sim_torch = torch.cosine_similarity(u_torch, v_torch, dim=0)

# Print result
print("PyTorch cosine similarity:", cosine_sim_torch.item())
```

**Output:**

```
PyTorch cosine similarity: 0.7071067690849304
```

This confirms PyTorch’s result matches NumPy’s, with minor floating-point
differences.

### Visualizing Cosine Similarity

Let’s compare vectors with different similarities:

```python
# Define vectors with varying similarity
v1 = np.array([1, 0])  # Same direction as u
v2 = np.array([0, 1])  # Orthogonal to v1
v3 = np.array([-1, 0])  # Opposite to v1

# Compute cosine similarities
cos_sim_v1 = np.dot(u, v1) / (np.linalg.norm(u) * np.linalg.norm(v1))
cos_sim_v2 = np.dot(u, v2) / (np.linalg.norm(u) * np.linalg.norm(v2))
cos_sim_v3 = np.dot(u, v3) / (np.linalg.norm(u) * np.linalg.norm(v3))

# Print results
print("Cosine similarity u, v1:", cos_sim_v1)
print("Cosine similarity u, v2:", cos_sim_v2)
print("Cosine similarity u, v3:", cos_sim_v3)

# Plot vectors
plot_2d_vectors(
    [u, v1, v2, v3],
    ['u', 'v1 (similar)', 'v2 (orthogonal)', 'v3 (opposite)'],
    ['blue', 'green', 'red', 'purple']
)
```

**Output:**

```
Cosine similarity u, v1: 0.4472135954999579
Cosine similarity u, v2: 0.8944271909999159
Cosine similarity u, v3: -0.4472135954999579
```

This shows how cosine similarity reflects directional relationships, with $ v1 $
being somewhat similar, $ v2 $ nearly orthogonal, and $ v3 $ opposite to $ u $.

---

## Exercises

Try these Python exercises to deepen your understanding. Solutions will be
discussed in the next post!

1. **Dot Product**: Create two 3D vectors with random integers between -5 and 5
   using NumPy. Compute their dot product and print the vectors and result.
2. **Cosine Similarity**: Compute the cosine similarity between the vectors from
   Exercise 1 using NumPy. Visualize the vectors in a 2D plot (use only the
   first two components).
3. **PyTorch Cosine Similarity**: Convert the vectors from Exercise 1 to PyTorch
   tensors and compute their cosine similarity using `torch.cosine_similarity`.
   Verify the result matches NumPy’s.
4. **Orthogonal Vectors**: Create two 2D vectors that are orthogonal (dot
   product = 0). Compute their dot product and cosine similarity, and plot them
   to confirm orthogonality.
5. **Word Vector Similarity**: Create a small dictionary of 3 “word” vectors
   (3D, random values). Compute pairwise cosine similarities using NumPy and
   print a similarity matrix.
6. **Projection**: Compute the projection of vector $ \mathbf{u} = [1, 2]
   $
   onto $ \mathbf{v} = [3, 1] $ using the formula $ \text{proj}\_{\mathbf{v}}
   \mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v}
   $.
   Plot $ \mathbf{u} $, $ \mathbf{v} $, and the projection in 2D.

---

## What’s Next?

In the next post, we’ll explore **linear independence and span**, key concepts
for understanding feature redundancy and the expressiveness of ML models. We’ll
provide more Python code and exercises to keep building your intuition.

Happy learning, and see you in Part 5!
