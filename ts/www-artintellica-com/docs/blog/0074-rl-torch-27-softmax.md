+++
title = "Learn Reinforcement Learning with PyTorch, Part 2.7: Softmax and Multiclass Classification"
author = "Artintellica"
date = "2024-06-11"
+++

## Introduction

Welcome! Thus far, you’ve used logistic regression for binary classification.
But in real-world RL and ML, problems usually involve **choosing between several
possible actions or classes**—not just two. This blog post will introduce you to
the **softmax function** and **multiclass (a.k.a. multinomial) classification**.

You will:

- Generate and visualize synthetic data for three classes.
- Implement softmax and categorical cross-entropy (multiclass loss) from scratch
  and with PyTorch.
- Train a multiclass classifier and plot its learned decision boundaries.
- Develop geometric and practical intuition for moving beyond binary decisions.

---

## Mathematics: Softmax, Cross-Entropy, and Multiclass Classification

### Softmax Function

Given an input vector $\mathbf{z} \in \mathbb{R}^K$ (the "logits" for $K$
classes), the **softmax** function outputs a probability vector $\mathbf{p}$:

$$
\mathrm{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

where $z_i$ is the logit for class $i$.

- Each $p_i \in (0, 1)$ and $\sum_i p_i = 1$ (so, valid probabilities).

### Cross-Entropy Loss (for Multiclass)

**One-hot true targets:** $y \in \{0, ..., K-1\}$ for each sample.

The **cross-entropy loss** for a sample with logits $\mathbf{z}$ and true class
$y$:

$$
L_{\text{CE}}(\mathbf{z}, y) = -\log \bigg( \frac{e^{z_{y}}}{\sum_{j} e^{z_j}} \bigg ) = -z_{y} + \log\left( \sum_{j} e^{z_j} \right )
$$

This penalizes the model for assigning low probability to the true class.

---

## Explanation: How the Math Maps to Code

Multiclass classification means the model estimates a probability for **each**
class, not just for "yes" or "no". Neural networks (and logistic regression
extensions) do this by outputting a vector of **logits**.

- The **softmax** transforms these arbitrary real outputs into proper predicted
  probabilities that sum to 1.
- The **cross-entropy loss** compares the softmaxed probabilities to the true
  class, measuring how well the model “focuses” probability mass on the correct
  answer.

**In practice:**

- Your final classifier “head” produces logits (raw outputs, no activation).
- For each sample, you use **PyTorch's `nn.CrossEntropyLoss`**, which both
  applies softmax and computes the correct cross-entropy loss (using logits
  directly for numerical stability).

In code:

- To classify three classes: model output is shape (_batch_size_, 3).
- True labels are integer class labels (not one-hot).
- Softmax and cross-entropy can be implemented from scratch via elementary
  tensor operations, giving you full insight into the mechanics of multiclass
  decision-making.

---

## Python Demonstrations

### Demo 1: Generate Synthetic Data for Three Classes

```python
import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)
N = 300
cov = torch.tensor([[1.2, 0.8], [0.8, 1.2]])
L = torch.linalg.cholesky(cov)
means = [torch.tensor([-2., 0.]), torch.tensor([2., 2.]), torch.tensor([0., -2.])]
X_list = []
y_list = []
for i, mu in enumerate(means):
    Xi = torch.randn(N//3, 2) @ L.T + mu
    X_list.append(Xi)
    y_list.append(torch.full((N//3,), i))
X = torch.cat(X_list)
y = torch.cat(y_list).long()
colors = ['b', 'r', 'g']
for i in range(3):
    plt.scatter(X_list[i][:,0], X_list[i][:,1], color=colors[i], alpha=0.5, label=f"Class {i}")
plt.legend(); plt.xlabel("x1"); plt.ylabel("x2")
plt.title("Synthetic 3-Class Data")
plt.show()
```

---

### Demo 2: Implement Softmax and Cross-Entropy Loss Manually

```python
import torch.nn.functional as F

def softmax(logits: torch.Tensor) -> torch.Tensor:
    # For numerical stability, subtract max
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits)
    return exp_logits / exp_logits.sum(dim=1, keepdim=True)

def cross_entropy_manual(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: (N, K), targets: (N,)
    N = logits.shape[0]
    log_probs = F.log_softmax(logits, dim=1)
    return -log_probs[torch.arange(N), targets].mean()

# Toy example
logits = torch.tensor([[2.0, 0.5, -1.0],[0.0,  3.0, 0.5]])
targets = torch.tensor([0, 1])
probs = softmax(logits)
manual_loss = cross_entropy_manual(logits, targets)
print("Probabilities:\n", probs)
print("Manual cross-entropy loss:", manual_loss.item())
```

---

### Demo 3: Train a Multiclass Classifier with PyTorch’s `nn.CrossEntropyLoss`

Let’s fit a linear classifier to the data above.

```python
# Model: simple linear (no bias for simplicity)
W = torch.zeros(2, 3, requires_grad=True)  # (features, classes)
b = torch.zeros(3, requires_grad=True)
lr = 0.05

loss_fn = torch.nn.CrossEntropyLoss()
losses = []
for epoch in range(400):
    logits = X @ W + b  # (N, 3)
    loss = loss_fn(logits, y)
    loss.backward()
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad
    W.grad.zero_(); b.grad.zero_()
    losses.append(loss.item())
    if epoch % 100 == 0 or epoch == 399:
        print(f"Epoch {epoch}: Cross-entropy loss={loss.item():.3f}")

plt.plot(losses)
plt.title("Multiclass Classifier Training Loss")
plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss"); plt.grid(True)
plt.show()
```

---

### Demo 4: Plot the Class Boundaries in 2D

```python
import numpy as np

with torch.no_grad():
    x1g, x2g = torch.meshgrid(torch.linspace(-6,6,200), torch.linspace(-6,6,200), indexing='ij')
    Xg = torch.stack([x1g.reshape(-1), x2g.reshape(-1)], dim=1)  # (n_grid, 2)
    logits_grid = Xg @ W + b
    y_pred_grid = logits_grid.argmax(dim=1).reshape(200,200)

plt.contourf(x1g, x2g, y_pred_grid.numpy(), levels=[-0.5,0.5,1.5,2.5], colors=['b','r','g'], alpha=0.15)
for i in range(3):
    plt.scatter(X_list[i][:,0], X_list[i][:,1], color=colors[i], alpha=0.6, label=f"Class {i}")
plt.title("Learned Class Boundaries (Linear)")
plt.xlabel("x1"); plt.ylabel("x2"); plt.legend(); plt.show()
```

---

## Exercises

### **Exercise 1:** Generate Synthetic Data for Three Classes

- Create three 2D Gaussian blobs with different means.
- Stack into features $X$ and labels $y$.
- Plot, color-coded by class.

---

### **Exercise 2:** Implement Softmax and Cross-Entropy Loss Manually

- Implement the softmax function on a batch of logits.
- Write manual cross-entropy loss for integer target labels.
- Test on a toy example and compare with PyTorch’s `F.cross_entropy`.

---

### **Exercise 3:** Train a Multiclass Classifier with PyTorch’s `nn.CrossEntropyLoss`

- Initialize a linear model (`W` and `b`).
- Train for 400 epochs using SGD on your three-class data.
- Plot the loss curve.

---

### **Exercise 4:** Plot the Class Boundaries in 2D

- Use your trained model to predict class on a grid.
- Use `contourf` or `imshow` to shade the 2D plane by predicted class.
- Overlay your dataset’s points.

---

### **Sample Starter Code for Exercises**

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# EXERCISE 1
torch.manual_seed(0)
N = 150
means = [torch.tensor([-2.0, 0.]), torch.tensor([2.0, 2.5]), torch.tensor([0., -2.])]
cov = torch.tensor([[1.1, 0.6], [0.6, 1.0]])
L = torch.linalg.cholesky(cov)
X_list = []
y_list = []
for i, mu in enumerate(means):
    Xi = torch.randn(N//3, 2) @ L.T + mu
    X_list.append(Xi)
    y_list.append(torch.full((N//3,), i))
X = torch.cat(X_list)
y = torch.cat(y_list).long()
for i, c in enumerate(['b', 'g', 'r']):
    plt.scatter(X_list[i][:,0], X_list[i][:,1], color=c, alpha=0.5, label=f"Class {i}")
plt.legend(); plt.title("Synthetic Data"); plt.show()

# EXERCISE 2
def softmax(logits):
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp = torch.exp(logits)
    return exp / exp.sum(dim=1, keepdim=True)
def cross_entropy(logits, targets):
    N = logits.shape[0]
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    return -log_probs[torch.arange(N), targets].mean()
# Test
logits = torch.tensor([[2.0, -1.0, 0.5], [0.2, 1.0, -2.0]])
targets = torch.tensor([0, 1])
probs = softmax(logits)
print("Softmax probabilities:\n", probs)
print("Manual cross-entropy:", cross_entropy(logits, targets).item())
print("PyTorch cross-entropy:", torch.nn.functional.cross_entropy(logits, targets).item())

# EXERCISE 3
W = torch.zeros(2, 3, requires_grad=True)
b = torch.zeros(3, requires_grad=True)
lr = 0.05
losses = []
for epoch in range(400):
    logits = X @ W + b
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad
    W.grad.zero_(); b.grad.zero_()
    losses.append(loss.item())
plt.plot(losses); plt.title("Classifier Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(); plt.show()

# EXERCISE 4
with torch.no_grad():
    grid_x, grid_y = torch.meshgrid(torch.linspace(-5, 5, 200), torch.linspace(-5, 5, 200), indexing='ij')
    Xg = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    logits_grid = Xg @ W + b
    pred_grid = logits_grid.argmax(dim=1).reshape(200,200)
plt.contourf(grid_x, grid_y, pred_grid.numpy(), levels=[-0.5,0.5,1.5,2.5], colors=['b','r','g'], alpha=0.15)
for i, c in enumerate(['b', 'g', 'r']):
    plt.scatter(X_list[i][:,0], X_list[i][:,1], color=c, alpha=0.6, label=f"Class {i}")
plt.title("Decision Boundaries"); plt.legend(); plt.show()
```

---

## Conclusion

You’ve now experienced multiclass classification end-to-end:

- Generating and visualizing 3-class data.
- Understanding and implementing softmax and cross-entropy from scratch.
- Training a multiclass linear model and visualizing its decision boundaries.
- Developing deeper intuition for how “choices” are separated in
  high-dimensional feature spaces.

**Next:** In the next post, you’ll use neural networks to model even more
complex (curved!) boundaries, and classify data that linear models can’t handle.
This is the last road before deep RL.

_Practice tweaking your classes, noise, and model—softmax is the backbone of
every multiclass RL agent! See you in Part 2.8!_
