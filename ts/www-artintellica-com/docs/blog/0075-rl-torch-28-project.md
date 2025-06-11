+++
title = "Learn Reinforcement Learning with PyTorch, Part 2.8: Mini-Project—Build, Train, and Visualize a Simple Classifier"
author = "Artintellica"
date = "2024-06-09"
+++

## Introduction

This post marks the end of Module 2! You’ve learned linear models, loss functions, optimization, and both binary and multiclass classification. Now let’s bring it all together in a *mini-project* where you’ll:

- Choose or generate your own 2D dataset for classification
- Train a simple classifier (logistic regression or a small neural network)
- Visualize the learned decision boundary in 2D
- Evaluate your model’s accuracy and highlight misclassified points

These hands-on skills are exactly what’s needed to prepare for truly RL-style decision boundaries and the transition to neural networks!

---

## Mathematics: Decision Boundaries and Classification Accuracy

Given a model $f_\theta:\mathbb{R}^2 \rightarrow \{0,1,\ldots,C-1\}$ that takes a point $\mathbf{x}$ and predicts a **class**:

- The **decision boundary** is the set of $\mathbf{x}$ where two (or more) classes are equally probable:
  $$
  \{\mathbf{x} : f_\theta(\mathbf{x}) = g\} := \{\mathbf{x} : \text{model is undecided between classes}\}
  $$

For neural networks and logistic regression, this can produce lines (linear), curves (nonlinear), or more complex boundaries (deep nets).

The **accuracy** is:
$$
\text{Accuracy} = \frac{\#\text{correct predictions}}{\#\text{examples}}
$$

We visualize the decision boundary by "coloring in" the 2D space by the model's prediction.

---

## Explanation: How the Math Connects to Code

In this project, you'll see **the direct link between mathematics and code**:

- **Data generation**: You create a dataset with known class structure (e.g. separate Gaussian blobs, concentric circles, or your own pattern).
- **Model training**: You choose a model (logistic regression or a simple neural network), parameterized by learnable weights $\theta$, and minimize cross-entropy loss so that outputs match the true labels.
- **Decision boundary visualization**: By evaluating your classifier on a grid of points, you color regions of the 2D plane according to the predicted class—drawing the “line” (or curve) where the classifier switches classes.
- **Accuracy and error visualization**: By comparing predictions to ground truth, you can mark which points were correctly or incorrectly classified, providing insight into how well the model separates the classes and where it might be confused.

Ultimately, this hands-on project distills both the geometry and mechanics of classification in machine learning, which will serve you well in both RL state/action problems and deep net applications.

---

## Python Demonstrations

### Demo 1: Choose or Generate a 2D Dataset for Classification

We'll generate a "two moons" dataset—a classic nonlinear classification task.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def make_moons(n_samples=200, noise=0.1):
    # Generate two interleaving half circles ("moons"), similar to sklearn.datasets.make_moons
    n = n_samples // 2
    theta = np.pi * np.random.rand(n)
    x0 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    x1 = np.stack([1 - np.cos(theta), 1 - np.sin(theta)], axis=1) + np.array([0.6, -0.4])
    X = np.vstack([x0, x1])
    X += noise * np.random.randn(*X.shape)
    y = np.hstack([np.zeros(n), np.ones(n)])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

torch.manual_seed(42)
np.random.seed(42)
X, y = make_moons(200, noise=0.2)
plt.scatter(X[y==0,0], X[y==0,1], c='r', label='Class 0', alpha=0.6)
plt.scatter(X[y==1,0], X[y==1,1], c='b', label='Class 1', alpha=0.6)
plt.legend(); plt.xlabel("x1"); plt.ylabel("x2"); plt.title("Two Moons Data"); plt.show()
```

---

### Demo 2: Train a Classifier (Neural Network) on the Data

Let’s use a small neural network (can use logistic regression if desired).

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 2)        # 2 outputs = logits for 2 classes
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()

# Training loop
losses = []
for epoch in range(200):
    logits = model(X)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 40 == 0 or epoch == 199:
        print(f"Epoch {epoch}: loss = {loss.item():.3f}")

plt.plot(losses)
plt.title("Loss During Training")
plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss"); plt.grid(True); plt.show()
```

---

### Demo 3: Visualize the Learned Decision Boundary

We color the background according to the predicted class.

```python
with torch.no_grad():
    # Create a grid of points covering the data
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.2, X[:,0].max()+0.2, 200),
                         np.linspace(X[:,1].min()-0.2, X[:,1].max()+0.2, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    logits_grid = model(grid_tensor)
    probas = torch.softmax(logits_grid, dim=1)
    preds = probas.argmax(dim=1).cpu().numpy().reshape(xx.shape)

plt.contourf(xx, yy, preds, alpha=0.2, cmap="coolwarm", levels=2)
plt.scatter(X[y==0,0], X[y==0,1], c='r', label='Class 0', alpha=0.7)
plt.scatter(X[y==1,0], X[y==1,1], c='b', label='Class 1', alpha=0.7)
plt.title("Learned Decision Boundary")
plt.xlabel("x1"); plt.ylabel("x2"); plt.legend(); plt.show()
```

---

### Demo 4: Evaluate Model Accuracy and Plot Misclassified Points

```python
with torch.no_grad():
    logits = model(X)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == y).float().mean().item()
    misclassified = (predictions != y)

print(f"Accuracy: {accuracy*100:.2f}%")

plt.scatter(X[y==0,0], X[y==0,1], c='r', label='Class 0', alpha=0.7)
plt.scatter(X[y==1,0], X[y==1,1], c='b', label='Class 1', alpha=0.7)
plt.scatter(X[misclassified,0], X[misclassified,1], 
            facecolors='none', edgecolors='k', linewidths=2, s=90, label='Misclassified')
plt.xlabel("x1"); plt.ylabel("x2")
plt.legend(); plt.title("Classification Results (Misclassifications circled)")
plt.show()
```

---

## Exercises

### **Exercise 1:** Choose or Generate a 2D Dataset for Classification

- Create or choose a 2D dataset with clear class structure (blobs, moons, circles, or your own idea).
- Visualize the data, coloring by class.

---

### **Exercise 2:** Train a Classifier (Logistic or Neural Network) on the Data

- Use either a logistic regression model (single linear layer) or a small neural network.
- Train using cross-entropy loss, keeping track of training loss.
- Plot the loss curve.

---

### **Exercise 3:** Visualize the Learned Decision Boundary

- After training, use your model to predict class on a dense grid covering your data range.
- Color the background according to the predicted class.
- Plot your data points over this background.

---

### **Exercise 4:** Evaluate Model Accuracy and Plot Misclassified Points

- Compute prediction accuracy on your dataset.
- Identify points classified incorrectly.
- Plot them with a distinctive marker (e.g., black outline or "X" marker) over your base plot.

---

### **Sample Starter Code for Exercises**

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# EXERCISE 1
def make_circles(n_samples=200, noise=0.05, factor=0.6):
    n = n_samples // 2
    theta = 2 * np.pi * np.random.rand(n)
    outer = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    inner = np.stack([factor * np.cos(theta), factor * np.sin(theta)], axis=1)
    X = np.vstack([outer, inner])
    X += noise * np.random.randn(*X.shape)
    y = np.hstack([np.zeros(n), np.ones(n)])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
torch.manual_seed(1); np.random.seed(1)
X, y = make_circles(200, noise=0.2)
plt.scatter(X[y==0,0], X[y==0,1], c='cyan', label='Class 0')
plt.scatter(X[y==1,0], X[y==1,1], c='magenta', label='Class 1')
plt.legend(); plt.xlabel("x1"); plt.ylabel("x2"); plt.title("Circle Data"); plt.show()

# EXERCISE 2
model = nn.Sequential(
    nn.Linear(2, 12), nn.ReLU(),
    nn.Linear(12, 2)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.07)
loss_fn = nn.CrossEntropyLoss()
losses = []
for epoch in range(250):
    logits = model(X)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
plt.plot(losses)
plt.title("Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(); plt.show()

# EXERCISE 3
with torch.no_grad():
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.2, X[:,0].max()+0.2, 200),
                         np.linspace(X[:,1].min()-0.2, X[:,1].max()+0.2, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    logits_grid = model(torch.tensor(grid_points, dtype=torch.float32))
    preds_grid = logits_grid.argmax(dim=1).cpu().numpy().reshape(xx.shape)
plt.contourf(xx, yy, preds_grid, alpha=0.2, cmap="coolwarm", levels=2)
plt.scatter(X[y==0,0], X[y==0,1], c='cyan', label='Class 0', alpha=0.8)
plt.scatter(X[y==1,0], X[y==1,1], c='magenta', label='Class 1', alpha=0.8)
plt.xlabel("x1"); plt.ylabel("x2"); plt.title("Learned Boundary"); plt.legend(); plt.show()

# EXERCISE 4
with torch.no_grad():
    logits = model(X)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == y).float().mean().item()
    misclassified = (predictions != y)
print(f"Accuracy: {accuracy*100:.2f}%")
plt.scatter(X[y==0,0], X[y==0,1], c='cyan', label='Class 0', alpha=0.8)
plt.scatter(X[y==1,0], X[y==1,1], c='magenta', label='Class 1', alpha=0.8)
plt.scatter(X[misclassified,0], X[misclassified,1], facecolors='none', edgecolors='k', linewidths=2, s=90, label='Misclassified')
plt.xlabel("x1"); plt.ylabel("x2"); plt.title("Classification with Misclassifications"); plt.legend(); plt.show()
```

---

## Conclusion

Congratulations, you’ve built and visualized your own machine learning classifier! You’ve learned to:

- Generate or choose 2D data tailored for classification problems.
- Train both logistic and simple neural classifiers on new data.
- Visualize decision boundaries—lines and nonlinear regions that separate classes.
- Evaluate and interpret accuracy and error, the key to model selection and iteration.

**Up next**: We step into the world of neural network architectures—moving from shallow to deep, and extending your classifier toolbox for RL and general ML alike.

*Experiment with your own datasets and models—true understanding comes from hands-on hacking. See you in Module 3!*
