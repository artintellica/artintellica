+++
title = "Learn Reinforcement Learning with PyTorch, Part 3.2: Feedforward Neural Networks from Scratch (No nn.Module)"
author = "Artintellica"
date = "2024-06-09"
+++

## Introduction

Welcome back! In the last lesson, you implemented and visualized the perceptron—the building block for all neural networks. Today, you'll learn how to construct and train **feedforward neural networks (multilayer perceptrons) from scratch**, using only tensors, matrix multiplications, and autograd—**no `nn.Module` or convenience libraries!**

This bootstraps your intuition for how NNs work under the hood—and will serve you well both in RL and practical ML engineering.

---

## Mathematics: Two-Layer Neural Network

A **two-layer neural network** for input $\mathbf{x} \in \mathbb{R}^{d_\text{in}}$, hidden dimension $h$, and output size $d_\text{out}$ is:

**Layer 1 (hidden):**
$$
\mathbf{h} = f(\mathbf{x} W_1 + \mathbf{b}_1)
$$

**Layer 2 (output):**
$$
\mathbf{o} = \mathbf{h} W_2 + \mathbf{b}_2
$$

Where
- $W_1 \in \mathbb{R}^{d_\text{in} \times h}$, $\mathbf{b}_1 \in \mathbb{R}^{h}$
- $W_2 \in \mathbb{R}^{h \times d_\text{out}}$, $\mathbf{b}_2 \in \mathbb{R}^{d_\text{out}}$
- $f$ is a nonlinear activation (e.g. ReLU, sigmoid, tanh)

For classification, $\mathbf{o}$ is sent to softmax (for cross-entropy loss).

---

## Explanation: How the Math Connects to Code

- **Weights and biases** are tensors you initialize and update (*not* `nn.Linear` modules).
- **Forward pass**:  
    1. Multiply inputs $X$ by $W_1$ and add $b_1$ (matrix multiply + broadcast).  
    2. Apply nonlinear activation, e.g. ReLU.  
    3. Multiply by $W_2$ and add $b_2$ for class logits.
- **Loss:** Use cross-entropy between logits and integer labels.
- **Backward pass:**  
    - If you want maximum insight, you can manually compute gradients using chain rule.
    - Or, let PyTorch `autograd` handle derivatives by calling `.backward()` after computing loss.

This is the backbone of modern ML and RL—autograd or not, everything boils down to math on tensors and gradients.

---

## Python Demonstrations

We’ll use a synthetic 2D classification dataset (moons or blobs) for demonstration.

### Demo 1: Implement a Two-Layer Neural Network Using Tensors and Matrix Ops

```python
import torch
import torch.nn.functional as F
from typing import Tuple

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)

def two_layer_forward(
    X: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor, 
    W2: torch.Tensor, b2: torch.Tensor
) -> torch.Tensor:
    # X: (N, d_in), W1: (d_in, h), b1: (h,), W2: (h, d_out), b2: (d_out,)
    hidden: torch.Tensor = relu(X @ W1 + b1)
    output: torch.Tensor = hidden @ W2 + b2
    return output  # logits (N, d_out)
```

---

### Demo 2: Forward Propagate and Compute Loss for a Batch of Inputs

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate 2D data (moons)
def make_moons(n_samples: int = 200, noise: float = 0.18) -> Tuple[torch.Tensor, torch.Tensor]:
    n: int = n_samples // 2
    theta: np.ndarray = np.pi * np.random.rand(n)
    x0: np.ndarray = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    x1: np.ndarray = np.stack([1 - np.cos(theta), 1 - np.sin(theta)], axis=1) + np.array([0.6, -0.4])
    X: np.ndarray = np.vstack([x0, x1])
    X = X + noise * np.random.randn(*X.shape)
    y: np.ndarray = np.hstack([np.zeros(n), np.ones(n)])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

torch.manual_seed(42)
np.random.seed(42)
X, y = make_moons(200, noise=0.22)

# Initialize weights/biases for 2-10-2 network
N, d_in = X.shape
h = 10
d_out = 2
W1: torch.Tensor = torch.randn(d_in, h, requires_grad=True) * 0.1
b1: torch.Tensor = torch.zeros(h, requires_grad=True)
W2: torch.Tensor = torch.randn(h, d_out, requires_grad=True) * 0.1
b2: torch.Tensor = torch.zeros(d_out, requires_grad=True)

# Forward
logits: torch.Tensor = two_layer_forward(X, W1, b1, W2, b2)
loss: torch.Tensor = F.cross_entropy(logits, y)
print("Initial loss:", loss.item())
```

---

### Demo 3: Backpropagate Gradients Manually (autograd or by hand)

#### (a) Using PyTorch autograd

```python
# Zero grads
for param in [W1, b1, W2, b2]:
    if param.grad is not None:
        param.grad.zero_()

loss.backward()
print("W1.grad shape:", W1.grad.shape)
print("b1.grad shape:", b1.grad.shape)
print("W2.grad shape:", W2.grad.shape)
print("b2.grad shape:", b2.grad.shape)
```

#### (b) (Optional) Manual Gradients (by hand)
Manually computing gradients for all weights is a classic exercise in chain rule. For most real tasks, use autograd for accuracy and simplicity.

---

### Demo 4: Train the Model and Plot Loss Over Epochs

```python
def train_nn(
    X: torch.Tensor, y: torch.Tensor, 
    W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor, 
    epochs: int = 300, lr: float = 0.08
) -> list[float]:
    loss_hist: list[float] = []
    for epoch in range(epochs):
        logits = two_layer_forward(X, W1, b1, W2, b2)
        loss = F.cross_entropy(logits, y)
        # Zero gradients
        for param in [W1, b1, W2, b2]:
            if param.grad is not None:
                param.grad.zero_()
        loss.backward()
        with torch.no_grad():
            W1 -= lr * W1.grad
            b1 -= lr * b1.grad
            W2 -= lr * W2.grad
            b2 -= lr * b2.grad
        loss_hist.append(loss.item())
        if epoch % 60 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch}: loss {loss.item():.4f}")
    return loss_hist

losses: list[float] = train_nn(X, y, W1, b1, W2, b2, epochs=300, lr=0.07)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)
plt.show()
```

---

## Exercises

### **Exercise 1:** Implement a Two-Layer Neural Network Using Tensors and Matrix Ops

- Explicitly define and initialize $W_1$, $b_1$, $W_2$, $b_2$ tensors (with gradients).
- Implement a function for forward pass with ReLU.

---

### **Exercise 2:** Forward Propagate and Compute Loss for a Batch of Inputs

- For your synthetic batch, compute logits (before softmax) and cross-entropy loss.

---

### **Exercise 3:** Backpropagate Gradients Manually (using autograd)

- Call `.backward()` on loss.
- Print gradients for each parameter.
- Zero the gradients and repeat.

---

### **Exercise 4:** Train the Model and Plot Loss Over Epochs

- Using your training loop, fit the network for 200+ epochs.
- Plot the loss curve.
- (Optional) Visualize the decision boundary by predicting over a grid.

---

### **Sample Starter Code for Exercises**

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)

# EXERCISE 1
def two_layer_forward(
    X: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor, 
    W2: torch.Tensor, b2: torch.Tensor
) -> torch.Tensor:
    h = relu(X @ W1 + b1)
    out = h @ W2 + b2
    return out

# EXERCISE 2
def make_blobs(
    n_samples: int = 200, centers: int = 2, random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    np.random.seed(random_state)
    X = np.vstack([
        np.random.randn(n_samples // centers, 2) + np.array([i*2, i*2]) 
        for i in range(centers)
    ])
    y = np.hstack([np.full(n_samples // centers, i) for i in range(centers)])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

X, y = make_blobs(200, 2)
N, d_in = X.shape
h = 12; d_out = 2
W1: torch.Tensor = torch.randn(d_in, h, requires_grad=True) * 0.05
b1: torch.Tensor = torch.zeros(h, requires_grad=True)
W2: torch.Tensor = torch.randn(h, d_out, requires_grad=True) * 0.05
b2: torch.Tensor = torch.zeros(d_out, requires_grad=True)

out: torch.Tensor = two_layer_forward(X, W1, b1, W2, b2)
loss: torch.Tensor = torch.nn.functional.cross_entropy(out, y)
print('Initial loss:', loss.item())

# EXERCISE 3
for p in [W1, b1, W2, b2]:
    if p.grad is not None:
        p.grad.zero_()
loss.backward()
for p in [W1, b1, W2, b2]:
    print(f"Grad shape {p.shape}: max abs {p.grad.abs().max().item()}")

# EXERCISE 4
def train_nn(
    X: torch.Tensor, y: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor, 
    W2: torch.Tensor, b2: torch.Tensor, 
    epochs: int = 200, lr: float = 0.07
) -> list[float]:
    loss_hist: list[float] = []
    for epoch in range(epochs):
        output = two_layer_forward(X, W1, b1, W2, b2)
        loss = torch.nn.functional.cross_entropy(output, y)
        # zero gradients
        for p in [W1, b1, W2, b2]:
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
        with torch.no_grad():
            W1 -= lr * W1.grad
            b1 -= lr * b1.grad
            W2 -= lr * W2.grad
            b2 -= lr * b2.grad
        loss_hist.append(loss.item())
    return loss_hist

loss_hist: list[float] = train_nn(X, y, W1, b1, W2, b2, epochs=250, lr=0.05)
plt.plot(loss_hist)
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.grid(); plt.show()
```

---

## Conclusion

You’ve constructed a real neural network *by hand* in PyTorch, built up the full forward and backward logic, and watched your model learn. These are the real nuts and bolts behind every deep reinforcement learning agent and modern classifier.

**Up next:** You’ll discover PyTorch’s higher-level abstractions (`nn.Module` and layers), making your life easier as you scale up your architectures.

*Congrats—now you know how the engine runs! See you in Part 3.3.*
