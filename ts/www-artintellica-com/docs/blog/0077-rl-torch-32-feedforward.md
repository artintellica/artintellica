+++
title = "Learn Reinforcement Learning with PyTorch, Part 3.2: Feedforward Neural Networks from Scratch (No nn.Module)"
author = "Artintellica"
date = "2024-06-11"
+++

## Introduction

Welcome to Part 3.2! You’ve mastered the perceptron. Now, you’re ready to build
a **multi-layer neural network**—_from scratch,_ with only tensors and matrix
operations. Understanding this core structure is crucial for deep RL, as modern
agents are built from stacks of such layers.

In this post, you'll:

- Implement a two-layer (one hidden layer) feedforward neural network using only
  PyTorch tensors and matrix math.
- Forward propagate a batch of data and compute the loss manually.
- Backpropagate gradients, by hand or using PyTorch’s autograd.
- Train the model and plot the loss over epochs to visualize learning.

Let’s pull back the curtain and see what makes neural networks “learn”!

---

## Mathematics: Feedforward Neural Networks

A simple **two-layer feedforward neural network** for classification with input
$\mathbf{x} \in \mathbb{R}^d$:

1. **Hidden Layer:**

   $$
   \mathbf{z}^{(1)} = W_1\mathbf{x} + \mathbf{b}_1 \\
   \mathbf{h} = \phi(\mathbf{z}^{(1)})
   $$

   Where $W_1$ is $(h, d)$ (hidden size $h$), $\mathbf{b}_1$ is $(h,)$, and
   $\phi$ is an activation function (like ReLU or sigmoid).

2. **Output Layer:**

   $$
   \mathbf{z}^{(2)} = W_2\mathbf{h} + \mathbf{b}_2 \\
   \hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{z}^{(2)})
   $$

   Where $W_2$ is $(c, h)$ (number of classes $c$), $\mathbf{b}_2$ is $(c,)$.

3. **Loss:**  
   Use cross-entropy between $\hat{\mathbf{y}}$ and the true one-hot label.

**Forward propagation** computes these layers in order; **backpropagation**
computes gradients to update $W_1, W_2, \mathbf{b}_1, \mathbf{b}_2$.

---

## Explanation: How the Math Connects to Code

A two-layer ("single hidden layer") neural net applies first a linear
transformation (matrix multiply + bias), then a nonlinear activation, then
another linear transformation + softmax. Modern frameworks like PyTorch's
`nn.Module` encapsulate this, but here we'll do everything "by hand" for
complete understanding.

- **Weights and biases** are tensors you create.
- **Forward pass:** For a batch, you use batch matrix multiplies (`@`) and
  broadcasting for activation and output.
- **Loss computation:** Use your own cross-entropy or PyTorch's, comparing
  outputs $\hat{y}$ to the true labels.
- **Backpropagation:** Let PyTorch’s autograd compute the gradients for
  you—although you could do it by hand for a small example.
- **Training:** Repeatedly update parameters using their gradients and plot the
  loss curve to visualize learning.

**Why do this?** You’ll strengthen your intuition and debugging ability for all
neural net code—knowing how all the wiring fits together under the hood.

---

## Python Demonstrations

### Demo 1: Implement a Two-Layer Neural Network Using Tensors and Matrix Ops

```python
import torch

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)

def softmax(logits: torch.Tensor) -> torch.Tensor:
    logits = logits - logits.max(dim=1, keepdim=True).values  # stability
    exp_logits = torch.exp(logits)
    return exp_logits / exp_logits.sum(dim=1, keepdim=True)

# Network sizes
input_dim: int = 2
hidden_dim: int = 8
output_dim: int = 2   # binary classification

# Weights and biases
W1: torch.Tensor = torch.randn(input_dim, hidden_dim, requires_grad=True)    # (2, 8)
b1: torch.Tensor = torch.zeros(hidden_dim, requires_grad=True)               # (8,)
W2: torch.Tensor = torch.randn(hidden_dim, output_dim, requires_grad=True)   # (8, 2)
b2: torch.Tensor = torch.zeros(output_dim, requires_grad=True)               # (2,)

# Forward, example for a batch X of shape (N, 2)
def forward(X: torch.Tensor) -> torch.Tensor:
    z1: torch.Tensor = X @ W1 + b1        # (N, 8)
    h: torch.Tensor = relu(z1)            # (N, 8)
    logits: torch.Tensor = h @ W2 + b2    # (N, 2)
    return logits

# Example input
X_example: torch.Tensor = torch.randn(5, 2)
logits_out: torch.Tensor = forward(X_example)
print("Logits (first 5):\n", logits_out)
```

---

### Demo 2: Forward Propagate and Compute Loss for a Batch of Inputs

```python
import torch.nn.functional as F

def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)

# Synthetic dataset
N: int = 80
X_data: torch.Tensor = torch.randn(N, 2)
# True decision boundary: if x0 + x1 > 0, class 1 else 0
y_data: torch.Tensor = ((X_data[:,0] + X_data[:,1]) > 0).long()

logits_batch: torch.Tensor = forward(X_data)
loss: torch.Tensor = cross_entropy_loss(logits_batch, y_data)
print("Loss for batch:", loss.item())
```

---

### Demo 3: Backpropagate Gradients Manually (Autograd Version)

```python
# Zero gradients (if pre-existing)
for param in [W1, b1, W2, b2]:
    if param.grad is not None:
        param.grad.zero_()

# Forward
logits_batch = forward(X_data)
loss = cross_entropy_loss(logits_batch, y_data)
# Backward: PyTorch computes all gradients automatically
loss.backward()

print("dL/dW1 (shape):", W1.grad.shape)
print("dL/dW2 (shape):", W2.grad.shape)
```

---

### Demo 4: Train the Model and Plot Loss over Epochs

```python
import matplotlib.pyplot as plt

lr: float = 0.07
epochs: int = 120
losses: list[float] = []

# Reinitialize weights/biases for training run
W1 = torch.randn(input_dim, hidden_dim, requires_grad=True)
b1 = torch.zeros(hidden_dim, requires_grad=True)
W2 = torch.randn(hidden_dim, output_dim, requires_grad=True)
b2 = torch.zeros(output_dim, requires_grad=True)

for epoch in range(epochs):
    # Forward
    logits = forward(X_data)
    loss = cross_entropy_loss(logits, y_data)
    # Backward
    loss.backward()
    # Gradient descent
    with torch.no_grad():
        W1 -= lr * W1.grad
        b1 -= lr * b1.grad
        W2 -= lr * W2.grad
        b2 -= lr * b2.grad
    for param in [W1, b1, W2, b2]:
        param.grad.zero_()
    losses.append(loss.item())
    if epoch % 30 == 0 or epoch == epochs-1:
        print(f"Epoch {epoch}: Loss={loss.item():.3f}")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Feedforward NN Training Loss")
plt.grid(True)
plt.show()
```

---

## Exercises

### **Exercise 1:** Implement a Two-Layer Neural Network Using Tensors and Matrix Ops

- Set input dimension $=2$, hidden size $=6$, output (class) size $=2$.
- Initialize weights and biases as tensors with `requires_grad=True`.
- Write forward propagation with a ReLU hidden layer and a linear output.

---

### **Exercise 2:** Forward Propagate and Compute Loss for a Batch of Inputs

- Generate a synthetic dataset $X$ (e.g., random points in 2D) and labels $y$.
- Compute the network output logits and use built-in or manual cross-entropy
  loss for a batch.

---

### **Exercise 3:** Backpropagate Gradients Manually (By Hand or via Autograd)

- Use PyTorch autograd as above to compute gradients (or, for a single sample,
  do it by hand).
- Print gradient shapes for each parameter.

---

### **Exercise 4:** Train the Model and Plot Loss Over Epochs

- Implement the full training loop: forward, loss, backward, parameter update,
  zero grads.
- Track and plot loss over epochs.

---

### **Sample Starter Code for Exercises**

```python
import torch
import matplotlib.pyplot as plt

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)

def forward(
    X: torch.Tensor,
    W1: torch.Tensor, b1: torch.Tensor,
    W2: torch.Tensor, b2: torch.Tensor
) -> torch.Tensor:
    z1: torch.Tensor = X @ W1 + b1
    h: torch.Tensor = relu(z1)
    logits: torch.Tensor = h @ W2 + b2
    return logits

# EXERCISE 1
input_dim: int = 2
hidden_dim: int = 6
output_dim: int = 2

W1: torch.Tensor = torch.randn(input_dim, hidden_dim, requires_grad=True)
b1: torch.Tensor = torch.zeros(hidden_dim, requires_grad=True)
W2: torch.Tensor = torch.randn(hidden_dim, output_dim, requires_grad=True)
b2: torch.Tensor = torch.zeros(output_dim, requires_grad=True)

# EXERCISE 2
N: int = 100
X: torch.Tensor = torch.randn(N, 2)
y: torch.Tensor = (X[:,0] - X[:,1] > 0).long()
logits: torch.Tensor = forward(X, W1, b1, W2, b2)
loss: torch.Tensor = torch.nn.functional.cross_entropy(logits, y)
print("Loss batch:", loss.item())

# EXERCISE 3
for param in [W1, b1, W2, b2]:
    if param.grad is not None:
        param.grad.zero_()
logits = forward(X, W1, b1, W2, b2)
loss = torch.nn.functional.cross_entropy(logits, y)
loss.backward()
for p in [W1, b1, W2, b2]:
    print(f"{p.shape}: grad shape {p.grad.shape}")

# EXERCISE 4
lr: float = 0.12
epochs: int = 140
losses: list[float] = []
W1 = torch.randn(input_dim, hidden_dim, requires_grad=True)
b1 = torch.zeros(hidden_dim, requires_grad=True)
W2 = torch.randn(hidden_dim, output_dim, requires_grad=True)
b2 = torch.zeros(output_dim, requires_grad=True)
for epoch in range(epochs):
    logits = forward(X, W1, b1, W2, b2)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    with torch.no_grad():
        W1 -= lr * W1.grad
        b1 -= lr * b1.grad
        W2 -= lr * W2.grad
        b2 -= lr * b2.grad
    for param in [W1, b1, W2, b2]:
        param.grad.zero_()
    losses.append(loss.item())
plt.plot(losses)
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("2-Layer NN Training Loss")
plt.grid(True); plt.show()
```

---

## Conclusion

You’ve coded a two-layer neural network "from scratch" using only tensors and
matrix math—no black boxes! Now you understand what powers deep RL agents (and
the rest of deep learning):

- How layers of linear transforms and nonlinear activations stack up.
- How outputs propagate forward, how gradients flow backward.
- How training (by hand!) improves the model over time.

**Up next:** We’ll use PyTorch’s `nn.Module` to build and train deeper networks
efficiently.

_Master this, and you’ll never be afraid to debug or augment your networks
again. See you in Part 3.3!_
