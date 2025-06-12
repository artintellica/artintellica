+++
title = "Learn Reinforcement Learning with PyTorch, Part 3.3: Building with torch.nn—The Convenient Way"
author = "Artintellica"
date = "2024-06-12"
+++

## Introduction

Hand-coding neural networks gives you intuition, but PyTorch's `torch.nn` module
is the **professional toolkit**—it provides higher-level abstractions, readable
code, and error-free scaling to deep architectures. In practice, nearly every RL
and ML practitioner uses `nn.Module` for defining models.

In this post, you’ll:

- Rewrite a two-layer neural network using `nn.Module` from scratch, with all
  definitions included.
- Compare line count and readability with the tensor-only approach.
- Add another hidden layer and see how easy network depth becomes.
- Learn to save and reload model weights for reproducibility and deployment.

Let’s see why—and how—PyTorch’s object-oriented approach saves time and
headaches.

---

## Mathematics: Feedforward Networks and Modularity

Recall the two-layer (one hidden layer) network from before. In `torch.nn` you
define each _layer_ as a linear transformation, with weights and biases _stored
for you_:

- For one hidden layer:
  $$
  \mathbf{h} = \phi(W_1 \mathbf{x} + \mathbf{b}_1) \\
  \mathbf{o} = W_2 \mathbf{h} + \mathbf{b}_2
  $$
- Extend this to more layers:
  $$
  \mathbf{h}_2 = \phi(W_2 \mathbf{h}_1 + \mathbf{b}_2)
  $$
- The object-oriented form means “define once, use everywhere,” with autograd,
  parameter management, and device handling all built-in.

---

## Explanation: How the Math Connects to Code

When you use `nn.Module` in PyTorch:

- **Each layer** becomes a class attribute (e.g., `self.fc1 = nn.Linear(...)`).
- The **forward pass** is defined as a method (`def forward(self, x): ...`),
  chaining the operations in order.
- You get all parameters (weights, biases) handled automatically: autograd will
  know about them, optimizers will manage updates, and serialization is
  automatic.
- Adding or removing layers, swapping activations, and reusing modules becomes
  just a matter of changing a line of code.
- Saving/loading models is one command (`torch.save`/`torch.load`).

You’ll see: switching from tensor code to `nn.Module` makes models more robust,
reusable, and production-ready.

---

## Python Demonstrations

### Demo 1: Rewrite Previous NN Using `nn.Module`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a two-layer neural network fully inside a class
class SimpleNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = F.relu(self.fc1(x))
        logits: torch.Tensor = self.fc2(h)
        return logits

# Example: instantiate and print the model
model: SimpleNet = SimpleNet(2, 8, 2)
print(model)
```

---

### Demo 2: Compare Number of Lines and Readability

Let’s train on some synthetic data and see how `nn.Module` streamlines the
process.

```python
import matplotlib.pyplot as plt

# Generate synthetic linearly separable data
torch.manual_seed(3)
N: int = 100
X: torch.Tensor = torch.randn(N, 2)
y: torch.Tensor = (X[:, 0] + X[:, 1] > 0).long()

model: SimpleNet = SimpleNet(2, 8, 2)
optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=0.07)
loss_fn: nn.Module = nn.CrossEntropyLoss()

losses: list[float] = []
for epoch in range(80):
    logits: torch.Tensor = model(X)
    loss: torch.Tensor = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 20 == 0 or epoch == 79:
        print(f"Epoch {epoch}: Loss={loss.item():.3f}")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("NN Training Loss (torch.nn)")
plt.grid(True)
plt.show()
```

It’s clear: training, updates, and device-handling are now concise and
readable—no need to hand-manage gradients!

---

### Demo 3: Add a Hidden Layer and Train on Data

```python
# Define a deeper feedforward network with two hidden layers
class DeepNet(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, output_dim: int) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_dim, hidden1)
        self.fc2: nn.Linear = nn.Linear(hidden1, hidden2)
        self.fc3: nn.Linear = nn.Linear(hidden2, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1: torch.Tensor = F.relu(self.fc1(x))
        h2: torch.Tensor = F.relu(self.fc2(h1))
        logits: torch.Tensor = self.fc3(h2)
        return logits

model_deep: DeepNet = DeepNet(2, 16, 8, 2)
optimizer_deep: torch.optim.Optimizer = torch.optim.Adam(model_deep.parameters(), lr=0.05)
losses_deep: list[float] = []
for epoch in range(100):
    logits: torch.Tensor = model_deep(X)
    loss: torch.Tensor = loss_fn(logits, y)
    optimizer_deep.zero_grad()
    loss.backward()
    optimizer_deep.step()
    losses_deep.append(loss.item())
    if epoch % 25 == 0 or epoch == 99:
        print(f"[DeepNet] Epoch {epoch}: Loss={loss.item():.3f}")

plt.plot(losses_deep)
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Deep NN Training Loss")
plt.grid(True)
plt.show()
```

---

### Demo 4: Save and Load Model Weights

```python
# Save model weights to disk
torch.save(model_deep.state_dict(), "deepnet_weights.pth")
print("Weights saved to deepnet_weights.pth")

# Load weights into a new instance (architecture must match)
model_loaded: DeepNet = DeepNet(2, 16, 8, 2)
model_loaded.load_state_dict(torch.load("deepnet_weights.pth"))
print("Weights loaded. Sample output:", model_loaded(X[:5]))
```

---

## Exercises

### **Exercise 1:** Define a Two-Layer Neural Network Using `nn.Module`

- Define a Python class `MyNet` that subclasses `torch.nn.Module`.
- It should have a hidden layer of size 6, ReLU activation, and an output layer
  for 2-class classification.
- Define and use a `forward()` method that passes an input tensor through your
  network.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 6, output_dim: int = 2) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = F.relu(self.fc1(x))
        out: torch.Tensor = self.fc2(h)
        return out

# Create dummy input and check shape
model: MyNet = MyNet()
x_sample: torch.Tensor = torch.randn(4, 2)
logits: torch.Tensor = model(x_sample)
print("Logits shape:", logits.shape)  # Should be (4, 2)
```

---

### **Exercise 2:** Compare Number of Lines and Readability

- Use the above `MyNet` as your base model.
- Compare this to a manual/tensor approach (as seen in previous posts) using
  equivalent input/output.
- Show and count lines in both versions for a forward pass and one training
  step.

```python
# Using MyNet
optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn: nn.Module = nn.CrossEntropyLoss()
x_batch: torch.Tensor = torch.randn(8, 2)
y_batch: torch.Tensor = torch.randint(0, 2, (8,))
logits: torch.Tensor = model(x_batch)
loss: torch.Tensor = loss_fn(logits, y_batch)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("Loss:", loss.item())
```

_Try to code an equivalent manual approach and compare for yourself!_

---

### **Exercise 3:** Add a Hidden Layer and Train on Data

- Add a _second_ hidden layer to your `MyNet` class with 5 units, activations
  between layers.
- Generate 100 random 2D data points and assign class labels (e.g., class 1 if
  x0 + x1 > 0).
- Train for 60 epochs and plot the loss curve.

```python
class MyDeepNet(nn.Module):
    def __init__(self, input_dim: int = 2, h1: int = 6, h2: int = 5, output_dim: int = 2) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_dim, h1)
        self.fc2: nn.Linear = nn.Linear(h1, h2)
        self.fc3: nn.Linear = nn.Linear(h2, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1: torch.Tensor = F.relu(self.fc1(x))
        h2: torch.Tensor = F.relu(self.fc2(h1))
        return self.fc3(h2)

# Data
torch.manual_seed(0)
N: int = 100
X: torch.Tensor = torch.randn(N, 2)
y: torch.Tensor = (X[:,0] + X[:,1] > 0).long()
net: MyDeepNet = MyDeepNet()
optim: torch.optim.Optimizer = torch.optim.Adam(net.parameters(), lr=0.06)
losses: list[float] = []
for epoch in range(60):
    logits: torch.Tensor = net(X)
    loss: torch.Tensor = nn.functional.cross_entropy(logits, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    losses.append(loss.item())
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("DeepNet Training Loss")
plt.grid(True); plt.show()
```

---

### **Exercise 4:** Save and Load Model Weights

- After training `MyDeepNet`, save model weights to disk.
- Instantiate a new `MyDeepNet`, load the weights, and verify that predictions
  are identical.

```python
# Save model
torch.save(net.state_dict(), "mydeepnet_weights.pth")
# Load into a new instance
net2: MyDeepNet = MyDeepNet()
net2.load_state_dict(torch.load("mydeepnet_weights.pth"))
# Check equality
out1: torch.Tensor = net(X[:5])
out2: torch.Tensor = net2(X[:5])
print("Predictions equal after reload:", torch.allclose(out1, out2))
```

---

## Conclusion

In this part, you’ve experienced the **transformational power of torch.nn and
nn.Module**. With just a few lines, you now:

- Build, train, and manage deeper neural nets.
- Quickly swap, scale, and save models—essentials in deep RL and large ML
  projects.
- Understand how high-level abstractions save time, reduce bugs, and let you
  focus more on ideas.

**Up next:** You’ll explore and visualize nonlinear neural network building
blocks—**activation functions**—and see how these unlock expressivity and speed
up learning.

_You’re now building neural nets “the real way.” Take pride in your
object-oriented power—see you in Part 3.4!_
