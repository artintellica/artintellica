+++
title = "Learn Reinforcement Learning with PyTorch, Part 3.3: Building with torch.nn—The Convenient Way"
author = "Artintellica"
date = "2024-06-11"
+++

## Introduction

Hand-coding neural networks gives you intuition, but PyTorch's `torch.nn` module
is the **professional toolkit**—it provides higher-level abstractions, readable
code, and error-free scaling to deep architectures. In practice, nearly every RL
and ML practitioner uses `nn.Module` for defining models.

In this post, you’ll:

- Rewrite a two-layer neural network using `nn.Module`
- Compare line count and readability with the tensor-only approach
- Add another hidden layer and see how easy network depth becomes
- Learn to save and reload model weights for reproducibility and deployment

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

class SimpleNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = F.relu(self.fc1(x))
        logits: torch.Tensor = self.fc2(h)
        return logits

# Instantiate and print
model: SimpleNet = SimpleNet(2, 8, 2)
print(model)
```

---

### Demo 2: Compare Number of Lines and Readability

Let’s train on some data with an optimizer, and see how little code is needed.

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

You can see: training, parameter updates, and device-handling are now concise
and readable—no more worrying about manually managing gradients!

---

### Demo 3: Add a Hidden Layer and Train on Data

```python
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
# Save
torch.save(model_deep.state_dict(), "deepnet_weights.pth")
print("Weights saved.")

# Load into new model instance (must match architecture)
model_loaded: DeepNet = DeepNet(2, 16, 8, 2)
model_loaded.load_state_dict(torch.load("deepnet_weights.pth"))
print("Weights loaded. Sample output:", model_loaded(X[:5]))
```

---

## Exercises

### **Exercise 1:** Rewrite Previous NN Using `nn.Module`

- Define a class (subclass `nn.Module`) with two layers, ReLU activation.
- Implement `forward` method.

---

### **Exercise 2:** Compare Number of Lines and Readability

- Write out both the tensor/matrix version and the `nn.Module` version of a
  two-layer network.
- Count lines and compare readability for a full training loop.

---

### **Exercise 3:** Add a Hidden Layer and Train on Data

- Extend your model to a three-layer network (with two hidden layers).
- Train for at least 60 epochs and plot loss.

---

### **Exercise 4:** Save and Load Model Weights

- Save your trained model’s weights.
- Instantiate a new (same architecture) model and load the weights.
- Verify that predictions before and after loading are identical.

---

### **Sample Starter Code for Exercises**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# EXERCISE 1
class TinyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = F.relu(self.fc1(x))
        return self.fc2(h)
model: TinyNet = TinyNet(2, 8, 2)
print(model)

# EXERCISE 2
# Compare with manual/tensor version (see last blog) vs this Module style

# EXERCISE 3
class DeeperNet(nn.Module):
    def __init__(self, input_dim: int, h1: int, h2: int, output_dim: int) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_dim, h1)
        self.fc2: nn.Linear = nn.Linear(h1, h2)
        self.fc3: nn.Linear = nn.Linear(h2, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1: torch.Tensor = F.relu(self.fc1(x))
        h2: torch.Tensor = F.relu(self.fc2(h1))
        return self.fc3(h2)
torch.manual_seed(0)
N: int = 100
X: torch.Tensor = torch.randn(N, 2)
y: torch.Tensor = (X[:,0] + X[:,1] > 0).long()
net: DeeperNet = DeeperNet(2, 12, 5, 2)
opt: torch.optim.Optimizer = torch.optim.Adam(net.parameters(), lr=0.06)
losses: list[float] = []
for epoch in range(70):
    logits: torch.Tensor = net(X)
    loss: torch.Tensor = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
plt.plot(losses)
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Three-Layer NN Loss")
plt.grid(True); plt.show()

# EXERCISE 4
# Save model
torch.save(net.state_dict(), "deepnn_weights.pth")
# Load into new model
net2: DeeperNet = DeeperNet(2, 12, 5, 2)
net2.load_state_dict(torch.load("deepnn_weights.pth"))
# Verify
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

**Up next:** You’ll explore and visualize the nonlinear building block of deep
nets: the **activation function**. You’ll see how these unlock expressivity and
speed up learning.

_You’re now building neural nets “the real way”. Take pride in your
object-oriented power—see you in Part 3.4!_
