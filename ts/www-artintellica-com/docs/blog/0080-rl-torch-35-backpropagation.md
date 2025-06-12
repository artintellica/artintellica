+++
title = "Learn Reinforcement Learning with PyTorch, Part 3.5: Backpropagation—Intuition and Hands-On Example"
author = "Artintellica"
date = "2024-06-12"
+++

## Introduction

Backpropagation ("backprop") is the heart of modern neural network training. It
enables us to efficiently compute gradients for all parameters, making deep
learning—and deep RL—practically feasible. In this post you'll:

- Intuitively understand what backpropagation does.
- Compute gradients for a small, two-layer neural net **by hand**.
- Use `.backward()` in PyTorch to automate and verify gradient calculations.
- Visualize how gradients flow through a network (and where/why they might
  vanish!).
- Debug a network suffering from vanishing gradients.

Grasping backprop is the difference between "using" neural nets and truly
understanding them!

---

## Mathematics: Backpropagation in a Two-Layer Neural Network

Consider a **two-layer** neural net (no bias for simplicity) for one sample
$x \in \mathbb{R}^2$:

$$
z_1 = W_1 x \\
h = \phi(z_1) \\
z_2 = W_2 h \\
y_{\mathrm{pred}} = \sigma(z_2)
$$

For a single output and target $y \in \{0,1\}$, use binary cross-entropy:

$$
L = -\left[y \log y_{\mathrm{pred}} + (1-y)\log(1-y_{\mathrm{pred}})\right]
$$

To optimize, we must compute **gradients**:

- $\frac{\partial L}{\partial W_2}$ (output weights)
- $\frac{\partial L}{\partial W_1}$ (input/hidden weights)

**Via chain rule**:

- $\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial y_{\mathrm{pred}}} \cdot \frac{\partial y_{\mathrm{pred}}}{\partial z_2} \cdot \frac{\partial z_2}{\partial W_2}$
- $\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial y_{\mathrm{pred}}} \cdot \frac{\partial y_{\mathrm{pred}}}{\partial z_2} \cdot \frac{\partial z_2}{\partial h} \cdot \frac{\partial h}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$

Each node ("layer output") passes gradients backward to previous layers—hence
the name **backpropagation**.

---

## Explanation: How the Math Connects to Code

In code, backprop means:

- **Forward pass:** Pass an input through the network, get output and loss.
- **Backward pass:** Start at loss, PyTorch computes gradients for all tensors
  _with respect to_ loss, using the computation graph and chain rule.
- For manual "by hand" computation, you calculate derivatives step-by-step for
  all layers and parameters.
- Comparing manual gradients and `.backward()` results gives confidence in your
  math and understanding.
- If any layer outputs or activations squash the gradients (e.g. sigmoid/tanh at
  large $|z|$), the gradient can become "vanishingly" small—this is the
  vanishing gradient problem.
- We can visualize "gradient flow" by plotting the mean/abs gradients at each
  layer parameter.

---

## Python Demonstrations

### Demo 1: Compute Gradients for a Two-Layer Network by Hand (Single Example)

Let's use simple numbers for hand calculation:  
Let $x = [1, 2]$, $W_1$ shape $(2, 2)$, $W_2$ shape $(1, 2)$, target $y=1$.

```python
import torch
import torch.nn.functional as F

# Input and target
x: torch.Tensor = torch.tensor([[1.0, 2.0]])      # (1, 2)
y: torch.Tensor = torch.tensor([1.0])             # (1,)

# Parameters (fixed small values for hand calc)
W1: torch.Tensor = torch.tensor([[0.1, -0.2],
                                 [0.3, 0.4]], requires_grad=True)  # (2,2)
W2: torch.Tensor = torch.tensor([[0.7, -0.5]], requires_grad=True) # (1,2)

# Forward pass (ReLU activation)
z1: torch.Tensor = x @ W1                         # (1,2)
h: torch.Tensor = F.relu(z1)                      # (1,2)
z2: torch.Tensor = h @ W2.T                       # (1,1)
y_pred: torch.Tensor = torch.sigmoid(z2).squeeze()# scalar

# Binary cross-entropy loss
eps: float = 1e-7
loss: torch.Tensor = - (y * torch.log(y_pred + eps) + (1 - y) * torch.log(1 - y_pred + eps))
print("Forward values:")
print("z1 =", z1.tolist())
print("h  =", h.tolist())
print("z2 =", z2.item())
print("y_pred =", y_pred.item())
print("loss =", loss.item())

# Manually compute:
# 1. dL/dy_pred = -1/y_pred
dL_dypred: float = float(-1.0 / y_pred.item())

# 2. dy_pred/dz2 = sigmoid'(z2)
dypred_dz2: float = float(y_pred.item() * (1 - y_pred.item()))

print("Manual dL/dy_pred:", dL_dypred)
print("Manual dy_pred/dz2:", dypred_dz2)
```

Now you can hand-multiply through the chain!

---

### Demo 2: Use `.backward()` to Compare with Manual Gradients

```python
# Backpropagation (PyTorch autograd)
# Zero gradients first
if W1.grad is not None: W1.grad.zero_()
if W2.grad is not None: W2.grad.zero_()
loss.backward()
print("PyTorch dL/dW2:\n", W2.grad)
print("PyTorch dL/dW1:\n", W1.grad)
```

You can now compare these with your manual chain calculation above!

---

### Demo 3: Visualize Gradient Flow in the Network

Let's use a bigger network and plot mean gradients at each parameter.

```python
import matplotlib.pyplot as plt
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, hidden: int = 6) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(2, hidden)
        self.fc2: nn.Linear = nn.Linear(hidden, 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))

torch.manual_seed(11)
N: int = 200
X: torch.Tensor = torch.randn(N, 2)
y: torch.Tensor = (X[:,0] + X[:,1] > 0).long()
mlp: TinyMLP = TinyMLP(10)
opt: torch.optim.Optimizer = torch.optim.Adam(mlp.parameters(), lr=0.1)
grad1: list[float] = []
grad2: list[float] = []
for epoch in range(60):
    logits = mlp(X)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    grad1.append(mlp.fc1.weight.grad.abs().mean().item())
    grad2.append(mlp.fc2.weight.grad.abs().mean().item())
    opt.step()
plt.plot(grad1, label="fc1")
plt.plot(grad2, label="fc2")
plt.xlabel("Epoch"); plt.ylabel("Mean Abs Grad")
plt.title("Gradient Flow in MLP")
plt.legend(); plt.grid(True); plt.show()
```

---

### Demo 4: Debug and Fix a Model with Vanishing Gradients

We'll deliberately cause vanishing gradients with a sigmoid activation.

```python
class DeepMLP(nn.Module):
    def __init__(self, hidden: int = 32, depth: int = 6) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList([nn.Linear(2 if i==0 else hidden, hidden) for i in range(depth)])
        self.out: nn.Linear = nn.Linear(hidden, 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = torch.sigmoid(l(x))   # Deliberate: will squash gradients
        return self.out(x)

torch.manual_seed(21)
deep_mlp: DeepMLP = DeepMLP()
opt: torch.optim.Optimizer = torch.optim.Adam(deep_mlp.parameters(), lr=0.07)
grad_hist: list[float] = []
for epoch in range(30):
    logits = deep_mlp(X)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    # Monitor average gradient in every layer
    mean_grad = torch.stack([l.weight.grad.abs().mean() for l in deep_mlp.layers]).mean().item()
    grad_hist.append(mean_grad)
    opt.step()
plt.plot(grad_hist)
plt.title("Vanishing Gradient in Deep Sigmoid Network")
plt.xlabel("Epoch"); plt.ylabel("Mean Gradient (all hidden layers)")
plt.grid(True); plt.show()

# Try switching to ReLU
class DeepMLPrelu(nn.Module):
    def __init__(self, hidden: int = 32, depth: int = 6) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList([nn.Linear(2 if i==0 else hidden, hidden) for i in range(depth)])
        self.out: nn.Linear = nn.Linear(hidden, 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = F.relu(l(x))
        return self.out(x)

deep_mlp_relu: DeepMLPrelu = DeepMLPrelu()
opt2: torch.optim.Optimizer = torch.optim.Adam(deep_mlp_relu.parameters(), lr=0.07)
grad_hist_relu: list[float] = []
for epoch in range(30):
    logits = deep_mlp_relu(X)
    loss = F.cross_entropy(logits, y)
    opt2.zero_grad()
    loss.backward()
    mean_grad = torch.stack([l.weight.grad.abs().mean() for l in deep_mlp_relu.layers]).mean().item()
    grad_hist_relu.append(mean_grad)
    opt2.step()
plt.plot(grad_hist, label='Sigmoid')
plt.plot(grad_hist_relu, label='ReLU')
plt.xlabel("Epoch"); plt.ylabel("Mean Gradient")
plt.title("Vanishing Gradients: Sigmoid vs ReLU")
plt.legend(); plt.grid(True); plt.show()
```

---

## Exercises

### **Exercise 1:** Compute Gradients for a Two-Layer Network by Hand (Single Example)

- Given $x = [2.0, 1.0]$,
  $W_1 = \begin{bmatrix}0.2 & -0.3 \\ 0.5 & 0.4\end{bmatrix}$,
  $W_2 = [0.6, -0.7]$, and $y=0$, using ReLU activation and sigmoid output.
- Perform the forward pass, compute the final loss, and hand-derive the
  gradients with respect to $W_1$ and $W_2$.

### **Exercise 2:** Use `.backward()` to Compare with Manual Gradients

- Implement the above example in PyTorch.
- Call `.backward()` on the loss and print the gradients for $W_1$ and $W_2$.
- Compare with your hand calculations.

### **Exercise 3:** Visualize Gradient Flow in the Network

- Train a 2-layer network on a random dataset, storing and plotting the mean
  gradient for each weight matrix on every epoch.

### **Exercise 4:** Debug and Fix a Model with Vanishing Gradients

- Build a deep (5+ layers) network with sigmoid/tanh activations.
- Train and plot gradient flow—note if gradients vanish.
- Swap ReLU for all activations and repeat—does this fix the issue?

---

### **Sample Starter Code for Exercises**

```python
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# EXERCISE 1/2
x: torch.Tensor = torch.tensor([[2.0, 1.0]])  # shape (1, 2)
y: torch.Tensor = torch.tensor([0.0])         # batch size 1

W1: torch.Tensor = torch.tensor([[0.2, -0.3], [0.5, 0.4]], requires_grad=True)  # (2,2)
W2: torch.Tensor = torch.tensor([[0.6, -0.7]], requires_grad=True)              # (1,2)

z1: torch.Tensor = x @ W1         # (1,2)
h: torch.Tensor = F.relu(z1)      # (1,2)
z2: torch.Tensor = h @ W2.T       # (1,1)
y_pred: torch.Tensor = torch.sigmoid(z2).squeeze()  # scalar
loss: torch.Tensor = - (y * torch.log(y_pred + 1e-7) + (1-y) * torch.log(1 - y_pred + 1e-7))
print("Loss:", loss.item())
if W1.grad is not None: W1.grad.zero_()
if W2.grad is not None: W2.grad.zero_()
loss.backward()
print("PyTorch dL/dW2:", W2.grad)
print("PyTorch dL/dW1:", W1.grad)

# EXERCISE 3
class Net2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(2, 10)
        self.fc2: nn.Linear = nn.Linear(10, 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))

N: int = 150
X: torch.Tensor = torch.randn(N, 2)
y: torch.Tensor = (X[:,0] - X[:,1] > 0).long()
net: Net2 = Net2()
opt: torch.optim.Optimizer = torch.optim.Adam(net.parameters(), lr=0.10)
g1: list[float] = []
g2: list[float] = []
for epoch in range(50):
    logits = net(X)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    g1.append(net.fc1.weight.grad.abs().mean().item())
    g2.append(net.fc2.weight.grad.abs().mean().item())
    opt.step()
plt.plot(g1, label="fc1 (input)")
plt.plot(g2, label="fc2 (out)")
plt.xlabel("Epoch"); plt.ylabel("Mean |grad|")
plt.legend(); plt.grid(True); plt.title("Gradient flow in NN"); plt.show()

# EXERCISE 4
class DeepSigNet(nn.Module):
    def __init__(self, hidden: int = 24, depth: int = 6) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList([nn.Linear(2 if i==0 else hidden, hidden) for i in range(depth)])
        self.out: nn.Linear = nn.Linear(hidden, 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = torch.sigmoid(l(x))
        return self.out(x)
deepnet = DeepSigNet()
opt = torch.optim.Adam(deepnet.parameters(), lr=0.09)
g_hist = []
for epoch in range(25):
    logits = deepnet(X)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    grads = [l.weight.grad.abs().mean().item() for l in deepnet.layers]
    g_hist.append(sum(grads)/len(grads))
    opt.step()
plt.plot(g_hist, label='Sigmoid')
plt.title("Vanishing Gradients with Sigmoid"); plt.xlabel("Epoch"); plt.ylabel("Mean Grad"); plt.legend(); plt.show()

# Fix: ReLU
class DeepReluNet(nn.Module):
    def __init__(self, hidden: int = 24, depth: int = 6) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList([nn.Linear(2 if i==0 else hidden, hidden) for i in range(depth)])
        self.out: nn.Linear = nn.Linear(hidden, 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = F.relu(l(x))
        return self.out(x)
deepnet_r = DeepReluNet()
opt_r = torch.optim.Adam(deepnet_r.parameters(), lr=0.09)
g_hist_r = []
for epoch in range(25):
    logits = deepnet_r(X)
    loss = F.cross_entropy(logits, y)
    opt_r.zero_grad()
    loss.backward()
    grads = [l.weight.grad.abs().mean().item() for l in deepnet_r.layers]
    g_hist_r.append(sum(grads)/len(grads))
    opt_r.step()
plt.plot(g_hist, label='Sigmoid'); plt.plot(g_hist_r, label='ReLU')
plt.title("Vanishing Gradients: Sigmoid vs ReLU")
plt.xlabel("Epoch"); plt.ylabel("Mean Grad"); plt.legend(); plt.grid(); plt.show()
```

---

## Conclusion

Now you:

- Understand what "backpropagation" means, both mathematically and in code.
- Have computed and checked gradients by hand and with autograd.
- Can visualize gradient flow and diagnose vanishing gradients.
- Know how activation choice and network depth can create or fix these issues.

**Next:** We’ll discuss overfitting, underfitting, and regularization—essentials
for making your models robust on real-world (not just training) data.

_Stick with these basics: understanding gradients and backprop is the foundation
of all deep learning and RL! See you in Part 3.6!_
