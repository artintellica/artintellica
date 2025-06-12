+++
title = "Learn Reinforcement Learning with PyTorch, Part 3.4: Activation Functions—Sigmoid, Tanh, ReLU, LeakyReLU, etc."
author = "Artintellica"
date = "2024-06-12"
+++

## Introduction

Neural networks are much more than stacks of linear layers. **Activation
functions** are the secret to their flexibility—they introduce nonlinearity,
enabling your model to represent complex patterns, boundaries, and behaviors
that no line can fit.

In this post, you’ll:

- Define and plot the key activation functions in modern ML.
- Train small neural nets with each activation and compare convergence.
- Visualize how activations affect gradient flow, leading to vanishing or
  exploding gradients.
- See what happens if you swap activations mid-training.

With this, you’ll understand why “the right activation” can make or break your
deep RL agent!

---

## Mathematics: Activation Functions and Their Gradients

**Activation functions** “squash” or transform each neuron’s output before
passing it to the next layer. Let $z \in \mathbb{R}$ be a neuron’s
pre-activation input.

- **Sigmoid:** $$ \sigma(z) = \frac{1}{1 + e^{-z}}

  $$
  Derivative: $\sigma'(z) = \sigma(z) (1 - \sigma(z))$
  $$

- **Tanh:** $$ \tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}

  $$
  Derivative: $1 - \tanh^2(z)$
  $$

- **ReLU:** $$ \mathrm{ReLU}(z) = \max(0, z)

  $$
  Derivative: $1$ if $z > 0$, $0$ otherwise
  $$

- **LeakyReLU:** $$ \mathrm{LeakyReLU}(z) = \begin{cases} z & \text{if } z \geq
  0 \\ \alpha z & \text{if } z < 0 \end{cases}
  $$
  Typical $\alpha = 0.01$.
  $$

**The effects:**

- Sigmoid and tanh “saturate” for large $|z|$—their gradients approach zero.
  This causes **vanishing gradients** in deep nets.
- ReLU avoids this for positive $z$, but can cause “dead neurons” if too many
  stay negative.
- LeakyReLU tries to fix dead neurons by using a small slope for $z < 0$.

---

## Explanation: How the Math Connects to Code

In code, the activation function is just a function applied elementwise to each
neuron’s output—at each layer.

- In PyTorch, you get access to built-in functions (`torch.sigmoid`,
  `torch.tanh`, `F.relu`, `F.leaky_relu`) and modules (e.g., `nn.ReLU()`).
- Changing the activation changes **how quickly the network converges, what
  patterns it learns**, and whether gradients can flow during backpropagation.
- Problems like **vanishing gradients** (when gradients become too small to
  update parameters in deep nets) or **exploding gradients** can be traced back
  to the choice and sequence of activations.

In the demos and exercises, you’ll:

- See (and plot!) what each activation does to its inputs.
- Train small networks with only the activation changed—watching as some
  converge easily and others get stuck.
- Monitor gradients to observe saturation (“flatlining”) or healthy learning.
- Try swapping activations mid-training to observe their effect in real time.

---

## Python Demonstrations

### Demo 1: Plot Different Activation Functions

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

z: torch.Tensor = torch.linspace(-5, 5, 200)
sigmoid: torch.Tensor = torch.sigmoid(z)
tanh: torch.Tensor = torch.tanh(z)
relu: torch.Tensor = F.relu(z)
leaky_relu: torch.Tensor = F.leaky_relu(z, negative_slope=0.1)

plt.plot(z.numpy(), sigmoid.numpy(), label='Sigmoid')
plt.plot(z.numpy(), tanh.numpy(), label='Tanh')
plt.plot(z.numpy(), relu.numpy(), label='ReLU')
plt.plot(z.numpy(), leaky_relu.numpy(), label='LeakyReLU')
plt.legend(); plt.xlabel('z'); plt.ylabel('Activation(z)')
plt.title("Activation Functions")
plt.grid(True); plt.show()
```

---

### Demo 2: Train Small NNs With Each Activation; Compare Convergence

```python
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self, activation: str) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(2, 8)
        self.fc2: nn.Linear = nn.Linear(8, 2)
        self.activation: str = activation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "sigmoid":
            h: torch.Tensor = torch.sigmoid(self.fc1(x))
        elif self.activation == "tanh":
            h = torch.tanh(self.fc1(x))
        elif self.activation == "relu":
            h = F.relu(self.fc1(x))
        elif self.activation == "leakyrelu":
            h = F.leaky_relu(self.fc1(x), negative_slope=0.05)
        else:
            raise ValueError("Unknown activation")
        return self.fc2(h)

torch.manual_seed(3)
N: int = 100
X: torch.Tensor = torch.randn(N, 2)
y: torch.Tensor = (X[:,0] + X[:,1] > 0).long()
acts: list[str] = ["sigmoid", "tanh", "relu", "leakyrelu"]
loss_hist: dict[str, list[float]] = {}
for act in acts:
    net: TinyNet = TinyNet(act)
    opt: torch.optim.Optimizer = torch.optim.Adam(net.parameters(), lr=0.08)
    losses: list[float] = []
    for epoch in range(80):
        logits: torch.Tensor = net(X)
        loss: torch.Tensor = nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    loss_hist[act] = losses

for act in acts:
    plt.plot(loss_hist[act], label=act)
plt.legend()
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Activation Function Comparison: Loss Curves")
plt.grid(True); plt.show()
```

---

### Demo 3: Observe Vanishing/Exploding Gradients by Visualizing Gradients

```python
nets: dict[str, TinyNet] = {act: TinyNet(act) for act in acts}
grads_by_act: dict[str, list[float]] = {act: [] for act in acts}
for act, net in nets.items():
    opt = torch.optim.Adam(net.parameters(), lr=0.07)
    for epoch in range(60):
        logits = net(X)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        # Sum the absolute value of all gradients in first layer
        grad_norm = net.fc1.weight.grad.abs().mean().item()
        grads_by_act[act].append(grad_norm)
        opt.step()
for act in acts:
    plt.plot(grads_by_act[act], label=act)
plt.legend()
plt.xlabel("Epoch"); plt.ylabel("Mean Grad |fc1|")
plt.title("Mean First-Layer Gradient Magnitude by Activation")
plt.grid(True); plt.show()
```

Observe that sigmoid/tanh may “flatline” (small gradients), while ReLU/LeakyReLU
retain gradient flow longer in deep nets.

---

### Demo 4: Swap Activation Mid-Training and Observe Changes

```python
net: TinyNet = TinyNet("tanh")
opt: torch.optim.Optimizer = torch.optim.Adam(net.parameters(), lr=0.08)
losses_swap: list[float] = []
for epoch in range(60):
    logits = net(X)
    loss = nn.functional.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses_swap.append(loss.item())
    if epoch == 35:
        net.activation = "relu"  # swap in-place!

plt.plot(losses_swap, label="Loss (tanh→relu@36)")
plt.axvline(35, color='gray', linestyle='--', label='Switched to ReLU')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Swapping Activations Mid-Training")
plt.legend(); plt.grid(True); plt.show()
```

---

## Exercises

### **Exercise 1:** Plot Different Activation Functions

- For each: Sigmoid, Tanh, ReLU, and LeakyReLU ($\alpha$=0.1)
- Plot in range $z \in [-5, 5]$. Label and compare their steepness, range, and
  flatness.

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

z: torch.Tensor = torch.linspace(-5, 5, 200)
sigmoid: torch.Tensor = torch.sigmoid(z)
tanh: torch.Tensor = torch.tanh(z)
relu: torch.Tensor = F.relu(z)
leaky_relu: torch.Tensor = F.leaky_relu(z, negative_slope=0.1)
plt.plot(z.numpy(), sigmoid.numpy(), label='Sigmoid')
plt.plot(z.numpy(), tanh.numpy(), label='Tanh')
plt.plot(z.numpy(), relu.numpy(), label='ReLU')
plt.plot(z.numpy(), leaky_relu.numpy(), label='LeakyReLU')
plt.legend(); plt.xlabel('z'); plt.ylabel('Activation(z)')
plt.title("Activation Functions"); plt.grid(True); plt.show()
```

---

### **Exercise 2:** Train Small NNs With Each Activation on the Same Task; Compare Convergence

- Build a `SmallNet` class that lets you switch activation via a string.
- Train four separate models (one for each activation) on the same synthetic
  data.
- Plot all loss curves together.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SmallNet(nn.Module):
    def __init__(self, activation: str) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(2, 6)
        self.fc2: nn.Linear = nn.Linear(6, 2)
        self.activation: str = activation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "sigmoid":
            h = torch.sigmoid(self.fc1(x))
        elif self.activation == "tanh":
            h = torch.tanh(self.fc1(x))
        elif self.activation == "relu":
            h = F.relu(self.fc1(x))
        elif self.activation == "leakyrelu":
            h = F.leaky_relu(self.fc1(x), negative_slope=0.08)
        else:
            raise ValueError("Unknown activation")
        return self.fc2(h)

N = 120
X = torch.randn(N, 2)
y = (X[:,0] * 1.1 - X[:,1] > 0).long()
acts = ["sigmoid", "tanh", "relu", "leakyrelu"]
loss_hist = {}
for act in acts:
    net = SmallNet(act)
    opt = torch.optim.Adam(net.parameters(), lr=0.1)
    losses = []
    for epoch in range(60):
        logits = net(X)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    loss_hist[act] = losses
for act in acts:
    plt.plot(loss_hist[act], label=act)
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves by Activation"); plt.grid(True); plt.show()
```

---

### **Exercise 3:** Observe Vanishing/Exploding Gradients by Visualizing Gradients

- For each trained model in Exercise 2, collect the mean absolute gradient of
  the first linear layer’s weights after each epoch.
- Plot all gradients’ traces.

```python
grads_by_act = {act: [] for act in acts}
for act in acts:
    net = SmallNet(act)
    opt = torch.optim.Adam(net.parameters(), lr=0.1)
    for epoch in range(40):
        logits = net(X)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        grad_norm = net.fc1.weight.grad.abs().mean().item()
        grads_by_act[act].append(grad_norm)
        opt.step()
for act in acts:
    plt.plot(grads_by_act[act], label=act)
plt.xlabel("Epoch"); plt.ylabel("Mean |grad|")
plt.title("Gradient Magnitude by Activation Function")
plt.legend(); plt.grid(True); plt.show()
```

---

### **Exercise 4:** Swap Activation Mid-Training and Observe Changes

- Train using Tanh for the first 30 epochs, then switch to ReLU and continue.
- Plot the loss curve and mark the swap epoch.

```python
net: SmallNet = SmallNet("tanh")
opt: torch.optim.Optimizer = torch.optim.Adam(net.parameters(), lr=0.11)
losses_swap: list[float] = []
for epoch in range(60):
    logits = net(X)
    loss = nn.functional.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses_swap.append(loss.item())
    if epoch == 30:
        net.activation = "relu"  # swap activation at epoch 31
plt.plot(losses_swap)
plt.axvline(30, linestyle='--', color='k', label='Swapped to ReLU')
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Changing Activation Function Mid-Training")
plt.legend(); plt.grid(True); plt.show()
```

---

## Conclusion

You’ve now:

- Graphed, compared, and internalized the impact of different activations.
- Watched their impact on network convergence and gradient flow.
- Learned about phenomena like vanishing/exploding gradients and “dead neurons.”
- Seen that activation functions can be hot-swapped—even mid-training!

**Up next:** You’ll dig deeper into what makes neural nets really
“train”—backpropagation. You’ll demystify how gradients flow, get calculated,
and sometimes stall in deep learning.

_Practice swapping, plotting, and analyzing on your own data! Understanding
activations will supercharge your RL and deep net intuition. See you in Part
3.5!_
