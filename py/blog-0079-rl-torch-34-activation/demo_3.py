import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
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
y: torch.Tensor = (X[:, 0] + X[:, 1] > 0).long()
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
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Activation Function Comparison: Loss Curves")
plt.grid(True)
plt.show()

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
        grad_norm = (
            net.fc1.weight.grad.abs().mean().item()
            if net.fc1.weight.grad is not None
            else 0.0
        )
        grads_by_act[act].append(grad_norm)
        opt.step()
for act in acts:
    plt.plot(grads_by_act[act], label=act)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Mean Grad |fc1|")
plt.title("Mean First-Layer Gradient Magnitude by Activation")
plt.grid(True)
plt.show()
