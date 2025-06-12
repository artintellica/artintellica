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
y = (X[:, 0] * 1.1 - X[:, 1] > 0).long()
acts = ["sigmoid", "tanh", "relu", "leakyrelu"]
loss_hist = {}
grads_by_act = {act: [] for act in acts}
for act in acts:
    net = SmallNet(act)
    opt = torch.optim.Adam(net.parameters(), lr=0.1)
    for epoch in range(40):
        logits = net(X)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        grad_norm = (
            net.fc1.weight.grad.abs().mean().item()
            if net.fc1.weight.grad is not None
            else 0.0
        )
        grads_by_act[act].append(grad_norm)
        opt.step()
for act in acts:
    plt.plot(grads_by_act[act], label=act)
plt.xlabel("Epoch")
plt.ylabel("Mean |grad|")
plt.title("Gradient Magnitude by Activation Function")
plt.legend()
plt.grid(True)
plt.show()
