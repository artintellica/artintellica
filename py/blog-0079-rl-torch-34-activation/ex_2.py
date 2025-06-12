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
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves by Activation")
plt.grid(True)
plt.show()
