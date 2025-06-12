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
