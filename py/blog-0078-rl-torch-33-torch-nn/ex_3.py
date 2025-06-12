from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MyNet(nn.Module):
    def __init__(
        self, input_dim: int = 2, hidden_dim: int = 6, output_dim: int = 2
    ) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = F.relu(self.fc1(x))
        out: torch.Tensor = self.fc2(h)
        return out


# create dummy input and check shape
model: MyNet = MyNet()
x_sample: torch.Tensor = torch.randn(4, 2)
logits: torch.Tensor = model(x_sample)
print("Logits shape:", logits.shape)

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


class MyDeepNet(nn.Module):
    def __init__(
        self, input_dim: int = 2, h1: int = 6, h2: int = 5, output_dim: int = 2
    ) -> None:
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
y: torch.Tensor = (X[:, 0] + X[:, 1] > 0).long()
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
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DeepNet Training Loss")
plt.grid(True)
plt.show()
