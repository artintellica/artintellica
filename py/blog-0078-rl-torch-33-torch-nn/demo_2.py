import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
