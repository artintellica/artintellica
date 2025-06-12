import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Generate synthetic linearly separable data
torch.manual_seed(3)
N: int = 100
X: torch.Tensor = torch.randn(N, 2)
y: torch.Tensor = (X[:, 0] + X[:, 1] > 0).long()

loss_fn: nn.Module = nn.CrossEntropyLoss()


class DeepNet(nn.Module):
    def __init__(
        self, input_dim: int, hidden1: int, hidden2: int, output_dim: int
    ) -> None:
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
optimizer_deep: torch.optim.Optimizer = torch.optim.Adam(
    model_deep.parameters(), lr=0.05
)

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

# Save
torch.save(model_deep.state_dict(), "deepnet_weights.pth")
print("Weights saved.")

# Load into new model instance (must match architecture)
model_loaded: DeepNet = DeepNet(2, 16, 8, 2)
model_loaded.load_state_dict(torch.load("deepnet_weights.pth"))
print("Weights loaded. Sample output:", model_loaded(X[:5]))
