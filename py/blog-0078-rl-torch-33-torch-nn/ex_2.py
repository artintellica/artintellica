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
