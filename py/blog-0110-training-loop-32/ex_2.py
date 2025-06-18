import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(0)

# Generate a simple synthetic dataset: y = 2x + 1 + noise
def make_data(n_samples: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(-1, 1, n_samples).unsqueeze(1)
    y = 2 * X + 1 + 0.2 * torch.randn_like(X)
    return X, y

X, y = make_data()

# Define a simple one-layer neural net (just linear regression)
class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(1, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

# Training loop template
def train(
    model: nn.Module, 
    X: torch.Tensor, 
    y: torch.Tensor, 
    lr: float, 
    epochs: int, 
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
) -> list[float]:
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return losses

# Settings
epochs = 100
lr = 0.1

# Training with constant learning rate
model_a = SimpleNet()
losses_const = train(model_a, X, y, lr=lr, epochs=epochs)

# Training with step LR scheduler (decay by 0.1 at epoch 50)
model_b = SimpleNet()
optimizer_b = SGD(model_b.parameters(), lr=lr)
scheduler_b = StepLR(optimizer_b, step_size=50, gamma=0.1)
def train_with_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int
) -> list[float]:
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return losses

losses_sched = train_with_optimizer(model_b, optimizer_b, scheduler_b, X, y, epochs) # type: ignore

# Plotting
plt.figure(figsize=(8,6))
plt.plot(losses_const, label="Constant LR (0.1)")
plt.plot(losses_sched, label="Step Decay LR (0.1 → 0.01 at 50)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Effect of Learning Rate Scheduling")
plt.legend()
plt.grid(True)
plt.show()

# Exercise 1 Solution

model_c = SimpleNet()
optimizer_c = SGD(model_c.parameters(), lr=lr)
scheduler_c = torch.optim.lr_scheduler.ExponentialLR(optimizer_c, gamma=0.95)

def train_exp(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int
) -> list[float]:
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return losses

losses_exp = train_exp(model_c, optimizer_c, scheduler_c, X, y, epochs) # type: ignore

plt.figure(figsize=(8,6))
plt.plot(losses_const, label="Constant LR (0.1)")
plt.plot(losses_sched, label="Step Decay LR")
plt.plot(losses_exp, label="Exponential Decay LR ($\gamma=0.95$)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Comparing Learning Rate Schedules")
plt.legend()
plt.grid(True)
plt.show()

# Exercise 2 Solution

model_d = SimpleNet()
optimizer_d = SGD(model_d.parameters(), lr=lr)
scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, patience=10, factor=0.5)

criterion = nn.MSELoss()
losses_plateau = []
for epoch in range(epochs):
    model_d.train()
    optimizer_d.zero_grad()
    out = model_d(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer_d.step()
    losses_plateau.append(loss.item())
    scheduler_d.step(loss.item())

plt.figure(figsize=(8,6))
plt.plot(losses_const, label="Constant LR (0.1)")
plt.plot(losses_plateau, label="ReduceLROnPlateau")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Effect of ReduceLROnPlateau")
plt.legend()
plt.grid(True)
plt.show()
