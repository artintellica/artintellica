import torch
import matplotlib.pyplot as plt
from typing import Tuple


# -- Data Preparation
def make_data(n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    X = torch.linspace(0, 1, n_samples).unsqueeze(1)
    y = 2 * X + 3 + 0.2 * torch.randn_like(X)
    return X, y


X, y = make_data()


# -- Simple Linear Model
class SimpleLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# -- Training function
def train_loop(
    optimizer_name: str, num_epochs: int = 100
) -> Tuple[list[float], SimpleLinear]:
    model = SimpleLinear()
    criterion = torch.nn.MSELoss()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.2, alpha=0.9)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.2, betas=(0.9, 0.999))
    else:
        raise ValueError("Unknown optimizer")

    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses, model


optimizers = ["sgd", "rmsprop", "adam"]
results = {}
for opt in optimizers:
    losses, model = train_loop(opt)
    results[opt] = {"losses": losses, "model": model}

# -- Plotting Loss Curves
plt.figure(figsize=(8, 5))
for opt in optimizers:
    plt.plot(results[opt]["losses"], label=opt.upper())
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Convergence of Different Optimizers")
plt.legend()
plt.show()

# -- Plotting Predictions
plt.figure(figsize=(8, 5))
plt.scatter(X.numpy(), y.numpy(), label="Data", alpha=0.6)
x_plot = torch.linspace(0, 1, 100).unsqueeze(1)
for opt in optimizers:
    y_pred = results[opt]["model"](x_plot).detach().numpy()
    plt.plot(x_plot.numpy(), y_pred, label=opt.upper())
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Learned Linear Fit by Optimizer")
plt.show()
