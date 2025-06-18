import torch
import matplotlib.pyplot as plt
from torch import nn
from typing import Tuple


def make_toy_data(
    n_samples: int = 30, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    X = torch.linspace(-3, 3, n_samples).reshape(-1, 1)
    # True function: y = sin(x), noise added
    y = torch.sin(X) + 0.3 * torch.randn_like(X)
    return X, y


X_train, y_train = make_toy_data()
X_test = torch.linspace(-3, 3, 100).reshape(-1, 1)
y_test = torch.sin(X_test)

plt.scatter(X_train.numpy(), y_train.numpy(), label="Train data")
plt.plot(X_test.numpy(), y_test.numpy(), label="True function", color="green")
plt.legend()
plt.show()


def train(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int,
    lr: float,
    l2: float = 0.0,
    dropout: bool = False,
) -> list:
    if dropout:
        # Replace activation layers with Dropout+activation
        new_layers = []
        for layer in model.layers:  # type: ignore
            new_layers.append(layer)
            if isinstance(layer, nn.Tanh):  # Insert Dropout after activation
                new_layers.append(nn.Dropout(p=0.3))
        model.layers = nn.Sequential(*new_layers)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.MSELoss()
    losses = []
    model.train()
    for epoch in range(n_epochs):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


deep_model = DeepNet()
train(deep_model, X_train, y_train, n_epochs=500, lr=0.01, dropout=True)
with torch.no_grad():
    y_pred_deep = deep_model(X_test)
plt.plot(X_test.numpy(), y_test.numpy(), label="True", color="green")
plt.plot(X_test.numpy(), y_pred_deep.numpy(), label="Dropout DeepNet", color="purple")
plt.scatter(X_train.numpy(), y_train.numpy(), alpha=0.4)
plt.legend()
plt.title("Deeper Net with Dropout")
plt.show()

both_model = TinyNet()
train(both_model, X_train, y_train, n_epochs=500, lr=0.01, l2=0.01, dropout=True)
with torch.no_grad():
    y_pred_both = both_model(X_test)
plt.plot(X_test.numpy(), y_test.numpy(), label="True", color="green")
plt.plot(X_test.numpy(), y_pred_both.numpy(), label="L2 + Dropout", color="orange")
plt.scatter(X_train.numpy(), y_train.numpy(), alpha=0.4)
plt.legend()
plt.title("L2 + Dropout Combined")
plt.show()
