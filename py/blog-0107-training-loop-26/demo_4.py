import torch
from torch import nn
import matplotlib.pyplot as plt
from typing import Tuple, List

# ==== 1. Data Generation ====
def make_toy_data(n_samples: int = 30, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    X = torch.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = torch.sin(X) + 0.3 * torch.randn_like(X)
    return X, y

X_train, y_train = make_toy_data()    # 30 points with noise
X_test = torch.linspace(-3, 3, 100).reshape(-1, 1)
y_test = torch.sin(X_test)            # true signal without noise

# ==== 2. Model Definition ====
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# ==== 3. Training Function ====
def train(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int,
    lr: float,
    l2: float = 0.0,
    dropout: bool = False,
    dropout_p: float = 0.3
) -> List[float]:
    if dropout:
        # insert Dropout after each activation
        new_layers = []
        for layer in model.layers:  # type: ignore
            new_layers.append(layer)
            if isinstance(layer, nn.Tanh):
                new_layers.append(nn.Dropout(p=dropout_p))
        model.layers = nn.Sequential(*new_layers)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.MSELoss()
    losses = []
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# ==== 4. Training Three Models ====

# (A) No Regularization
model_no_reg = TinyNet()
losses_no_reg = train(model_no_reg, X_train, y_train, n_epochs=500, lr=0.01)

# (B) L2 Regularization (weight decay)
model_l2 = TinyNet()
losses_l2 = train(model_l2, X_train, y_train, n_epochs=500, lr=0.01, l2=0.1)

# (C) Dropout
model_dropout = TinyNet()
losses_dropout = train(model_dropout, X_train, y_train, n_epochs=500, lr=0.01, dropout=True, dropout_p=0.3)

# ==== 5. Plotting Loss Curves ====
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(losses_no_reg)
plt.title("No Regularization")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")

plt.subplot(1,3,2)
plt.plot(losses_l2)
plt.title("L2 Regularization")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")

plt.subplot(1,3,3)
plt.plot(losses_dropout)
plt.title("Dropout")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.tight_layout()
plt.show()

# ==== 6. Plotting Predictions ====
plt.figure(figsize=(8,6))
plt.plot(X_test.numpy(), y_test.numpy(), label='True Function', color='green', linewidth=2)
with torch.no_grad():
    plt.plot(X_test.numpy(), model_no_reg(X_test).numpy(), label='No Reg', linestyle='--', color='orange')
    plt.plot(X_test.numpy(), model_l2(X_test).numpy(), label='L2', linestyle='-.', color='blue')
    plt.plot(X_test.numpy(), model_dropout(X_test).numpy(), label='Dropout', linestyle=':', color='red')
plt.scatter(X_train.numpy(), y_train.numpy(), label='Train Data', color='black', alpha=0.5)
plt.legend()
plt.title("Regularization Comparison: Model Predictions")
plt.xlabel("x"); plt.ylabel("y")
plt.show()
