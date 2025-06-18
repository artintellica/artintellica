import torch
import matplotlib.pyplot as plt
from torch import nn
from typing import Tuple

def make_toy_data(n_samples: int=30, seed: int=42) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    X = torch.linspace(-3, 3, n_samples).reshape(-1, 1)
    # True function: y = sin(x), noise added
    y = torch.sin(X) + 0.3 * torch.randn_like(X)
    return X, y

X_train, y_train = make_toy_data()
X_test = torch.linspace(-3, 3, 100).reshape(-1, 1)
y_test = torch.sin(X_test)

plt.scatter(X_train.numpy(), y_train.numpy(), label='Train data')
plt.plot(X_test.numpy(), y_test.numpy(), label='True function', color='green')
plt.legend()
plt.show()

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

def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, 
          n_epochs: int, lr: float, l2: float=0.0, dropout: bool=False) -> list:
    if dropout:
        # Replace activation layers with Dropout+activation
        new_layers = []
        for layer in model.layers: # type: ignore
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

model_l2 = TinyNet()
losses_l2 = train(model_l2, X_train, y_train, n_epochs=500, lr=0.01, l2=0.1)

plt.plot(losses_l2, label='Train Loss (L2)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss (L2 Regularization)')
plt.show()

# Plot predictions
model_l2.eval()
with torch.no_grad():
    y_pred_test_l2 = model_l2(X_test)
plt.scatter(X_train.numpy(), y_train.numpy(), label='Train')
plt.plot(X_test.numpy(), y_test.numpy(), label='True', color='green')
plt.plot(X_test.numpy(), y_pred_test_l2.numpy(), label='Predicted (L2)', color='red')
plt.legend()
plt.title('Model Fit (L2 Regularization)')
plt.show()
