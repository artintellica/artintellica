import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Synthetic dataset (nonlinear boundary)
torch.manual_seed(7)
N: int = 200
X: torch.Tensor = torch.randn(N, 2)
y: torch.Tensor = ((X[:, 0] ** 2 + 0.8 * X[:, 1]) > 0.5).long()


# Split into train/val
train_idx = torch.arange(0, int(0.7 * N))
val_idx = torch.arange(int(0.7 * N), N)
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

loss_fn = nn.CrossEntropyLoss()

train_acc: list[float] = []
val_acc: list[float] = []


# Model with no regularization
class PlainNet(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(2, hidden)
        self.fc2: nn.Linear = nn.Linear(hidden, hidden)
        self.fc3: nn.Linear = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


net_plain = PlainNet()
optimizer_plain = torch.optim.Adam(net_plain.parameters(), lr=0.08)
train_acc_plain: list[float] = []
val_acc_plain: list[float] = []

for epoch in range(80):
    net_plain.train()
    logits = net_plain(X_train)
    loss = loss_fn(logits, y_train)
    optimizer_plain.zero_grad()
    loss.backward()
    optimizer_plain.step()

    pred_train = logits.argmax(dim=1)
    acc_t = (pred_train == y_train).float().mean().item()
    train_acc_plain.append(acc_t)

    net_plain.eval()
    with torch.no_grad():
        val_logits = net_plain(X_val)
        pred_val = val_logits.argmax(dim=1)
        acc_v = (pred_val == y_val).float().mean().item()
        val_acc_plain.append(acc_v)

plt.plot(train_acc_plain, label="Train Acc (No Reg)")
plt.plot(val_acc_plain, label="Val Acc (No Reg)")
plt.plot(train_acc, label="Train Acc (Dropout)")
plt.plot(val_acc, label="Val Acc (Dropout)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Regularization vs. No Regularization")
plt.legend()
plt.grid(True)
plt.show()
