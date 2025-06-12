import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Data (Nonlinear)
torch.manual_seed(11)
N = 180
X = torch.randn(N, 2)
y = ((X[:, 0] ** 2 - 0.7 * X[:, 1]) > 0.3).long()
train_idx = torch.arange(0, int(0.6 * N))
val_idx = torch.arange(int(0.6 * N), N)
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]


# EXERCISE 1
class RegNet(nn.Module):
    def __init__(self, hidden: int = 28, pdrop: float = 0.4) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(2, hidden)
        self.drop: nn.Dropout = nn.Dropout(pdrop)
        self.fc2: nn.Linear = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


net = RegNet()
optimizer = torch.optim.Adam(net.parameters(), lr=0.07)
loss_fn = nn.CrossEntropyLoss()
train_accs, val_accs = [], []
for epoch in range(60):
    net.train()
    logits = net(X_train)
    loss = loss_fn(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc = (logits.argmax(dim=1) == y_train).float().mean().item()
    train_accs.append(acc)
    net.eval()
    with torch.no_grad():
        val_logits = net(X_val)
        acc_v = (val_logits.argmax(dim=1) == y_val).float().mean().item()
        val_accs.append(acc_v)
plt.plot(train_accs, label="Train")
plt.plot(val_accs, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Dropout Regularization")
plt.legend()
plt.grid()
plt.show()

# EXERCISE 2
net_l2 = RegNet()
optimizer_l2 = torch.optim.Adam(net_l2.parameters(), lr=0.07, weight_decay=0.02)
val_accs_l2 = []
for epoch in range(60):
    net_l2.train()
    logits = net_l2(X_train)
    loss = loss_fn(logits, y_train)
    optimizer_l2.zero_grad()
    loss.backward()
    optimizer_l2.step()
    net_l2.eval()
    with torch.no_grad():
        val_logits = net_l2(X_val)
        acc_v = (val_logits.argmax(dim=1) == y_val).float().mean().item()
        val_accs_l2.append(acc_v)
plt.plot(val_accs, label="Dropout Only")
plt.plot(val_accs_l2, label="Dropout + L2")
plt.title("Weight Decay in Optimizer")
plt.xlabel("Epoch")
plt.ylabel("Val Accuracy")
plt.legend()
plt.grid()
plt.show()

# EXERCISE 3
# See Demo 3 for full comparison. Try repeating with and without dropout/L2.

# EXERCISE 4
rates = [0.0, 0.2, 0.5, 0.7]
for p in rates:
    regnet = RegNet(pdrop=p)
    optimizer = torch.optim.Adam(regnet.parameters(), lr=0.07)
    val_accs_this = []
    for epoch in range(40):
        regnet.train()
        logits = regnet(X_train)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        regnet.eval()
        with torch.no_grad():
            val_logits = regnet(X_val)
            acc_val = (val_logits.argmax(dim=1) == y_val).float().mean().item()
            val_accs_this.append(acc_val)
    plt.plot(val_accs_this, label=f"Dropout={p}")
plt.title("Dropout Rate Effect")
plt.xlabel("Epoch")
plt.ylabel("Val Acc")
plt.legend()
plt.grid()
plt.show()
