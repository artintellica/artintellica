import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def gen_data(N: int, noise: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    x: torch.Tensor = torch.linspace(-3, 3, N)
    y_true: torch.Tensor = torch.sin(x) + 0.5 * x
    y: torch.Tensor = y_true + noise * torch.randn(N)
    return x.unsqueeze(1), y


class TinyNet(nn.Module):
    def __init__(self, hidden: int = 12) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(1, hidden)
        self.fc2: nn.Linear = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = F.relu(self.fc1(x))
        return self.fc2(h).squeeze(1)


def train_and_eval(
    model: nn.Module,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xte: torch.Tensor,
    yte: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.05,
    l2: float = 0.0,
) -> tuple[list[float], list[float]]:
    opt: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=l2
    )
    train_losses, test_losses = [], []
    for _ in range(epochs):
        model.train()
        pred: torch.Tensor = model(xtr)
        loss: torch.Tensor = F.mse_loss(pred, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            test_pred = model(xte)
            test_loss = F.mse_loss(test_pred, yte)
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
    return train_losses, test_losses


# EXERCISE 1
x_train, y_train = gen_data(20)
x_test, y_test = gen_data(100)
net = TinyNet()
train_loss, test_loss = train_and_eval(net, x_train, y_train, x_test, y_test)
plt.plot(train_loss, label="Train")
plt.plot(test_loss, label="Test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Overfitting on Small Data")
plt.show()

# EXERCISE 2
x_train2, y_train2 = gen_data(20, noise=1.0)
net2 = TinyNet()
train2, test2 = train_and_eval(net2, x_train2, y_train2, x_test, y_test)
plt.plot(train2, label="Train")
plt.plot(test2, label="Test")
plt.legend()
plt.title("Overfitting Worse With Noise")
plt.show()
with torch.no_grad():
    plt.scatter(x_train2, y_train2, c="b", label="Train (noisy)")
    plt.plot(x_test, y_test, "k:", label="True")
    plt.plot(x_test, net2(x_test), "r-", label="Model")
    plt.legend()
    plt.title("Prediction on Noisy Data")
    plt.show()

# EXERCISE 3
losses_l2 = {}
for l2 in [0.0, 0.01, 0.1]:
    net_l2 = TinyNet()
    t_l, te_l = train_and_eval(net_l2, x_train2, y_train2, x_test, y_test, l2=l2)
    losses_l2[l2] = (t_l, te_l)
for l2, (t_l, te_l) in losses_l2.items():
    plt.plot(te_l, label=f"L2={l2}")
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.legend()
plt.title("L2 Regularization Effect")
plt.show()

# EXERCISE 4
sizes = [2, 8, 32, 64]
accs = []
for h in sizes:
    net_c = TinyNet(hidden=h)
    t_l, te_l = train_and_eval(net_c, x_train, y_train, x_test, y_test, epochs=90)
    with torch.no_grad():
        preds = net_c(x_test)
        yte_mean = y_test.mean()
        ss_res = torch.sum((y_test - preds) ** 2)
        ss_tot = torch.sum((y_test - yte_mean) ** 2)
        r2 = 1 - ss_res / ss_tot
    accs.append(r2.item())
plt.plot(sizes, accs, marker="o")
plt.xlabel("Hidden Layer Size (Model Complexity)")
plt.ylabel("Test R^2 Score")
plt.title("Accuracy vs. Model Complexity")
plt.grid(True)
plt.show()
