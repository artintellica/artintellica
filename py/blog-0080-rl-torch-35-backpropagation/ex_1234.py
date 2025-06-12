import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# EXERCISE 1/2
x: torch.Tensor = torch.tensor([[2.0, 1.0]])  # shape (1, 2)
y: torch.Tensor = torch.tensor([0.0])  # batch size 1

W1: torch.Tensor = torch.tensor([[0.2, -0.3], [0.5, 0.4]], requires_grad=True)  # (2,2)
W2: torch.Tensor = torch.tensor([[0.6, -0.7]], requires_grad=True)  # (1,2)

z1: torch.Tensor = x @ W1  # (1,2)
h: torch.Tensor = F.relu(z1)  # (1,2)
z2: torch.Tensor = h @ W2.T  # (1,1)
y_pred: torch.Tensor = torch.sigmoid(z2).squeeze()  # scalar
loss: torch.Tensor = -(
    y * torch.log(y_pred + 1e-7) + (1 - y) * torch.log(1 - y_pred + 1e-7)
)
print("Loss:", loss.item())
if W1.grad is not None:
    W1.grad.zero_()
if W2.grad is not None:
    W2.grad.zero_()
loss.backward()
print("PyTorch dL/dW2:", W2.grad)
print("PyTorch dL/dW1:", W1.grad)


# EXERCISE 3
class Net2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(2, 10)
        self.fc2: nn.Linear = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


N: int = 150
X: torch.Tensor = torch.randn(N, 2)
y: torch.Tensor = (X[:, 0] - X[:, 1] > 0).long()
net: Net2 = Net2()
opt: torch.optim.Optimizer = torch.optim.Adam(net.parameters(), lr=0.10)
g1: list[float] = []
g2: list[float] = []
for epoch in range(50):
    logits = net(X)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    g1.append(
        net.fc1.weight.grad.abs().mean().item()
        if net.fc1.weight.grad is not None
        else 0.0
    )
    g2.append(
        net.fc2.weight.grad.abs().mean().item()
        if net.fc2.weight.grad is not None
        else 0.0
    )
    opt.step()
plt.plot(g1, label="fc1 (input)")
plt.plot(g2, label="fc2 (out)")
plt.xlabel("Epoch")
plt.ylabel("Mean |grad|")
plt.legend()
plt.grid(True)
plt.title("Gradient flow in NN")
plt.show()


# EXERCISE 4
class DeepSigNet(nn.Module):
    def __init__(self, hidden: int = 24, depth: int = 6) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList(
            [nn.Linear(2 if i == 0 else hidden, hidden) for i in range(depth)]
        )
        self.out: nn.Linear = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = torch.sigmoid(l(x))
        return self.out(x)


deepnet = DeepSigNet()
opt = torch.optim.Adam(deepnet.parameters(), lr=0.09)
g_hist = []
for epoch in range(25):
    logits = deepnet(X)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    # grads = [l.weight.grad.abs().mean().item() for l in deepnet.layers]
    grads = [
        param.grad.detach().abs().mean()
        for param in deepnet.parameters()
        if param.grad is not None
    ]
    g_hist.append(sum(grads) / len(grads))
    opt.step()
plt.plot(g_hist, label="Sigmoid")
plt.title("Vanishing Gradients with Sigmoid")
plt.xlabel("Epoch")
plt.ylabel("Mean Grad")
plt.legend()
plt.show()


# Fix: ReLU
class DeepReluNet(nn.Module):
    def __init__(self, hidden: int = 24, depth: int = 6) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList(
            [nn.Linear(2 if i == 0 else hidden, hidden) for i in range(depth)]
        )
        self.out: nn.Linear = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = F.relu(l(x))
        return self.out(x)


deepnet_r = DeepReluNet()
opt_r = torch.optim.Adam(deepnet_r.parameters(), lr=0.09)
g_hist_r = []
for epoch in range(25):
    logits = deepnet_r(X)
    loss = F.cross_entropy(logits, y)
    opt_r.zero_grad()
    loss.backward()
    # grads = [l.weight.grad.abs().mean().item() for l in deepnet_r.layers]
    grads = [
        param.grad.detach().abs().mean()
        for param in deepnet_r.parameters()
        if param.grad is not None
    ]
    g_hist_r.append(sum(grads) / len(grads))
    opt_r.step()
plt.plot(g_hist, label="Sigmoid")
plt.plot(g_hist_r, label="ReLU")
plt.title("Vanishing Gradients: Sigmoid vs ReLU")
plt.xlabel("Epoch")
plt.ylabel("Mean Grad")
plt.legend()
plt.grid()
plt.show()
