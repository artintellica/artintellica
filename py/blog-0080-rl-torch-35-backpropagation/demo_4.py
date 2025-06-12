import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

# Input and target
x: torch.Tensor = torch.tensor([[1.0, 2.0]])  # (1, 2)
y: torch.Tensor = torch.tensor([1.0])  # (1,)

# Parameters (fixed small values for hand calc)
W1: torch.Tensor = torch.tensor([[0.1, -0.2], [0.3, 0.4]], requires_grad=True)  # (2,2)
W2: torch.Tensor = torch.tensor([[0.7, -0.5]], requires_grad=True)  # (1,2)

# Forward pass (ReLU activation)
z1: torch.Tensor = x @ W1  # (1,2)
h: torch.Tensor = F.relu(z1)  # (1,2)
z2: torch.Tensor = h @ W2.T  # (1,1)
y_pred: torch.Tensor = torch.sigmoid(z2).squeeze()  # scalar

# Binary cross-entropy loss
eps: float = 1e-7
loss: torch.Tensor = -(
    y * torch.log(y_pred + eps) + (1 - y) * torch.log(1 - y_pred + eps)
)
print("Forward values:")
print("z1 =", z1.tolist())
print("h  =", h.tolist())
print("z2 =", z2.item())
print("y_pred =", y_pred.item())
print("loss =", loss.item())

# Manually compute:
# 1. dL/dy_pred = -1/y_pred
dL_dypred: float = float(-1.0 / y_pred.item())

# 2. dy_pred/dz2 = sigmoid'(z2)
dypred_dz2: float = float(y_pred.item() * (1 - y_pred.item()))

print("Manual dL/dy_pred:", dL_dypred)
print("Manual dy_pred/dz2:", dypred_dz2)

# Backpropagation (PyTorch autograd)
# Zero gradients first
if W1.grad is not None:
    W1.grad.zero_()
if W2.grad is not None:
    W2.grad.zero_()
loss.backward()
print("PyTorch dL/dW2:\n", W2.grad)
print("PyTorch dL/dW1:\n", W1.grad)


class TinyMLP(nn.Module):
    def __init__(self, hidden: int = 6) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(2, hidden)
        self.fc2: nn.Linear = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


torch.manual_seed(11)
N: int = 200
X: torch.Tensor = torch.randn(N, 2)
y: torch.Tensor = (X[:, 0] + X[:, 1] > 0).long()
mlp: TinyMLP = TinyMLP(10)
opt: torch.optim.Optimizer = torch.optim.Adam(mlp.parameters(), lr=0.1)
grad1: list[float] = []
grad2: list[float] = []


class DeepMLP(nn.Module):
    def __init__(self, hidden: int = 32, depth: int = 6) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList(
            [nn.Linear(2 if i == 0 else hidden, hidden) for i in range(depth)]
        )
        self.out: nn.Linear = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = torch.sigmoid(l(x))  # Deliberate: will squash gradients
        return self.out(x)


torch.manual_seed(21)
deep_mlp: DeepMLP = DeepMLP()
opt: torch.optim.Optimizer = torch.optim.Adam(deep_mlp.parameters(), lr=0.07)
grad_hist: list[float] = []
for epoch in range(30):
    logits = deep_mlp(X)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    grads = [
        param.grad.detach().abs().mean()
        for param in deep_mlp.parameters()
        if param.grad is not None
    ]
    mean_grad = torch.stack(grads).mean().item() if grads else 0.0
    grad_hist.append(mean_grad)
    opt.step()
plt.plot(grad_hist)
plt.title("Vanishing Gradient in Deep Sigmoid Network")
plt.xlabel("Epoch")
plt.ylabel("Mean Gradient (all hidden layers)")
plt.grid(True)
plt.show()


# Try switching to ReLU
class DeepMLPrelu(nn.Module):
    def __init__(self, hidden: int = 32, depth: int = 6) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList(
            [nn.Linear(2 if i == 0 else hidden, hidden) for i in range(depth)]
        )
        self.out: nn.Linear = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = F.relu(l(x))
        return self.out(x)


deep_mlp_relu: DeepMLPrelu = DeepMLPrelu()
opt2: torch.optim.Optimizer = torch.optim.Adam(deep_mlp_relu.parameters(), lr=0.07)
grad_hist_relu: list[float] = []
for epoch in range(30):
    logits = deep_mlp(X)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    grads = [
        param.grad.detach().abs().mean()
        for param in deep_mlp.parameters()
        if param.grad is not None
    ]
    mean_grad = torch.stack(grads).mean().item() if grads else 0.0
    grad_hist.append(mean_grad)
    opt.step()
plt.plot(grad_hist, label="Sigmoid")
plt.plot(grad_hist_relu, label="ReLU")
plt.xlabel("Epoch")
plt.ylabel("Mean Gradient")
plt.title("Vanishing Gradients: Sigmoid vs ReLU")
plt.legend()
plt.grid(True)
plt.show()
