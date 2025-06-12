import torch
import torch.nn.functional as F

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
