import torch
import torch.nn.functional as F

y_true = torch.tensor([2.0, 3.5, 5.0])
y_pred = torch.tensor([2.5, 2.8, 4.6])

# Manual MSE
mse_manual = ((y_true - y_pred) ** 2).mean()
print("Manual MSE:", mse_manual.item())

# PyTorch MSE
mse_builtin = F.mse_loss(y_pred, y_true)
print("PyTorch MSE:", mse_builtin.item())
