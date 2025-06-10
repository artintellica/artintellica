import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ### **Exercise 1:** Implement Mean Squared Error (MSE) Loss Manually and with PyTorch

# - Given $y_\text{true} = [1.5, 3.0, 4.0]$, $y_\text{pred} = [2.0, 2.5, 3.5]$,
#   compute MSE manually.
# - Verify it matches `torch.nn.functional.mse_loss`.

y_true = torch.tensor([1.5, 3.0, 4.0])
y_pred = torch.tensor([2.0, 2.5, 3.5])
# Manual MSE calculation
mse_manual = ((y_true - y_pred) ** 2).mean()
print("Manual MSE:", mse_manual.item())
# PyTorch MSE calculation
mse_builtin = F.mse_loss(y_pred, y_true)
print("PyTorch MSE:", mse_builtin.item())
# Verify that both manual and PyTorch MSE match
# assert np.isclose(mse_manual.item(), mse_builtin.item()), "MSE values do not match!"
print("MSE values match:", np.isclose(mse_manual.item(), mse_builtin.item()))
