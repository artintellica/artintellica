import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

y_true = torch.tensor([1, 1, 1, 10], dtype=torch.float32)
x_pred = np.linspace(0, 20, 120)
mse_vals = [F.mse_loss(torch.tensor([1, 1, 1, val]), y_true).item() for val in x_pred]
mae_vals = [F.l1_loss(torch.tensor([1, 1, 1, val]), y_true).item() for val in x_pred]
plt.plot(x_pred, mse_vals, label="MSE")
plt.plot(x_pred, mae_vals, label="MAE")
plt.xlabel("Predicted outlier value")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("Effect of Outlier on MSE vs MAE")
plt.show()
