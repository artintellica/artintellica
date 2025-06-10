import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

y_true = torch.tensor([1.0, 1.0, 1.0, 1.0, 10.0])
errs = np.linspace(-5, 10, 200)  # Error for the last (potential outlier) element

mse_vals = [
    F.mse_loss(torch.tensor([*y_true[:-1], y_true[-1] + e]), y_true).item()
    for e in errs
]
mae_vals = [
    F.l1_loss(torch.tensor([*y_true[:-1], y_true[-1] + e]), y_true).item() for e in errs
]

plt.plot(errs, mse_vals, label="MSE")
plt.plot(errs, mae_vals, label="MAE")
plt.xlabel("Outlier error")
plt.ylabel("Loss")
plt.title("Losses vs Outlier Error")
plt.legend()
plt.grid(True)
plt.show()
