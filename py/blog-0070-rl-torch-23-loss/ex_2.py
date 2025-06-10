import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

x = np.array([0, 1, 2, 3])
y = np.array([1, 2, 2, 4])
ws = np.linspace(-1, 3, 100)
loss_curve = [np.mean((w * x - y) ** 2) for w in ws]
plt.plot(ws, loss_curve)
plt.xlabel("w")
plt.ylabel("MSE Loss")
plt.title("Loss Surface for Linear Model")
plt.grid(True)
plt.show()
