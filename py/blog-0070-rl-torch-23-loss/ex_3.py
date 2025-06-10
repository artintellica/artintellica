import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

p = np.linspace(0.01, 0.99, 100)
bce_0 = -np.log(1 - p)
bce_1 = -np.log(p)
plt.plot(p, bce_0, label="y=0")
plt.plot(p, bce_1, label="y=1")
plt.xlabel("Predicted probability ($\\hat{y}$)")
plt.ylabel("BCE Loss")
plt.legend(); plt.grid(True); plt.title("BCE Loss as Function of Prediction"); plt.show()
