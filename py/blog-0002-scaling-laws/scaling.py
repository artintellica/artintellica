import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)
print("Using", device)


class MLP(nn.Module):
    """Three‑layer tanh MLP for 1‑D regression."""

    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x)


def gen_batch(n: int):
    """x ∈ [‑π, π],  y = sin x ."""
    x = torch.rand(n, 1) * (2 * math.pi) - math.pi
    y = torch.sin(x)
    return x.to(device), y.to(device)


@torch.no_grad()
def mse(model, n_val=1_000):
    x_val, y_val = gen_batch(n_val)
    return nn.functional.mse_loss(model(x_val), y_val).item()


def train_once(
    width: int, n_train: int, steps: int = 2000, batch_sz: int = 64, lr: float = 2e-3
):
    model = MLP(width).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    # One big dataset, but we’ll draw random batches each step
    x_full, y_full = gen_batch(n_train)
    for _ in range(steps):
        idx = torch.randint(0, n_train, (batch_sz,))
        x_b, y_b = x_full[idx], y_full[idx]
        opt.zero_grad()
        nn.functional.mse_loss(model(x_b), y_b).backward()
        opt.step()

    return mse(model)  # validation MSE on 1 000 fresh points

print("Error vs. data size ...")

data_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
fixed_width = 64  # ≈ 20 k parameters
data_err = [train_once(fixed_width, n) for n in data_sizes]

plt.figure(figsize=(6, 4))
plt.loglog(data_sizes, data_err, "o-")
plt.xlabel("training samples D (log)")
plt.ylabel("val MSE (log)")
plt.title("Fixed model, growing data")
plt.grid(True, which="both", ls="--")
plt.show()

print("Error vs. model size ...")

widths = [2, 4, 8, 16, 32, 64, 128, 256]  # model “size” dial
fixed_data = 2048
model_err = [train_once(w, fixed_data) for w in widths]

n_params = [3 * w * w + 2 * w + 1 for w in widths]  # rough param count
plt.figure(figsize=(6, 4))
plt.loglog(n_params, model_err, "s-")
plt.xlabel("parameters N (log)")
plt.ylabel("val MSE (log)")
plt.title("Fixed data, growing model")
plt.grid(True, which="both", ls="--")
plt.show()
