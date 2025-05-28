"""
exercise_4_noise_robustness.py
-------------------------------------------------
Add Gaussian noise to spiral data and train a Neural ODE.
Plot fit to noisy vs. true data.
"""

import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

torch.manual_seed(123)
device = torch.device("cpu")

# --- Spiral ODE parameters (as before)
alpha, beta = -0.4, 1.2
t = torch.linspace(0, 7, 160)
h0 = torch.tensor([2.0, 0.7])


def true_field(t, h):
    x, y = h[..., 0], h[..., 1]
    dx = alpha * x - beta * y
    dy = beta * x + alpha * y
    return torch.stack([dx, dy], -1)


# --- Generate (noiseless) spiral trajectory
with torch.no_grad():
    true_traj = odeint(true_field, h0, t)

# --- Add Gaussian noise to the data
noise_std = 0.15
noisy_traj = true_traj + noise_std * torch.randn_like(true_traj)


# --- Neural ODE model (time-invariant for simplicity)
class ODEFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 2),
        )

    def forward(self, t, h):
        return self.net(h)


odefunc = ODEFunc()
optimizer = torch.optim.Adam(odefunc.parameters(), lr=0.01)

# --- Training on noisy data
for epoch in range(300):
    pred_traj = odeint(odefunc, h0, t)
    loss = ((pred_traj - noisy_traj) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.5f}")

# --- Plot
plt.plot(true_traj[:, 0], true_traj[:, 1], label="True Spiral", lw=3, alpha=0.7)
plt.plot(noisy_traj[:, 0], noisy_traj[:, 1], ".", label="Noisy Data", alpha=0.5)
plt.plot(
    pred_traj.detach()[:, 0],
    pred_traj.detach()[:, 1],
    "--",
    label="Neural ODE Fit",
    lw=2,
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Neural ODE Fit with Noisy Spiral Data")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
