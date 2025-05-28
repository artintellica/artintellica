"""
exercise_2_multiple_initial_states.py
-------------------------------------------------
Plot spiral ODE solutions for 4 different h0,
overlaying the true trajectory and the Neural ODE fit.
Requires: torchdiffeq (pip install torchdiffeq)
"""

import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint

# --- Spiral ODE parameters (same as earlier demo)
alpha, beta = -0.4, 1.2
t = torch.linspace(0, 7, 160)


def true_field(t, h):
    x, y = h[..., 0], h[..., 1]
    dx = alpha * x - beta * y
    dy = beta * x + alpha * y
    return torch.stack([dx, dy], -1)


# --- Neural ODE definition (copy from training, retrain if needed)
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


# NOTE: You should load a pretrained model for a real test.
# For demo, we quickly (and poorly) train on just one h0.
odefunc = ODEFunc()
optimizer = torch.optim.Adam(odefunc.parameters(), lr=0.01)
h0_train = torch.tensor([2.0, 0.7])
with torch.no_grad():
    true_traj = odeint(true_field, h0_train, t)
for epoch in range(200):
    pred_traj = odeint(odefunc, h0_train, t)
    loss = ((pred_traj - true_traj) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- Now plot for multiple initial states
h0s = [
    torch.tensor([2.0, 0.7]),
    torch.tensor([1.0, -1.5]),
    torch.tensor([-2.0, 0.2]),
    torch.tensor([0.0, 2.0]),
]

plt.figure(figsize=(7, 7))

for h0 in h0s:
    with torch.no_grad():
        true_traj = odeint(true_field, h0, t)
        nn_traj = odeint(odefunc, h0, t)
    plt.plot(
        true_traj[:, 0], true_traj[:, 1], label=f"True $h_0$={h0.tolist()}", linewidth=2
    )
    plt.plot(
        nn_traj[:, 0], nn_traj[:, 1], "--", label=f"Neural ODE $h_0$={h0.tolist()}"
    )

plt.title("Spiral ODEs: True vs Neural ODE for Multiple $h_0$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(fontsize=8, loc="best")
plt.axis("equal")
plt.tight_layout()
plt.show()
