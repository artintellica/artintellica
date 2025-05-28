"""
exercise_3_time_varying_field.py
-------------------------------------------------
Neural ODE where the neural net gets (h, t) as input.
Demonstrates time-varying vector field with torchdiffeq.
"""

import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

torch.manual_seed(0)

# --- Data: Spiral ODE (for demonstration)
alpha, beta = -0.4, 1.2
t = torch.linspace(0, 7, 160)
h0 = torch.tensor([2.0, 0.7])

def true_field(t, h):
    x, y = h[..., 0], h[..., 1]
    dx = alpha * x - beta * y
    dy = beta * x + alpha * y
    return torch.stack([dx, dy], -1)

with torch.no_grad():
    true_traj = odeint(true_field, h0, t)

# --- Neural ODE with time input
class TimeVaryingODEFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 2),
        )
    def forward(self, t, h):
        # h: shape (..., 2), t: scalar
        # Expand t to match batch dimension
        t_ = t * torch.ones_like(h[..., :1])
        inp = torch.cat([h, t_], dim=-1)
        return self.net(inp)

odefunc = TimeVaryingODEFunc()
optimizer = torch.optim.Adam(odefunc.parameters(), lr=0.01)

# --- Training loop
for epoch in range(250):
    pred_traj = odeint(odefunc, h0, t)
    loss = ((pred_traj - true_traj) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.5f}")

# --- Plot
plt.plot(true_traj[:, 0], true_traj[:, 1], label="True Spiral", lw=3)
plt.plot(pred_traj.detach()[:, 0], pred_traj.detach()[:, 1], "--", label="Neural ODE (with t)", lw=2)
plt.xlabel("x"); plt.ylabel("y")
plt.title("Neural ODE with Time-Varying Field (input t)")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
