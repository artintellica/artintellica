# calc-12-ode/neural_ode_spiral.py
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

torch.manual_seed(42)
device = torch.device("cpu")

# --- Spiral data generation
alpha, beta = -0.4, 1.2


def true_field(t, h):
    x, y = h[..., 0], h[..., 1]
    dx = alpha * x - beta * y
    dy = beta * x + alpha * y
    return torch.stack([dx, dy], -1)


t = torch.linspace(0, 7, 160)
h0 = torch.tensor([2.0, 0.7])

with torch.no_grad():
    true_traj = odeint(true_field, h0, t)


# --- Neural ODE model
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

# --- Training loop
for epoch in range(250):
    pred_traj = odeint(odefunc, h0, t)
    loss = ((pred_traj - true_traj) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 40 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.5f}")

# --- Plot
plt.plot(true_traj[:, 0], true_traj[:, 1], label="True", lw=3)
plt.plot(
    pred_traj.detach()[:, 0], pred_traj.detach()[:, 1], "--", label="Neural ODE", lw=2
)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Neural ODE fits Spiral Trajectory")
plt.axis("equal")
plt.tight_layout()
plt.show()
