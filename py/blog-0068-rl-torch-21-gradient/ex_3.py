import torch
import numpy as np
import matplotlib.pyplot as plt


def grad_fvec_shifted(x: torch.Tensor) -> torch.Tensor:
    return 2 * (x - torch.tensor([2.0, -1.0]))


xv = torch.tensor([5.0, 5.0])
eta_vec = 0.1
traj_v = [xv.clone()]
for _ in range(20):
    xv = xv - eta_vec * grad_fvec_shifted(xv)
    traj_v.append(xv.clone())

traj_v_np = torch.stack(traj_v).numpy()
target = np.array([2.0, -1.0])
plt.plot(traj_v_np[:, 0], traj_v_np[:, 1], "o-", label="GD Path")
plt.plot(target[0], target[1], "rx", label="Minimum")
plt.title("Gradient Descent Path: Vector")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
