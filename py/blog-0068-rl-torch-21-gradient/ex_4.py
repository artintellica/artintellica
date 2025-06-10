import torch
import numpy as np
import matplotlib.pyplot as plt


def grad_fx_shifted(x: float) -> float:
    return 2 * (x - 3)


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

# lrs = [0.01, 0.1, 1.0, 1.5]
# lrs = [0.01, 0.1, 1.0]
lrs = [0.01, 0.02, 0.05, 0.1]
plt.figure()
for lr in lrs:
    x = -7.0
    h = [x]
    for _ in range(15):
        x = x - lr * grad_fx_shifted(x)
        h.append(x)
    plt.plot(h, [(hx - 3) ** 2 for hx in h], "o-", label=f"LR={lr}")
plt.plot(3, 0, "kx", markersize=10, label="Minimum")
plt.legend()
plt.grid(True)
plt.title("Learning Rate Effect: Scalar")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
