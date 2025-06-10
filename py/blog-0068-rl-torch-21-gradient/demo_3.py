import torch

def grad_f_vec(x: torch.Tensor) -> torch.Tensor:
    return 2 * x

x: torch.Tensor = torch.tensor([5.0, -3.0], dtype=torch.float32)  # Initial point in 2D
eta_vec = 0.2
trajectory_vec = [x.clone()]

for step in range(15):
    x = x - eta_vec * grad_f_vec(x)
    trajectory_vec.append(x.clone())

trajectory_vec = torch.stack(trajectory_vec)

print("Final x:", x)
print("Norm at end:", torch.norm(x).item())
