import torch
import math


def rotation_matrix(theta: float) -> torch.Tensor:
    return torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=torch.float32,
    )


def scaling_matrix(sx: float, sy: float) -> torch.Tensor:
    return torch.tensor([[sx, 0.0], [0.0, sy]], dtype=torch.float32)


theta = math.pi / 4  # 45 degrees
R = rotation_matrix(theta)
S = scaling_matrix(2.0, 0.5)
print("Rotation (45°):\n", R)
print("Scaling (2, 0.5):\n", S)
