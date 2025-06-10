import torch
import math


def rotation_matrix(theta: float) -> torch.Tensor:
    return torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=torch.float32,
    )


def scaling_matrix(sx: float, sy: float) -> torch.Tensor:
    return torch.tensor([[sx, 0.0], [0.0, sy]], dtype=torch.float32)


# Array of 2D points (shape Nx2)
points: torch.Tensor = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
theta = math.pi / 2  # 90 degrees
R = rotation_matrix(theta)  # Counterclockwise
rotated: torch.Tensor = points @ R.T
print("Original points:\n", points)
print("Rotated points (90°):\n", rotated)
