import torch
import math


def rotation_matrix(theta: float) -> torch.Tensor:
    return torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=torch.float32,
    )


def scaling_matrix(sx: float, sy: float) -> torch.Tensor:
    return torch.tensor([[sx, 0.0], [0.0, sy]], dtype=torch.float32)


# - Define three points: $(1,0)$, $(0,1)$, $(1,1)$.
# - Rotate them by 45 degrees using your rotation matrix.
# - Print the original and rotated coordinates.
points: torch.Tensor = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
theta = math.radians(45)  # Convert degrees to radians
R = rotation_matrix(theta)
rotated: torch.Tensor = points @ R.T
print("Original points:\n", points)
print("Rotated points (45°):\n", rotated)
