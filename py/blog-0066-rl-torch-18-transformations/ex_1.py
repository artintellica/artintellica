import torch
import math


# - Create a 2D rotation matrix for 60 degrees.
# - Create a scaling matrix that doubles $x$ and halves $y$.
# - Print both matrices.
def rotation_matrix(theta: float) -> torch.Tensor:
    return torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=torch.float32,
    )


def scaling_matrix(sx: float, sy: float) -> torch.Tensor:
    return torch.tensor([[sx, 0.0], [0.0, sy]], dtype=torch.float32)


theta = math.radians(60)  # Convert degrees to radians
R = rotation_matrix(theta)
S = scaling_matrix(2.0, 0.5)
print("Rotation matrix (60°):\n", R)
print("Scaling matrix (2, 0.5):\n", S)
