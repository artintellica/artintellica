import torch
import math


def rotation_matrix(theta: float) -> torch.Tensor:
    return torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=torch.float32,
    )


def scaling_matrix(sx: float, sy: float) -> torch.Tensor:
    return torch.tensor([[sx, 0.0], [0.0, sy]], dtype=torch.float32)


# Define points: corners of the square (counterclockwise)
square: torch.Tensor = torch.tensor(
    [
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],  # close the square for the plot
    ]
)
# Transformation: rotate by 30° and scale x2 in x, 0.5 in y
theta = math.radians(30)
R = rotation_matrix(theta)
S = scaling_matrix(2.0, 0.5)

# Chain: first scaling, then rotation (note order!)
composite1 = R @ S
# Chain: first rotation, then scaling
composite2 = S @ R

point = torch.tensor([[1.0, 0.0]])
result1 = (point @ composite1.T).squeeze()
result2 = (point @ composite2.T).squeeze()
print("Scaling THEN rotation:", result1)
print("Rotation THEN scaling:", result2)
