import torch
import math
import matplotlib.pyplot as plt


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
# Apply scaling THEN rotation
transformed: torch.Tensor = (square @ S.T) @ R.T

plt.figure(figsize=(6, 6))
plt.plot(square[:, 0], square[:, 1], "bo-", label="Original")
plt.plot(transformed[:, 0], transformed[:, 1], "ro-", label="Transformed")
plt.title("Transforming a Square: Rotation and Scaling")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
