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


# - Define coordinates for the corners of a square.
# - Apply a scaling transformation with $s_x = 1.5$, $s_y = 0.5$.
# - Plot the original and scaled squares.
square: torch.Tensor = torch.tensor(
    [
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],  # close the square for the plot
    ]
)

R90 = rotation_matrix(math.radians(90))
S3 = scaling_matrix(0.5, 2.0)
# rotate then scale
sq_rot_scale = (square @ R90.T) @ S3.T
# scale then rotate
sq_scale_rot = (square @ S3.T) @ R90.T

plt.figure(figsize=(6, 6))
plt.plot(square[:, 0], square[:, 1], "b--o", label="Original")
plt.plot(sq_rot_scale[:, 0], sq_rot_scale[:, 1], "r-o", label="Rotate->Scale")
plt.plot(sq_scale_rot[:, 0], sq_scale_rot[:, 1], "g-o", label="Scale->Rotate")
plt.title("Chained Transformations on a Square")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
