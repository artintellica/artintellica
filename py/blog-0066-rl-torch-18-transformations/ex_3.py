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


square = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]], dtype=torch.float32)
S2 = scaling_matrix(1.5, 0.5)
scaled_sq = square @ S2.T
plt.figure(figsize=(5, 5))
plt.plot(square[:, 0], square[:, 1], "b-o", label="Original Square")
plt.plot(scaled_sq[:, 0], scaled_sq[:, 1], "r-o", label="Scaled Square")
plt.title("Scaling a Square")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
