import torch
import matplotlib.pyplot as plt
import math

# ### **Exercise 2:** Apply Various Matrix Transformations and Visualize Before/After


# - Create a rotation matrix for 90°, and a scaling matrix (scale x by 2, y by
#   0.5).
# - Apply both (try scaling then rotating).
# - Plot the original and transformed datasets side-by-side.
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
theta = math.radians(90)
R = rotation_matrix(theta)
S = scaling_matrix(2.0, 0.5)
square_transf = (square @ S.T) @ R.T  # scale then rotate
plt.subplot(1, 2, 1)
plt.scatter(square[:, 0], square[:, 1], alpha=0.5, label="Original")
plt.axis("equal")
plt.title("Original")
plt.subplot(1, 2, 2)
plt.scatter(
    square_transf[:, 0],
    square_transf[:, 1],
    alpha=0.5,
    color="orange",
    label="Transformed",
)
plt.axis("equal")
plt.title("Transformed")
plt.tight_layout()
plt.show()
