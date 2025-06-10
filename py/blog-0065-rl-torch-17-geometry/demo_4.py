import torch
import matplotlib.pyplot as plt

# 2D vectors for visualization
a2d: torch.Tensor = torch.tensor([3.0, 1.0])
b2d: torch.Tensor = torch.tensor([2.0, 0.0])
# Compute projection
proj_length: torch.Tensor = torch.dot(a2d, b2d) / torch.dot(b2d, b2d)
proj_vec: torch.Tensor = proj_length * b2d
print("a2d:", a2d)
print("b2d:", b2d)
print("Projection of a2d onto b2d:", proj_vec)

# Plot
plt.figure(figsize=(6, 6))
plt.quiver(
    0, 0, a2d[0], a2d[1], angles="xy", scale_units="xy", scale=1, color="b", label="a2d"
)
plt.quiver(
    0, 0, b2d[0], b2d[1], angles="xy", scale_units="xy", scale=1, color="r", label="b2d"
)
plt.quiver(
    0,
    0,
    proj_vec[0],
    proj_vec[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="g",
    label="proj_b(a)",
)
plt.legend()
plt.xlim(-1, 5)
plt.ylim(-1, 3)
plt.grid(True)
plt.axhline(0, color="black", linewidth=0.3)
plt.axvline(0, color="black", linewidth=0.3)
plt.title("Vector Projection in 2D")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
