import torch
import matplotlib.pyplot as plt

# 2D vectors
u_2d: torch.Tensor = torch.tensor([1.0, 2.0])
v_2d: torch.Tensor = torch.tensor([2.0, 1.0])
sum_uv: torch.Tensor = u_2d + v_2d
scaled_u: torch.Tensor = 1.5 * u_2d

# Plotting
plt.figure(figsize=(6, 6))
plt.quiver(
    0, 0, u_2d[0], u_2d[1], angles="xy", scale_units="xy", scale=1, color="b", label="u"
)
plt.quiver(
    0, 0, v_2d[0], v_2d[1], angles="xy", scale_units="xy", scale=1, color="r", label="v"
)
plt.quiver(
    0,
    0,
    sum_uv[0],
    sum_uv[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="g",
    label="u + v",
)
plt.quiver(
    0,
    0,
    scaled_u[0],
    scaled_u[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="purple",
    label="1.5 * u",
)

plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.grid(True)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.title("Vector Addition and Scalar Multiplication in 2D")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
