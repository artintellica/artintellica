import torch
import matplotlib.pyplot as plt

# - Create $2$D vectors `u = [4, 3]` and `v = [5, 0]`.
# - Compute the projection of `u` onto `v`.
# - Plot `u`, `v`, and the projection vector from the origin.
u: torch.Tensor = torch.tensor([4.0, 3.0])
v: torch.Tensor = torch.tensor([5.0, 0.0])
# Compute projection
proj_length: torch.Tensor = torch.dot(u, v) / torch.dot(v, v)
proj_vec: torch.Tensor = proj_length * v
print("u:", u)
print("v:", v)
print("Projection of u onto v:", proj_vec)
# Plot
plt.figure(figsize=(6, 6))
plt.quiver(
    0, 0, u[0], u[1], angles="xy", scale_units="xy", scale=1, color="b", label="u"
)
plt.quiver(
    0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1, color="r", label="v"
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
    label="proj_v(u)",
)
plt.legend()
plt.xlim(-1, 6)
plt.ylim(-1, 4)
plt.grid(True)
plt.axhline(0, color="black", linewidth=0.3)
plt.axvline(0, color="black", linewidth=0.3)
plt.title("Vector Projection in 2D")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

