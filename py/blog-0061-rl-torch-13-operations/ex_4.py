import torch
import matplotlib.pyplot as plt

d: torch.Tensor = torch.tensor([1, 3], dtype=torch.float32)
e: torch.Tensor = torch.tensor([2, 1], dtype=torch.float32)
sum_de: torch.Tensor = d + e

scaled_sum_de: torch.Tensor = sum_de * 0.5

print("d:", d)
print("e:", e)
print("d + e:", sum_de)
print("Scaled (d + e) by 0.5:", scaled_sum_de)
# Plotting the vectors
plt.quiver(
    0, 0, d[0], d[1], angles="xy", scale_units="xy", scale=1, color="r", label="d"
)
plt.quiver(
    0, 0, e[0], e[1], angles="xy", scale_units="xy", scale=1, color="b", label="e"
)
plt.quiver(
    0,
    0,
    sum_de[0],
    sum_de[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="g",
    label="d + e",
)
plt.quiver(
    0,
    0,
    scaled_sum_de[0],
    scaled_sum_de[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="purple",
    label="0.5 * (d + e)",
)
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.grid(True)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.title("Vector Addition and Scalar Multiplication")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
