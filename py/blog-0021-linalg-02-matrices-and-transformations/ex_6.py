import numpy as np
import matplotlib.pyplot as plt

# The vector to rotate
vector = np.array([1, 0])

# Angles in degrees and radians
angles_deg = [0, 45, 90]
angles_rad = [np.deg2rad(a) for a in angles_deg]

# Colors and labels for plotting
colors = ["blue", "green", "red"]
labels = [f"{a}°" for a in angles_deg]

# Compute rotated vectors
rotated_vectors = []
for theta in angles_rad:
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    rotated = rotation_matrix @ vector
    rotated_vectors.append(rotated)

# Plot all vectors
plt.figure(figsize=(6, 6))
origin = np.zeros(2)

for vec, color, label in zip(rotated_vectors, colors, labels):
    plt.quiver(
        *origin, *vec, color=color, angles="xy", scale=1, scale_units="xy", label=label
    )
    plt.text(vec[0] * 1.1, vec[1] * 1.1, label, color=color, fontsize=12)

# Formatting
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.axhline(0, color="gray", linewidth=0.5)
plt.axvline(0, color="gray", linewidth=0.5)
plt.grid(True)
plt.title("Rotation of [1, 0] by 0°, 45°, and 90°")
plt.legend()
plt.gca().set_aspect("equal")
plt.show()
