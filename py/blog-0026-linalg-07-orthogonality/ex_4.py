import numpy as np
import matplotlib.pyplot as plt

# Define two 2D vectors
v1 = np.array([2, 1])
v2 = np.array([1, 2])

# Gram-Schmidt process
# Step 1: Normalize v1 to get u1
u1 = v1 / np.linalg.norm(v1)

# Step 2: Orthogonalize v2 against u1
w2 = v2 - np.dot(v2, u1) * u1

# Step 3: Normalize w2 to get u2
u2 = w2 / np.linalg.norm(w2)

# Verify orthonormality
dot_u1_u2 = np.dot(u1, u2)
norm_u1 = np.linalg.norm(u1)
norm_u2 = np.linalg.norm(u2)

# Print results
print("Original vector v1:", v1)
print("Original vector v2:", v2)
print("\nOrthonormal vectors:")
print("u1:", u1)
print("u2:", u2)
print("\nVerification of orthonormality:")
print("Dot product u1 · u2 (should be ~0):", dot_u1_u2)
print("Norm of u1 (should be ~1):", norm_u1)
print("Norm of u2 (should be ~1):", norm_u2)
print("Orthogonal?", np.isclose(dot_u1_u2, 0, atol=1e-10))
print("u1 unit vector?", np.isclose(norm_u1, 1, atol=1e-10))
print("u2 unit vector?", np.isclose(norm_u2, 1, atol=1e-10))


# Visualize original and orthonormal vectors
def plot_2d_vectors(vectors, labels, colors):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)
    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(*origin, *vec, color=color, scale=1, scale_units="xy", angles="xy")
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)
    plt.grid(True)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Gram-Schmidt Orthonormal Basis")
    plt.show()


plot_2d_vectors(
    [v1, v2, u1, u2], ["v1", "v2", "u1", "u2"], ["blue", "red", "green", "purple"]
)
