import numpy as np
import matplotlib.pyplot as plt

# Define a 2x2 matrix
A = np.array([[2, 1], [1, 2]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)


# Visualize eigenvectors and their transformations
def plot_eigenvectors(A, eigenvalues, eigenvectors):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)

    # Plot original and transformed eigenvectors
    colors = ["blue", "red"]
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        Av = A @ v
        plt.quiver(
            *origin,
            *v,
            color=colors[i],
            scale=1,
            scale_units="xy",
            angles="xy",
            alpha=0.5,
        )
        plt.quiver(
            *origin, *Av, color=colors[i], scale=1, scale_units="xy", angles="xy"
        )
        plt.text(v[0], v[1], f"v{i+1}", color=colors[i], fontsize=12)
        plt.text(Av[0], Av[1], f"Av{i+1}", color=colors[i], fontsize=12)

    plt.grid(True)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Eigenvectors and Their Transformations")
    plt.show()


plot_eigenvectors(A, eigenvalues, eigenvectors)

# Verify eigenvalue equation
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    Av = A @ v
    lambda_v = eigenvalues[i] * v
    print(f"\nEigenvector {i+1}:", v)
    print(f"A @ v_{i+1}:", Av)
    print(f"λ_{i+1} * v_{i+1}:", lambda_v)
    print("Match?", np.allclose(Av, lambda_v))
