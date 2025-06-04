import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data  # 4D data (sepal length, sepal width, petal length, petal width)
y = iris.target  # Class labels (0: setosa, 1: versicolor, 2: virginica)
class_names = iris.target_names
print("Shape of Iris dataset:", X.shape)
print("Number of classes:", len(class_names))

# Standardize the data (mean=0, variance=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce from 4D to 2D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
print("Shape of reduced dataset (2D):", X_2d.shape)

# Compute and print explained variance ratio for the top 2 components
explained_variance_ratio = pca.explained_variance_ratio_
print("\nExplained variance ratio for top 2 components:")
print(f"Principal Component 1: {explained_variance_ratio[0]:.4f}")
print(f"Principal Component 2: {explained_variance_ratio[1]:.4f}")
print(f"Total explained variance (PC1 + PC2): {sum(explained_variance_ratio):.4f}")

# Visualize the 2D projection with different colors for each class
plt.figure(figsize=(8, 6))
colors = ["blue", "orange", "green"]
for i in range(len(class_names)):
    plt.scatter(
        X_2d[y == i, 0],
        X_2d[y == i, 1],
        c=colors[i],
        alpha=0.6,
        label=class_names[i].capitalize(),
    )
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D Projection of Iris Dataset after PCA")
plt.legend()
plt.grid(True)
plt.show()
