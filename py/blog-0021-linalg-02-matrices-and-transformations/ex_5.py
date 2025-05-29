import numpy as np

# Create a synthetic dataset: 5 samples, 3 features (height, weight, age)
# (Values are made up for demonstration)
#           height  weight  age
data = np.array(
    [[170, 65, 25], [160, 70, 30], [180, 80, 28], [175, 75, 35], [165, 60, 22]]
)

print("Dataset (rows: samples, columns: features):\n", data)

# Compute the mean of each feature (i.e., mean by column)
feature_means = data.mean(axis=0)

print("\nMean of each feature (height, weight, age):", feature_means)
