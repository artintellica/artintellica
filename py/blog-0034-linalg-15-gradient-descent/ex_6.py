import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the California Housing dataset
california = fetch_california_housing()
X = california.data
y = california.target
feature_names = california.feature_names
print("Shape of full dataset:", X.shape)
print("Feature names:", feature_names)

# Select 3 features for simplicity (e.g., 'MedInc', 'HouseAge', 'AveRooms')
selected_features = [0, 1, 2]  # Indices for 'MedInc', 'HouseAge', 'AveRooms'
X_selected = X[:, selected_features]
selected_names = [feature_names[i] for i in selected_features]
print("Shape of selected dataset (3 features):", X_selected.shape)
print("Selected features:", selected_names)

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=51
)
print("Shape of training set:", X_train.shape, y_train.shape)
print("Shape of test set:", X_test.shape, y_test.shape)

# Add a column of ones to X_train and X_test for the bias term
X_train_with_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_with_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
print("Shape of training set with bias:", X_train_with_bias.shape)

# Initialize weights
n_features = X_train_with_bias.shape[1]  # 4 (3 features + bias)
w_init = np.zeros(n_features)
eta = 0.001  # Learning rate (smaller due to unscaled data)
batch_size = 32
n_epochs = 100  # Full passes through the data
n_samples_train = X_train_with_bias.shape[0]
n_batches_per_epoch = n_samples_train // batch_size

# Mini-Batch Gradient Descent
w = w_init.copy()
losses = []

for epoch in range(n_epochs):
    # Shuffle the training data for each epoch
    indices = np.random.permutation(n_samples_train)
    X_shuffled = X_train_with_bias[indices]
    y_shuffled = y_train[indices]

    # Process mini-batches
    for i in range(0, n_samples_train, batch_size):
        X_batch = X_shuffled[i : i + batch_size]
        y_batch = y_shuffled[i : i + batch_size]
        # Compute gradient for mini-batch: (2/batch_size) * X^T * (Xw - y)
        gradient = (2 / batch_size) * X_batch.T @ (X_batch @ w - y_batch)
        # Update weights
        w = w - eta * gradient

    # Compute and store loss (MSE) on full training set after each epoch
    loss = np.mean((X_train_with_bias @ w - y_train) ** 2)
    losses.append(loss)

print("\nFinal weights (bias, coef_MedInc, coef_HouseAge, coef_AveRooms):", w)

# Compute predictions and MSE on test set
y_pred_test = X_test_with_bias @ w
test_mse = mean_squared_error(y_test, y_pred_test)
print("Final MSE on test set:", test_mse)

# Plot loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(n_epochs), losses, label="Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Training Loss Over Epochs (Mini-Batch GD)")
plt.legend()
plt.grid(True)
plt.show()
