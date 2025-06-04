from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the California Housing dataset
california = fetch_california_housing()
X = california.data
y = california.target
feature_names = california.feature_names
print("Shape of full dataset:", X.shape)
print("Feature names:", feature_names)

# Select 2 features for simplicity (e.g., 'MedInc' and 'HouseAge')
selected_features = [0, 1]  # Indices for 'MedInc' (median income) and 'HouseAge'
X_selected = X[:, selected_features]
selected_names = [feature_names[i] for i in selected_features]
print("Shape of selected dataset (2 features):", X_selected.shape)
print("Selected features:", selected_names)

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=48
)
print("Shape of training set:", X_train.shape, y_train.shape)
print("Shape of test set:", X_test.shape, y_test.shape)

# Fit a linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Extract and print the coefficients
coefficients = model.coef_
intercept = model.intercept_
print("\nLearned coefficients:")
for name, coef in zip(selected_names, coefficients):
    print(f"{name}: {coef:.4f}")
print(f"Intercept (bias): {intercept:.4f}")

# Predict on the test set
y_pred = model.predict(X_test)

# Compute and print the mean squared error on the test set
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error on test set:", mse)
