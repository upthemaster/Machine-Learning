import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Step 1: Load dataset
file_path = r"E:\ENGG\THIRD YEAR\MACHINE LEARNING LAB\EXP02\california_housing_test.csv"
df = pd.read_csv(file_path)

# Step 2: Check and rename target column
if 'median_house_value' in df.columns:
    df.rename(columns={'median_house_value': 'price'}, inplace=True)
else:
    raise ValueError("Expected column 'median_house_value' not found. Available columns: ", df.columns)

# Step 3: Normalize features (excluding target)
features = df.columns[df.columns != 'price']
df[features] = (df[features] - df[features].mean()) / df[features].std()

# Step 4: Feature matrix and target vector
X = df.drop('price', axis=1).values
y = df['price'].values.reshape(-1, 1)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Initialize weights and bias
w = np.random.randn(1, X_train.shape[1])
b = 0
learning_rate = 0.01

# Forward propagation
def for_prop(w, b, X):
    return np.dot(w, X.T).T + b  # Output shape: (n_samples, 1)

# Cost function (Mean Squared Error)
def cost(z, y):
    m = y.shape[0]
    return (1 / (2 * m)) * np.sum((z - y) ** 2)

# Backward propagation
def back_prop(z, y, X):
    m = y.shape[0]
    dz = z - y
    dw = (1 / m) * np.dot(dz.T, X)
    db = (1 / m) * np.sum(dz)
    return dw, db

# Gradient descent update
def gradient_descent(w, b, dw, db, learning_rate):
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

# Training function
def linear_model(X_train, y_train, epochs=1000):
    global w, b
    losses = []
    for i in range(epochs):
        z = for_prop(w, b, X_train)
        c = cost(z, y_train)
        dw, db = back_prop(z, y_train, X_train)
        w, b = gradient_descent(w, b, dw, db, learning_rate)
        losses.append(c)
        if i % 100 == 0:
            print(f"Epoch {i} - Cost: {c:.4f}")
    return w, b, losses

# Step 7: Train the model
w, b, losses = linear_model(X_train, y_train, epochs=1000)

# Step 8: Predict on test set
y_pred = for_prop(w, b, X_test)

# Step 9: Evaluation metrics
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")

# Step 10: Plot loss curve
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Loss during Training")
plt.grid(True)
plt.show()

# Step 11: Regression plot for one feature (e.g., median_income)
feature_name = 'median_income'
if feature_name in df.columns:
    feature_index = df.columns.get_loc(feature_name)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, feature_index], y_test, label='Actual')
    plt.scatter(X_test[:, feature_index], y_pred, color='red', alpha=0.5, label='Predicted')
    plt.xlabel(f"{feature_name} (Standardized)")
    plt.ylabel("Price")
    plt.title(f"Regression: {feature_name} vs Price")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print(f"Feature '{feature_name}' not found in dataset.")

# Step 12: Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.grid(True)
plt.show()
