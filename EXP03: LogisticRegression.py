import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load data
df = pd.read_csv('E:/ENGG/THIRD YEAR/MACHINE LEARNING LAB/EXP03/diabetes.csv')

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# Scale features (important for gradient descent)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize parameters
w = np.random.randn(X_train.shape[1])  # shape (n_features,)
b = 0
learning_rate = 0.05
epochs = 1000

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward propagation
def forward_prop(w, b, X):
    return sigmoid(np.dot(X, w) + b)

# Cost function (Log loss)
def cost(y_hat, y):
    m = len(y)
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)  # Avoid log(0)
    return -(1/m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# Backward propagation
def back_prop(X, y, y_hat):
    m = len(y)
    dw = (1/m) * np.dot(X.T, (y_hat - y))
    db = (1/m) * np.sum(y_hat - y)
    return dw, db

# Training loop
for epoch in range(epochs):
    y_hat = forward_prop(w, b, X_train)
    loss = cost(y_hat, y_train)
    dw, db = back_prop(X_train, y_train, y_hat)
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Predict function
def predict(X, w, b, threshold=0.5):
    prob = forward_prop(w, b, X)
    return (prob >= threshold).astype(int)

# Evaluate on test set
y_pred = predict(X_test, w, b)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
