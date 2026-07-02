# ================================
# TASK 3 - LINEAR REGRESSION
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Load Dataset
# -------------------------------
# Change the filename if necessary
df = pd.read_csv("dataset/house_price.csv")
print(df.columns.tolist())
print(df.head())
print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

# -------------------------------
# Select Features and Target
# -------------------------------
# Change column names according to your dataset

X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model = LinearRegression()

model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics")
print("-----------------------")
print("MAE :", mae)
print("MSE :", mse)
print("R2 Score :", r2)

# -------------------------------
# Coefficients
# -------------------------------
print("\nModel Coefficients")

for feature, coef in zip(X.columns, model.coef_):
    print(feature, ":", coef)

print("Intercept :", model.intercept_)

# -------------------------------
# Actual vs Predicted Plot
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# -------------------------------
# Regression Line (Simple Regression)
# -------------------------------

X_simple = df[['area']]
y_simple = df['price']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple,
    y_simple,
    test_size=0.2,
    random_state=42
)

simple_model = LinearRegression()

simple_model.fit(X_train_s, y_train_s)

plt.figure(figsize=(8,6))
plt.scatter(X_test_s, y_test_s, color='blue')

plt.plot(
    X_test_s,
    simple_model.predict(X_test_s),
    color='red',
    linewidth=2
)

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Simple Linear Regression")
plt.show()

print("\nTask Completed Successfully!")