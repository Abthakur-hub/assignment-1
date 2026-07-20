# ==========================================
# AI-ML Assignment 1
# Medical Insurance Cost Prediction
# Using Multiple Linear Regression
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("dataset/insurance.csv")

print("Dataset Loaded Successfully!\n")

print("First Five Records:")
print(df.head())


# ==========================================
# Data Understanding
# ==========================================

print("\nDataset Shape:", df.shape)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())


# ==========================================
# Data Preprocessing
# ==========================================

print("\nMissing Values\n")
print(df.isnull().sum())

label_encoder = LabelEncoder()

for column in ["sex", "smoker", "region"]:
    df[column] = label_encoder.fit_transform(df[column])

print("\nCategorical Variables Encoded Successfully!")


# ==========================================
# Split Dataset
# ==========================================

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# Model Development
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")


# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# Model Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-" * 30)
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"R²  : {r2:.4f}")


# ==========================================
# Scatter Plot
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred, alpha=0.7)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)

plt.title("Actual vs Predicted Insurance Charges")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")

plt.grid(True)

plt.savefig("images/actual_vs_predicted.png", dpi=300)

plt.show()