# src/train.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from preprocessing import load_data, preprocess_data


# -----------------------------
# 1. Load and preprocess data
# -----------------------------
df = load_data("../data/crop_data.csv")
df = preprocess_data(df)

# Change 'label' to your target column name if different
X = df.drop("label", axis=1)
y = df["label"]

# -----------------------------
# 2. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Feature Scaling (important for SVM, Logistic)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
os.makedirs("../models", exist_ok=True)
joblib.dump(scaler, "../models/scaler.pkl")

# -----------------------------
# 4. Initialize Models
# -----------------------------
models = {
    "linear_regression": LinearRegression(),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "svm": SVC(),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# -----------------------------
# 5. Train and Save Models
# -----------------------------
trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")

    # Use scaled data for some models
    if name in ["logistic_regression", "svm"]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)

    trained_models[name] = model

    # Save model
    model_path = f"../models/{name}.pkl"
    joblib.dump(model, model_path)

    print(f"{name} saved at {model_path}")

print("\n All models trained and saved successfully!")

# -----------------------------
# 6. Notes
# -----------------------------
print("\nNote:")
print("- Logistic Regression and SVM use scaled features.")
print("- Random Forest does not require scaling.")
print("- Linear Regression included for comparison purposes.")
