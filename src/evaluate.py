# src/evaluate.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from preprocessing import load_data, preprocess_data


# -----------------------------
# 1. Load and preprocess data
# -----------------------------
df = load_data("../data/crop_data.csv")
df = preprocess_data(df)

# Change 'label' if needed
X = df.drop("label", axis=1)
y = df["label"]

# -----------------------------
# 2. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Load scaler
# -----------------------------
scaler = joblib.load("../models/scaler.pkl")

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 4. Load models
# -----------------------------
models = {
    "Linear Regression": joblib.load("../models/linear_regression.pkl"),
    "Logistic Regression": joblib.load("../models/logistic_regression.pkl"),
    "SVM": joblib.load("../models/svm.pkl"),
    "Random Forest": joblib.load("../models/random_forest.pkl")
}

# -----------------------------
# 5. Evaluate models
# -----------------------------
results = []

for name, model in models.items():
    print(f"Evaluating {name}...")

    # Use scaled data where required
    if name in ["Logistic Regression", "SVM"]:
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)

    # Handle Linear Regression output (convert to class)
    if name == "Linear Regression":
        y_pred = [round(val) for val in y_pred]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted')
    })

# -----------------------------
# 6. Results DataFrame
# -----------------------------
results_df = pd.DataFrame(results)

print("\n Model Comparison Results:\n")
print(results_df)

# Save results
results_df.to_csv("../results/model_comparison.csv", index=False)

# -----------------------------
# 7. Visualization
# -----------------------------
results_df.set_index("Model")[["Accuracy", "F1 Score"]].plot(kind="bar")

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("../results/model_comparison.png")
plt.show()

print("\n Results saved in /results folder")
