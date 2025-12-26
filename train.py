import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# 0. Ensure folders exist
# =====================================================
os.makedirs("models", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

# =====================================================
# 1. MLflow configuration
# =====================================================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CustomerGuard-Churn")

# =====================================================
# 2. Load data
# =====================================================
data = pd.read_excel("data/churn_data.xlsx")
print("✅ Data loaded:", data.shape)

# =====================================================
# 3. Drop unnecessary columns
# =====================================================
data = data.drop(
    ["CustomerID", "Count", "Lat Long", "Latitude", "Longitude", "Zip Code"],
    axis=1, errors="ignore"
)

# =====================================================
# 4. Encode binary columns
# =====================================================
binary_cols = [
    "Gender", "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Multiple Lines", "Online Security",
    "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies", "Paperless Billing"
]

for col in binary_cols:
    if col in data.columns:
        data[col] = data[col].map({
            "Yes": 1, "No": 0,
            "Male": 1, "Female": 0
        })

# =====================================================
# 5. One-hot encode all remaining object (string) columns
# =====================================================
categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
# Exclude target columns
categorical_cols = [c for c in categorical_cols if c not in ["Churn Label", "Churn Reason"]]

print("🔎 One-hot encoding columns:", categorical_cols)
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# =====================================================
# 6. Handle missing values
# =====================================================
# Option 1: Fill missing numeric values with 0
data = data.fillna(0)

# Option 2 (alternative): drop rows with NaN
# data = data.dropna()

# =====================================================
# 7. Split features & target
# =====================================================
y = data["Churn Value"]

X = data.drop(
    ["Churn Value", "Churn Label", "Churn Score", "CLTV", "Churn Reason"],
    axis=1, errors="ignore"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 8. Define models
# =====================================================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42)
}

# =====================================================
# 9. Train & log with MLflow
# =====================================================
best_model = None
best_accuracy = 0.0

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print("\n==============================")
        print("Model:", name)
        print("Accuracy:", acc)
        print(classification_report(y_test, preds))

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

# =====================================================
# 10. Save best model
# =====================================================
joblib.dump(best_model, "models/churn_model.pkl")

print("\n✅ Training finished successfully")
print("🏆 Best accuracy:", best_accuracy)
print("💾 Model saved to models/churn_model.pkl")
