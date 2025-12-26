from prefect import task, flow
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# Tasks
# =========================

@task
def load_data():
    return pd.read_excel("data/churn_data.xlsx")

@task
def preprocess(data):
    # Drop ID / useless columns
    data = data.drop(
        ["CustomerID", "Count", "Lat Long", "Latitude", "Longitude", "Zip Code"],
        axis=1,
        errors="ignore"
    )

    # Encode binary columns
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

    # One-hot encode remaining categorical columns
    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ["Churn Label", "Churn Reason"]]

    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Handle missing values
    data = data.fillna(0)

    # Split target
    y = data["Churn Value"]
    X = data.drop(
        ["Churn Value", "Churn Label", "Churn Score", "CLTV", "Churn Reason"],
        axis=1,
        errors="ignore"
    )

    return X, y


@task
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, acc

@task
def log_experiment(model, acc):
    mlflow.set_experiment("CustomerGuard-Churn")

    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "models/churn_model.pkl")

# =========================
# Flow
# =========================

@flow(name="churn-training-pipeline")
def training_pipeline():
    data = load_data()
    X, y = preprocess(data)
    model, acc = train_model(X, y)
    log_experiment(model, acc)

if __name__ == "__main__":
    training_pipeline()
