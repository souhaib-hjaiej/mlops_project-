from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="CustomerGuard Churn API")

model = joblib.load("models/churn_model.pkl")
features = joblib.load("models/features.pkl")  # Load saved features

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # Encode binary columns
    binary_cols = [
        "Gender", "Senior Citizen", "Partner", "Dependents",
        "Phone Service", "Multiple Lines", "Online Security",
        "Online Backup", "Device Protection", "Tech Support",
        "Streaming TV", "Streaming Movies", "Paperless Billing"
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({
                "Yes": 1, "No": 0,
                "Male": 1, "Female": 0
            })

    # Add missing one-hot columns as 0
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct column order
    df = df[features]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0].max()

    return {
        "churn_prediction": int(prediction),
        "confidence": float(probability)
    }
