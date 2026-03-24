from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib


# -----------------------------
# 1. Load model artifacts
# -----------------------------
model = joblib.load("models/random_forest_model.joblib")
target_encoder = joblib.load("models/target_encoder.joblib")
feature_encoders = joblib.load("models/feature_encoders.joblib")


# -----------------------------
# 2. Create FastAPI app
# -----------------------------
app = FastAPI(title="Loan Approval Prediction API")


# -----------------------------
# 3. Define request schema
# -----------------------------
class LoanApplication(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int


# -----------------------------
# 4. Health check endpoint
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# -----------------------------
# 5. Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_loan(application: LoanApplication):
    input_df = pd.DataFrame([application.dict()])

    for col, encoder in feature_encoders.items():
        input_df[col] = input_df[col].str.strip()

        valid_values = set(encoder.classes_)
        input_value = input_df[col].iloc[0]

        if input_value not in valid_values:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value '{input_value}' for column '{col}'. Expected one of: {list(valid_values)}"
            )

        input_df[col] = encoder.transform(input_df[col])

    prediction_encoded = model.predict(input_df)[0]
    prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]

    prediction_proba = model.predict_proba(input_df)[0]
    class_labels = target_encoder.inverse_transform([0, 1])

    probabilities = {
        label: round(float(prob) * 100, 2)
        for label, prob in zip(class_labels, prediction_proba)
    }

    return {
        "prediction": prediction_label,
        "probabilities": probabilities
    }
