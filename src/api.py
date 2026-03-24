from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.model_utils import predict_loan_application


app = FastAPI(title="Loan Approval Prediction API")


class LoanApplication(BaseModel):
    no_of_dependents: int = Field(..., ge=0, le=20)
    education: str
    self_employed: str
    income_annum: int = Field(..., ge=0)
    loan_amount: int = Field(..., ge=0)
    loan_term: int = Field(..., gt=0, le=50)
    cibil_score: int = Field(..., ge=300, le=900)
    residential_assets_value: int = Field(..., ge=0)
    commercial_assets_value: int = Field(..., ge=0)
    luxury_assets_value: int = Field(..., ge=0)
    bank_asset_value: int = Field(..., ge=0)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_loan(application: LoanApplication):
    try:
        result = predict_loan_application(application.dict())

        rounded_probabilities = {
            label: round(prob * 100, 2)
            for label, prob in result["probabilities"].items()
        }

        predicted_label = result["prediction"]
        confidence = rounded_probabilities[predicted_label]

        if confidence < 70:
            decision_type = "manual_review"
        else:
            decision_type = "automatic"

        return {
            "prediction": predicted_label,
            "confidence": confidence,
            "decision_type": decision_type,
            "model_version": "random_forest_v1",
            "probabilities": rounded_probabilities
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
