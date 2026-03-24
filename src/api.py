from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.model_utils import predict_loan_application


app = FastAPI(title="Loan Approval Prediction API")


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


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_loan(application: LoanApplication):
    try:
        result = predict_loan_application(application.dict())
        result["probabilities"] = {
            label: round(prob * 100, 2)
            for label, prob in result["probabilities"].items()
        }
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
