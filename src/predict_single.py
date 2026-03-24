from src.model_utils import predict_loan_application


strong_applicant = {
    "no_of_dependents": 1,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 15000000,
    "loan_amount": 5000000,
    "loan_term": 8,
    "cibil_score": 820,
    "residential_assets_value": 10000000,
    "commercial_assets_value": 7000000,
    "luxury_assets_value": 12000000,
    "bank_asset_value": 9000000
}

weak_applicant = {
    "no_of_dependents": 5,
    "education": "Not Graduate",
    "self_employed": "Yes",
    "income_annum": 2000000,
    "loan_amount": 35000000,
    "loan_term": 20,
    "cibil_score": 350,
    "residential_assets_value": 500000,
    "commercial_assets_value": 0,
    "luxury_assets_value": 1000000,
    "bank_asset_value": 300000
}

borderline_applicant = {
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "Yes",
    "income_annum": 5500000,
    "loan_amount": 15000000,
    "loan_term": 16,
    "cibil_score": 650,
    "residential_assets_value": 3000000,
    "commercial_assets_value": 2000000,
    "luxury_assets_value": 2500000,
    "bank_asset_value": 1500000
}


for name, applicant in [
    ("Strong Applicant", strong_applicant),
    ("Weak Applicant", weak_applicant),
    ("Borderline Applicant", borderline_applicant)
]:
    result = predict_loan_application(applicant)

    print("\n" + "=" * 40)
    print(name)
    print("=" * 40)

    print("Applicant data:")
    for key, value in applicant.items():
        print(f"{key}: {value}")

    print("\nPrediction:", result["prediction"])
    print("Class probabilities:")
    for label, prob in result["probabilities"].items():
        print(f"{label}: {round(prob * 100, 2)}%")
