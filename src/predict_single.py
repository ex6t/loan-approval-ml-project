import pandas as pd
import joblib


# -----------------------------
# 1. Load saved model + encoders
# -----------------------------
model = joblib.load("models/random_forest_model.joblib")
target_encoder = joblib.load("models/target_encoder.joblib")
feature_encoders = joblib.load("models/feature_encoders.joblib")


# -----------------------------
# 2. Prediction function
# -----------------------------
def predict_loan_application(applicant_data):
    input_df = pd.DataFrame([applicant_data])

    for col, encoder in feature_encoders.items():
        input_df[col] = input_df[col].str.strip()

        valid_values = set(encoder.classes_)
        input_value = input_df[col].iloc[0]

        if input_value not in valid_values:
            raise ValueError(
                f"Invalid value '{input_value}' for column '{col}'. "
                f"Expected one of: {list(valid_values)}"
            )

        input_df[col] = encoder.transform(input_df[col])

    prediction_encoded = model.predict(input_df)[0]
    prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]

    prediction_proba = model.predict_proba(input_df)[0]
    class_labels = target_encoder.inverse_transform([0, 1])

    return {
        "prediction": prediction_label,
        "probabilities": {
            label: float(prob) for label, prob in zip(class_labels, prediction_proba)
        }
    }


# -----------------------------
# 3. Test applicants
# -----------------------------
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


# -----------------------------
# 4. Run predictions
# -----------------------------
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
