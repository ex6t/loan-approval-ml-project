import pandas as pd
import joblib


# -----------------------------
# 1. Load saved model + encoders
# -----------------------------
model = joblib.load("models/random_forest_model.joblib")
target_encoder = joblib.load("models/target_encoder.joblib")
feature_encoders = joblib.load("models/feature_encoders.joblib")


# -----------------------------
# 2. Create one sample applicant
# -----------------------------
sample_applicant = {
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


# -----------------------------
# 3. Convert to DataFrame
# -----------------------------
input_df = pd.DataFrame([sample_applicant])


# -----------------------------
# 4. Encode categorical columns
# -----------------------------
for col, encoder in feature_encoders.items():
    input_df[col] = input_df[col].str.strip()
    input_df[col] = encoder.transform(input_df[col])


# -----------------------------
# 5. Make prediction
# -----------------------------
prediction_encoded = model.predict(input_df)[0]
prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]

prediction_proba = model.predict_proba(input_df)[0]
class_labels = target_encoder.inverse_transform([0, 1])


# -----------------------------
# 6. Print result
# -----------------------------
print("Loan application prediction result")
print("----------------------------------")

print("Applicant data:")
for key, value in sample_applicant.items():
    print(f"{key}: {value}")

print("\nPrediction:", prediction_label)

print("\nClass probabilities:")
for label, prob in zip(class_labels, prediction_proba):
    print(f"{label}: {round(prob * 100, 2)}%")
