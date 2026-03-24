import pandas as pd
import joblib


# -----------------------------
# 1. Load model artifacts once
# -----------------------------
model = joblib.load("models/random_forest_model.joblib")
target_encoder = joblib.load("models/target_encoder.joblib")
feature_encoders = joblib.load("models/feature_encoders.joblib")


# -----------------------------
# 2. Validate + encode input
# -----------------------------
def prepare_input(applicant_data):
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

    return input_df


# -----------------------------
# 3. Predict one applicant
# -----------------------------
def predict_loan_application(applicant_data):
    input_df = prepare_input(applicant_data)

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
