import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💸", layout="centered")

st.title("💸 Loan Approval Predictor")
st.write("Enter a loan application below and click **Predict**.")

with st.form("loan_form"):
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    income_annum = st.number_input("Annual Income", min_value=0, value=5000000, step=100000)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=10000000, step=100000)
    loan_term = st.number_input("Loan Term", min_value=1, max_value=50, value=12)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
    residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=2000000, step=100000)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=1000000, step=100000)
    luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=1500000, step=100000)
    bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=1000000, step=100000)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "no_of_dependents": int(no_of_dependents),
        "education": education,
        "self_employed": self_employed,
        "income_annum": int(income_annum),
        "loan_amount": int(loan_amount),
        "loan_term": int(loan_term),
        "cibil_score": int(cibil_score),
        "residential_assets_value": int(residential_assets_value),
        "commercial_assets_value": int(commercial_assets_value),
        "luxury_assets_value": int(luxury_assets_value),
        "bank_asset_value": int(bank_asset_value)
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()

            prediction = result["prediction"]
            confidence = result["confidence"]
            decision_type = result["decision_type"]
            model_version = result["model_version"]
            probabilities = result["probabilities"]

            st.subheader("Result")

            if prediction == "Approved":
                st.success(f"Prediction: {prediction}")
            elif prediction == "Rejected":
                st.error(f"Prediction: {prediction}")
            else:
                st.info(f"Prediction: {prediction}")

            st.write(f"**Confidence:** {confidence}%")
            st.write(f"**Decision Type:** {decision_type}")
            st.write(f"**Model Version:** {model_version}")

            st.subheader("Class Probabilities")
            for label, prob in probabilities.items():
                st.write(f"**{label}:** {prob}%")

            if decision_type == "manual_review":
                st.warning("This application is borderline and should be reviewed manually.")

            with st.expander("Show submitted application data"):
                st.json(payload)

        else:
            st.error(f"API Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the FastAPI backend. Make sure the API is running on http://127.0.0.1:8000")
    except requests.exceptions.Timeout:
        st.error("The request timed out.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
