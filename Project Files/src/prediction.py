# Description: Load trained model and predict loan status
import joblib
import pandas as pd
# Load Saved Model and Scaler
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")
# Feature Names (Must match training dataset)
FEATURE_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area"
]
# Prediction Function
def predict_loan(input_data):
    """
    Predict loan approval.
    Parameters
    ----------
    input_data : list
        List containing the 11 input values.
    Returns
    -------
    str
        Loan Approved ✅ or Loan Rejected ❌
    """
    # Create DataFrame with feature names
    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURE_COLUMNS
    )
    # Scale input
    input_scaled = scaler.transform(input_df)
    # Convert back to DataFrame to preserve column names
    input_scaled_df = pd.DataFrame(
        input_scaled,
        columns=FEATURE_COLUMNS
    )
    # Predict
    prediction = model.predict(input_scaled_df)
    if prediction[0] == 1:
        return "Loan Approved ✅"
    return "Loan Rejected ❌"
# Test Prediction
if __name__ == "__main__":
    print("=" * 60)
    print("SMART LENDER PREDICTION SYSTEM")
    print("=" * 60)
    # Sample Applicant
    sample_data = [
        1,      # Gender
        1,      # Married
        0,      # Dependents
        1,      # Education
        0,      # Self Employed
        5000,   # Applicant Income
        2000,   # Coapplicant Income
        150,    # Loan Amount
        360,    # Loan Amount Term
        1,      # Credit History
        2       # Property Area
    ]
    result = predict_loan(sample_data)
    print("\nSample Applicant Details")
    for feature, value in zip(FEATURE_COLUMNS, sample_data):
        print(f"{feature:<20}: {value}")
    print("\nPrediction Result")
    print(result)
    print("\nPrediction Completed Successfully.")