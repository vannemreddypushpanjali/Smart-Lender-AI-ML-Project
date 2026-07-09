def convert_input(form_data):
    """
    Convert HTML form data into numeric values
    required by the machine learning model.
    """
    gender = 1 if form_data["Gender"] == "Male" else 0
    married = 1 if form_data["Married"] == "Yes" else 0
    dependents = int(form_data["Dependents"])
    education = 1 if form_data["Education"] == "Graduate" else 0
    self_employed = 1 if form_data["Self_Employed"] == "Yes" else 0
    applicant_income = float(form_data["ApplicantIncome"])
    coapplicant_income = float(form_data["CoapplicantIncome"])
    loan_amount = float(form_data["LoanAmount"])
    loan_term = float(form_data["Loan_Amount_Term"])
    credit_history = int(form_data["Credit_History"])
    property_area_dict = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }
    property_area = property_area_dict[form_data["Property_Area"]]
    return [
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]
# Validate User Input
def validate_input(data):
    """
    Validate numerical values entered by the user.
    """
    if data[5] < 0:
        return False, "Applicant Income cannot be negative."
    if data[6] < 0:
        return False, "Coapplicant Income cannot be negative."
    if data[7] <= 0:
        return False, "Loan Amount must be greater than zero."
    if data[8] <= 0:
        return False, "Loan Term must be greater than zero."
    return True, "Valid Input"
# Convert Prediction to Message
def prediction_message(result):
    """
    Return a user-friendly prediction message.
    """
    if "Approved" in result:
        return "🎉 Congratulations! Your loan is likely to be Approved."
    return "❌ Sorry! Your loan is likely to be Rejected."
# Test Utility Functions
if __name__ == "__main__":
    sample = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": "5000",
        "CoapplicantIncome": "2000",
        "LoanAmount": "150",
        "Loan_Amount_Term": "360",
        "Credit_History": "1",
        "Property_Area": "Urban"
    }
    converted = convert_input(sample)
    print("Converted Input")
    print(converted)
    status, message = validate_input(converted)
    print("\nValidation")
    print(status)
    print(message)