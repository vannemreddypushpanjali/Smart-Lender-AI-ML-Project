# Smart Lender - AI Powered Loan Prediction System
from flask import Flask, render_template, request
from src.prediction import predict_loan
from src.utils import convert_input, validate_input, prediction_message
app = Flask(__name__)
# Home Page
@app.route("/")
def home():
    return render_template("home.html")
# Prediction Form Page
@app.route("/predict")
def predict():
    return render_template("predict.html")
# Predict Loan
@app.route("/submit", methods=["POST"])
def submit():
    try:
        form_data = request.form
        input_data = convert_input(form_data)
        status, message = validate_input(input_data)
        if not status:
            return render_template(
                "submit.html",
                prediction=message
            )
        result = predict_loan(input_data)
        final_message = prediction_message(result)
        return render_template(
            "submit.html",
            prediction=result,
            message=final_message
        )
    except Exception as e:
        return render_template(
            "submit.html",
            prediction="Error",
            message=str(e)
        )
# Run Flask App
if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )