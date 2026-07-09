# Smart Lender - AI Powered Loan Prediction System
## Project Overview
The Smart Lender project is an Artificial Intelligence and Machine Learning based web application that predicts whether a loan application is likely to be approved or rejected based on applicant information.
The system helps financial institutions automate the loan approval process by analyzing applicant details such as income, education, employment status, credit history, loan amount, and property area. It reduces manual effort and improves decision-making using machine learning algorithms.
---
# Objectives
- Automate loan approval prediction.
- Reduce manual verification time.
- Improve decision-making using Machine Learning.
- Compare multiple classification algorithms.
- Deploy the trained model using Flask.
---
# Technologies Used
- Python
- Visual Studio Code
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- Flask
- Joblib
- HTML
- CSS
---
# Machine Learning Algorithms
The following algorithms are implemented and compared:
1. Decision Tree Classifier
2. Random Forest Classifier
3. K-Nearest Neighbors (KNN)
4. XGBoost Classifier
The best-performing model is selected based on evaluation metrics and saved for deployment.
---
# Dataset
Dataset Name:
Loan Prediction Dataset
Features:
- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status
Target Variable:
Loan_Status
- Y → Loan Approved
- N → Loan Rejected
---
# Project Structure
```
SMART_LENDER_PROJET
│
├── dataset/
│      loan_prediction.csv
├── model/
├── src/
│      visualization.py
│      data_preprocessing.py
│      train_models.py
│      prediction.py
│      utils.py
├── static/
│      style.css
├── templates/
│      home.html
│      predict.html
│      submit.html
├── app.py
├── requirements.txt
└── README.md
```
---
# Machine Learning Workflow
1. Import Dataset
2. Data Visualization
- Univariate Analysis
- Bivariate Analysis
- Multivariate Analysis
3. Data Preprocessing
- Missing Value Handling
- Label Encoding
- SMOTE Balancing
- Feature Scaling
4. Train-Test Split
5. Model Training
- Decision Tree
- Random Forest
- KNN
- XGBoost
6. Model Evaluation
- Accuracy
- Confusion Matrix
- Classification Report
- Cross Validation
7. Save Best Model
8. Flask Deployment
9. Loan Prediction
---
# Installation
Install the required packages using:
```bash
pip install -r requirements.txt
```
---
# Run the Project
Step 1
Train the models:
```bash
python src/train_models.py
```
Step 2
Start the Flask server:
```bash
python app.py
```
Step 3
Open your browser and visit:
```
http://127.0.0.1:5000
```
---
# Expected Output
The user enters applicant information through a web interface.
↓
The trained machine learning model predicts whether the loan should be approved.
↓
The prediction result is displayed on the screen.
---
# Future Enhancements
- Real-time Credit Score API Integration
- Deep Learning Models
- Explainable AI (XAI)
- Cloud Deployment
- Mobile Application
- Database Integration
---
# Author
Smart Lender - AI Powered Loan Prediction System
Developed using Python, Machine Learning, Flask, HTML and CSS.