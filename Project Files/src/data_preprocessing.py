import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
# Create Required Folders
os.makedirs("model", exist_ok=True)
os.makedirs("dataset", exist_ok=True)
# Load Dataset
print("=" * 60)
print("SMART LENDER - DATA PREPROCESSING")
print("=" * 60)
df = pd.read_csv("dataset/loan_prediction.csv")
print("\nDataset Loaded Successfully")
print("Dataset Shape :", df.shape)
# Remove Loan_ID
if "Loan_ID" in df.columns:
    df.drop(columns=["Loan_ID"], inplace=True)
# Missing Values
print("\nHandling Missing Values...")
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(df[column].mean())
print("Missing Values Removed")
# Remove Duplicates
df.drop_duplicates(inplace=True)
print("Duplicate Records Removed")
# Label Encoding
print("\nEncoding Categorical Features...")
encoders = {}
categorical_columns = df.select_dtypes(include="object").columns
for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder
joblib.dump(encoders, "model/label_encoder.pkl")
print("Label Encoder Saved")
# Separate Features and Target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]
# Apply SMOTE
print("\nBalancing Dataset using SMOTE...")
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
print("Balanced Shape :", X.shape)
# Feature Scaling
print("\nScaling Features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "model/scaler.pkl")
print("Scaler Saved")
# Create Processed Dataset
# Preserve original feature names
feature_names = df.drop("Loan_Status", axis=1).columns
processed_df = pd.DataFrame(
    X,
    columns=feature_names
)
processed_df["Loan_Status"] = y

processed_df.to_csv(
    "dataset/processed_dataset.csv",
    index=False
)
print("\nDATA PREPROCESSING COMPLETED SUCCESSFULLY")