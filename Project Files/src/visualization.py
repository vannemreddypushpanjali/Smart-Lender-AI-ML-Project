# Description: Dataset Loading and Visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Plot Style
plt.style.use("fivethirtyeight")
sns.set_theme(style="whitegrid")
# Load Dataset
df = pd.read_csv("dataset/loan_prediction.csv")
print("SMART LENDER - DATA VISUALIZATION")
# Basic Information
print("\nDataset Shape:")
print(df.shape)
print("\nFirst 5 Records:")
print(df.head())
print("\nDataset Information:")
df.info()
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Records:")
print(df.duplicated().sum())
# Univariate Analysis
print("\nGenerating Univariate Analysis...")
# Numerical Columns
numerical_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount"
]
for column in numerical_columns:
    plt.figure(figsize=(8,4))
    sns.histplot(
        df[column].dropna(),
        bins=30,
        kde=True,
        color="steelblue"
    )
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
# Categorical Columns
categorical_columns = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
    "Loan_Status"
]
for column in categorical_columns:
    plt.figure(figsize=(6,4))
    sns.countplot(
        data=df,
        x=column,
        palette="Set2"
    )
    plt.title(f"Count Plot of {column}")
    plt.tight_layout()
    plt.show()
# Bivariate Analysis
print("\nGenerating Bivariate Analysis...")
plots = [
    ("Gender", "Loan_Status"),
    ("Married", "Loan_Status"),
    ("Education", "Loan_Status"),
    ("Property_Area", "Loan_Status"),
    ("Credit_History", "Loan_Status")
]
for x, hue in plots:
    plt.figure(figsize=(6,4))
    sns.countplot(
        data=df,
        x=x,
        hue=hue,
        palette="viridis"
    )
    plt.title(f"{x} vs {hue}")
    plt.tight_layout()
    plt.show()
# Multivariate Analysis
print("\nGenerating Multivariate Analysis...")
# Correlation Heatmap
numeric_df = df.select_dtypes(include=["number"])
plt.figure(figsize=(10,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
# Pair Plot
sns.pairplot(
    df[
        [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Status"
        ]
    ],
    hue="Loan_Status"
)
plt.show()
# Box Plot (Replaces Swarm Plot)
plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Education",
    y="ApplicantIncome",
    hue="Loan_Status",
    palette="Set3"
)
plt.title("Education vs Applicant Income")
plt.tight_layout()
plt.show()
print("\nVisualization Completed Successfully.")