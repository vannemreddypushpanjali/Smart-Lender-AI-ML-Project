# Smart Lender - AI Powered Loan Prediction System
import joblib
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")
# Load Processed Dataset
print("SMART LENDER - MODEL TRAINING")
df = pd.read_csv("dataset/processed_dataset.csv")
print("\nProcessed Dataset Loaded Successfully.")
print("\nDataset Shape :", df.shape)
print("\nFirst Five Records")
print(df.head())
# Separate Features and Target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]
print("\nFeatures Shape :", X.shape)
print("Target Shape   :", y.shape)
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\nTrain-Test Split Completed Successfully.")
print("\nTraining Samples")
print(X_train.shape)
print("\nTesting Samples")
print(X_test.shape)
print("\nModel Training Begins...\n")
# Dictionary to store model accuracies
model_scores = {}
# Decision Tree Classifier
print("DECISION TREE CLASSIFIER")
# Create Decision Tree Model
dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)
# Train Model
dt_model.fit(X_train, y_train)
# Prediction
dt_pred = dt_model.predict(X_test)
# Accuracy
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"\nDecision Tree Accuracy : {dt_accuracy:.4f}")
# Cross Validation
dt_cv = cross_val_score(
    dt_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)
print("\n5-Fold Cross Validation Scores")
print(dt_cv)
print(f"\nAverage CV Accuracy : {dt_cv.mean():.4f}")
# Confusion Matrix
print("\nConfusion Matrix")
cm = confusion_matrix(y_test, dt_pred)
print(cm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)
disp.plot(cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.show()
# Classification Report
print("\nClassification Report")
print(classification_report(y_test, dt_pred))
# Decision Tree Visualization
plt.figure(figsize=(18,10))
plot_tree(
    dt_model,
    filled=True,
    rounded=True,
    fontsize=9,
    feature_names=X.columns,
    class_names=["Rejected", "Approved"]
)
plt.title("Decision Tree")
plt.show()
print("\nDecision Tree Model Completed Successfully.\n")
# Random Forest Classifier
print("RANDOM FOREST CLASSIFIER")
# Create Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=200,
    criterion="gini",
    max_depth=10,
    random_state=42
)
# Train Model
rf_model.fit(X_train, y_train)
# Prediction
rf_pred = rf_model.predict(X_test)
# Accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"\nRandom Forest Accuracy : {rf_accuracy:.4f}")
# Cross Validation
rf_cv = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)
print("\n5-Fold Cross Validation Scores")
print(rf_cv)
print(f"\nAverage CV Accuracy : {rf_cv.mean():.4f}")
# Confusion Matrix
print("\nConfusion Matrix")
rf_cm = confusion_matrix(y_test, rf_pred)
print(rf_cm)
rf_disp = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm
)
rf_disp.plot(cmap="Greens")
plt.title("Random Forest Confusion Matrix")
plt.show()
# Classification Report
print("\nClassification Report")
print(classification_report(y_test, rf_pred))
# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})
importance = importance.sort_values(
    by="Importance",
    ascending=False
)
print("\nFeature Importance")
print(importance)
plt.figure(figsize=(10,6))
sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
print("\nRandom Forest Model Completed Successfully.\n")
# K-Nearest Neighbors (KNN) Classifier
print("K-NEAREST NEIGHBORS (KNN) CLASSIFIER")
# Create KNN Model
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    metric="minkowski",
    p=2
)
# Train Model
knn_model.fit(X_train, y_train)
# Prediction
knn_pred = knn_model.predict(X_test)
# Accuracy
knn_accuracy = accuracy_score(y_test, knn_pred)
print(f"\nKNN Accuracy : {knn_accuracy:.4f}")
# Cross Validation
knn_cv = cross_val_score(
    knn_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)
print("\n5-Fold Cross Validation Scores")
print(knn_cv)
print(f"\nAverage CV Accuracy : {knn_cv.mean():.4f}")
# Confusion Matrix
print("\nConfusion Matrix")
knn_cm = confusion_matrix(y_test, knn_pred)
print(knn_cm)
knn_disp = ConfusionMatrixDisplay(
    confusion_matrix=knn_cm
)
knn_disp.plot(cmap="Oranges")
plt.title("KNN Confusion Matrix")
plt.show()
# Classification Report
print("\nClassification Report")
print(classification_report(y_test, knn_pred))
print("\nKNN Model Completed Successfully.\n")
# Store Model Accuracy (Used Later for Comparison)
model_scores = {}
model_scores["Decision Tree"] = dt_accuracy
model_scores["Random Forest"] = rf_accuracy
model_scores["KNN"] = knn_accuracy
# XGBoost Classifier
print("XGBOOST CLASSIFIER")
# Create XGBoost Model
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)
# Train Model
xgb_model.fit(X_train, y_train)
# Prediction
xgb_pred = xgb_model.predict(X_test)
# Accuracy
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"\nXGBoost Accuracy : {xgb_accuracy:.4f}")
# Cross Validation
xgb_cv = cross_val_score(
    xgb_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)
print("\n5-Fold Cross Validation Scores")
print(xgb_cv)
print(f"\nAverage CV Accuracy : {xgb_cv.mean():.4f}")
# Confusion Matrix
print("\nConfusion Matrix")
xgb_cm = confusion_matrix(y_test, xgb_pred)
print(xgb_cm)
xgb_disp = ConfusionMatrixDisplay(
    confusion_matrix=xgb_cm
)
xgb_disp.plot(cmap="Purples")
plt.title("XGBoost Confusion Matrix")
plt.show()
# Classification Report
print("\nClassification Report")
print(classification_report(y_test, xgb_pred))
print("\nXGBoost Model Completed Successfully.\n")
# Store Accuracy
model_scores["XGBoost"] = xgb_accuracy
# Hyperparameter Tuning using GridSearchCV
print("GRID SEARCH CV - RANDOM FOREST")
# Parameter Grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
# Grid Search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
# Train Grid Search
grid_search.fit(X_train, y_train)
# Best Model
best_rf = grid_search.best_estimator_
print("\nBest Parameters")
print(grid_search.best_params_)
print("\nBest Cross Validation Accuracy")
print(f"{grid_search.best_score_:.4f}")
# Prediction
best_rf_pred = best_rf.predict(X_test)
# Accuracy
best_rf_accuracy = accuracy_score(y_test, best_rf_pred)
print("\nTuned Random Forest Accuracy")
print(f"{best_rf_accuracy:.4f}")
# Classification Report
print("\nClassification Report")
print(classification_report(y_test, best_rf_pred))
# Confusion Matrix
best_rf_cm = confusion_matrix(y_test, best_rf_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=best_rf_cm
)
disp.plot(cmap="Greens")
plt.title("Tuned Random Forest Confusion Matrix")
plt.show()
# Update Score Dictionary
model_scores["Tuned Random Forest"] = best_rf_accuracy
print("\nGrid Search Completed Successfully.")
# Model Comparison
print("MODEL COMPARISON")
# Display Model Accuracies
accuracy_df = pd.DataFrame({
    "Model": model_scores.keys(),
    "Accuracy": model_scores.values()
})
accuracy_df = accuracy_df.sort_values(
    by="Accuracy",
    ascending=False
)
print("\nModel Accuracy Comparison\n")
print(accuracy_df)
# Accuracy Comparison Graph
plt.figure(figsize=(10,6))
sns.barplot(
    data=accuracy_df,
    x="Accuracy",
    y="Model",
    palette="viridis"
)
plt.title("Model Accuracy Comparison")
plt.xlabel("Accuracy")
plt.ylabel("Machine Learning Models")
plt.xlim(0.70,1.00)
plt.tight_layout()
plt.show()
# Best Model Selection
best_model_name = accuracy_df.iloc[0]["Model"]
print("\nBest Model :", best_model_name)
if best_model_name == "Decision Tree":
    best_model = dt_model
elif best_model_name == "Random Forest":
    best_model = rf_model
elif best_model_name == "KNN":
    best_model = knn_model
elif best_model_name == "XGBoost":
    best_model = xgb_model
elif best_model_name == "Tuned Random Forest":
    best_model = best_rf
# Save Best Model
joblib.dump(
    best_model,
    "model/best_model.pkl"
)
print("\nModels Trained Successfully")