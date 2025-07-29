
"""
main.py - Credit Scoring Model Training and Evaluation
Author: <Your Name>
Date: 2025-07-27
Description: Loads data, preprocesses, trains Logistic Regression and Random Forest models, evaluates, and saves the best model.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
DATA_PATH = "data/credit_score.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Initial exploration
print("\n🔍 First 5 rows:\n", df.head())
print("\n📊 Data info:\n")
print(df.info())
print("\n📉 Missing values:\n", df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Encode target label
le = LabelEncoder()
df['Credit_Score'] = le.fit_transform(df['Credit_Score'])


# Separate features and target
X = df.drop("Credit_Score", axis=1)
y = df["Credit_Score"]
# 👉 One-hot encode categorical columns (like Payment_Behaviour)
X = pd.get_dummies(X)

# ✅ Then scale numeric values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
print("\n🧠 Training Logistic Regression...")
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\n📋 Logistic Regression Report:\n", classification_report(y_test, y_pred_lr))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_lr))

# Random Forest Model
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\n📋 Random Forest Report:\n", classification_report(y_test, y_pred_rf))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_rf))

# Save best model (Random Forest)
joblib.dump(rf, "models/credit_model.pkl")
print("\n✅ Best model saved to models/credit_model.pkl")

def plot_confusion_matrix(y_true, y_pred, title):
    """Plot confusion matrix using seaborn heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot confusion matrix for Random Forest
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest - Confusion Matrix")

# Plot ROC Curve for Random Forest
if hasattr(rf, "predict_proba"):
    fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
    plt.plot(fpr, tpr, label='Random Forest')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
else:
    print("Random Forest model does not support predict_proba.")
