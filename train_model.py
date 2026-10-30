"""Reproduce the project's Logistic Regression model and save its artifacts."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "heart_disease_cleaned.csv"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

df = pd.read_csv(DATA_PATH)
# Match the notebook's binary target and missing-value treatment.
if "disease" not in df.columns:
    df["disease"] = (df["target"] != 0).astype(int)
X = df[FEATURES].copy()
y = df["disease"].astype(int)
# The supplied cleaned CSV retains missing ca/thal values; reproduce notebook imputation.
X["ca"] = X["ca"].fillna(X["ca"].median())
X["thal"] = X["thal"].fillna(X["thal"].mode()[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)
prob = model.predict_proba(X_test_scaled)[:, 1]
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, prob):.4f}")
joblib.dump(model, BASE_DIR / "model.pkl")
joblib.dump(scaler, BASE_DIR / "scaler.pkl")
print("Saved model.pkl and scaler.pkl")
