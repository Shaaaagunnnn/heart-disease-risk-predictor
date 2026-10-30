import joblib
import gradio as gr
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    raise FileNotFoundError("model.pkl and scaler.pkl must be present. Run train_model.py first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

def risk_predictor(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    patient = pd.DataFrame([values], columns=FEATURES)
    probability = float(model.predict_proba(scaler.transform(patient))[0][1])
    risk = probability * 100
    category = "LOW RISK" if risk < 30 else "MODERATE RISK" if risk < 60 else "HIGH RISK"
    return f"{risk:.2f}%", category

with gr.Blocks(title="Heart Disease Risk Predictor") as app:
    gr.Markdown("""# ❤️ Heart Disease Risk Predictor
### Probability & Statistics Project
This application uses Logistic Regression trained on the UCI Cleveland Heart Disease dataset to estimate a model probability.

**Model Accuracy:** 86.89% &nbsp;&nbsp; **ROC-AUC:** 95.13%

## 👤 Patient Information""")
    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Age", value=55, minimum=1, maximum=100)
            sex = gr.Dropdown([( "Female", 0), ("Male", 1)], value=1, type="value", label="Sex")
            cp = gr.Dropdown([("Typical Angina", 1), ("Atypical Angina", 2), ("Non-anginal Pain", 3), ("Asymptomatic", 4)], value=1, type="value", label="Chest Pain Type")
            trestbps = gr.Number(label="Resting Blood Pressure (mm Hg)", value=140, minimum=50, maximum=250)
            chol = gr.Number(label="Serum Cholesterol (mg/dl)", value=250, minimum=50, maximum=700)
            fbs = gr.Dropdown([("No (≤ 120 mg/dl)", 0), ("Yes (> 120 mg/dl)", 1)], value=0, type="value", label="Fasting Blood Sugar")
            restecg = gr.Dropdown([("Normal", 0), ("ST-T Wave Abnormality", 1), ("Left Ventricular Hypertrophy", 2)], value=1, type="value", label="Resting ECG")
        with gr.Column():
            thalach = gr.Number(label="Maximum Heart Rate Achieved", value=150, minimum=50, maximum=250)
            exang = gr.Dropdown([("No", 0), ("Yes", 1)], value=0, type="value", label="Exercise-Induced Angina")
            oldpeak = gr.Number(label="ST Depression (Oldpeak)", value=1.0, minimum=0, maximum=10)
            slope = gr.Dropdown([("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)], value=2, type="value", label="ST Segment Slope")
            ca = gr.Dropdown([("0 vessels", 0), ("1 vessel", 1), ("2 vessels", 2), ("3 vessels", 3)], value=0, type="value", label="Number of Major Vessels")
            thal = gr.Dropdown([("Normal", 3), ("Fixed Defect", 6), ("Reversible Defect", 7)], value=3, type="value", label="Thalassemia Result")
    predict = gr.Button("🔍 Predict Heart Disease Risk", variant="primary")
    gr.Markdown("## 📊 Prediction Result")
    with gr.Row():
        risk_out = gr.Textbox(label="Estimated Heart Disease Risk", interactive=False)
        category_out = gr.Textbox(label="Risk Category", interactive=False)
    predict.click(risk_predictor, [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal], [risk_out, category_out])
    gr.Markdown("---\n### ⚠️ Disclaimer\nThis is an academic demonstration using a statistical machine-learning model. It is not a medical diagnosis, a clinically validated risk score, or a substitute for professional medical advice.")

if __name__ == "__main__":
    app.launch()
