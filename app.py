import os
import joblib
import pandas as pd
import gradio as gr


# ==========================================
# Load trained model and scaler
# ==========================================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# ==========================================
# Prediction function
# ==========================================

def predict_heart_disease(
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
):

    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # Scale using the SAME scaler used during training
    patient_scaled = scaler.transform(patient)

    # Probability of disease
    probability = model.predict_proba(patient_scaled)[0][1]

    risk_percentage = probability * 100

    # Project-defined risk categories
    if risk_percentage < 30:
        category = "Low"
    elif risk_percentage < 60:
        category = "Moderate"
    else:
        category = "High"

    result = f"""
## ❤️ Heart Disease Risk Result

### Estimated Risk: **{risk_percentage:.2f}%**

### Risk Category: **{category}**

This is an analytical prediction produced by the machine-learning
model trained on the UCI Cleveland Heart Disease dataset.

⚠️ **Disclaimer:** This result is for academic/project purposes only
and is not a medical diagnosis.
"""

    return result


# ==========================================
# Gradio Interface
# ==========================================

with gr.Blocks(title="Heart Disease Risk Predictor") as app:

    gr.Markdown(
        """
        # ❤️ Heart Disease Risk Predictor

        ### Probability and Statistics Project

        Enter the patient's clinical information below to estimate
        the probability of heart disease.
        """
    )

    with gr.Row():

        with gr.Column():

            age = gr.Number(
                label="Age",
                value=55,
                minimum=1,
                maximum=120
            )

            sex = gr.Dropdown(
                choices=[
                    ("Female", 0),
                    ("Male", 1)
                ],
                value=1,
                label="Sex"
            )

            cp = gr.Dropdown(
                choices=[
                    ("Typical Angina", 1),
                    ("Atypical Angina", 2),
                    ("Non-anginal Pain", 3),
                    ("Asymptomatic", 4)
                ],
                value=1,
                label="Chest Pain Type"
            )

            trestbps = gr.Number(
                label="Resting Blood Pressure (mm Hg)",
                value=140,
                minimum=50,
                maximum=250
            )

            chol = gr.Number(
                label="Serum Cholesterol (mg/dl)",
                value=250,
                minimum=50,
                maximum=700
            )

            fbs = gr.Dropdown(
                choices=[
                    ("No (≤ 120 mg/dl)", 0),
                    ("Yes (> 120 mg/dl)", 1)
                ],
                value=0,
                label="Fasting Blood Sugar"
            )

            restecg = gr.Dropdown(
                choices=[
                    ("Normal", 0),
                    ("ST-T Wave Abnormality", 1),
                    ("Left Ventricular Hypertrophy", 2)
                ],
                value=1,
                label="Resting ECG"
            )

        with gr.Column():

            thalach = gr.Number(
                label="Maximum Heart Rate",
                value=150,
                minimum=50,
                maximum=250
            )

            exang = gr.Dropdown(
                choices=[
                    ("No", 0),
                    ("Yes", 1)
                ],
                value=0,
                label="Exercise-Induced Angina"
            )

            oldpeak = gr.Number(
                label="ST Depression (Oldpeak)",
                value=1.0,
                minimum=0,
                maximum=10,
                step=0.1
            )

            slope = gr.Dropdown(
                choices=[
                    ("Upsloping", 1),
                    ("Flat", 2),
                    ("Downsloping", 3)
                ],
                value=1,
                label="Slope of Peak Exercise ST Segment"
            )

            ca = gr.Dropdown(
                choices=[
                    ("0 vessels", 0),
                    ("1 vessel", 1),
                    ("2 vessels", 2),
                    ("3 vessels", 3)
                ],
                value=0,
                label="Number of Major Vessels"
            )

            thal = gr.Dropdown(
                choices=[
                    ("Normal", 3),
                    ("Fixed Defect", 6),
                    ("Reversible Defect", 7)
                ],
                value=3,
                label="Thalassemia"
            )

    predict_button = gr.Button(
        "🔍 Predict Heart Disease Risk",
        variant="primary"
    )

    output = gr.Markdown()

    predict_button.click(
        fn=predict_heart_disease,
        inputs=[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ],
        outputs=output
    )

    gr.Markdown(
        """
        ---
        **Model:** Logistic Regression  
        **Dataset:** UCI Heart Disease – Cleveland  
        **Test Accuracy:** 86.89%  
        **ROC-AUC:** 95.13%

        *Risk categories are project-defined and are not clinical thresholds.*
        """
    )


# ==========================================
# Render deployment
# ==========================================

port = int(os.environ.get("PORT", 7860))

app.launch(
    server_name="0.0.0.0",
    server_port=port
)
