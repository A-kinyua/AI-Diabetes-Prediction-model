import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Diabetes Prediction Dashboard", page_icon="🩺", layout="wide")

@st.cache_resource
def load_assets():
    models = {
        "Logistic Regression": joblib.load("logistic_regression.pkl"),
        "K-Nearest Neighbors": joblib.load("k-nearest_neighbors.pkl"),
        "Support Vector Machine": joblib.load("support_vector_machine.pkl")
    }
    background_data = joblib.load("background_data.pkl")
    return models, background_data

models, background_data = load_assets()

st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home", 
    "🔮 Patient Prediction", 
    "📊 Model Performance", 
    " Explainable AI (XAI)"
])

def get_recommendation(prob):
    if prob <= 0.30:
        return "🟢 **Low Risk (0-30%)**: Maintain a healthy lifestyle with regular exercise and a balanced diet."
    elif prob <= 0.50:
        return "🟡 **Moderate Risk (31-50%)**: Consider regular health check-ups and monitor blood sugar levels."
    elif prob <= 0.70:
        return "🟠 **High Risk (51-70%)**: Consult a healthcare provider soon for further evaluation."
    else:
        return "🔴 **Very High Risk (71-100%)**: Immediate medical consultation is strongly advised."

if page == "🏠 Home":
    st.title("🩺 Diabetes Prediction Dashboard")
    st.markdown("Welcome to the **Diabetes Prediction System**. This dashboard leverages machine learning to assess the likelihood of diabetes based on patient health metrics.")
    st.markdown("### 🎯 Project Objectives")
    st.markdown("- Early identification of diabetic patients to enable timely treatment.")
    st.markdown("- Reduce severe complications (kidney disease, heart problems, nerve damage, vision loss).")
    st.markdown("- Provide explainable AI (XAI) insights so healthcare providers understand *why* a prediction was made.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Features", "7")
    col2.metric("Models Available", "3")
    col3.metric("Target Variable", "Binary (0/1)")

elif page == "🔮 Patient Prediction":
    st.title("🔮 New Patient Prediction")
    st.markdown("Enter the patient's health metrics below to get a real-time diabetes risk assessment.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0.0, max_value=100.0, value=40.0)
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)
        with col2:
            hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
            glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=100)
            smoking = st.selectbox("Smoking History", ["No Info", "never", "former", "current", "not current", "ever"])
        
        selected_model = st.selectbox("Choose Prediction Model", list(models.keys()))
        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        input_data = pd.DataFrame({
            'age': [age], 'hypertension': [hypertension], 'heart_disease': [heart_disease],
            'bmi': [bmi], 'HbA1c_level': [hba1c], 'blood_glucose_level': [glucose],
            'smoking_history': [smoking]
        })
        
        model = models[selected_model]
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"**Diabetic** (Probability: {probability:.2%})")
            else:
                st.success(f"**Non-Diabetic** (Probability: {probability:.2%})")
        
        with col2:
            st.subheader("Clinical Recommendation")
            st.markdown(get_recommendation(probability))

elif page == " Model Performance":
    st.title("📊 Model Performance Metrics")
    
    model_metrics = {
        "Logistic Regression": {"Accuracy": 0.96, "Precision": 0.86, "Recall": 0.62, "F1-Score": 0.72},
        "K-Nearest Neighbors": {"Accuracy": 0.95, "Precision": 0.97, "Recall": 0.47, "F1-Score": 0.63},
        "Support Vector Machine": {"Accuracy": 0.94, "Precision": 1.00, "Recall": 0.37, "F1-Score": 0.54}
    }
    
    selected_model = st.selectbox("Select Model to View Details", list(models.keys()))
    metrics = model_metrics[selected_model]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
    col2.metric("Precision", f"{metrics['Precision']:.2f}")
    col3.metric("Recall", f"{metrics['Recall']:.2f}")
    col4.metric("F1-Score", f"{metrics['F1-Score']:.2f}")
    
    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    if selected_model == "Logistic Regression":
        cm = np.array([[16850, 167], [642, 1064]])
    elif selected_model == "K-Nearest Neighbors":
        cm = np.array([[16995, 22], [905, 801]])
    else:
        cm = np.array([[17017, 0], [1070, 636]])
        
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {selected_model}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

elif page == "🧠 Explainable AI (XAI)":
    st.title("🧠 Explainable AI (SHAP)")
    st.markdown("SHAP explains the output of the machine learning model for a specific patient prediction.")
    
    with st.expander("Enter Patient Data for Local Explanation", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            x_age = st.number_input("Age", value=65.0, key="xai_age")
            x_htn = st.selectbox("Hypertension", [0, 1], key="xai_htn")
        with col2:
            x_glucose = st.number_input("Blood Glucose", value=180, key="xai_glucose")
            x_hba1c = st.number_input("HbA1c Level", value=7.5, key="xai_hba1c")
            
        explain_btn = st.button("Generate SHAP Explanation")
        
    if explain_btn:
        with st.spinner("Calculating SHAP values..."):
            input_df = pd.DataFrame({
                'age': [x_age], 'hypertension': [x_htn], 'heart_disease': [0],
                'bmi': [30.0], 'HbA1c_level': [x_hba1c], 'blood_glucose_level': [x_glucose],
                'smoking_history': ['never']
            })
            
            model = models["Logistic Regression"]
            pred_prob = model.predict_proba(input_df)[0][1]
            
            st.markdown(f"### Prediction Probability: **{pred_prob:.2%}**")
            st.markdown(get_recommendation(pred_prob))
            
            explainer = shap.Explainer(model, background_data)
            shap_values = explainer(input_df)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.title(f"SHAP Explanation: Prediction Probability = {pred_prob:.2%}")
            st.pyplot(fig)
            
            st.markdown("""
            ### How to read this plot:
            - **Base value**: The average model prediction.
            - **Red bars**: Features that push the prediction higher (towards Diabetes).
            - **Blue bars**: Features that push the prediction lower (towards Non-Diabetes).
            """)