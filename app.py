import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load model and feature columns
model = joblib.load("breast_cancer_risk_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# App Title
st.title("Early Breast Cancer Risk Prediction App")
st.markdown("This app uses clinical, radiological, and lifestyle data to estimate early breast cancer risk.")

st.markdown("---")

# Section 1: Mean Measurements
st.subheader("📊 Mean Cell Measurements")
mean_radius = st.number_input("Mean Radius (mm)", value=14.0)
mean_texture = st.number_input("Mean Texture", value=20.0)
mean_perimeter = st.number_input("Mean Perimeter (mm)", value=90.0)
mean_area = st.number_input("Mean Area (mm²)", value=600.0)
mean_smoothness = st.number_input("Mean Smoothness", value=0.1)
mean_compactness = st.number_input("Mean Compactness", value=0.1)
mean_concavity = st.number_input("Mean Concavity", value=0.1)
mean_concave_points = st.number_input("Mean Concave Points", value=0.1)
mean_symmetry = st.number_input("Mean Symmetry", value=0.2)
mean_fractal_dimension = st.number_input("Mean Fractal Dimension", value=0.06)

# Section 2: Errors
st.subheader("📈 Error Metrics")
radius_error = st.number_input("Radius Error", value=0.5)
texture_error = st.number_input("Texture Error", value=1.0)
perimeter_error = st.number_input("Perimeter Error", value=3.0)
area_error = st.number_input("Area Error", value=40.0)
smoothness_error = st.number_input("Smoothness Error", value=0.005)
compactness_error = st.number_input("Compactness Error", value=0.02)
concavity_error = st.number_input("Concavity Error", value=0.03)
concave_points_error = st.number_input("Concave Points Error", value=0.02)
symmetry_error = st.number_input("Symmetry Error", value=0.02)
fractal_dimension_error = st.number_input("Fractal Dimension Error", value=0.003)

# Section 3: Worst Measurements
st.subheader("🧪 Worst Cell Measurements")
worst_radius = st.number_input("Worst Radius (mm)", value=17.0)
worst_texture = st.number_input("Worst Texture", value=25.0)
worst_perimeter = st.number_input("Worst Perimeter (mm)", value=110.0)
worst_area = st.number_input("Worst Area (mm²)", value=800.0)
worst_smoothness = st.number_input("Worst Smoothness", value=0.15)
worst_compactness = st.number_input("Worst Compactness", value=0.3)
worst_concavity = st.number_input("Worst Concavity", value=0.4)
worst_concave_points = st.number_input("Worst Concave Points", value=0.2)
worst_symmetry = st.number_input("Worst Symmetry", value=0.3)
worst_fractal_dimension = st.number_input("Worst Fractal Dimension", value=0.09)

# Section 4: History & Lifestyle
st.subheader("🧝 Patient History & Lifestyle")
likely_malignant = st.selectbox("Initial Diagnosis", options=[0, 1], format_func=lambda x: "Benign (0)" if x == 0 else "Malignant (1)")

# Binary inputs
family_history = st.selectbox("Family History", options=["No", "Yes"])
menopause_status = st.selectbox("Menopause Status", options=["Pre", "Post"])
alcohol = st.selectbox("Alcohol Intake Per Week", options=["Light", "Moderate", "Heavy"])
physical_activity = st.selectbox("Physical Activity", options=["Active", "Moderate", "Sedentary"])
nipple_discharge = st.selectbox("Nipple Discharge", options=["No", "Yes"])
palpable_lump = st.selectbox("Palpable Lump", options=["No", "Yes"])
localized_pain = st.selectbox("Localized Breast Pain", options=["No", "Yes"])

# --- Input Dictionary ---
input_dict = {
    "mean radius": mean_radius,
    "mean texture": mean_texture,
    "mean perimeter": mean_perimeter,
    "mean area": mean_area,
    "mean smoothness": mean_smoothness,
    "mean compactness": mean_compactness,
    "mean concavity": mean_concavity,
    "mean concave points": mean_concave_points,
    "mean symmetry": mean_symmetry,
    "mean fractal dimension": mean_fractal_dimension,
    "radius error": radius_error,
    "texture error": texture_error,
    "perimeter error": perimeter_error,
    "area error": area_error,
    "smoothness error": smoothness_error,
    "compactness error": compactness_error,
    "concavity error": concavity_error,
    "concave points error": concave_points_error,
    "symmetry error": symmetry_error,
    "fractal dimension error": fractal_dimension_error,
    "worst radius": worst_radius,
    "worst texture": worst_texture,
    "worst perimeter": worst_perimeter,
    "worst area": worst_area,
    "worst smoothness": worst_smoothness,
    "worst compactness": worst_compactness,
    "worst concavity": worst_concavity,
    "worst concave points": worst_concave_points,
    "worst symmetry": worst_symmetry,
    "worst fractal dimension": worst_fractal_dimension,
    "likely_malignant": likely_malignant,
    f"family_history_breast_cancer_{family_history}": 1,
    f"menopause_status_{menopause_status}": 1,
    f"alcohol_intake_per_week_{alcohol}": 1,
    f"physical_activity_level_{physical_activity}": 1,
    f"nipple_discharge_{nipple_discharge}": 1,
    f"palpable_lump_{palpable_lump}": 1,
    f"localized_breast_pain_{localized_pain}": 1,
}

# Convert dict to DataFrame
input_df = pd.DataFrame([input_dict])

# Ensure all features match the trained model (fill 0 if missing)
input_encoded = input_df.reindex(columns=feature_columns, fill_value=0)

# --- Prediction ---
if st.button("🔍 Predict Risk"):
    prediction = model.predict(input_encoded)[0]
    if prediction == 1:
        st.error("🩺 **High Risk of Early Breast Cancer**")
    else:
        st.success("✅ **Low Risk of Early Breast Cancer**")

# --- SHAP Explanation ---
if st.button("🔬 Explain Prediction"):
    st.subheader("🔍 SHAP Explanation (Top 10 Features)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_encoded)
    expected_val = explainer.expected_value
    shap_val = shap_values[0]  # single instance

    fig = shap.plots._waterfall.waterfall_legacy(
        expected_val,
        shap_val,
        feature_names=feature_columns,
        max_display=10
    )
    st.pyplot(fig)