import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load model
model = joblib.load("breast_cancer_risk_model.pkl")

# App Title
st.title("🧬 Early Breast Cancer Risk Prediction")
st.markdown("This app uses 40 features to estimate the likelihood of early breast cancer based on patient data.")

# Styling
st.markdown("---")

# Section 1: Mean Measurements
st.subheader("📊 Mean Cell Measurements")
mean_radius = st.number_input("Mean Radius (mm)", value=14.0, help="Average radius of cell nuclei")
mean_texture = st.number_input("Mean Texture", value=20.0, help="Standard deviation of gray-scale values")
mean_perimeter = st.number_input("Mean Perimeter (mm)", value=90.0)
mean_area = st.number_input("Mean Area (mm²)", value=600.0)
mean_smoothness = st.number_input("Mean Smoothness", value=0.1, help="Local variation in radius lengths")
mean_compactness = st.number_input("Mean Compactness", value=0.1, help="Perimeter² / Area - 1.0")
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

# Section 3: Worst Case Measurements
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

# Section 4: Patient History & Lifestyle
st.subheader("🧝 Patient History & Lifestyle")
diagnosis_label = st.selectbox("Initial Diagnosis", options=[0, 1], format_func=lambda x: "Benign (0)" if x == 0 else "Malignant (1)")
family_history = st.checkbox("Family History of Breast Cancer")
menopause_status = st.selectbox("Menopause Status", options=[0, 1], format_func=lambda x: "Premenopause (0)" if x == 0 else "Postmenopause (1)")
nipple_discharge = st.checkbox("Nipple Discharge")
palpable_lump = st.checkbox("Palpable Lump")
localized_pain = st.checkbox("Localized Breast Pain")
alcohol_light = st.checkbox("Alcohol Intake: Light")
alcohol_moderate = st.checkbox("Alcohol Intake: Moderate")
physical_activity_moderate = st.checkbox("Physical Activity: Moderate")
physical_activity_sedentary = st.checkbox("Physical Activity: Sedentary")

# Feature vector
input_data = np.array([[
    mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
    mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension,
    radius_error, texture_error, perimeter_error, area_error, smoothness_error,
    compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
    worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
    worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension,
    diagnosis_label,
    int(family_history), menopause_status,
    int(nipple_discharge), int(palpable_lump), int(localized_pain),
    int(alcohol_light), int(alcohol_moderate),
    int(physical_activity_moderate), int(physical_activity_sedentary)
]])

import shap
import matplotlib.pyplot as plt

# --- Prediction ---
if st.button("🔍 Predict Risk", key="predict_button"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("🩺 **High Risk of Early Breast Cancer**")
    else:
        st.success("✅ **Low Risk of Early Breast Cancer**")

# --- SHAP Explanation ---
import shap
import matplotlib.pyplot as plt

if st.button("🔬 Explain Prediction", key="shap_button"):
    st.subheader("🔍 Detailed SHAP Explanation (Top 10 Features)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # Get SHAP values for class 1
    shap_val = shap_values[0, :, 1]  # shape: (40,)
    expected_val = explainer.expected_value[1]

    # Real 40 feature names in order
    feature_names = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
        'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension',
        'diagnosis_label', 'family_history_breast_cancer', 'menopause_status',
        'nipple_discharge', 'palpable_lump', 'localized_breast_pain',
        'alcohol_intake_per_week_Light', 'alcohol_intake_per_week_Moderate',
        'physical_activity_level_Moderate', 'physical_activity_level_Sedentary'
    ]

    # Plot waterfall chart
    fig = shap.plots._waterfall.waterfall_legacy(
        expected_val,
        shap_val,
        feature_names=feature_names,
        max_display=10
    )

    st.pyplot(fig)
