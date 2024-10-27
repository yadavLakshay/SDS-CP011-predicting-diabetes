# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost
import shap
import category_encoders

# Define the path to your pipeline file
pipeline_path = 'web-app/hemant-gulati/pipeline.pkl'

# Load the pipeline
try:
    pipeline = joblib.load('web-app/hemant-gulati/pipeline.pkl')
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading pipeline: {e}")

# Title of the app
st.title("Diabetes Prediction App")

# User inputs
st.header("Enter Patient's Clinical and Demographic Details to Assess Diabetes Risk and Receive Personalized Recommendations.")

hbA1c_level = st.number_input("HbA1c Level (e.g., 5.5)", min_value=0.0, max_value=15.0, value=5.5, step=0.1)
blood_glucose_level = st.number_input("Blood Glucose Level (e.g., 100)", min_value=0, max_value=400, value=100, step=1)
bmi = st.number_input("BMI (e.g., 24.5)", min_value=0.0, max_value=70.0, value=24.5, step=0.1)
age = st.number_input("Age (e.g., 45)", min_value=0, max_value=120, value=45, step=1)
hypertension = st.selectbox("Hypertension (0 for No, 1 for Yes)", options=[0, 1])
smoking_history = st.selectbox("Smoking History (0 for No, 1 for Yes)", options=[0, 1])
heart_disease = st.selectbox("Heart Disease (0 for No, 1 for Yes)", options=[0, 1])

# Prediction button
if st.button("Predict"):
    # Prepare user data in a dictionary format
    user_data = {
        'hbA1c_level': [hbA1c_level],
        'blood_glucose_level': [blood_glucose_level],
        'bmi': [bmi],
        'age': [age],
        'hypertension': [hypertension],
        'smoking_history': [smoking_history],
        'heart_disease': [heart_disease]
    }

    # Convert user input to DataFrame
    user_df = pd.DataFrame(user_data)

    # Automatically generate binned columns
    user_df['binned_blood_glucose_Diabetes'] = (user_df['blood_glucose_level'] > 125).astype(int)
    user_df['binned_hba1c_Severe Diabetes'] = (user_df['hbA1c_level'] > 7.5).astype(int)
    user_df['binned_bmi_Obese'] = (user_df['bmi'] >= 30).astype(int)

    # Define the required feature columns
    top_features = [
        'hbA1c_level', 'blood_glucose_level', 'bmi', 'age', 
        'hypertension', 'smoking_history', 'heart_disease',
        'binned_blood_glucose_Diabetes', 'binned_hba1c_Severe Diabetes', 'binned_bmi_Obese'
    ]

    # Ensure the DataFrame has all required columns
    user_df_final = user_df[top_features]

    # Make prediction with the loaded pipeline
    prediction = pipeline.predict(user_df_final)[0]
    prediction_proba = pipeline.predict_proba(user_df_final)[0]

    # Interpret the prediction
    if prediction == 0:
        prediction_label = "Non-diabetic"
    elif prediction == 1:
        prediction_label = "Pre-diabetic"
    else:
        prediction_label = "Diabetic"

    # Display the prediction and confidence level
    st.subheader("Prediction Results")
    st.write(f"**Prediction:** {prediction_label}")
    st.write(f"**Confidence Level:** {np.max(prediction_proba) * 100:.2f}%")

    # Suggest the next steps based on the prediction
    if prediction_label == "Non-diabetic":
        st.write("Patient is likely non-diabetic. Maintain a healthy lifestyle to reduce future risks.")
    elif prediction_label == "Pre-diabetic":
        st.write("Patient is  likely pre-diabetic. Consider consulting a healthcare provider for lifestyle advice.")
    else:
        st.write("Patient is likely diabetic. Seek professional medical guidance for appropriate treatment.")
