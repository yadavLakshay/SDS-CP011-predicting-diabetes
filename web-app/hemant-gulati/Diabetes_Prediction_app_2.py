# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils import preprocess_and_predict_with_clinical_thresholds
import base64

# Import necessary libraries
import streamlit as st
import base64

# Set background with overlay
def set_background():
    background_image_path = "web-app/hemant-gulati/app_background.jpg"  # Replace with your image path
    with open(background_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # Apply background with a dark overlay
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .title-box, .header-box {{
            background-color: rgba(255, 255, 255, 0.8); /* Light semi-transparent background */
            padding: 10px;
            border-radius: 8px;
            width: fit-content;
            margin: 20px auto; /* Center align */
        }}
        .title {{
            color: #1a3d6d; /* Dark blue */
            font-size: 2.5em;
            font-weight: bold;
        }}
        .header {{
            color: #333333; /* Charcoal gray */
            font-size: 1.75em;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to set the background
set_background()

# Apply title and header with the styled boxes
st.markdown("<div class='title-box'><p class='title'>Diabetes Risk & Health Assessment</p></div>", unsafe_allow_html=True)
st.markdown("<div class='header-box'><p class='header'>Enter Patient's Clinical and Demographic details to assess Diabetes risk and receive personalized recommendations.</p></div>", unsafe_allow_html=True)


# Define the path to your pipeline file
pipeline_path = "web-app/hemant-gulati/pipeline.pkl"

# Load the pipeline
try:
    pipeline = joblib.load(pipeline_path)
except Exception as e:
    st.write(f"Error loading pipeline: {e}")

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

    # Make predictions
    results = preprocess_and_predict_with_clinical_thresholds(pipeline, user_df)

    # Display the prediction and confidence level
    st.subheader("Prediction Results")
    for index, row in results.iterrows():
        st.write(f"**Prediction:** {row['Prediction']}")
        #st.write(f"**Confidence Level:** {row['Probability'] * 100:.2f}%")

    # Suggest the next steps based on the prediction
    if row['Prediction'] == "Non-diabetic":
        st.write("Patient is likely non-diabetic. Maintain a healthy lifestyle to reduce future risks.")
    elif row['Prediction'] == "Pre-diabetic":
        st.write("Patient is likely pre-diabetic. Consider consulting a healthcare provider for lifestyle advice.")
    else:
        st.write("Patient is likely diabetic. Seek professional medical guidance for appropriate treatment.")
