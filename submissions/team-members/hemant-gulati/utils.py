# utils.py
import pandas as pd
from collections import Counter

def preprocess_and_predict_with_clinical_thresholds(pipeline, new_data, non_diabetic_prob=0.15, pre_diabetic_prob=0.65):
    # Step 1: Automatically generate binned columns
    new_data['binned_blood_glucose_Diabetes'] = (new_data['blood_glucose_level'] > 125).astype(int)
    new_data['binned_hba1c_Severe Diabetes'] = (new_data['hbA1c_level'] > 7.5).astype(int)
    new_data['binned_bmi_Obese'] = (new_data['bmi'] >= 30).astype(int)
    
    # Step 2: Transform and predict using the loaded pipeline
    transformed_data = pipeline['preprocessor'].transform(new_data)
    probabilities = pipeline['model'].predict_proba(transformed_data)[:, 1]
    
    # Step 3: Classify based on clinical and probability thresholds
    categories = []
    for prob, row in zip(probabilities, new_data.itertuples()):
        hba1c = row.hbA1c_level
        glucose = row.blood_glucose_level

        # Use clinical guidelines as primary thresholds
        if hba1c < 5.7 and glucose <= 99:
            categories.append("Non-diabetic")
        elif 5.7 <= hba1c < 6.5 or (100 <= glucose <= 125):
            if prob >= pre_diabetic_prob:
                categories.append("Diabetic")
            else:
                categories.append("Pre-diabetic")
        else:
            categories.append("Diabetic")

    results = pd.DataFrame({
        'Probability': probabilities,
        'Prediction': categories
    })
    
    print("Classification Counts:", Counter(categories))
    return results
