import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load pre-trained models and encoders ---
one_hot_encoder = joblib.load('/content/one_hot_encoder 1.joblib')
min_max_scaler = joblib.load('/content/min_max_scaler 1.joblib')
voting_hard_model = joblib.load('/content/voting_hard_model.joblib')
label_encoder = joblib.load('/content/label_encoder 1.joblib')

# --- 2. Streamlit App Title and Description ---
st.title('Course Approval Prediction')
st.write('Enter the student details to predict if they will approve the course.')

# --- 3. Input Widgets ---

# 'Felder' input (assuming 'Felder' has categories from the encoder)
felder_categories = one_hot_encoder.categories_[0].tolist()
selected_felder = st.selectbox('Felder Learning Style', felder_categories)

# 'Examen_admisión' input
# Assuming typical range for exam scores, adjust min/max/value as per actual data
examen_min = 0.0
examen_max = 5.0
examen_default = 3.5
examen_admision = st.slider('Admission Exam Score (0.0 - 5.0)', min_value=examen_min, max_value=examen_max, value=examen_default, step=0.01)

# --- 4. Prediction Button ---
if st.button('Predict Approval'):
    # --- 5. Preprocessing Input Data ---
    # a. Process 'Felder'
    felder_df = pd.DataFrame({'Felder': [selected_felder]})
    felder_encoded = one_hot_encoder.transform(felder_df[['Felder']])
    felder_encoded_df = pd.DataFrame(felder_encoded, columns=one_hot_encoder.get_feature_names_out(['Felder']))

    # b. Process 'Examen_admisión'
    examen_df = pd.DataFrame({'Examen_admisión': [examen_admision]})
    examen_scaled = min_max_scaler.transform(examen_df[['Examen_admisión']])
    examen_scaled_df = pd.DataFrame(examen_scaled, columns=['Examen_admisión'])

    # c. Combine preprocessed data
    # The order of columns and presence of all expected one-hot encoded columns is crucial
    # Create a DataFrame with all possible Felder columns, initialized to 0
    all_felder_cols = [f'Felder_{cat}' for cat in felder_categories]
    input_df_combined = pd.DataFrame(0, index=[0], columns=all_felder_cols + ['Examen_admisión'])

    # Fill in the scaled exam score
    input_df_combined['Examen_admisión'] = examen_scaled_df['Examen_admisión'].iloc[0]

    # Fill in the one-hot encoded Felder values
    for col in felder_encoded_df.columns:
        if col in input_df_combined.columns:
            input_df_combined[col] = felder_encoded_df[col].iloc[0]

    # Reorder columns to match the training data if necessary (based on df_second_sheet columns from previous steps)
    # The exact column order used for model training should be maintained.
    # For simplicity, assuming the `one_hot_encoder.get_feature_names_out()` provides columns in the correct order
    # for Felder, and 'Examen_admisión' is the first.
    # This needs to be precisely matched to the model's training features.
    # Based on notebook cells like 4GsQV1n7HeUy and the Kernel State for df_second_sheet,
    # the order seems to be Examen_admisión first, then Felder_*. Let's try to infer it.

    # Inferred feature order from df_second_sheet kernel state:
    # ['Examen_admisión', 'Felder_activo', 'Felder_equilibrio', 'Felder_intuitivo', 'Felder_reflexivo', 'Felder_secuencial', 'Felder_sensorial', 'Felder_verbal', 'Felder_visual']

    model_features_order = ['Examen_admisión'] + sorted(one_hot_encoder.get_feature_names_out(['Felder']).tolist())

    # Ensure the combined input DataFrame has columns in the exact order the model expects
    final_input_for_prediction = input_df_combined[model_features_order]

    # --- 6. Make Prediction ---
    prediction_encoded = voting_hard_model.predict(final_input_for_prediction)

    # --- 7. Decode Prediction ---
    prediction_decoded = label_encoder.inverse_transform(prediction_encoded)

    # --- 8. Display Result ---
    if prediction_decoded[0] == 'si':
        st.success(f'Prediction: The student is likely to approve the course ({prediction_decoded[0]}).')
    else:
        st.error(f'Prediction: The student is likely to not approve the course ({prediction_decoded[0]}).')

st.write('---')
st.write('This is a simple Streamlit application for course approval prediction.')
