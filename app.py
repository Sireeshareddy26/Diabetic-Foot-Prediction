
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the order of columns used during training
# This is crucial for consistent input to the model
feature_columns = ['Gender', 'Age', 'Duration of DM ', 'Family H/O DM', 'BM1', 'FBS', 'HbA1C', 
                   'ITLNI ELISA', 'NTN1 ELISA', 'STAGES', 'TG', 'HDL', 'LDL', 'CHOLESTEROL', 'VLDL']

# Define columns that were scaled
scaling_columns = ['Age', 'Duration of DM ', 'BM1', 'FBS', 'HbA1C', 'ITLNI ELISA', 'NTN1 ELISA', 'TG', 'HDL', 'LDL', 'CHOLESTEROL', 'VLDL']

st.title('Diabetic Foot Development Prediction')
st.write('Enter the patient\u2019s details to predict the risk of Diabetic Foot development.')

# Input fields for features
gender = st.selectbox('Gender', [1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')
age = st.slider('Age', 17, 80, 50)
duration_dm = st.slider('Duration of DM', 1.0, 30.0, 10.0)
family_ho_dm = st.selectbox('Family H/O DM', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
bm1 = st.slider('BM1', 13.0, 31.0, 23.0)
fbs = st.slider('FBS', 50, 400, 150)
hba1c = st.slider('HbA1C', 5.0, 19.0, 8.0)
itlni_elisa = st.slider('ITLNI ELISA', 0.0, 1200.0, 150.0)
ntn1_elisa = st.slider('NTN1 ELISA', 0.0, 1400.0, 200.0)
stages = st.selectbox('STAGES', [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
tg = st.slider('TG', 30.0, 600.0, 120.0)
hdl = st.slider('HDL', 5.0, 80.0, 35.0)
ldl = st.slider('LDL', 0.0, 300.0, 95.0)
cholesterol = st.slider('CHOLESTEROL', 20.0, 350.0, 150.0)
vldl = st.slider('VLDL', 0.0, 120.0, 13.0)

# Collect inputs into a DataFrame
input_data = pd.DataFrame([[gender, age, duration_dm, family_ho_dm, bm1, fbs, hba1c,
                            itlni_elisa, ntn1_elisa, stages, tg, hdl, ldl, cholesterol, vldl]],
                          columns=feature_columns)

# Scale numerical features
input_data[scaling_columns] = scaler.transform(input_data[scaling_columns])

if st.button('Predict'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0]

    if prediction[0] == 1:
        st.error(f"Prediction: High Risk of Diabetic Foot Development (Probability: {prediction_proba[1]:.2f})")
    else:
        st.success(f"Prediction: Low Risk of Diabetic Foot Development (Probability: {prediction_proba[0]:.2f})")
