import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Load model and label encoders
model = pickle.load(open("svm.pkl", "rb"))
label_encoders = joblib.load("label_encoder.joblib")  # dict: {col_name: LabelEncoder}

st.title("🧠 Customer Attrition Prediction")

st.markdown("Fill out the form below to predict whether a customer is likely to attrite.")

# Define categorical and numerical columns
categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
numerical_cols = [
    'Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
    'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
]

# Options for categorical columns (from dataset)
gender_options = ['M', 'F']
education_options = ['High School', 'Graduate', 'Uneducated', 'Unknown', 'College', 'Post-Graduate', 'Doctorate']
marital_options = ['Married', 'Single', 'Unknown', 'Divorced']
income_options = ['$60K - $80K', 'Less than $40K', '$80K - $120K', '$40K - $60K', '$120K +', 'Unknown']
card_options = ['Blue', 'Gold', 'Silver', 'Platinum']

# Create input form
with st.form("attrition_form"):
    # Categorical inputs
    gender = st.selectbox("Gender", gender_options)
    education = st.selectbox("Education Level", education_options)
    marital = st.selectbox("Marital Status", marital_options)
    income = st.selectbox("Income Category", income_options)
    card = st.selectbox("Card Category", card_options)

    # Numerical inputs
    numerical_inputs = {}
    for col in numerical_cols:
        default = 0.0 if col != "Customer_Age" else 40
        numerical_inputs[col] = st.number_input(col.replace('_', ' '), value=default)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input
    input_data = {
        'Gender': gender,
        'Education_Level': education,
        'Marital_Status': marital,
        'Income_Category': income,
        'Card_Category': card,
        **numerical_inputs
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in categorical_cols:
        le = label_encoders.get(col)
        if le:
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError:
                st.error(f"Invalid value for {col}. Please choose from: {list(le.classes_)}")
                st.stop()

    # Predict
    prediction = model.predict(input_df)[0]
    prediction_label = "Attrited Customer" if prediction == 1 else "Existing Customer"

    # Display result
    st.success(f"🧾 Prediction: **{prediction_label}**")
