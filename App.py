import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("svm.joblib")  # Use .joblib instead of .pkl
label_encoders = joblib.load("label_encoder.joblib")  # Dictionary of LabelEncoders

st.title("🧠 Customer Attrition Prediction")

st.markdown("Fill in the details to predict whether a customer is likely to attrite.")

# Categorical and numerical column definitions
categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
numerical_cols = [
    'Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
    'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
]

# Value options for dropdowns
gender_options = ['M', 'F']
education_options = ['High School', 'Graduate', 'Uneducated', 'Unknown', 'College', 'Post-Graduate', 'Doctorate']
marital_options = ['Married', 'Single', 'Unknown', 'Divorced']
income_options = ['$60K - $80K', 'Less than $40K', '$80K - $120K', '$40K - $60K', '$120K +', 'Unknown']
card_options = ['Blue', 'Gold', 'Silver', 'Platinum']

# Streamlit form
with st.form("attrition_form"):
    st.subheader("📋 Categorical Information")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", gender_options)
        education = st.selectbox("Education Level", education_options)
        marital = st.selectbox("Marital Status", marital_options)
    with col2:
        income = st.selectbox("Income Category", income_options)
        card = st.selectbox("Card Category", card_options)

    st.subheader("🔢 Numerical Information")
    with st.expander("Click to expand numerical inputs"):
        numerical_inputs = {}
        numerical_labels = {
            'Customer_Age': 'Customer Age',
            'Dependent_count': 'Number of Dependents',
            'Months_on_book': 'Months on Book',
            'Total_Relationship_Count': 'Total Relationship Count',
            'Months_Inactive_12_mon': 'Inactive Months (Last 12)',
            'Contacts_Count_12_mon': 'Contact Count (Last 12 Months)',
            'Credit_Limit': 'Credit Limit',
            'Total_Revolving_Bal': 'Total Revolving Balance',
            'Avg_Open_To_Buy': 'Average Open to Buy',
            'Total_Amt_Chng_Q4_Q1': 'Transaction Amount Change (Q4-Q1)',
            'Total_Trans_Amt': 'Total Transaction Amount',
            'Total_Trans_Ct': 'Total Transaction Count',
            'Total_Ct_Chng_Q4_Q1': 'Transaction Count Change (Q4-Q1)',
            'Avg_Utilization_Ratio': 'Average Utilization Ratio'
        }

        for col in numerical_cols:
            label = numerical_labels.get(col, col.replace("_", " "))
            default_val = 40.0 if col == "Customer_Age" else 0.0
            numerical_inputs[col] = st.number_input(label, value=default_val, step=1.0)

    submitted = st.form_submit_button("🚀 Predict")

if submitted:
    input_data = {
        'Gender': gender,
        'Education_Level': education,
        'Marital_Status': marital,
        'Income_Category': income,
        'Card_Category': card,
        **numerical_inputs
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical values
    for col in categorical_cols:
        le = label_encoders.get(col)
        if le:
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError:
                st.error(f"Invalid input for {col}. Choose from: {list(le.classes_)}")
                st.stop()

    # Predict
    prediction = model.predict(input_df)[0]
    label = "Attrited Customer" if prediction == 1 else "Existing Customer"
    st.success(f"🧾 Prediction: **{label}**")
