import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("svm.joblib")
label_encoders = joblib.load("label_encoder.joblib")

st.set_page_config(page_title="Customer Attrition Predictor", page_icon="🧠")
st.title("🧠 Customer Attrition Prediction")
st.markdown("Fill in the customer's key information to predict whether they are likely to attrite.")

# Categorical options (user-friendly)
gender_options = ['Male', 'Female']
education_options = ['High School', 'College', 'Graduate', 'Doctorate']
marital_options = ['Single', 'Married', 'Divorced']
income_options = ['< $40K', '$40K–$60K', '$60K–$80K', '$80K–$120K', '$120K+']

# Streamlit form
with st.form("attrition_form"):
    st.subheader("👤 Customer Info")

    gender = st.selectbox("Gender", gender_options)
    education = st.selectbox("Education Level", education_options)
    marital = st.selectbox("Marital Status", marital_options)
    income = st.selectbox("Income Bracket", income_options)

    st.subheader("💳 Financial Indicators")

    age = st.number_input("Customer Age", value=40, min_value=18, max_value=100)
    trans_count = st.number_input("Total Transactions (last 12 months)", value=60, min_value=0)
    trans_amt = st.number_input("Total Amount Transacted ($)", value=5000.0, min_value=0.0)
    utilization = st.number_input("Average Utilization Ratio", value=0.2, min_value=0.0, max_value=1.0)

    submitted = st.form_submit_button("🚀 Predict")

# On form submit
if submitted:
    # Map friendly inputs to model format
    input_data = {
        'Gender': 'M' if gender == 'Male' else 'F',
        'Education_Level': education,
        'Marital_Status': marital,
        'Income_Category': income,
        'Customer_Age': age,
        'Total_Trans_Ct': trans_count,
        'Total_Trans_Amt': trans_amt,
        'Avg_Utilization_Ratio': utilization
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical values
    for col in ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category']:
        le = label_encoders.get(col)
        if le:
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError:
                st.error(f"Invalid input for {col}. Expected one of: {list(le.classes_)}")
                st.stop()

    # Predict
    prediction = model.predict(input_df)[0]
    label = "Attrited Customer" if prediction == 1 else "Existing Customer"
    st.success(f"🧾 Prediction: **{label}**")
