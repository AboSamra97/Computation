import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import joblib

# Load encoders and model
label_encoders = joblib.load('label_encoder.joblib')  # dict of LabelEncoders for categorical columns
model = joblib.load('svm.joblib')  # pre-trained pipeline (scaler, PCA, SVC)

# Winsorization limits matching training
WINSOR_LIMITS = {
    'Months_on_book': 0.03,
    'Credit_Limit': 0.05,
    'Avg_Open_To_Buy': 0.05,
    'Total_Amt_Chng_Q4_Q1': 0.02,
    'Total_Trans_Amt': 0.04,
    'Total_Ct_Chng_Q4_Q1': 0.01
}

# Preprocessing function
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Encode categorical
    for col, le in label_encoders.items():
        df[col] = le.transform(df[col])
    # Winsorize numeric
    for col, limit in WINSOR_LIMITS.items():
        df[col] = winsorize(df[col], limits=(limit, limit))
    return df


def main():
    st.set_page_config(page_title="Churn Predictor", layout="centered")
    st.title("Bank Customer Churn Prediction")
    st.markdown("Enter customer details below and click **Predict** to see churn probability.")

    with st.form(key='input_form'):
        # Numeric inputs
        age = st.number_input('Customer Age', min_value=18, max_value=100, value=40)
        dependents = st.number_input('Dependent Count', min_value=0, max_value=10, value=1)
        mos_on_book = st.number_input('Months on Book', min_value=0, max_value=120, value=12)
        credit_lim = st.number_input('Credit Limit', min_value=0.0, max_value=50000.0, value=10000.0)
        revolve_bal = st.number_input('Total Revolving Balance', min_value=0.0, max_value=5000.0, value=1000.0)
        open_to_buy = st.number_input('Avg Open To Buy', min_value=0.0, max_value=50000.0, value=9000.0)
        amt_chng = st.number_input('Total Amt Change Q4/Q1', min_value=0.0, value=1.0)
        trans_amt = st.number_input('Total Transaction Amount', min_value=0.0, value=1000.0)
        ct_chng = st.number_input('Total Count Change Q4/Q1', min_value=0.0, value=1.0)
        util_ratio = st.slider('Avg Utilization Ratio', min_value=0.0, max_value=1.0, value=0.3)

        # Categorical inputs using encoder classes
        gender = st.selectbox('Gender', label_encoders['Gender'].classes_)
        education = st.selectbox('Education Level', label_encoders['Education_Level'].classes_)
        marital = st.selectbox('Marital Status', label_encoders['Marital_Status'].classes_)
        income = st.selectbox('Income Category', label_encoders['Income_Category'].classes_)
        card = st.selectbox('Card Category', label_encoders['Card_Category'].classes_)

        submit = st.form_submit_button('Predict')

    if submit:
        # Build DataFrame
        data = {
            'Customer_Age': [age],
            'Dependent_count': [dependents],
            'Months_on_book': [mos_on_book],
            'Credit_Limit': [credit_lim],
            'Total_Revolving_Bal': [revolve_bal],
            'Avg_Open_To_Buy': [open_to_buy],
            'Total_Amt_Chng_Q4_Q1': [amt_chng],
            'Total_Trans_Amt': [trans_amt],
            'Total_Ct_Chng_Q4_Q1': [ct_chng],
            'Avg_Utilization_Ratio': [util_ratio],
            'Gender': [gender],
            'Education_Level': [education],
            'Marital_Status': [marital],
            'Income_Category': [income],
            'Card_Category': [card]
        }
        input_df = pd.DataFrame(data)

        # Preprocess
        processed_df = preprocess(input_df.copy())

        # Predict
        prob = model.predict_proba(processed_df)[:, 1][0]
        pred = model.predict(processed_df)[0]

        # Display results
        st.subheader('Prediction Results')
        st.write(f"**Churn Probability:** {prob:.2%}")
        st.write(f"**Predicted Label:** {pred}")

if __name__ == '__main__':
    main()
