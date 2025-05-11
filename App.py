import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import joblib

# Load encoders and model
try:
    label_encoders = joblib.load('label_encoder.joblib')  # dict of LabelEncoders for categorical columns
except Exception as e:
    label_encoders = {}
    st.error("Failed to load label encoders. Ensure 'label_encoder.joblib' is available.")

try:
    model_dict = joblib.load('svm.joblib')  # dict with keys: 'model', 'scaler', 'pca'
    model = model_dict['model']
except Exception as e:
    model = None
    st.error("Failed to load model. Ensure 'svm.joblib' contains a trained model.")

# Winsorization limits
WINSOR_LIMITS = {
    'months_on_book': 0.03,
    'credit_limit': 0.05,
    'avg_open_to_buy': 0.05,
    'total_amt_chng_q4_q1': 0.02,
    'total_trans_amt': 0.04,
    'total_ct_chng_q4_q1': 0.01
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

    if not label_encoders or model is None:
        st.stop()

    try:
        gender_options = label_encoders['gender'].classes_
        education_options = label_encoders['education_level'].classes_
        marital_options = label_encoders['marital_status'].classes_
        income_options = label_encoders['income_category'].classes_
        card_options = label_encoders['card_category'].classes_
    except KeyError as e:
        st.error(f"Missing encoder for: {e}. Please check your encoder file.")
        return

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
        gender = st.selectbox('Gender', gender_options)
        education = st.selectbox('Education Level', education_options)
        marital = st.selectbox('Marital Status', marital_options)
        income = st.selectbox('Income Category', income_options)
        card = st.selectbox('Card Category', card_options)

        submit = st.form_submit_button('Predict')

    if submit:
        # Build DataFrame with correct lowercase keys
        data = {
            'customer_age': [age],
            'dependent_count': [dependents],
            'months_on_book': [mos_on_book],
            'credit_limit': [credit_lim],
            'total_revolving_bal': [revolve_bal],
            'avg_open_to_buy': [open_to_buy],
            'total_amt_chng_q4_q1': [amt_chng],
            'total_trans_amt': [trans_amt],
            'total_ct_chng_q4_q1': [ct_chng],
            'avg_utilization_ratio': [util_ratio],
            'gender': [gender],
            'education_level': [education],
            'marital_status': [marital],
            'income_category': [income],
            'card_category': [card]
        }

        input_df = pd.DataFrame(data)

        try:
            processed_df = preprocess(input_df.copy())

            # Ensure column order matches model's training
            if hasattr(model, 'feature_names_in_'):
                processed_df = processed_df[model.feature_names_in_]

            prob = model.predict_proba(processed_df)[:, 1][0]
            pred = model.predict(processed_df)[0]

            st.subheader('Prediction Results')
            st.write(f"**Churn Probability:** {prob:.2%}")
            st.write(f"**Predicted Label:** {pred}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

if __name__ == '__main__':
    main()
