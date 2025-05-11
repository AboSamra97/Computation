import streamlit as st
import pandas as pd
from scipy.stats.mstats import winsorize
import joblib

# Page configuration
st.set_page_config(
    page_title="Bank Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load encoders and trained pipeline
try:
    label_encoders = joblib.load('label_encoders.joblib')      # dict of LabelEncoder
    model = joblib.load('sklearn_pipeline.joblib')             # fitted Pipeline with scaler, PCA, SVC
except Exception as e:
    st.sidebar.error(f"Error loading model artifacts: {e}")
    st.stop()

# Winsorization limits from training
WINSOR_LIMITS = {
    'months_on_book': 0.03,
    'credit_limit': 0.05,
    'avg_open_to_buy': 0.05,
    'total_amt_chng_q4_q1': 0.02,
    'total_trans_amt': 0.04,
    'total_ct_chng_q4_q1': 0.01
}

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Encode categories
    for col, le in label_encoders.items():
        df[col] = le.transform(df[col])
    # Winsorize numeric
    for col, limit in WINSOR_LIMITS.items():
        df[col] = winsorize(df[col], limits=(limit, limit))
    return df

# Sidebar title
st.sidebar.header("Input Customer Details")

with st.sidebar.form(key='customer_form'):
    st.subheader("Demographic Info")
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=45)
    gender = st.selectbox("Gender", label_encoders['gender'].classes_)

    st.subheader("Account Details")
    months_on_book = st.number_input("Tenure (months)", min_value=0, max_value=120, value=36)
    total_relationship_count = st.number_input("Total Relationships", min_value=0, max_value=10, value=4)
    months_inactive = st.number_input("Inactive Months (last 12)", min_value=0, max_value=12, value=2)
    contacts_count = st.number_input("Contacts (last 12 months)", min_value=0, max_value=20, value=3)

    st.subheader("Transaction Metrics")
    total_trans_amt = st.number_input("Total Transaction Amount ($)", min_value=0.0, value=3000.0)
    total_trans_ct = st.number_input("Transaction Count", min_value=0, value=50)
    avg_util_ratio = st.slider("Avg Utilization Ratio", min_value=0.0, max_value=1.0, value=0.25)

    st.subheader("Financial Metrics")
    credit_limit = st.number_input("Credit Limit ($)", min_value=0.0, value=10000.0)
    total_revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, value=1000.0)
    avg_open_to_buy = st.number_input("Avg Open-to-Buy ($)", min_value=0.0, value=9000.0)
    amt_chng_q4_q1 = st.number_input("Amt Change Q4/Q1", min_value=0.0, value=1.0)
    ct_chng_q4_q1 = st.number_input("Count Change Q4/Q1", min_value=0.0, value=1.0)

    st.subheader("Personal Attributes")
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=2)
    education = st.selectbox("Education Level", label_encoders['education_level'].classes_)
    marital = st.selectbox("Marital Status", label_encoders['marital_status'].classes_)
    income = st.selectbox("Income Bracket", label_encoders['income_category'].classes_)
    card = st.selectbox("Card Type", label_encoders['card_category'].classes_)

    submitted = st.form_submit_button("Predict Churn")

# Main page
st.title("🏦 Bank Customer Churn Predictor")
st.markdown(
    "Use the form in the sidebar to enter customer details. "
    "Click **Predict Churn** to see the probability of attrition."
)

if submitted:
    # Collect inputs
    data = {
        'customer_age': [age],
        'gender': [gender],
        'dependent_count': [dependents],
        'education_level': [education],
        'marital_status': [marital],
        'income_category': [income],
        'card_category': [card],
        'months_on_book': [months_on_book],
        'total_relationship_count': [total_relationship_count],
        'months_inactive_12_mon': [months_inactive],
        'contacts_count_12_mon': [contacts_count],
        'credit_limit': [credit_limit],
        'total_revolving_bal': [total_revol_bal],
        'avg_open_to_buy': [avg_open_to_buy],
        'total_amt_chng_q4_q1': [amt_chng_q4_q1],
        'total_trans_amt': [total_trans_amt],
        'total_trans_ct': [total_trans_ct],
        'total_ct_chng_q4_q1': [ct_chng_q4_q1],
        'avg_utilization_ratio': [avg_util_ratio]
    }
    input_df = pd.DataFrame(data)

    # Preprocess & reorder
    processed = preprocess(input_df.copy())
    processed = processed[model.feature_names_in_]

    # Predict
    prob = model.predict_proba(processed)[:, 1][0]
    label = model.predict(processed)[0]

    # Display results
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{prob:.2%}")
    col2.metric("Predicted Outcome", label)
