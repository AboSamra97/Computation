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

# Load LabelEncoders and trained Pipeline
try:
    label_encoders = joblib.load('label_encoders.joblib')  # dict of LabelEncoder objects
    model = joblib.load('sklearn_pipeline.joblib')         # fitted Pipeline with scaler, PCA, SVC
except Exception as e:
    st.sidebar.error(f"Error loading model artifacts: {e}")
    st.stop()

# Mapping for readability
GENDER_MAP = {'M': 'Male', 'F': 'Female'}
INV_GENDER_MAP = {v: k for k, v in GENDER_MAP.items()}

# Winsorization limits used during training
WINSOR_LIMITS = {
    'months_on_book': 0.03,
    'credit_limit': 0.05,
    'avg_open_to_buy': 0.05,
    'total_amt_chng_q4_q1': 0.02,
    'total_trans_amt': 0.04,
    'total_ct_chng_q4_q1': 0.01
}

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Encode categories back to model labels
    for col, le in label_encoders.items():
        df[col] = le.transform(df[col])
    # Winsorize numeric fields
    for col, limit in WINSOR_LIMITS.items():
        df[col] = winsorize(df[col], limits=(limit, limit))
    return df

# Sidebar input form
st.sidebar.header("Enter Customer Details")
with st.sidebar.form(key='input_form'):
    st.subheader("👤 Demographics")
    age = st.number_input(
        "Age (years)", min_value=18, max_value=100, value=45,
        help="Customer's age in completed years"
    )
    gender_read = st.selectbox(
        "Gender", list(GENDER_MAP.values()),
        help="Select 'Male' or 'Female'"
    )
    gender = INV_GENDER_MAP[gender_read]

    st.subheader("💼 Account Details")
    months_on_book = st.number_input(
        "Account Tenure (months)", min_value=0, max_value=120, value=36,
        help="Number of months since account opening"
    )
    total_relationship_count = st.number_input(
        "Total Relationships", min_value=0, max_value=10, value=4,
        help="Number of products held with the bank"
    )
    months_inactive = st.number_input(
        "Inactive Months (last 12)", min_value=0, max_value=12, value=2,
        help="Count of months with no transactions in the past year"
    )
    contacts_count = st.number_input(
        "Contacts (last 12 months)", min_value=0, max_value=20, value=3,
        help="Number of customer-service calls in the past year"
    )

    st.subheader("💳 Transaction Metrics")
    total_trans_amt = st.number_input(
        "Total Transaction Amount ($)", min_value=0.0, value=3000.0,
        help="Sum of all transaction amounts over all channels"
    )
    total_trans_ct = st.number_input(
        "Transaction Count", min_value=0, value=50,
        help="Total number of transactions made"
    )
    avg_util_ratio = st.slider(
        "Avg Utilization Ratio", min_value=0.0, max_value=1.0, value=0.25,
        help="Avg. balance / credit limit over time"
    )

    st.subheader("💰 Financial Metrics")
    credit_limit = st.number_input(
        "Credit Limit ($)", min_value=0.0, value=10000.0,
        help="Maximum credit limit available"
    )
    total_revol_bal = st.number_input(
        "Revolving Balance ($)", min_value=0.0, value=1000.0,
        help="Outstanding balance that is carried over"
    )
    avg_open_to_buy = st.number_input(
        "Avg Open-to-Buy ($)", min_value=0.0, value=9000.0,
        help="Average available credit over time"
    )
    amt_chng_q4_q1 = st.number_input(
        "Amt Change Q4/Q1", min_value=0.0, value=1.0,
        help="Ratio of transaction amount: Q4 vs Q1"
    )
    ct_chng_q4_q1 = st.number_input(
        "Count Change Q4/Q1", min_value=0.0, value=1.0,
        help="Ratio of transaction counts: Q4 vs Q1"
    )

    st.subheader("🔖 Personal Attributes")
    dependents = st.number_input(
        "Dependents", min_value=0, max_value=10, value=2,
        help="Number of dependents"
    )
    education = st.selectbox(
        "Education Level",
        label_encoders['education_level'].classes_,
        help="Highest level of education attained"
    )
    marital = st.selectbox(
        "Marital Status",
        label_encoders['marital_status'].classes_,
        help="Customer's marital status"
    )
    income = st.selectbox(
        "Income Bracket",
        label_encoders['income_category'].classes_,
        help="Annual household income range"
    )
    card = st.selectbox(
        "Card Type",
        label_encoders['card_category'].classes_,
        help="Type of credit card product"
    )

    submitted = st.form_submit_button("Predict Churn")

# Main page
st.title("🏦 Bank Customer Churn Predictor")
st.markdown(
    "Use the sidebar to enter customer details and click **Predict Churn**."
)

if submitted:
    # Build DataFrame
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
    prob = model.predict_proba(processed)[:, 1][1]
    label = model.predict(processed)[0]

    # Display results side-by-side
    st.subheader("🔮 Prediction Results")
    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{prob:.2%}")
    col2.metric("Predicted Outcome", label)
