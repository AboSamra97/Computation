import streamlit as st
import pandas as pd
from scipy.stats.mstats import winsorize
import joblib

# 1) MUST be the first Streamlit call in the script
st.set_page_config(page_title="Churn Predictor", layout="centered")

# 2) Load LabelEncoders and trained Pipeline
try:
    label_encoders = joblib.load('label_encoders.joblib')      # dict of LabelEncoder
    model = joblib.load('sklearn_pipeline.joblib')             # fitted Pipeline with scaler, PCA, SVC
except Exception as e:
    st.error(f"Failed to load saved artifacts: {e}")
    st.stop()

# Optional: show expected feature order
st.write("🔑 Model expects features (in order):")
st.write(list(model.feature_names_in_))

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
    # 1) Label-encode all categorical columns
    for col, le in label_encoders.items():
        df[col] = le.transform(df[col])
    # 2) Winsorize numeric columns
    for col, limit in WINSOR_LIMITS.items():
        df[col] = winsorize(df[col], limits=(limit, limit))
    return df

def main():
    st.title("🏦 Bank Customer Churn Prediction")
    st.markdown("Fill in the customer details and click **Predict** to see churn probability.")

    # Pull out encoder classes for selectboxes
    try:
        gender_options    = label_encoders['gender'].classes_
        education_options = label_encoders['education_level'].classes_
        marital_options   = label_encoders['marital_status'].classes_
        income_options    = label_encoders['income_category'].classes_
        card_options      = label_encoders['card_category'].classes_
    except KeyError as e:
        st.error(f"Missing encoder for: {e}. Check `label_encoders.joblib`.")
        st.stop()

    with st.form(key='input_form'):
        # — Numeric inputs —
        age         = st.number_input('Customer Age', min_value=18, max_value=100, value=40)
        dependents  = st.number_input('Dependent Count', min_value=0, max_value=10, value=1)
        mos_on_book = st.number_input('Months on Book', min_value=0, max_value=120, value=12)
        credit_lim  = st.number_input('Credit Limit', min_value=0.0, max_value=50000.0, value=10000.0)
        revolve_bal = st.number_input('Total Revolving Balance', min_value=0.0, max_value=5000.0, value=1000.0)
        open_to_buy = st.number_input('Avg Open To Buy', min_value=0.0, max_value=50000.0, value=9000.0)
        amt_chng    = st.number_input('Total Amt Change Q4/Q1', min_value=0.0, value=1.0)
        trans_amt   = st.number_input('Total Transaction Amount', min_value=0.0, value=1000.0)
        ct_chng     = st.number_input('Total Count Change Q4/Q1', min_value=0.0, value=1.0)
        util_ratio  = st.slider('Avg Utilization Ratio', min_value=0.0, max_value=1.0, value=0.3)

        # — Additional numeric inputs —
        rel_count       = st.number_input('Total Relationship Count', min_value=0, max_value=10, value=3)
        inactive_months = st.number_input('Months Inactive (Last 12 months)', min_value=0, max_value=12, value=2)
        contacts_count  = st.number_input('Contacts Count (Last 12 months)', min_value=0, max_value=20, value=3)
        trans_ct        = st.number_input('Total Transaction Count', min_value=0, max_value=200, value=50)

        # — Categorical inputs —
        gender    = st.selectbox('Gender', gender_options)
        education = st.selectbox('Education Level', education_options)
        marital   = st.selectbox('Marital Status', marital_options)
        income    = st.selectbox('Income Category', income_options)
        card      = st.selectbox('Card Category', card_options)

        submit = st.form_submit_button('Predict')

    if submit:
        # Build raw DataFrame
        input_df = pd.DataFrame({
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
            'card_category': [card],
            'total_relationship_count': [rel_count],
            'months_inactive_12_mon': [inactive_months],
            'contacts_count_12_mon': [contacts_count],
            'total_trans_ct': [trans_ct]
        })

        # Preprocess and reorder columns
        processed_df = preprocess(input_df.copy())
        processed_df = processed_df[model.feature_names_in_]

        # Predict
        prob = model.predict_proba(processed_df)[:, 1][0]
        pred = model.predict(processed_df)[0]

        # Display
        st.subheader('🔮 Prediction Results')
        st.write(f"**Churn Probability:** {prob:.2%}")
        st.write(f"**Predicted Label:** {pred}")

if __name__ == '__main__':
    main()
