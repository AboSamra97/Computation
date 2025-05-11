import streamlit as st
import pandas as pd
import joblib

# Load encoders and pipeline
try:
    label_encoders = joblib.load('label_encoders.joblib')
    model = joblib.load('sklearn_pipeline.joblib')  # pipeline with scaler, PCA, SVC
except Exception as e:
    st.error(f"Failed to load saved artifacts: {e}")
    st.stop()

# Preprocessing function
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Encode categorical features
    for col, le in label_encoders.items():
        df[col] = le.transform(df[col])
    return df

# Main function to display the app
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

        # New required fields
        rel_count = st.number_input('Total Relationship Count', min_value=0, max_value=10, value=3)
        inactive_months = st.number_input('Months Inactive (Last 12 months)', min_value=0, max_value=12, value=2)
        contacts_count = st.number_input('Contacts Count (Last 12 months)', min_value=0, max_value=20, value=3)
        trans_ct = st.number_input('Total Transaction Count', min_value=0, max_value=200, value=50)

        # Categorical inputs using encoder classes
        gender = st.selectbox('Gender', gender_options)
        education = st.selectbox('Education Level', education_options)
        marital = st.selectbox('Marital Status', marital_options)
        income = st.selectbox('Income Category', income_options)
        card = st.selectbox('Card Category', card_options)

        submit = st.form_submit_button('Predict')

    if submit:
        # Prepare the input data for prediction
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

        # Preprocess the data
        processed_df = preprocess(input_df)

        # Print feature names of the model to compare with input data
        st.write("Model expected feature names:", model.named_steps['preprocessor'].get_feature_names_out())

        # Check the columns in the processed DataFrame
        st.write("Processed DataFrame columns:", processed_df.columns)

        # Get prediction probabilities and label
        try:
            prob = model.predict_proba(processed_df)
            st.write("Prediction Probabilities:", prob)

            # Handle binary classification or multi-class
            if len(prob.shape) > 1 and prob.shape[1] > 1:
                prob = prob[:, 1]  # Positive class probability
            else:
                prob = prob[:, 0]  # Class 0 probability (negative)

            # Make prediction
            pred = model.predict(processed_df)[0]
            st.write("Predicted Label:", pred)  # Check label output

            # Display results
            st.subheader('Prediction Results')
            st.write(f"**Churn Probability:** {prob:.2%}")
            st.write(f"**Predicted Label:** {pred}")
        except ValueError as e:
            st.error(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
