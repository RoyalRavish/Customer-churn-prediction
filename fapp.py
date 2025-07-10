import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load external CSS file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# Set page configuration
st.set_page_config(page_title="Customer Churn ++++Prediction", layout="centered")

# Define input fields
st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict churn:")

# Create form for better user experience
with st.form(key="churn_form"):
    # Create two columns for inputs
    col1, col2 = st.columns(2)

    # Input fields
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1, key="tenure")
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="InternetService")
        OnlineSecurity = st.selectbox("Online Security", ["No internet service", "No", "Yes"], key="OnlineSecurity")
        TechSupport = st.selectbox("Tech Support", ["No internet service", "No", "Yes"], key="TechSupport")

    with col2:
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="Contract")
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key="PaymentMethod")
        MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=1000.0, value=0.0, step=0.01, key="MonthlyCharges")
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=0.0, step=0.01, key="TotalCharges")

    # Submit button
    submit_button = st.form_submit_button(label="Predict Churn")

# Prepare input for model
input_dict = {
    "tenure": tenure,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "TechSupport": TechSupport,
    "Contract": Contract,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

input_df = pd.DataFrame([input_dict])

# Encode categorical features
for col in encoders:
    if col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])

# Predict and display result
if submit_button:
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] if prediction == 1 else model.predict_proba(input_df)[0][0]
    if prediction == 1:
        st.markdown(f"<div class='prediction error'>The customer is likely to churn (Probability: {probability:.2%}).</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='prediction success'>The customer is not likely to churn (Probability: {probability:.2%}).</div>", unsafe_allow_html=True)

# Add footer
st.markdown("<div class='footer'>Built with Streamlit | Customer Churn Prediction Model</div>", unsafe_allow_html=True)