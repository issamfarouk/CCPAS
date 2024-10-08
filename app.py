import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model (make sure to save your model after training)
model = joblib.load('E:/DEPI/big project/random_forest_model.pkl')

# Load the encoder for categorical features
encoders = joblib.load('E:/DEPI/big project/label_encoders.pkl')

# Function to preprocess input data
def preprocess_input(data):
    for col, encoder in encoders.items():
        if col in data:
            # Apply the label encoding for each categorical feature
            data[col] = encoder.transform([data[col]])[0]
    return data

# Streamlit UI components
st.title("Customer Churn Prediction")
st.write("Enter the customer details to predict churn:")

# User input for all relevant features
customer_data = {
    'gender': st.selectbox("Gender", options=['Male', 'Female']),
    'SeniorCitizen': st.selectbox("Senior Citizen", options=[0, 1]),
    'Partner': st.selectbox("Partner", options=['Yes', 'No']),
    'Dependents': st.selectbox("Dependents", options=['Yes', 'No']),
    'tenure': st.number_input("Tenure (in months)", min_value=0, max_value=100, value=1),
    'PhoneService': st.selectbox("Phone Service", options=['Yes', 'No']),
    'MultipleLines': st.selectbox("Multiple Lines", options=['Yes', 'No', 'No phone service']),
    'InternetService': st.selectbox("Internet Service", options=['DSL', 'Fiber optic', 'No']),
    'OnlineSecurity': st.selectbox("Online Security", options=['Yes', 'No', 'No internet service']),
    'OnlineBackup': st.selectbox("Online Backup", options=['Yes', 'No', 'No internet service']),
    'DeviceProtection': st.selectbox("Device Protection", options=['Yes', 'No', 'No internet service']),
    'TechSupport': st.selectbox("Tech Support", options=['Yes', 'No', 'No internet service']),
    'StreamingTV': st.selectbox("Streaming TV", options=['Yes', 'No', 'No internet service']),
    'StreamingMovies': st.selectbox("Streaming Movies", options=['Yes', 'No', 'No internet service']),
    'Contract': st.selectbox("Contract", options=['Month-to-month', 'One year', 'Two year']),
    'PaperlessBilling': st.selectbox("Paperless Billing", options=['Yes', 'No']),
    'PaymentMethod': st.selectbox("Payment Method", options=[
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ]),
    'MonthlyCharges': st.number_input("Monthly Charges", min_value=0.0, value=0.0),
    'TotalCharges': st.number_input("Total Charges", min_value=0.0, value=0.0),
}

# Convert to DataFrame
input_df = pd.DataFrame(customer_data, index=[0])

# Preprocess input data (encode categorical features)
input_df = preprocess_input(input_df)

# Make prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_df)
    st.write("Churn Prediction:", "Yes" if prediction[0] else "No")
