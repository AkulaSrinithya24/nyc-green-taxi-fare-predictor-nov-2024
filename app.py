
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model
model = joblib.load("model.pkl")

# Page Config
st.set_page_config(page_title="NYC Green Taxi Fare Predictor", layout="wide")

# Sidebar Navigation
app_mode = st.sidebar.selectbox("Choose Application Mode", ["Prediction Tool", "Model Performance"])

# Custom Styling
st.markdown("""
    <style>
    .main {background-color: #111827; color: white;}
    h1, h2, h3, h4, h5, h6 {color: #10B981;}
    .stButton>button {background-color: #10B981; color: white;}
    .stSelectbox>div>div {color: black;}
    </style>
""", unsafe_allow_html=True)

# Sample Input Features (match training)
features = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
            'tolls_amount', 'improvement_surcharge', 'congestion_surcharge',
            'trip_duration', 'passenger_count']

# UI for Prediction Tool
if app_mode == "Prediction Tool":
    st.title("🚖 NYC Green Taxi Trip Amount Predictor")
    st.info("This application predicts the total amount for a NYC green taxi trip based on various trip features.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            pickup_date = st.date_input("Pickup Date")
            pickup_time = st.time_input("Pickup Time")
            trip_distance = st.number_input("Trip Distance (miles)", value=2.5)
            fare_amount = st.number_input("Fare Amount ($)", value=10.0)
        with col2:
            mta_tax = st.number_input("MTA Tax ($)", value=0.5)
            tolls_amount = st.number_input("Tolls Amount ($)", value=0.0)
            tip_amount = st.number_input("Tip Amount ($)", value=2.0)
            passenger_count = st.slider("Number of Passengers", 1, 6, 1)
        with col3:
            extra = st.number_input("Extra Amount ($)", value=0.0)
            improvement_surcharge = st.number_input("Improvement Surcharge ($)", value=0.3)
            congestion_surcharge = st.number_input("Congestion Surcharge ($)", value=2.5)

        submitted = st.form_submit_button("Predict Fare")
        
        if submitted:
            trip_duration = 15  # Assume 15 mins (or calculate from timestamps if available)
            input_data = pd.DataFrame([[
                trip_distance, fare_amount, extra, mta_tax, tip_amount,
                tolls_amount, improvement_surcharge, congestion_surcharge,
                trip_duration, passenger_count
            ]], columns=features)
            
            prediction = model.predict(input_data)[0]
            st.success(f"Estimated Total Fare: ${prediction:.2f}")

# Model Performance UI
else:
    st.title("📊 Model Performance Analysis")

    # Example static metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Squared Error", "24.59")
    col2.metric("RMSE", "4.96")
    col3.metric("MAE", "3.89")
    col4.metric("R² Score", "0.90")

    st.subheader("Feature Importance")
    st.bar_chart({
        "Importance": {
            "payment_type": 0.14,
            "RatecodeID": 0.13,
            "mta_tax": 0.12,
            "month": 0.11,
            "trip_type": 0.10,
            "tip_amount": 0.09,
            "day_of_week": 0.08,
            "hour_of_day": 0.07,
            "trip_distance": 0.06
        }
    })
