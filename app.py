import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model and feature list
model_bundle = joblib.load("green_taxi_fare_model.pkl")

# Unpack model and feature names
if isinstance(model_bundle, tuple):
    model, feature_names = model_bundle
else:
    model = model_bundle
    feature_names = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else []

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
            # Timestamp features
            pickup_datetime = datetime.combine(pickup_date, pickup_time)
            dropoff_datetime = pickup_datetime + pd.Timedelta(minutes=15)
            weekday = pickup_datetime.strftime("%A")
            hour = pickup_datetime.hour

            # Default/dummy values
            vendor_id = 1
            ratecode_id = 1.0
            payment_type = 1.0
            trip_type = 1.0
            store_and_fwd_flag_Y = 0

            # Base input dict
            raw_input = {
                'VendorID': vendor_id,
                'lpep_pickup_datetime': pickup_datetime,
                'lpep_dropoff_datetime': dropoff_datetime,
                'PULocationID': 130,
                'DOLocationID': 205,
                'passenger_count': passenger_count,
                'trip_distance': trip_distance,
                'fare_amount': fare_amount,
                'extra': extra,
                'mta_tax': mta_tax,
                'tip_amount': tip_amount,
                'tolls_amount': tolls_amount,
                'improvement_surcharge': improvement_surcharge,
                'congestion_surcharge': congestion_surcharge,
                'pickup_datetime': pickup_datetime,
                'dropoff_datetime': dropoff_datetime,
                'trip_duration': 15,
                'store_and_fwd_flag_Y': store_and_fwd_flag_Y
            }

            # One-hot encoding
            for col in feature_names:
                if "RatecodeID_" in col:
                    raw_input[col] = 1.0 if col == f"RatecodeID_{float(ratecode_id)}" else 0.0
                elif "payment_type_" in col:
                    raw_input[col] = 1.0 if col == f"payment_type_{float(payment_type)}" else 0.0
                elif "trip_type_" in col:
                    raw_input[col] = 1.0 if col == f"trip_type_{float(trip_type)}" else 0.0
                elif "weekday_" in col:
                    raw_input[col] = 1.0 if col == f"weekday_{weekday}" else 0.0
                elif "hourofday_" in col:
                    raw_input[col] = 1.0 if col == f"hourofday_{hour}" else 0.0
                elif col not in raw_input:
                    raw_input[col] = 0.0

            # Build input DataFrame
            input_df = pd.DataFrame([raw_input])[feature_names]

            # Predict
            prediction = model.predict(input_df)[0]
            st.success(f"Estimated Total Fare: ${prediction:.2f}")

# Model Performance UI
else:
    st.title("📊 Model Performance Analysis")

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
