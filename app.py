import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model and extract feature names safely
model_bundle = joblib.load("green_taxi_fare_model.pkl")

if isinstance(model_bundle, tuple) and len(model_bundle) == 2:
    model, feature_names = model_bundle
elif hasattr(model_bundle, 'feature_names_in_'):
    model = model_bundle
    feature_names = list(model.feature_names_in_)
else:
    st.error("Model is missing required feature name metadata.")
    st.stop()

# Page Config
st.set_page_config(page_title="NYC Green Taxi Fare Predictor", layout="wide")

# Custom Styling for dark theme and visible selectbox
st.markdown("""
    <style>
    /* Main content styling */
    .main {
        background-color: #111827;
        color: white;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #10B981;
    }

    /* Button styling */
    .stButton>button {
        background-color: #10B981;
        color: white;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1f2937;
        color: white;
    }

    /* Sidebar label text */
    .stSidebar label {
        color: white;
    }

    /* Dropdown/selectbox text and background */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #374151;
        color: white;
        border: 1px solid #10B981;
        border-radius: 5px;
    }

    .stSelectbox div[data-baseweb="select"] * {
        color: white !important;
    }

    /* Dropdown arrow icon */
    .stSelectbox svg {
        fill: white;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
app_mode = st.sidebar.selectbox("Choose Application Mode", ["Prediction Tool", "Model Performance"])

# Prediction Tool UI
if app_mode == "Prediction Tool":
    st.title("🚖 NYC Green Taxi Trip Amount Predictor")
    st.info("This application predicts the total fare amount for a NYC green taxi trip.")

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
            trip_duration = 15  # Static for now; optionally compute from timestamps

            raw_input = {
                'trip_distance': trip_distance,
                'fare_amount': fare_amount,
                'extra': extra,
                'mta_tax': mta_tax,
                'tip_amount': tip_amount,
                'tolls_amount': tolls_amount,
                'improvement_surcharge': improvement_surcharge,
                'congestion_surcharge': congestion_surcharge,
                'trip_duration': trip_duration,
                'passenger_count': passenger_count
            }

            input_df = pd.DataFrame([raw_input])

            # Add missing features as zeros
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0.0

            # Reorder and ensure float dtype
            input_df = input_df[feature_names].astype(np.float64)

            try:
                prediction = model.predict(input_df)[0]
                st.success(f"💵 Estimated Total Fare: ${prediction:.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

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
