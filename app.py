"""Streamlit interface for the house-price prediction model."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/house_price_model.joblib")
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")
st.title("🏡 House Price Prediction System")
st.write("Enter property details to estimate a house price using a trained Linear Regression model.")
st.caption("Educational demo only. Predictions are not professional property valuations.")

if not MODEL_PATH.exists():
    st.warning("The trained model was not found. Train it first by running: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
with st.form("property_form"):
    area = st.number_input("Area (sq ft)", min_value=100, max_value=20000, value=1200, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
    age = st.number_input("Property age (years)", min_value=0, max_value=150, value=5, step=1)
    location = st.selectbox("Location", ["Hyderabad", "Bengaluru", "Pune", "Chennai"])
    submitted = st.form_submit_button("Predict house price")

if submitted:
    features = pd.DataFrame([{
        "area_sqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "location": location,
    }])
    estimate = float(model.predict(features)[0])
    st.subheader("Estimated price")
    st.success(f"₹{estimate:,.0f}")
    st.caption("This estimate depends on the training data and may not reflect current market prices.")
