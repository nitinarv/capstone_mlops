import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib


USERNAME = os.getenv("YOUR_USERNAME")
HF_TOKEN = os.getenv("HF_TOKEN")

if not USERNAME:
    raise ValueError("YOUR_USERNAME environment variable not set")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")
        

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id=f"{USERNAME}/car-engine-predictive-maintenence-model", filename="best_model.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Engine Failure Prediction
st.title("Engine Failure Prediction App")
st.write("The Engine Failure Prediction App is an internal tool for the automotive company staff that predicts whether an engine is likely to fail based on sensor data.")
st.write("Kindly enter the engine sensor readings to check the engine condition.")

# Collect user input
engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=3000, value=800, step=10, help="Engine revolutions per minute")
lub_oil_pressure = st.number_input("Lubricating Oil Pressure (bar)", min_value=0.0, max_value=10.0, value=3.5, step=0.1, format="%.2f")
fuel_pressure = st.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=25.0, value=6.5, step=0.1, format="%.2f")
coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=10.0, value=2.5, step=0.1, format="%.2f")
lub_oil_temp = st.number_input("Lubricating Oil Temperature (°C)", min_value=0.0, max_value=150.0, value=80.0, step=1.0, format="%.1f")
coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=0.0, max_value=200.0, value=80.0, step=1.0, format="%.1f")
# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': lub_oil_pressure,
    'Fuel pressure': fuel_pressure,
    'Coolant pressure': coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "likely to fail" if prediction == 1 else "likely to operate normally"
    st.write(f"Based on the information provided, the engine is {result}.")