import streamlit as st
import pandas as pd
import os
from huggingface_hub import hf_hub_download
import joblib

# Page configuration
st.set_page_config(
    page_title="Engine Health Monitor",
    page_icon="🔧",
    layout="centered"
)

# Download and load the model from Hugging Face Model Hub
@st.cache_resource
def load_model():
    try:
        USERNAME = os.getenv("YOUR_USERNAME")
        HF_TOKEN = os.getenv("HF_TOKEN")
        
        if not USERNAME:
            raise ValueError("YOUR_USERNAME environment variable not set")
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN environment variable not set")
        
        model_path = hf_hub_download(
            repo_id=f"{USERNAME}/car-engine-predictive-maintenence-model",
            filename="best_model.joblib",
            token=HF_TOKEN
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Header section
st.title("🔧 Engine Predictive Maintenance System")
st.markdown("Analyze engine sensor data to predict engine condition")

st.divider()

# Main input section
st.subheader("📝 Enter Engine Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    engine_rpm = st.number_input(
        "Engine RPM",
        min_value=0,
        max_value=3000,
        value=800,
        step=10,
        help="Engine revolutions per minute"
    )

    lub_oil_pressure = st.number_input(
        "Lubricating Oil Pressure (bar)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1,
        format="%.2f"
    )

    fuel_pressure = st.number_input(
        "Fuel Pressure (bar)",
        min_value=0.0,
        max_value=25.0,
        value=6.5,
        step=0.1,
        format="%.2f"
    )

with col2:
    coolant_pressure = st.number_input(
        "Coolant Pressure (bar)",
        min_value=0.0,
        max_value=10.0,
        value=2.5,
        step=0.1,
        format="%.2f"
    )

    lub_oil_temp = st.number_input(
        "Lubricating Oil Temperature (°C)",
        min_value=0.0,
        max_value=150.0,
        value=80.0,
        step=1.0,
        format="%.1f"
    )

    coolant_temp = st.number_input(
        "Coolant Temperature (°C)",
        min_value=0.0,
        max_value=200.0,
        value=80.0,
        step=1.0,
        format="%.1f"
    )

# Quick Presets
st.markdown("### 🎯 Quick Presets")
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🟢 Normal Operation", use_container_width=True):
        st.session_state.update({
            'engine_rpm': 800,
            'lub_oil_pressure': 3.5,
            'fuel_pressure': 6.5,
            'coolant_pressure': 2.5,
            'lub_oil_temp': 80.0,
            'coolant_temp': 80.0
        })
        st.rerun()

with col_btn2:
    if st.button("🔴 Faulty Scenario", use_container_width=True):
        st.session_state.update({
            'engine_rpm': 150,
            'lub_oil_pressure': 0.5,
            'fuel_pressure': 1.0,
            'coolant_pressure': 0.2,
            'lub_oil_temp': 88.0,
            'coolant_temp': 120.0
        })
        st.rerun()

st.divider()

# Create input dataframe
input_data = pd.DataFrame([{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': lub_oil_pressure,
    'Fuel pressure': fuel_pressure,
    'Coolant pressure': coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp
}])

# Prediction section
if st.button("🚀 Predict Engine Condition", type="primary", use_container_width=True):
    if model is not None:
        with st.spinner("Analyzing sensor data..."):
            try:
                # Set classification threshold (optimized for high recall)
                classification_threshold = 0.40

                # Get prediction probabilities
                prediction_proba = model.predict_proba(input_data)[0, 1]
                prediction = int(prediction_proba >= classification_threshold)

                # Display results
                st.divider()
                
                # Map prediction to engine condition
                # 0 = Normal, 1 = Faulty
                if prediction == 0:
                    st.success("### ✅ Engine Condition: NORMAL")
                    st.markdown("The engine is operating within normal parameters.")
                else:
                    st.error("### ⚠️ Engine Condition: FAULTY")
                    st.markdown("**Maintenance required immediately.**")

                # Show prediction details
                st.markdown("---")
                col_prob1, col_prob2 = st.columns(2)
                
                with col_prob1:
                    st.metric(
                        "Faulty Probability",
                        f"{prediction_proba*100:.1f}%"
                    )
                
                with col_prob2:
                    st.metric(
                        "Normal Probability",
                        f"{(1-prediction_proba)*100:.1f}%"
                    )

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    else:
        st.error("Model is not loaded. Please check the configuration.")

# Footer
st.divider()
st.caption("🔧 Engine Predictive Maintenance System | Powered by Machine Learning")
