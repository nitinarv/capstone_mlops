import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Engine Health Monitor",
    page_icon="🔧",
    layout="wide"
)

# Download and load the model from Hugging Face Model Hub
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id=f"{os.getenv('YOUR_USERNAME')}/car-engine-predictive-maintenence-model",
            filename="best_model.joblib"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Header section
st.title("🔧 Engine Predictive Maintenance System")
st.markdown("""
**Real-time engine health monitoring for fleet operators and maintenance teams**

This tool analyzes engine sensor data to predict potential failures before they occur,
enabling proactive maintenance and preventing costly breakdowns.
""")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **Purpose:** Predict engine failure risk based on real-time sensor readings

    **Target Users:**
    - Fleet managers
    - Maintenance technicians
    - Service center operators

    **Model Performance:**
    - Recall: 89.8%
    - F1-Score: 77.0%
    - Catches 9 out of 10 failures
    """)

    st.header("📊 Risk Levels")
    st.success("✅ **Normal**: Engine operating normally")
    st.error("⚠️ **Faulty**: Maintenance required")

# Main input section
st.header("📝 Enter Engine Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Speed & Performance")
    engine_rpm = st.number_input(
        "Engine RPM",
        min_value=0,
        max_value=3000,
        value=800,
        step=10,
        help="Engine revolutions per minute (typical range: 600-2000 RPM)"
    )

    st.subheader("Pressure Readings (bar)")
    lub_oil_pressure = st.number_input(
        "Lubricating Oil Pressure",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1,
        format="%.2f",
        help="Oil pressure for engine lubrication (typical: 2-6 bar)"
    )

    fuel_pressure = st.number_input(
        "Fuel Pressure",
        min_value=0.0,
        max_value=25.0,
        value=6.5,
        step=0.1,
        format="%.2f",
        help="Fuel supply pressure (typical: 3-12 bar)"
    )

    coolant_pressure = st.number_input(
        "Coolant Pressure",
        min_value=0.0,
        max_value=10.0,
        value=2.5,
        step=0.1,
        format="%.2f",
        help="Coolant system pressure (typical: 1-5 bar)"
    )

with col2:
    st.subheader("Temperature Readings (°C)")
    lub_oil_temp = st.number_input(
        "Lubricating Oil Temperature",
        min_value=0.0,
        max_value=150.0,
        value=80.0,
        step=1.0,
        format="%.1f",
        help="Oil temperature (typical: 70-90°C)"
    )

    coolant_temp = st.number_input(
        "Coolant Temperature",
        min_value=0.0,
        max_value=200.0,
        value=80.0,
        step=1.0,
        format="%.1f",
        help="Engine coolant temperature (typical: 75-95°C)"
    )

    st.subheader("Quick Presets")
    if st.button("🟢 Normal Operation"):
        st.session_state.update({
            'engine_rpm': 800,
            'lub_oil_pressure': 3.5,
            'fuel_pressure': 6.5,
            'coolant_pressure': 2.5,
            'lub_oil_temp': 80.0,
            'coolant_temp': 80.0
        })
        st.rerun()

    if st.button("🔴 Failure Scenario"):
        st.session_state.update({
            'engine_rpm': 150,
            'lub_oil_pressure': 0.5,
            'fuel_pressure': 1.0,
            'coolant_pressure': 0.2,
            'lub_oil_temp': 88.0,
            'coolant_temp': 120.0
        })
        st.rerun()

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
st.header("🔍 Engine Health Analysis")

if st.button("🚀 Analyze Engine Health", type="primary", use_container_width=True):
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

                col_result1, col_result2, col_result3 = st.columns([2, 1, 1])

                with col_result1:
                    if prediction == 0:
                        st.success("### ✅ Engine Status: NORMAL")
                        st.markdown("**The engine is operating within normal parameters.**")
                    else:
                        st.error("### ⚠️ Engine Status: MAINTENANCE REQUIRED")
                        st.markdown("**Potential failure detected. Schedule maintenance immediately.**")

                with col_result2:
                    st.metric(
                        "Failure Risk",
                        f"{prediction_proba*100:.1f}%",
                        delta=f"{'High' if prediction_proba > 0.7 else 'Moderate' if prediction_proba > 0.4 else 'Low'} Risk"
                    )

                with col_result3:
                    st.metric(
                        "Confidence",
                        f"{max(prediction_proba, 1-prediction_proba)*100:.1f}%"
                    )

                # Detailed analysis
                st.subheader("📋 Detailed Analysis")

                # Check for anomalies
                anomalies = []
                if engine_rpm < 200:
                    anomalies.append("⚠️ **Critical RPM**: Engine RPM dangerously low (stalling risk)")
                if engine_rpm > 2000:
                    anomalies.append("⚠️ **High RPM**: Engine over-revving (wear risk)")
                if lub_oil_pressure < 1.0:
                    anomalies.append("⚠️ **Low Oil Pressure**: Risk of engine damage")
                if fuel_pressure < 2.0:
                    anomalies.append("⚠️ **Low Fuel Pressure**: Combustion issues possible")
                if coolant_temp > 100:
                    anomalies.append("🔥 **Overheating**: Coolant temperature critical")
                if lub_oil_temp > 90:
                    anomalies.append("🔥 **High Oil Temperature**: Reduced lubrication efficiency")

                if anomalies:
                    st.warning("**Sensor Anomalies Detected:**")
                    for anomaly in anomalies:
                        st.markdown(anomaly)
                else:
                    st.info("✓ All sensor readings within normal range")

                # Recommendations
                st.subheader("💡 Recommendations")
                if prediction == 1:
                    st.markdown("""
                    **Immediate Actions:**
                    1. 🛑 Schedule maintenance inspection within 24 hours
                    2. 📊 Monitor engine closely for deteriorating conditions
                    3. 🔍 Check sensor readings that triggered the alert
                    4. 📝 Document current operating conditions
                    5. 🚨 Consider temporary load reduction
                    """)
                else:
                    st.markdown("""
                    **Preventive Measures:**
                    - ✅ Continue regular monitoring schedule
                    - 🔧 Perform routine maintenance as planned
                    - 📊 Log sensor data for trend analysis
                    """)

                # Show input summary
                with st.expander("📊 View Input Summary"):
                    st.dataframe(input_data, use_container_width=True)

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("⚠️ Model not loaded. Please check your Hugging Face configuration.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <small>
    🔧 Engine Predictive Maintenance System | Powered by Machine Learning<br>
    Model: Bagging Classifier | Recall: 89.8% | F1-Score: 77.0%
    </small>
</div>
""", unsafe_allow_html=True)
