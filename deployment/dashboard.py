import streamlit as st
import requests
import time
import json
import pandas as pd
from datetime import datetime
import random

# --- 1. SENSOR SIMULATOR LOGIC (Reusing your structure) ---
def generate_sensor_data():
    """
    Simulates raw sensor readings from a CNC machine.
    """
    while True:
        air_temp = random.uniform(295.0, 305.0)
        process_temp = air_temp + random.uniform(10, 12)
        rotational_speed = random.uniform(1300, 1600)
        torque = random.uniform(30.0, 60.0)
        tool_wear = random.uniform(0, 250)
        
        data_point = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Air temperature [K]": round(air_temp, 2),
            "Process temperature [K]": round(process_temp, 2),
            "Rotational speed [rpm]": round(rotational_speed, 1),
            "Torque [Nm]": round(torque, 2),
            "Tool wear [min]": round(tool_wear, 2)
        }
        yield data_point
        time.sleep(1)

# --- 2. STREAMLIT UI SETUP ---
st.set_page_config(page_title="Machine Health Monitor", layout="wide")
st.title("🏭 CNC Machine Predictive Maintenance")

# Flask API URL
API_URL = "http://localhost:5000/predict"

# Layout: Metrics at the top
col1, col2, col3 = st.columns(3)
prob_metric = col1.empty()
anomaly_metric = col2.empty()
alert_placeholder = col3.empty()

# Layout: Data visualization below
st.subheader("Live Sensor Data Stream")
data_placeholder = st.empty()
chart_placeholder = st.empty()

# Persistent history for plotting
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = pd.DataFrame(columns=['Time', 'Prob'])

# --- 3. THE PIPELINE EXECUTION ---
def run_dashboard():
    try:
        # Start iterating through your simulator generator
        for raw_data in generate_sensor_data():
            
            # A. CALL FLASK API
            try:
                response = requests.post(API_URL, json=raw_data)
                result = response.json()
            except Exception as e:
                st.error(f"Waiting for Flask API... Ensure app.py is running on port 5000.")
                continue

            # B. EXTRACT RESULTS
            prob = result.get("failure_probability", 0.0)
            is_anomaly = result.get("anomaly_flag", False)
            status_msg = result.get("pipeline_status", "Init")
            
            # C. UPDATE METRICS
            prob_metric.metric("Failure Probability", f"{prob*100:.2f}%", delta_color="inverse")
            
            anomaly_text = "⚠️ ANOMALY" if is_anomaly else "✅ NORMAL"
            anomaly_metric.metric("Anomaly Status", anomaly_text)

            # D. SHOW ALERT MESSAGE
            if prob > 0.5:
                alert_placeholder.error("🚨 Maintenance Needed")
            else:
                alert_placeholder.success("✅ System Healthy")

            # E. UPDATE TABLE & CHART
            # Show the raw sensor data in a clean table
            data_placeholder.table(pd.DataFrame([raw_data]).set_index('timestamp'))

            # Update history for the line chart
            new_row = pd.DataFrame({'Time': [raw_data['timestamp']], 'Prob': [prob]})
            st.session_state.chart_data = pd.concat([st.session_state.chart_data, new_row]).tail(20)
            chart_placeholder.line_chart(st.session_state.chart_data.set_index('Time'))

    except KeyboardInterrupt:
        st.write("Stopped")

if __name__ == "__main__":
    run_dashboard()
