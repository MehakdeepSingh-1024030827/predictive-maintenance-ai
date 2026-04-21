import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
from collections import deque

app = Flask(__name__)

# --- 1. LOAD SAVED ASSETS ---
# Ensure these files are in the same folder as this script
try:
    scaler = joblib.load('scaler.joblib')
    autoencoder = tf.keras.models.load_model('autoencoder_model.h5')
    lstm_model = tf.keras.models.load_model('best_lstm_attention_model_v2.keras')
    print("✅ System Ready: Models and Scaler loaded.")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

# --- 2. CONFIGURATION ---
# Replace this with the threshold you calculated in your 03_anomaly_and_model_design notebook
ANOMALY_THRESHOLD = 0.025  
SEQUENCE_LENGTH = 10
FEATURE_COUNT = 8

# Persistent buffer to store the last 10 scaled readings
# This lives in the server's memory while it's running
data_buffer = deque(maxlen=SEQUENCE_LENGTH)

# --- 3. REUSE PREPROCESSING LOGIC ---
def preprocess_sensor_data(raw_json):
    """
    Applies the exact Feature Engineering and Scaling from your EDA.
    """
    # Extract raw values
    air_temp = raw_json['Air temperature [K]']
    proc_temp = raw_json['Process temperature [K]']
    rpm = raw_json['Rotational speed [rpm]']
    torque = raw_json['Torque [Nm]']
    tool_wear = raw_json['Tool wear [min]']

    # Apply your Feature Engineering logic
    temp_diff = proc_temp - air_temp
    stress_index = torque / (rpm + 1e-5)
    torque_wear = torque * tool_wear

    # Order must match the scaler's training order exactly!
    feature_vector = [
        air_temp, proc_temp, rpm, torque, tool_wear, 
        temp_diff, stress_index, torque_wear
    ]

    # Return scaled 2D array: (1, 8)
    return scaler.transform([feature_vector])

# --- 4. PREDICTION ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get incoming data from simulator
        sensor_data = request.get_json()
        
        # A. Preprocess and Scale
        scaled_point = preprocess_sensor_data(sensor_data)
        
        # B. Run Anomaly Detection (Immediate 1-second check)
        reconstructed = autoencoder.predict(scaled_point, verbose=0)
        mse = np.mean(np.power(scaled_point - reconstructed, 2))
        is_anomaly = bool(mse > ANOMALY_THRESHOLD)
        
        # C. Run Predictive Maintenance (10-second trend check)
        # Add the current point to our 10-step buffer
        data_buffer.append(scaled_point[0])
        
        failure_prob = 0.0
        status = "Stabilizing Buffer..."
        
        # LSTM only runs if we have a full window of 10 steps
        if len(data_buffer) == SEQUENCE_LENGTH:
            # Reshape buffer to 3D for LSTM: (Samples, Time_Steps, Features)
            input_seq = np.array(data_buffer).reshape(1, SEQUENCE_LENGTH, FEATURE_COUNT)
            failure_prob = float(lstm_model.predict(input_seq, verbose=0)[0][0])
            status = "Monitoring Active"
        
        # D. Return Unified Response
        return jsonify({
            "timestamp": sensor_data.get('timestamp'),
            "status": status,
            "anomaly_flag": is_anomaly,
            "reconstruction_error": round(float(mse), 5),
            "failure_probability": round(failure_prob, 4),
            "alert_level": "CRITICAL" if failure_prob > 0.5 else "NOMINAL"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Running on local port 5000
    app.run(host='0.0.0.0', port=5000)
