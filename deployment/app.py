import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
from collections import deque

app = Flask(__name__)

# --- 1. LOAD SAVED ASSETS ---
try:
    scaler = joblib.load('scaler.joblib')
    autoencoder = tf.keras.models.load_model('autoencoder_model.h5')
    lstm_model = tf.keras.models.load_model('best_lstm_attention_model_v2.keras')
    print("✅ Pipeline Ready: Assets Loaded.")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

# --- 2. CONFIGURATION ---
ANOMALY_THRESHOLD = 0.025  
SEQUENCE_LENGTH = 10
FEATURE_COUNT = 8
data_buffer = deque(maxlen=SEQUENCE_LENGTH)

# --- 3. HELPER FUNCTIONS FOR PIPELINE ---

def run_feature_engineering(raw_json):
    """Step 3: Calculate new features from raw input"""
    air_temp = raw_json['Air temperature [K]']
    proc_temp = raw_json['Process temperature [K]']
    rpm = raw_json['Rotational speed [rpm]']
    torque = raw_json['Torque [Nm]']
    tool_wear = raw_json['Tool wear [min]']

    # Engineering logic
    temp_diff = proc_temp - air_temp
    stress_index = torque / (rpm + 1e-5)
    torque_wear = torque * tool_wear

    return [air_temp, proc_temp, rpm, torque, tool_wear, temp_diff, stress_index, torque_wear]

def run_preprocessing(feature_vector):
    """Step 2: Scale the engineered features"""
    # Returns a scaled 2D array: (1, 8)
    return scaler.transform([feature_vector])


# --- 4. THE PIPELINE ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. INPUT: Get raw JSON from simulator
        sensor_data = request.get_json()
        
        # 2. FEATURE ENGINEERING: Calculate derived metrics
        engineered_features = run_feature_engineering(sensor_data)
        
        # 3. PREPROCESS: Scale the data for the models
        scaled_point = run_preprocessing(engineered_features)
        
        # 4. SEQUENCE: Update the 10-step sliding window buffer
        data_buffer.append(scaled_point[0])
        
        # 5. ANOMALY DETECTION: Run the Autoencoder check
        reconstructed = autoencoder.predict(scaled_point, verbose=0)
        mse = np.mean(np.power(scaled_point - reconstructed, 2))
        is_anomaly = bool(mse > ANOMALY_THRESHOLD)
        
        # 6. LSTM PREDICTION: Run the failure probability check
        failure_prob = 0.0
        status = "Stabilizing Buffer..."
        
        if len(data_buffer) == SEQUENCE_LENGTH:
            # Reshape buffer to 3D for LSTM: (1 Sample, 10 Time_Steps, 8 Features)
            input_seq = np.array(data_buffer).reshape(1, SEQUENCE_LENGTH, FEATURE_COUNT)
            failure_prob = float(lstm_model.predict(input_seq, verbose=0)[0][0])
            status = "Monitoring Active"
        
        # 7. RETURN RESULT: Return unified JSON response
        return jsonify({
            "timestamp": sensor_data.get('timestamp'),
            "pipeline_status": status,
            "anomaly_detected": is_anomaly,
            "reconstruction_error": round(float(mse), 6),
            "failure_probability": round(failure_prob, 4),
            "alert_level": "CRITICAL" if failure_prob > 0.5 else "NOMINAL",
            "message": "Maintenance Needed" if failure_prob > 0.5 else "System Healthy"
        })

    except Exception as e:
        return jsonify({"pipeline_error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
