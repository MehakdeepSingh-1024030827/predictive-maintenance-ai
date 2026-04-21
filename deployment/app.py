import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
from collections import deque

app = Flask(__name__)

# --- 1. LOAD ASSETS ---
try:
    scaler = joblib.load('scaler.joblib')
    autoencoder = tf.keras.models.load_model('autoencoder_model.h5')
    lstm_model = tf.keras.models.load_model('best_lstm_attention_model_v2.keras')
    print("✅ BACKEND LIVE: All models loaded.")
except Exception as e:
    print(f"❌ LOAD ERROR: {e}")

# --- 2. CONFIG ---
data_buffer = deque(maxlen=10)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 🛡️ THE FIX: Simple mapping to avoid KeyErrors
        # We look for simple keys sent by the new dashboard
        f1 = float(data.get('air_t', 0))
        f2 = float(data.get('proc_t', 0))
        f3 = float(data.get('rpm', 0))
        f4 = float(data.get('torque', 0))
        f5 = float(data.get('wear', 0))
        
        # Engineering
        f6 = f2 - f1
        f7 = f4 / (f3 + 1e-5)
        f8 = f4 * f5
        
        features = [f1, f2, f3, f4, f5, f6, f7, f8]
        scaled = scaler.transform([features])
        
        # Models
        reconstructed = autoencoder.predict(scaled, verbose=0)
        mse = np.mean(np.power(scaled - reconstructed, 2))
        
        data_buffer.append(scaled[0])
        prob = 0.0
        if len(data_buffer) == 10:
            seq = np.array(data_buffer).reshape(1, 10, 8)
            prob = float(lstm_model.predict(seq, verbose=0)[0][0])
            
        print(f"📥 Point Processed | Fail Prob: {prob:.2%}")
        
        return jsonify({
            "failure_probability": round(prob, 4),
            "anomaly_flag": bool(mse > 0.025)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)
