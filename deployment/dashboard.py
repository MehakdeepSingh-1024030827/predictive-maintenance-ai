import time
import random
import json
from datetime import datetime

def generate_sensor_data():
    """
    Simulates raw sensor readings from a CNC machine.
    Values are loosely based on the AI4I 2020 dataset distributions.
    """
    # 1. Base sensor values
    air_temp = random.uniform(295.0, 305.0)  # [K]
    process_temp = air_temp + random.uniform(10.0, 12.0)  # [K]
    rotational_speed = random.uniform(1300, 1600)  # [rpm]
    torque = random.uniform(30.0, 60.0)  # [Nm]
    tool_wear = random.uniform(0, 250)  # [min]
    
    # 2. Feature Engineering (Important: The model expects these 8 features)
    temp_diff = process_temp - air_temp
    stress_index = torque / (rotational_speed + 1e-5)
    torque_wear = torque * tool_wear
    
    # 3. Create the JSON-like dictionary
    data_point = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "Air temperature [K]": round(air_temp, 2),
        "Process temperature [K]": round(process_temp, 2),
        "Rotational speed [rpm]": round(rotational_speed, 2),
        "Torque [Nm]": round(torque, 2),
        "Tool wear [min]": round(tool_wear, 2),
        "temp_diff": round(temp_diff, 2),
        "stress_index": round(stress_index, 6),
        "torque_wear": round(torque_wear, 2)
    }
    return data_point

def stream_data():
    print("🚀 Sensor stream started... Press Ctrl+C to stop.")
    try:
        while True:
            # Generate the point
            reading = generate_sensor_data()
            
            # Print to console (simulating the 'send' action)
            print(f"📡 Sending Data: {json.dumps(reading)}")
            
            # This is where we will eventually add: 
            # requests.post('http://localhost:5000/predict', json=reading)
            
            # Wait for 1 second
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Simulator stopped.")

if __name__ == "__main__":
    stream_data()
