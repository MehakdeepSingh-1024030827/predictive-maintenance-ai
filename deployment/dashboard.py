import time
import random
import json
from datetime import datetime

def generate_sensor_data():
    """
    Simulates raw sensor readings from a CNC machine.
    Ranges are based on the AI4I 2020 dataset distributions.
    """
    while True:
        # 1. Generate Raw Sensor Values
        air_temp = random.uniform(295.0, 305.0)       # Air temperature [K]
        process_temp = air_temp + random.uniform(10, 12) # Process temperature [K]
        rotational_speed = random.uniform(1300, 1600)   # Rotational speed [rpm]
        torque = random.uniform(30.0, 60.0)             # Torque [Nm]
        tool_wear = random.uniform(0, 250)              # Tool wear [min]
        
        # 2. Format as a JSON-compatible dictionary
        data_point = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Air temperature [K]": round(air_temp, 2),
            "Process temperature [K]": round(process_temp, 2),
            "Rotational speed [rpm]": round(rotational_speed, 1),
            "Torque [Nm]": round(torque, 2),
            "Tool wear [min]": round(tool_wear, 2)
        }
        
        # 3. Yield the data (Iteratable stream)
        yield data_point
        
        # 4. Heartbeat: 1-second interval
        time.sleep(1)

def run_simulator():
    print("🚀 Starting Production Sensor Stream (Press Ctrl+C to stop)...")
    print("Format: JSON Dictionary")
    print("-" * 50)
    
    try:
        for reading in generate_sensor_data():
            # In the next step, we will use 'requests' to send this to the Flask API
            print(json.dumps(reading, indent=4))
    except KeyboardInterrupt:
        print("\n🛑 Simulator shutdown gracefully.")

if __name__ == "__main__":
    run_simulator()
