import streamlit as st
import requests
import time
import pandas as pd
import numpy as np

st.set_page_config(page_title="Production Monitor", layout="wide")
st.title("🏭 Machine Health Live")

if 'hist' not in st.session_state:
    st.session_state.hist = pd.DataFrame(columns=['Time', 'Prob'])

placeholder = st.empty()

while True:
    # Generate data with SIMPLE KEYS
    val_air = np.random.uniform(295, 305)
    payload = {
        "air_t": round(val_air, 2),
        "proc_t": round(val_air + 11, 2),
        "rpm": round(np.random.uniform(1300, 1600), 1),
        "torque": round(np.random.uniform(30, 60), 2),
        "wear": round(np.random.uniform(0, 250), 2)
    }

    try:
        r = requests.post("http://localhost:5000/predict", json=payload)
        res = r.json()
        
        p = res['failure_probability']
        new_row = pd.DataFrame({'Time': [time.strftime("%H:%M:%S")], 'Prob': [p]})
        st.session_state.hist = pd.concat([st.session_state.hist, new_row]).tail(20)
        
        with placeholder.container():
            c1, c2 = st.columns(2)
            c1.metric("Failure Probability", f"{p:.2%}")
            c2.line_chart(st.session_state.hist.set_index('Time'))
            
            if p > 0.5:
                st.error("🚨 MAINTENANCE NEEDED NOW")
            else:
                st.success("✅ Machine Operating Normally")
                
    except:
        st.warning("Connecting to pipeline...")
        
    time.sleep(1)
