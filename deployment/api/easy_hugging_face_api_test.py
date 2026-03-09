import requests
import numpy as np
import time

API_URL = "https://black-hole-diver-har-deployment.hf.space/predict"

print(f"📡 Pinging cloud API at: {API_URL}")

# THE FIX: Generate exactly 100 timesteps by 9 sensor channels (a 2D array)
# This perfectly matches data: list[list[float]]
dummy_raw_data = np.random.rand(100, 9).tolist()

payload = {
    "data": dummy_raw_data
}

start_time = time.time()

try:
    response = requests.post(API_URL, json=payload, timeout=120)
    
    elapsed_time = round(time.time() - start_time, 2)
    response.raise_for_status() 
    
    result = response.json()
    print(f"\n✅ Success! Server responded in {elapsed_time} seconds.")
    print(f"Raw Activity: {result.get('raw_activity')}")
    print(f"Smoothed Activity: {result.get('smoothed_activity')}")
    
except requests.exceptions.RequestException as e:
    print(f"\n❌ Connection failed: {e}")
    if 'response' in locals() and response is not None:
        print(f"Server Details: {response.text}")
