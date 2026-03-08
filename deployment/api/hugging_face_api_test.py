import os
import time
import requests
import pandas as pd
from src.settings import Config, SensorChannel

PREDICT_URL = "https://black-hole-diver-har-deployment.hf.space/predict"
RESET_URL = "https://black-hole-diver-har-deployment.hf.space/reset"

SENSOR_COLS = [
    SensorChannel.UACC_X,
    SensorChannel.UACC_Y,
    SensorChannel.UACC_Z,
    SensorChannel.GYR_X,
    SensorChannel.GYR_Y,
    SensorChannel.GYR_Z,
    SensorChannel.GRAV_X,
    SensorChannel.GRAV_Y,
    SensorChannel.GRAV_Z
]

def simulate_watch_stream(csv_file_path):
    print(f"📂 Loading local sensor data from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)

    df[SENSOR_COLS] = df[SENSOR_COLS].interpolate(method='linear', limit=2).dropna()

    window_size = 100
    step_size = 50

    print("\n🧹 Resetting the cloud model's smoothing buffer...")
    requests.post(RESET_URL)

    print("🚀 Starting real-time Watch simulation...\n")

    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window_df = df[SENSOR_COLS].iloc[start:end]

        payload_data = window_df.values.tolist()

        try:
            start_time = time.time()
            response = requests.post(PREDICT_URL, json={"data": payload_data}, timeout=10)
            response.raise_for_status()

            result = response.json()
            raw = result.get("raw_activity", "Unknown")
            smooth = result.get("smoothed_activity", "Unknown")

            ping = round((time.time() - start_time) * 1000)

            print(f"⏱️ Window [{start:04d}:{end:04d}] | 📶 {ping}ms | 🧠 Raw: {raw:<15} | 🌊 Smoothed: {smooth}")

            time.sleep(1.0)

        except requests.exceptions.RequestException as e:
            print(f"\n❌ Connection failed: {e}")
            if 'response' in locals() and response is not None:
                print(f"Server Detail: {response.text}")
            break

    print("\n✅ Streaming complete!")

if __name__ == "__main__":
    target_csv = Config.RAW_DATA_DIR + "/Seal/Seal_1.csv"
    if not os.path.exists(target_csv):
        print(f"⚠️ File not found: {target_csv}")
    else:
        simulate_watch_stream(target_csv)