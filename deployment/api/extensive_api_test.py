import os
import glob
import time
import random
import requests
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from src.utils import clean_class_name

from src.settings import Config, MAPPING, BLACKLIST

PREDICT_URL = "https://black-hole-diver-har-deployment.hf.space/predict"
RESET_URL = "https://black-hole-diver-har-deployment.hf.space/reset"
MOVEMENTS_DIR = Config.RAW_DATA_DIR

SENSOR_COLS = Config.SENSOR_FEATURES

FILES_PER_CLASS = 3
WINDOW_SIZE = 100
STEP_SIZE = 50

def run_extensive_api_test():
    print("🌍 Booting Extensive API Stress Test...")
    print(f"🎯 Target URL: {PREDICT_URL}\n")

    if not os.path.exists(MOVEMENTS_DIR):
        print(f"❌ Error: Could not find '{MOVEMENTS_DIR}'. Run this script from the project root!")
        return

    session = requests.Session()

    y_true = []
    y_pred_raw = []
    y_pred_smooth = []

    total_ping = 0
    total_requests = 0

    activity_folders = [f for f in os.listdir(MOVEMENTS_DIR) if os.path.isdir(os.path.join(MOVEMENTS_DIR, f))]
    activity_folders.sort()

    for activity in activity_folders:
        raw_label = clean_class_name(activity)
        if any(raw_label == b.value for b in BLACKLIST):
            print(f"⏭️ Skipping Blacklisted Class: {raw_label}")
            continue
        mapped_enum = MAPPING.get(raw_label, raw_label)
        true_label = mapped_enum.value if hasattr(mapped_enum, 'value') else str(mapped_enum)
        csv_files = glob.glob(os.path.join(MOVEMENTS_DIR, activity, "*.csv"))

        random.shuffle(csv_files)
        test_files = csv_files[:FILES_PER_CLASS]

        if not test_files:
            continue

        print(f"📂 Testing Class: {raw_label} (Mapped to: {true_label}) | {len(test_files)} files selected")

        for file_path in test_files:
            session.post(RESET_URL)

            df = pd.read_csv(file_path)

            if not all(col in df.columns for col in SENSOR_COLS):
                continue

            df[SENSOR_COLS] = df[SENSOR_COLS].interpolate(method='linear', limit=2).dropna()

            if len(df) < WINDOW_SIZE:
                continue

            for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
                end = start + WINDOW_SIZE
                window_data = df[SENSOR_COLS].iloc[start:end].values.tolist()

                try:
                    start_time = time.time()
                    response = session.post(PREDICT_URL, json={"data": window_data}, timeout=10)
                    response.raise_for_status()

                    ping = (time.time() - start_time) * 1000
                    total_ping += ping
                    total_requests += 1

                    result = response.json()

                    y_true.append(true_label)
                    y_pred_raw.append(result.get("raw_activity", "Unknown"))
                    y_pred_smooth.append(result.get("smoothed_activity", "Unknown"))

                except Exception as e:
                    print(f"⚠️ API Request failed on {file_path}: {e}")
                    continue

    if total_requests == 0:
        print("\n❌ No requests were successfully completed.")
        return

    print("\n" + "="*50)
    print("🏆 ELITE API TEST RESULTS")
    print("="*50)

    avg_ping = total_ping / total_requests
    print(f"📡 Total Windows Processed : {total_requests}")
    print(f"⚡ Average Network Latency : {avg_ping:.2f} ms per request\n")

    raw_acc = accuracy_score(y_true, y_pred_raw)
    smooth_acc = accuracy_score(y_true, y_pred_smooth)

    raw_f1 = f1_score(y_true, y_pred_raw, average='weighted')
    smooth_f1 = f1_score(y_true, y_pred_smooth, average='weighted')

    print(f"🧠 Raw AI Stream Accuracy : {raw_acc * 100:.2f}%")
    print(f"🌊 Smoothed Live Accuracy : {smooth_acc * 100:.2f}%\n")

    print(f"📊 Raw Weighted F1        : {raw_f1:.4f}")
    print(f"🎖️ Smoothed Weighted F1   : {smooth_f1:.4f}")
    print("="*50)

    print("\n📑 DETAILED CLASSIFICATION REPORT (Smoothed)")
    print(classification_report(y_true, y_pred_smooth, zero_division=0))

if __name__ == "__main__":
    run_extensive_api_test()