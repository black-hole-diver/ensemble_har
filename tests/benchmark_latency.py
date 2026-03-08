from src.settings import Config, BLACKLIST, FileNames
from src.utils import extract_physics

import os
import glob
import time
import random
import numpy as np
import pandas as pd
import joblib
from collections import deque
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

class LatencyBenchmarker:
    def __init__(self):
        print("Loading Elite Ensemble for Latency Testing...")
        model_path = os.path.join(Config.MODELS_DIR, FileNames.MODEL_NAME)
        scaler_path = os.path.join(Config.MODELS_DIR, FileNames.SCALER_NAME)
        labels_path = os.path.join(Config.MODELS_DIR, FileNames.LABELS_NAME)
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(labels_path)
        self.prediction_buffer = deque(maxlen=3)
        self.blacklist = BLACKLIST

    def predict_pipeline(self, window_data):
        """Runs the exact pipeline used in the live API.
        Feature extraction time accounted. Return latency in millisecond"""

        t0 = time.perf_counter()
        features = extract_physics(window_data).reshape(1, -1)

        t1 = time.perf_counter()
        scaled = self.scaler.transform(features)
        pred_encoded = self.model.predict(scaled)
        raw_pred = self.label_encoder.inverse_transform(pred_encoded)[0]

        t2 = time.perf_counter()
        self.prediction_buffer.append(raw_pred)

        t3 = time.perf_counter()

        extract_ms = (t1 - t0) * 1000
        predict_ms = (t2 - t1) * 1000
        smooth_ms = (t3 - t2) * 1000
        total_ms = (t3 - t0) * 1000

        return extract_ms, predict_ms, smooth_ms, total_ms

    def get_real_windows(self, num_windows=1000):
        """Extracts real 2-second windows from your raw CSVs to test realistic math loads.
        Keep grabbing files until we hit window target, grab a random slice"""
        all_csvs = glob.glob(os.path.join(Config.RAW_DATA_DIR, '**', '*.csv'), recursive=True)
        valid_csvs = [f for f in all_csvs if os.path.basename(os.path.dirname(f)) not in self.blacklist]

        windows = []
        win_size = int(Config.SAMPLING_RATE_HZ * Config.WINDOW_SEC)
        while len(windows) < num_windows:
            file = random.choice(valid_csvs)
            df = pd.read_csv(file)
            if len(df) >= win_size and all(col in df.columns for col in Config.SENSOR_FEATURES):
                df = df[Config.SENSOR_FEATURES].interpolate().dropna()
                start = random.randint(0, len(df) - win_size)
                windows.append(df.iloc[start:start+win_size].values)
        return windows

    def run_benchmark(self, num_windows=1000):
        """ML models are lazy-loaded. The first few predictions are always artificially slow."""
        print("\n--- Generating Real Test Data ---")
        test_windows = self.get_real_windows(num_windows)
        print(f"Loaded {len(test_windows)} real sensor windows.")
        print("\n--- Warming Up Model (10 Iterations) ---")

        for i in range(10):
            self.predict_pipeline(test_windows[i])

        print("\n--- Running Latency Benchmark ---")
        total_times = []
        extract_times = []
        predict_times = []

        for window in test_windows:
            ext, pred, _, tot = self.predict_pipeline(window)
            extract_times.append(ext)
            predict_times.append(pred)
            total_times.append(tot)

        best = np.min(total_times)
        worst = np.max(total_times)
        avg = np.mean(total_times)
        p99 = np.percentile(total_times, 99) # 99th percentile: realistic worst-case

        avg_ext = np.mean(extract_times)
        avg_pred = np.mean(predict_times)

        print("=" * 45)
        print(f"INFERENCE LATENCY REPORT ({num_windows} Windows)")
        print("=" * 45)
        print(f"Best (Min) Time:     {best:.2f} ms")
        print(f"Average Time:        {avg:.2f} ms")
        print(f"99th Percentile:     {p99:.2f} ms")
        print(f"Worst (Max) Time:    {worst:.2f} ms")
        print("-" * 45)
        print("BREAKDOWN (Average):")
        print(f" - Feature Math:     {avg_ext:.2f} ms")
        print(f" - ML Ensemble:      {avg_pred:.2f} ms")
        print(f" - Smoothing:        {np.mean([t - ext - pred for t, ext, pred in zip(total_times, extract_times, predict_times)]):.3f} ms")
        print("=" * 45)

        print("\nDEVICE DEPLOYMENT CHECK:")
        if worst < 100:
            print("✅ EXCELLENT: Peak latency is well under 100ms. Will run flawlessly on wearable APIs.")
        elif worst < 500:
            print("⚠️ ACCEPTABLE: Latency is fine, but might cause minor API blocking under heavy load.")
        else:
            print("❌ DANGER: Worst-case latency is too high. You need to drop Random Forest or reduce features.")

if __name__ == "__main__":
    """Test 1k real sensor windows for a statistically sound benchmark."""
    benchmarker = LatencyBenchmarker()
    benchmarker.run_benchmark(num_windows=1000)