from src.settings import Config, BLACKLIST, MAPPING, FileNames
from src.utils import extract_physics

import os
import random
import glob
import pandas as pd
import joblib
from collections import Counter, deque

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

class LiveSmoothedPredictor:
    def __init__(self, buffer_size=3):
        print("Loading Elite Ensemble and pre-processors...")

        model_path = os.path.join(Config.MODELS_DIR, FileNames.MODEL_NAME)
        scaler_path = os.path.join(Config.MODELS_DIR, FileNames.SCALER_NAME)
        labels_path = os.path.join(Config.MODELS_DIR, FileNames.LABELS_NAME)

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(labels_path)
        self.prediction_buffer = deque(maxlen=buffer_size)

    def predict_live_window(self, window_data):
        """Returns BOTH the raw prediction and the smoothed prediction for analysis."""
        features = extract_physics(window_data).reshape(1, -1)
        scaled = self.scaler.transform(features)

        pred_encoded = self.model.predict(scaled)
        raw_prediction = self.label_encoder.inverse_transform(pred_encoded)[0]

        self.prediction_buffer.append(raw_prediction)

        if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
            smoothed_prediction = Counter(self.prediction_buffer).most_common(1)[0][0]
        else:
            smoothed_prediction = raw_prediction

        return raw_prediction, smoothed_prediction

    def clear_buffer(self):
        """Clears the temporal buffer between testing different files."""
        self.prediction_buffer.clear()

class EliteRandomTester:
    def __init__(self):
        self.predictor = LiveSmoothedPredictor(buffer_size=3)
        self.mapping = MAPPING
        self.blacklist = BLACKLIST

    def get_random_files(self, num_files=20):
        all_csvs = glob.glob(os.path.join(Config.RAW_DATA_DIR, '**', '*.csv'), recursive=True)
        # Remove blacklisted directories
        valid_csvs = [f for f in all_csvs if os.path.basename(os.path.dirname(f)) not in self.blacklist]
        return random.sample(valid_csvs, min(num_files, len(valid_csvs)))

    def test_file_stream(self, file_path):
        """Simulates a continuous data stream from a single file.
        Always reset buffer for new stream"""
        self.predictor.clear_buffer()

        raw_label = os.path.basename(os.path.dirname(file_path))
        true_label = self.mapping.get(raw_label, raw_label)

        df = pd.read_csv(file_path)
        if not all(col in df.columns for col in Config.SENSOR_FEATURES):
            return None

        df = df[Config.SENSOR_FEATURES].interpolate().dropna()
        win_size = int(Config.SAMPLING_RATE_HZ * Config.WINDOW_SEC)
        step = int(win_size * 0.5) # 50% overlap for fluid streaming

        if len(df) < win_size:
            return None

        raw_correct = 0
        smooth_correct = 0
        total_windows = 0

        # Simulate the live in the CSV
        for i in range(0, len(df) - win_size + 1, step):
            window_data = df.iloc[i:i+win_size].values
            raw_pred, smooth_pred = self.predictor.predict_live_window(window_data)

            if raw_pred == true_label:
                raw_correct += 1
            if smooth_pred == true_label:
                smooth_correct += 1
            total_windows += 1

        return {
            'true_label': true_label,
            'windows': total_windows,
            'raw_correct': raw_correct,
            'smooth_correct': smooth_correct
        }

    def run_extensive_test(self, num_files=50):
        print(f"\n--- Starting Extensive Random Test ({num_files} files) ---")
        test_files = self.get_random_files(num_files)

        total_windows = 0
        total_raw_hits = 0
        total_smooth_hits = 0

        print("\n{:<25} | {:<10} | {:<10} | {:<10}".format("True Activity", "Windows", "Raw Acc", "Smooth Acc"))
        print("-" * 65)

        for file in test_files:
            results = self.test_file_stream(file)
            if not results:
                continue

            raw_acc = (results['raw_correct'] / results['windows']) * 100
            smooth_acc = (results['smooth_correct'] / results['windows']) * 100

            total_windows += results['windows']
            total_raw_hits += results['raw_correct']
            total_smooth_hits += results['smooth_correct']

            print("{:<25} | {:<10} | {:<9.1f}% | {:<9.1f}%".format(
                results['true_label'][:24],
                results['windows'],
                raw_acc,
                smooth_acc
            ))

        final_raw = (total_raw_hits / total_windows) * 100
        final_smooth = (total_smooth_hits / total_windows) * 100

        print("-" * 65)
        print(f"TOTAL RAW ACCURACY:      {final_raw:.2f}%")
        print(f"TOTAL SMOOTHED ACCURACY: {final_smooth:.2f}%")
        print(f"NET GAIN FROM SMOOTHING: +{final_smooth - final_raw:.2f}%")

if __name__ == "__main__":
    """Test 50 random files"""
    tester = EliteRandomTester()
    tester.run_extensive_test(num_files=120)