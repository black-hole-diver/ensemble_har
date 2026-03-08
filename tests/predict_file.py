from src.settings import Config, FileNames
from src.utils import extract_batch_features

import os
import warnings
import pandas as pd
import joblib
from collections import Counter

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")



class ElitePredictor:
    def __init__(self):
        print("--- Loading Elite Stacked Ensemble Pipeline ---")

        model_path = os.path.join(Config.MODELS_DIR, FileNames.MODEL_NAME)
        scaler_path = os.path.join(Config.MODELS_DIR, FileNames.SCALER_NAME)
        labels_path = os.path.join(Config.MODELS_DIR, FileNames.LABELS_NAME)

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(labels_path)
        except FileNotFoundError as e:
            raise RuntimeError(f"❌ Missing artifact: {e}. Did you run src.train_elite first?")

    def predict_file(self, file_path):
        if not os.path.exists(file_path):
            print(f"❌ ERROR: File not found -> {file_path}")
            return None

        df = pd.read_csv(file_path)

        if not all(col in df.columns for col in Config.SENSOR_FEATURES):
            print(f"❌ ERROR: Missing sensor columns in -> {file_path}")
            return None

        df = df[Config.SENSOR_FEATURES].interpolate().dropna()

        win_size = int(Config.SAMPLING_RATE_HZ * Config.WINDOW_SEC)
        step = int(win_size * 0.5)

        if len(df) < win_size:
            print(f"❌ ERROR: File too short for a {Config.WINDOW_SEC}s window -> {file_path}")
            return None

        raw_windows = [df.iloc[i:i+win_size].values for i in range(0, len(df)-win_size+1, step)]

        features = extract_batch_features(raw_windows)
        scaled = self.scaler.transform(features)
        preds = self.model.predict(scaled)
        labels = self.label_encoder.inverse_transform(preds)

        final = Counter(labels).most_common(1)[0][0]
        print(f"✅ Final File-Level Classification: {final.upper()}")
        return final

if __name__ == "__main__":
    predictor = ElitePredictor()
    test_file = os.path.join(Config.RAW_DATA_DIR, "Bear", "Bear_0.csv")
    print(f"\nAnalyzing file: {test_file}")
    predictor.predict_file(test_file)