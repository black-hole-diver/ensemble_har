import os
import coremltools as ct
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from src.settings import Config, MAPPING, BLACKLIST
from src.data_processor import DataProcessor
from src.utils import extract_batch_features, clean_class_name

def create_watch_model():
    print("--- 1. Apple Core ML Workaround: Loading Cleaned Data ---")
    processor = DataProcessor()
    X, y = processor.process_all_files(Config.RAW_DATA_DIR)

    y_mapped = np.array([clean_class_name(MAPPING.get(label, label)) for label in y])
    keep_indices = [i for i, label in enumerate(y_mapped) if label not in BLACKLIST]

    X_clean = X[keep_indices]
    y_clean = y_mapped[keep_indices]

    X_features = extract_batch_features(X_clean)

    print("--- 2. Training Fast, Watch-Optimized Random Forest (< 3 seconds) ---")

    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_features, y_clean)

    print("--- 3. Translating to Apple Core ML ---")
    mlmodel = ct.converters.sklearn.convert(
        rf,
        input_features="physics_features",
        output_feature_names="activity_label"
    )

    mlmodel.author = "Elite HAR Architecture"
    mlmodel.short_description = "Pediatric Human Activity Recognition (WatchOS Edition)"

    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    out_path = os.path.join(Config.MODELS_DIR, 'WatchActivityPredictor.mlpackage')
    mlmodel.save(out_path)
    print(f"✅ Success! Xcode MLPackage saved to: {out_path}")

if __name__ == "__main__":
    create_watch_model()