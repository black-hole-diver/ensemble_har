from src.settings import Config, MAPPING, BLACKLIST, FileNames
from src.data_processor import DataProcessor
from src.utils import extract_batch_features, clean_class_name


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os

class ConfusionMatrixGenerator:
    def __init__(self):
        print("--- 1. Loading Pre-Trained Elite Artifacts ---")
        try:
            model_path = os.path.join(Config.MODELS_DIR, FileNames.MODEL_NAME)
            scaler_path = os.path.join(Config.MODELS_DIR, FileNames.SCALER_NAME)
            labels_path = os.path.join(Config.MODELS_DIR, FileNames.LABELS_NAME)
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(labels_path)
        except FileNotFoundError as e:
            raise RuntimeError(f"Could not find artifact: {e}. Ensure .pkl files are in this directory.")

        self.processor = DataProcessor()

        self.mapping = MAPPING
        self.blacklist = BLACKLIST

    def get_test_data(self):
        print("--- 2. Recreating Exact Test Split ---")
        X, y = self.processor.process_all_files(Config.RAW_DATA_DIR)
        y_mapped = np.array([clean_class_name(self.mapping.get(label, label)) for label in y])

        keep_indices = [i for i, label in enumerate(y_mapped) if label not in self.blacklist]
        X_clean = X[keep_indices]
        y_clean = y_mapped[keep_indices]

        # .transform() instead of .fit_transform(): the encoder is already trained!
        y_encoded = self.label_encoder.transform(y_clean)
        y_encoded = np.asarray(y_encoded, dtype=int).reshape(-1)
        X_features = extract_batch_features(X_clean)

        _, X_test, _, y_test = train_test_split(
            X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        X_test_s = self.scaler.transform(X_test)
        return X_test_s, y_test

    def generate_matrix(self):
        X_test_s, y_test = self.get_test_data()

        print("--- 3. Running Predictions ---")
        y_pred = self.model.predict(X_test_s)

        print("--- 4. Plotting Confusion Matrix ---")
        cm = confusion_matrix(y_test, y_pred)
        classes = self.label_encoder.classes_

        plt.figure(figsize=(18, 14))

        # cmap='Blues': the high numbers dark blue and low numbers light/white
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes,
                    cbar_kws={'label': 'Number of Predictions'})

        plt.title('Elite Ensemble Confusion Matrix', fontsize=20, fontweight='bold')
        plt.ylabel('True Activity (What the child actually did)', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Activity (What the AI guessed)', fontsize=14, fontweight='bold')

        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, fontsize=11)
        plt.tight_layout()

        save_path = f"{Config.VISUALS_DIR}/elite_confusion_matrix.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"\n✅ Success! Confusion matrix saved to: {save_path}")

if __name__ == "__main__":
    generator = ConfusionMatrixGenerator()
    generator.generate_matrix()