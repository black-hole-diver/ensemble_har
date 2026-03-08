from src.settings import Config, MAPPING, BLACKLIST
from src.data_processor import DataProcessor
from src.utils import extract_batch_features

import os
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class LearningCurveGenerator:
    def __init__(self):
        self.mapping = MAPPING
        self.blacklist = BLACKLIST
        self.processor = DataProcessor()

    def prepare_data(self):
        print("--- 1. Loading and Cleaning Raw Sensor Data ---")
        X_raw, y_raw = self.processor.process_all_files(Config.RAW_DATA_DIR)

        y_mapped = np.array([self.mapping.get(label, label) for label in y_raw])

        keep_indices = [i for i, label in enumerate(y_mapped) if label not in self.blacklist]
        X_clean = X_raw[keep_indices]
        y_clean = y_mapped[keep_indices]

        print("--- 2. Extracting 72-Feature Physics Signatures ---")

        X_features = extract_batch_features(X_clean)

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y_clean)

        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        return X_train_s, X_test_s, y_train, y_test, len(encoder.classes_)

    def visualize_bias_variance(self):
        X_train, X_test, y_train, y_test, num_classes = self.prepare_data()

        print("--- 3. Training Model to Extract Learning Curves ---")
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=num_classes,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            min_child_samples=30
        )

        evals_result = {}

        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_names=['Training (Memorization)', 'Validation (Generalization)'],
            eval_metric='multi_logloss',
            callbacks=[lgb.record_evaluation(evals_result)]
        )

        print("\n--- 4. Plotting Bias and Variance ---")
        train_loss = evals_result['Training (Memorization)']['multi_logloss']
        val_loss = evals_result['Validation (Generalization)']['multi_logloss']
        rounds = range(len(train_loss))

        plt.figure(figsize=(12, 8))
        plt.plot(rounds, train_loss, label='Training Loss (Lower is better)', color='blue', linewidth=2)
        plt.plot(rounds, val_loss, label='Validation Loss (Lower is better)', color='red', linewidth=2)

        plt.fill_between(rounds, train_loss, val_loss, color='red', alpha=0.1, label='Variance (Overfitting Gap)')

        plt.title('Bias vs. Variance in Activity Recognition', fontsize=18, fontweight='bold')
        plt.xlabel('Boosting Rounds (Number of Trees)', fontsize=14)
        plt.ylabel('Multi-Logloss Error', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        os.makedirs(Config.VISUALS_DIR, exist_ok=True)
        save_path = os.path.join(Config.VISUALS_DIR, "bias_variance_curve.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Graph saved securely to: {save_path}")

if __name__ == "__main__":
    generator = LearningCurveGenerator()
    generator.visualize_bias_variance()