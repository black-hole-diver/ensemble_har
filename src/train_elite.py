from src.settings import Config, MAPPING, BLACKLIST, FileNames
from src.data_processor import DataProcessor
from src.utils import extract_batch_features, clean_class_name

import numpy as np
import optuna
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

class EliteEnsembleManager:
    def __init__(self):
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        os.makedirs(Config.VISUALS_DIR, exist_ok=True)

        self.processor = DataProcessor()
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        self.mapping = MAPPING

        self.blacklist = BLACKLIST

    def prepare_and_augment(self):
        print("--- 1. Extracting and Mapping ---")
        X, y = self.processor.process_all_files(Config.RAW_DATA_DIR)
        y_mapped = np.array([clean_class_name(self.mapping.get(label, label)) for label in y])

        keep_indices = [i for i, label in enumerate(y_mapped) if label not in self.blacklist]
        X_clean = X[keep_indices]
        y_clean = y_mapped[keep_indices]

        y_encoded = self.label_encoder.fit_transform(y_clean)
        y_encoded = np.asarray(y_encoded, dtype=int).reshape(-1)
        X_features = extract_batch_features(X_clean)

        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print("--- 2. Dynamic Data Augmentation (SMOTE) ---")
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))

        smote_strategy = {}
        for cls, count in class_counts.items():
            if count < 200:
                smote_strategy[cls] = 200

        if smote_strategy:
            smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=3, random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            boosted_names = [self.label_encoder.inverse_transform([c])[0] for c in smote_strategy.keys()]
            print(f"Synthesized new data for: {boosted_names}")

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        return X_train_s, X_test_s, y_train.flatten(), y_test.flatten()

    def optimize_lgbm(self, X_train, y_train):
        print("--- 3. Optuna: Aggressive Variance Regularization ---")
        def objective(trial):
            params = {
                'objective': 'multiclass',
                'num_class': len(self.label_encoder.classes_),
                'n_estimators': 1500,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 7), # Forced shallower trees
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'min_child_samples': trial.suggest_int('min_child_samples', 50, 150), # THE VARIANCE KILLER
                'subsample': trial.suggest_float('subsample', 0.5, 0.9), # Random row dropout
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9), # Random feature dropout
                'class_weight': 'balanced',
                'n_jobs': -1,
                'verbose': -1,
                'random_state': 42
            }
            X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            model = lgb.LGBMClassifier(**params)

            model.fit(
                X_opt_train, y_opt_train,
                eval_set=[(X_opt_val, y_opt_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )

            trial.set_user_attr("optimal_trees", model.best_iteration_)
            y_pred = model.predict(X_opt_val)
            return f1_score(y_opt_val, y_pred, average='weighted')

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=15)

        best_params = study.best_params
        best_params['n_estimators'] = study.best_trial.user_attrs['optimal_trees']
        print(f"✅ Best Params Found! Model perfectly converged at {best_params['n_estimators']} trees.")
        return best_params

    def train_elite_ensemble(self, X_train, y_train, X_test, y_test, best_lgbm_params):
        """Add L2 Regularization to CatBoost to fight variance
        Shallower Random Forest"""
        print("\n--- 4. Training Elite Ensemble ---")
        best_lgbm_params['verbose'] = -1
        best_lgbm_params['class_weight'] = 'balanced'
        lgbm = lgb.LGBMClassifier(**best_lgbm_params)

        cat = CatBoostClassifier(
            iterations=600,
            depth=5,
            learning_rate=0.05,
            l2_leaf_reg=5,
            auto_class_weights='Balanced',
            random_state=42,
            verbose=0
        )

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=1
        )

        stack_model = StackingClassifier(
            estimators=[('lgbm', lgbm), ('cat', cat), ('rf', rf)],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3,
            n_jobs=1
        )

        stack_model.fit(X_train, y_train)

        print("\n--- Evaluation (Raw Accuracy, No Smoothing) ---")
        y_pred = stack_model.predict(X_test)
        print(f"\nELITE ENSEMBLE Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        print("\n--- Saving ML Artifacts to /models ---")
        os.makedirs(Config.MODELS_DIR, exist_ok=True)

        model_path = os.path.join(Config.MODELS_DIR, FileNames.MODEL_NAME)
        labels_path = os.path.join(Config.MODELS_DIR, FileNames.LABELS_NAME)
        scaler_path = os.path.join(Config.MODELS_DIR, FileNames.SCALER_NAME)

        joblib.dump(stack_model, model_path)
        joblib.dump(self.label_encoder, labels_path)
        joblib.dump(self.scaler, scaler_path)

if __name__ == "__main__":
    manager = EliteEnsembleManager()
    X_train_s, X_test_s, y_train, y_test = manager.prepare_and_augment()
    best_params = manager.optimize_lgbm(X_train_s, y_train)
    manager.train_elite_ensemble(X_train_s, y_train, X_test_s, y_test, best_params)