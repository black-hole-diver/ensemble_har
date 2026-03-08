
from src.settings import Config, MAPPING
from src.data_processor import DataProcessor
from src.utils import extract_batch_features

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class MappingDiagnoser:
    def __init__(self):
        self.processor = DataProcessor()
        self.scaler = StandardScaler()
        self.mapping = MAPPING

    def diagnose_internal_similarity(self):
        """Calculate the centroid of every movement.
        Reverse mapping dictionary to group raw class by Super-Class"""
        print("--- Extracting Raw Data ---")
        X, y = self.processor.process_all_files(Config.RAW_DATA_DIR)

        X_features = extract_batch_features(X)
        X_scaled = self.scaler.fit_transform(X_features)

        df = pd.DataFrame(X_scaled)
        df['Raw_Target'] = y

        centroids = df.groupby('Raw_Target').mean()

        super_groups = defaultdict(list)
        for raw_class, super_class in self.mapping.items():
            if raw_class in centroids.index:
                super_groups[super_class].append(raw_class)

        print("\n--- Diagnostic Results: Should these be grouped? ---")
        print("Rule of Thumb:")
        print("  > 0.90 : Virtually identical to the sensor. MUST group.")
        print("  0.75 - 0.89 : Very similar. HIGHLY RECOMMENDED to group.")
        print("  < 0.75 : Distinct physical signatures. CONSIDER SPLITTING.\n")

        for super_class, raw_classes in super_groups.items():
            if len(raw_classes) > 1:
                print(f"[{super_class.upper()}]")
                group_centroids = centroids.loc[raw_classes]
                sim_matrix = cosine_similarity(group_centroids)

                for i in range(len(raw_classes)):
                    for j in range(i + 1, len(raw_classes)):
                        class_a = raw_classes[i]
                        class_b = raw_classes[j]
                        score = sim_matrix[i, j]

                        warning = "  <-- SPLIT WARNING" if score < 0.75 else ""
                        print(f"  {score:.4f} | {class_a} vs {class_b}{warning}")
                print("-" * 40)

if __name__ == "__main__":
    diagnoser = MappingDiagnoser()
    diagnoser.diagnose_internal_similarity()