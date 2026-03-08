from src.settings import Config, MAPPING, BLACKLIST
from src.data_processor import DataProcessor
from src.utils import extract_batch_features

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

class MovementDiagnoser:
    def __init__(self):
        self.processor = DataProcessor()
        self.scaler = StandardScaler()
        self.mapping = MAPPING
        self.blacklist = BLACKLIST

    def load_and_extract(self):
        print("--- Extracting Data for Diagnosis ---")
        X, y = self.processor.process_all_files(Config.RAW_DATA_DIR)
        y_mapped = np.array([self.mapping.get(label, label) for label in y])
        keep_indices = [i for i, label in enumerate(y_mapped) if label not in self.blacklist]
        X_clean = X[keep_indices]
        self.y_labels = y_mapped[keep_indices]
        X_features = extract_batch_features(X_clean)
        self.X_scaled = self.scaler.fit_transform(X_features)

        self.df = pd.DataFrame(self.X_scaled)
        self.df['Target'] = self.y_labels

    def diagnose_similarity(self):
        """Find centroid for each movement.
        Calculate cosine similarity between all centroids
        Find the most identical pairs
        Sort by similarity and generate heatmap"""
        print("\n--- Calculating Centroid Similarities ---")
        # 1. Find the "average" feature vector (centroid) for each movement
        centroids = self.df.groupby('Target').mean()
        class_names = centroids.index.tolist()

        sim_matrix = cosine_similarity(centroids)

        pairs = []
        for i in range(len(class_names)):
            for j in range(i + 1, len(class_names)):
                pairs.append((class_names[i], class_names[j], sim_matrix[i, j]))

        pairs.sort(key=lambda x: x[2], reverse=True)

        print("\n🚨 RED ALERT: Top 10 Most Mathematically Similar Movements 🚨")
        print("If similarity is > 0.90, the sensor cannot easily tell them apart.")
        for class_a, class_b, score in pairs[:10]:
            print(f"{score:.4f} | {class_a} <--> {class_b}")

        plt.figure(figsize=(16, 14))
        sns.heatmap(sim_matrix, xticklabels=class_names, yticklabels=class_names,
                    cmap='coolwarm', annot=False)
        plt.title("Movement Physics Similarity Matrix", fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{Config.VISUALS_DIR}/similarity_heatmap.png", dpi=300)
        plt.close()

    def generate_tsne_map(self):
        """Sample data down a bit to run t-SNE faster
        Compress 72 dims to 2 dims"""
        print("\n--- Generating t-SNE 2D Map (This takes a minute) ---")
        sample_size = min(5000, len(self.df))
        df_sampled = self.df.sample(n=sample_size, random_state=42)

        features = df_sampled.drop('Target', axis=1).values
        labels = df_sampled['Target'].values

        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
        tsne_results = tsne.fit_transform(features)

        plt.figure(figsize=(18, 12))
        sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            hue=labels, palette=sns.color_palette("husl", len(np.unique(labels))),
            legend="full", alpha=0.7, s=30
        )
        plt.title("t-SNE Map: Visualizing Overlapping Movements", fontsize=18, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(f"{Config.VISUALS_DIR}/tsne_map.png", dpi=300)
        plt.close()
        print(f"Visualizations saved to {Config.VISUALS_DIR}/")

if __name__ == "__main__":
    diagnoser = MovementDiagnoser()
    diagnoser.load_and_extract()
    diagnoser.diagnose_similarity()
    diagnoser.generate_tsne_map()