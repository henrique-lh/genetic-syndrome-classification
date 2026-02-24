import os
import json
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


INPUT_PATH = "artifacts/processed/dataset.parquet"
OUTPUT_DIR = "artifacts/visualization"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def data_visualization() -> None:
    """
    Generates different plots to visualize the dataset. It is being considered:
    - Class Distribution Plot
    - Standardization
    - PCA
    - t-SNE
    - Silhouette Score

    Saves the plots to the `visualization` directory.
    """
    logger.info("Starting data visualization...")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("Processed dataset not found. Run data_processing first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)

    X = np.vstack(df["embedding"].apply(np.array).values)
    y = df["syndrome_id"].values

    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} dimensions.")

    class_counts = df["syndrome_id"].value_counts().sort_values(ascending=False)
    class_percentages = class_counts / len(df) * 100

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=class_counts.index,
        y=class_counts.values,
        palette="viridis"
    )

    for i, (count, pct) in enumerate(zip(class_counts.values, class_percentages.values)):
        plt.text(i, count, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.title("Class Distribution of Genetic Syndromes", fontsize=14)
    plt.xlabel("Syndrome ID")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.svg"), format="svg")
    plt.close()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_.sum()

    pca_df = pd.DataFrame({
        "pca_1": X_pca[:, 0],
        "pca_2": X_pca[:, 1],
        "syndrome_id": y
    })

    pca_df.to_csv(os.path.join(OUTPUT_DIR, "pca_coordinates.csv"), index=False)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="pca_1",
        y="pca_2",
        hue="syndrome_id",
        palette="tab10",
        s=30,
        alpha=0.7
    )

    plt.title(f"PCA Projection (Explained Variance: {explained_variance:.2%})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Syndrome", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_projection.svg"), format="svg")
    plt.close()

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    X_tsne = tsne.fit_transform(X_scaled)

    tsne_df = pd.DataFrame({
        "tsne_1": X_tsne[:, 0],
        "tsne_2": X_tsne[:, 1],
        "syndrome_id": y
    })

    tsne_df.to_csv(os.path.join(OUTPUT_DIR, "tsne_coordinates.csv"), index=False)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=tsne_df,
        x="tsne_1",
        y="tsne_2",
        hue="syndrome_id",
        palette="tab10",
        s=30,
        alpha=0.7
    )

    plt.title("t-SNE Projection of 320-D Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Syndrome", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_projection.svg"), format="svg")
    plt.close()

    silhouette_original = silhouette_score(X_scaled, y)
    silhouette_pca = silhouette_score(X_pca, y)

    summary = {
        "explained_variance_pca_2d": float(explained_variance),
        "silhouette_score_original_space": float(silhouette_original),
        "silhouette_score_pca_space": float(silhouette_pca)
    }

    with open(os.path.join(OUTPUT_DIR, "visualization_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    logger.info("Data visualization completed successfully.")
    logger.info(f"PCA explained variance (2D): {explained_variance:.2%}")
    logger.info(f"Silhouette score (original space): {silhouette_original:.4f}")
    logger.info(f"Silhouette score (PCA space): {silhouette_pca:.4f}")
