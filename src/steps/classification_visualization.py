import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERIMENTS_PATH = "artifacts/classification/knn_experiments.json"
OUTPUT_DIR = "artifacts/visualization"


def run_knn_visualization() -> None:
    """Generates a plot comparing the performance of KNN models with different distances."""
    logger.info("Generating KNN comparison plot...")

    with open(EXPERIMENTS_PATH, "r") as f:
        experiments = json.load(f)

    ks = list(range(1, 16))

    f1_euclidean = [
        experiments["euclidean"][f"k={k}"]["f1_macro_mean"]
        for k in ks
    ]

    f1_cosine = [
        experiments["cosine"][f"k={k}"]["f1_macro_mean"]
        for k in ks
    ]

    best_k_euc = ks[np.argmax(f1_euclidean)]
    best_k_cos = ks[np.argmax(f1_cosine)]

    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(10, 6))

    plt.plot(
        ks,
        f1_euclidean,
        marker="o",
        label="Euclidean Distance"
    )

    plt.plot(
        ks,
        f1_cosine,
        marker="s",
        label="Cosine Distance"
    )

    plt.scatter(
        best_k_euc,
        max(f1_euclidean),
        s=150
    )

    plt.scatter(
        best_k_cos,
        max(f1_cosine),
        s=150
    )

    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Macro F1 Score")
    plt.title("KNN Performance Across k Values: Euclidean vs Cosine Distance")
    plt.legend()


    plt.savefig(os.path.join(OUTPUT_DIR, "knn_f1_comparison.svg"), format="svg")

    plt.close()

    logger.info("KNN comparison plot saved successfully.")
