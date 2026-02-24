import os
import json
import joblib
import logging
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, normalize, label_binarize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    top_k_accuracy_score,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_PATH = "artifacts/processed/dataset.parquet"
OUTPUT_DIR = "artifacts/classification"


def load_data() -> tuple:
    """
    Loads data from a Parquet file, extracts embeddings, target labels, and grouping
    information, and encodes target labels using a LabelEncoder.

    Raises:
        ValueError: If the dataset contains fewer than two distinct classes.

    Returns:
        tuple: A tuple containing the following elements:
            - X (np.ndarray): The extracted embeddings as a 2D array.
            - y_encoded (np.ndarray): Encoded labels as a 1D array.
            - groups (np.ndarray): Grouping information as a 1D array.
            - label_encoder (LabelEncoder): Fitted LabelEncoder instance for
              decoding and encoding labels in the dataset.
    """
    df = pd.read_parquet(INPUT_PATH)
    X = np.array(df["embedding"].tolist(), dtype=np.float32)
    y = df["syndrome_id"].values
    groups = df["subject_id"].values

    if len(np.unique(y)) < 2:
        raise ValueError("Dataset must contain at least two classes.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, groups, label_encoder


def evaluate_knn(
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        metric_name: str,
        normalize_embeddings: bool = False
) -> dict:
    """
    Evaluates a KNN model using cross-validation.

    Args:
        X (np.ndarray): Embeddings to evaluate.
        y (np.ndarray): Target labels.
        groups (np.ndarray): Grouping information.
        metric_name (str): Distance metric to use for KNN.
        normalize_embeddings (bool, optional): Whether to normalize embeddings before training. Defaults to False.

    Returns:
        dict: Dictionary containing evaluation metrics for each k value.
    """
    logger.info(f"Evaluating metric: {metric_name}")

    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    results = {}

    for k in range(1, 16):
        acc_scores, f1_scores, auc_scores, top3_scores = [], [], [], []

        for train_idx, val_idx in skf.split(X, y, groups=groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if normalize_embeddings:
                X_train = normalize(X_train, norm="l2")
                X_val = normalize(X_val, norm="l2")

            model = KNeighborsClassifier(n_neighbors=k, metric=metric_name)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)

            acc_scores.append(accuracy_score(y_val, y_pred))
            f1_scores.append(f1_score(y_val, y_pred, average="macro"))

            y_val_bin = label_binarize(y_val, classes=np.unique(y))
            auc_scores.append(roc_auc_score(y_val_bin, y_proba, multi_class="ovr"))
            top3_scores.append(top_k_accuracy_score(y_val, y_proba, k=3))

        results[f"k={k}"] = {
            "accuracy_mean": float(np.mean(acc_scores)),
            "accuracy_std": float(np.std(acc_scores)),
            "f1_macro_mean": float(np.mean(f1_scores)),
            "f1_macro_std": float(np.std(f1_scores)),
            "auc_ovr_mean": float(np.mean(auc_scores)),
            "auc_ovr_std": float(np.std(auc_scores)),
            "top3_mean": float(np.mean(top3_scores)),
            "top3_std": float(np.std(top3_scores)),
        }
        logger.info(f"{metric_name} | k={k} | F1={results[f'k={k}']['f1_macro_mean']:.4f}")

    return results


def select_best_model(experiments: dict) -> tuple:
    """
    Selects the best model based on F1 score.

    Args:
        experiments (dict): Dictionary containing evaluation metrics for each k value.

    Returns:
        tuple: A tuple containing the best metric name, best k value, and best F1 score.
    """
    best_metric, best_k, best_f1 = None, None, -1
    for metric_name, metric_results in experiments.items():
        for k_name, metrics in metric_results.items():
            if metrics["f1_macro_mean"] > best_f1:
                best_f1 = metrics["f1_macro_mean"]
                best_metric = metric_name
                best_k = int(k_name.split("=")[1])
    return best_metric, best_k


def run_knn_classification() -> None:
    """
    Run a KNN Classification step, find the best metric and k value, and save the results.
    """
    logger.info("Starting KNN classification...")
    X, y, groups, label_encoder = load_data()

    experiments = {
        "euclidean": evaluate_knn(X, y, groups, metric_name="euclidean", normalize_embeddings=False),
        "cosine": evaluate_knn(X, y, groups, metric_name="cosine", normalize_embeddings=True)
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "knn_experiments.json"), "w") as f:
        json.dump(experiments, f, indent=4)

    best_metric, best_k = select_best_model(experiments)
    normalize_flag = (best_metric == "cosine")

    X_final = normalize(X, norm="l2") if normalize_flag else X
    final_model = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
    final_model.fit(X_final, y)

    joblib.dump({"model": final_model, "label_encoder": label_encoder, "normalization": normalize_flag},
                os.path.join(OUTPUT_DIR, "knn_best_model.pkl"))

    metadata = {
        "model_type": "knn",
        "distance_metric": best_metric,
        "k": best_k,
        "normalization": normalize_flag,
        "f1_macro": experiments[best_metric][f"k={best_k}"]["f1_macro_mean"],
        "auc_ovr": experiments[best_metric][f"k={best_k}"]["auc_ovr_mean"]
    }

    with open(os.path.join(OUTPUT_DIR, "best_model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"KNN classification completed. Best metric: {best_metric}, Best k: {best_k}")
