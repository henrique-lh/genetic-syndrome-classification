import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, roc_auc_score, top_k_accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = "artifacts/processed/dataset.parquet"
EXPERIMENTS_PATH = "artifacts/classification/knn_experiments.json"
OUTPUT_DIR = "artifacts/visualization"


def load_data() -> tuple:
    """
    Load dataset from a parquet file and return embeddings, target labels, and grouping information.
    """
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    X = np.array(df["embedding"].tolist(), dtype=np.float32)
    y = df["syndrome_id"].values
    groups = df["subject_id"].values
    return X, y, groups


def get_best_ks() -> tuple[int, int]:
    """Get the best k values for Euclidean and Cosine metrics from the experiments dictionary."""
    with open(EXPERIMENTS_PATH, "r") as f:
        experiments = json.load(f)

    best_k_euc = max(experiments["euclidean"].keys(), key=lambda k: experiments["euclidean"][k]["f1_macro_mean"])
    best_k_cos = max(experiments["cosine"].keys(), key=lambda k: experiments["cosine"][k]["f1_macro_mean"])

    return int(best_k_euc.split("=")[1]), int(best_k_cos.split("=")[1])


def compute_cv_metrics_and_roc(
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        metric_name: str,
        k: int,
        normalize_flag: bool
) -> dict:
    """
    Compute metrics and ROC curves for cross-validation.

    Args:
        X (np.ndarray): Embeddings to evaluate.
        y (np.ndarray): Target labels.
        groups (np.ndarray): Grouping information.
        metric_name (str): Distance metric to use for KNN.
        k (int): Number of neighbors to use for KNN.
        normalize_flag (bool): Whether to normalize embeddings before training.

    Returns:
        dict: Dictionary containing evaluation metrics and ROC curves.
    """
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs, acc_scores, f1_scores, auc_scores, top3_scores = [], [], [], [], [], []
    classes = np.unique(y)

    for train_idx, val_idx in skf.split(X, y, groups=groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if normalize_flag:
            X_train = normalize(X_train, norm="l2")
            X_val = normalize(X_val, norm="l2")

        model = KNeighborsClassifier(n_neighbors=k, metric=metric_name)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        acc_scores.append(accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred, average="macro"))

        y_val_bin = label_binarize(y_val, classes=classes)
        auc_scores.append(roc_auc_score(y_val_bin, y_proba, multi_class="ovr"))
        top3_scores.append(top_k_accuracy_score(y_val, y_proba, k=3))

        fpr_dict, tpr_dict = {}, {}
        for i in range(len(classes)):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_val_bin[:, i], y_proba[:, i])

        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(len(classes))]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= len(classes)

        interp_tpr = np.interp(mean_fpr, all_fpr, mean_tpr)
        tprs.append(interp_tpr)
        aucs.append(auc(mean_fpr, interp_tpr))

    return {
        "mean_fpr": mean_fpr, "mean_tpr": np.mean(tprs, axis=0), "std_tpr": np.std(tprs, axis=0),
        "auc_mean": np.mean(aucs), "accuracy_mean": np.mean(acc_scores), "accuracy_std": np.std(acc_scores),
        "f1_mean": np.mean(f1_scores), "f1_std": np.std(f1_scores),
        "auc_ovr_mean": np.mean(auc_scores), "auc_ovr_std": np.std(auc_scores),
        "top3_mean": np.mean(top3_scores), "top3_std": np.std(top3_scores),
    }


def run_evaluation() -> None:
    """Runs evaluation and saves the results."""
    logger.info("Starting evaluation...")
    X, y, groups = load_data()
    k_euclidean, k_cosine = get_best_ks()

    results_euclidean = compute_cv_metrics_and_roc(X, y, groups, "euclidean", k_euclidean, False)
    results_cosine = compute_cv_metrics_and_roc(X, y, groups, "cosine", k_cosine, True)

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 7))

    for results, label in [(results_euclidean, f"Euclidean (k={k_euclidean})"),
                           (results_cosine, f"Cosine (k={k_cosine})")]:
        plt.plot(results["mean_fpr"], results["mean_tpr"], label=f"{label} | AUC={results['auc_mean']:.3f}")
        plt.fill_between(results["mean_fpr"], results["mean_tpr"] - results["std_tpr"],
                         results["mean_tpr"] + results["std_tpr"], alpha=0.2)

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Macro-Averaged ROC Curves: Euclidean vs Cosine KNN")
    plt.legend()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_comparison.svg"), format="svg")
    plt.close()

    summary_df = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Macro", "AUC (OvR)", "Top-3 Accuracy"],
        "Euclidean (mean ± std)": [f"{results_euclidean[m + '_mean']:.4f} ± {results_euclidean[m + '_std']:.4f}" for m
                                   in ["accuracy", "f1", "auc_ovr", "top3"]],
        "Cosine (mean ± std)": [f"{results_cosine[m + '_mean']:.4f} ± {results_cosine[m + '_std']:.4f}" for m in
                                ["accuracy", "f1", "auc_ovr", "top3"]],
    })
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)
    logger.info("Evaluation completed successfully.")
