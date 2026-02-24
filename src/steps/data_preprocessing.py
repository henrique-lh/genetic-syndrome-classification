import os
import json
import pickle
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd


RAW_DATA_PATH = "mini_gm_public_v0.1.p"
OUTPUT_DIR = "artifacts/processed"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_valid_embedding(embedding: Any) -> bool:
    """
    Validate embedding:
    - array-like
    - length 320
    - numeric
    - no NaN

    Args:
        embedding: Embedding to validate. Loaded from pickle file

    Returns:
        bool: True if embedding is valid, False otherwise
    """
    try:
        arr = np.asarray(embedding)

        if arr.shape != (320,):
            return False

        if np.isnan(arr).any():
            return False

        arr.astype(np.float32)
        return True

    except Exception:
        return False


def data_processing() -> None:
    """
    Load, flatten, validate and persist processed dataset.
    """

    logger.info("Starting data processing...")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"{RAW_DATA_PATH} not found.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(RAW_DATA_PATH, "rb") as f:
        raw_data: Dict[str, Dict[str, Dict[str, Any]]] = pickle.load(f)

    rows = []
    removed_invalid = 0

    for syndrome_id, subjects in raw_data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():

                if not _is_valid_embedding(embedding):
                    removed_invalid += 1
                    continue

                embedding_array = np.asarray(embedding, dtype=np.float32)

                rows.append(
                    {
                        "syndrome_id": str(syndrome_id),
                        "subject_id": str(subject_id),
                        "image_id": str(image_id),
                        "embedding": embedding_array.tolist(),
                    }
                )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("Processed dataset is empty after validation.")

    n_classes = df["syndrome_id"].nunique()
    if n_classes < 2:
        raise ValueError(
            "Dataset must contain at least 2 classes for classification."
        )

    total_syndromes = df["syndrome_id"].nunique()
    total_subjects = df["subject_id"].nunique()
    total_images = len(df)

    images_per_syndrome = df.groupby("syndrome_id").size()

    metadata = {
        "total_syndromes": int(total_syndromes),
        "total_subjects": int(total_subjects),
        "total_images": int(total_images),
        "images_per_syndrome_mean": float(images_per_syndrome.mean()),
        "images_per_syndrome_std": float(images_per_syndrome.std()),
        "max_images_syndrome": int(images_per_syndrome.max()),
        "min_images_syndrome": int(images_per_syndrome.min()),
        "removed_invalid_embeddings": int(removed_invalid),
    }

    dataset_path = os.path.join(OUTPUT_DIR, "dataset.parquet")
    df.to_parquet(dataset_path, index=False)

    metadata_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    class_distribution = (
        images_per_syndrome.reset_index()
        .rename(columns={0: "image_count"})
    )
    class_distribution["percentage"] = (
        class_distribution["image_count"] / total_images * 100
    )

    distribution_path = os.path.join(OUTPUT_DIR, "class_distribution.csv")
    class_distribution.to_csv(distribution_path, index=False)

    logger.info("Data processing completed successfully.")
    logger.info(f"Total images: {total_images}")
    logger.info(f"Total syndromes: {total_syndromes}")
    logger.info(f"Removed invalid embeddings: {removed_invalid}")
