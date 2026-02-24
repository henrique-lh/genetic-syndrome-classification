import os
import joblib
import logging
import numpy as np
from sklearn.preprocessing import normalize
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the loaded KNN model and performs predictions."""

    NUMBER_OF_EMBEDDING = 320

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.normalization = False
        self.model_loaded = False
        self.metadata = None

    def load_model(
        self, model_path: str = "artifacts/classification/knn_best_model.pkl"
    ) -> bool:
        """
        Load the trained KNN model and associated metadata.

        Args:
            model_path (str): Path to the pickled model file.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False

            logger.info(f"Loading model from {model_path}")
            model_data = joblib.load(model_path)

            self.model = model_data["model"]
            self.label_encoder = model_data["label_encoder"]
            self.normalization = model_data["normalization"]

            metadata_path = "artifacts/classification/best_model_metadata.json"
            if os.path.exists(metadata_path):
                import json

                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {
                    "model_type": "knn",
                    "distance_metric": self.model.metric,
                    "k": self.model.n_neighbors,
                    "normalization": self.normalization,
                    "n_features": self.model.n_features_in_,
                    "n_classes": len(self.label_encoder.classes_),
                }

            self.model_loaded = True
            logger.info("Model loaded successfully")
            logger.info(f"Model metadata: {self.metadata}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict(self, embedding: List[float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Make a prediction for a single embedding.

        Args:
            embedding (List[float]): 320-dimensional embedding vector.

        Returns:
            Tuple containing (predicted_class, confidence, probabilities)
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        X = np.array([embedding], dtype=np.float32)

        if self.normalization:
            X = normalize(X, norm="l2")

        predicted_idx = self.model.predict(X)[0]
        predicted_class = self.label_encoder.classes_[predicted_idx]

        proba = self.model.predict_proba(X)[0]
        confidence = float(proba[predicted_idx])

        probabilities = {
            self.label_encoder.classes_[i]: float(prob) for i, prob in enumerate(proba)
        }

        return predicted_class, confidence, probabilities

    def batch_predict(
        self, embeddings: List[List[float]]
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Make predictions for multiple embeddings.

        Args:
            embeddings (List[List[float]]): List of 320-dimensional embedding vectors.

        Returns:
            List of tuples containing (predicted_class, confidence, probabilities)
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        X = np.array(embeddings, dtype=np.float32)

        if self.normalization:
            X = normalize(X, norm="l2")

        predictions = self.model.predict(X)
        probas = self.model.predict_proba(X)

        results = []
        for pred_idx, proba in zip(predictions, probas):
            predicted_class = self.label_encoder.classes_[pred_idx]
            confidence = float(proba[pred_idx])
            probabilities = {
                self.label_encoder.classes_[i]: float(prob)
                for i, prob in enumerate(proba)
            }
            results.append((predicted_class, confidence, probabilities))

        return results

    def get_metadata(self) -> Dict:
        """Get metadata about the loaded model."""
        if not self.model_loaded:
            return {}
        return self.metadata

    def get_expected_embedding_dim(self) -> int:
        """Get the expected embedding dimension."""
        if self.model_loaded:
            return self.model.n_features_in_
        return self.NUMBER_OF_EMBEDDING
