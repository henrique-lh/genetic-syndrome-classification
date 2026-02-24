"""
Inference service for genetic syndrome classification.
"""

from .app import app
from .model_manager import ModelManager
from .models import (
    EmbeddingInput,
    PredictionResult,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheckResponse,
    ErrorResponse,
)

__all__ = [
    "app",
    "ModelManager",
    "EmbeddingInput",
    "PredictionResult",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "HealthCheckResponse",
    "ErrorResponse",
]
