from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class EmbeddingInput(BaseModel):
    """Model for a single embedding input."""

    embedding: List[float] = Field(
        ...,
        description="A 320-dimensional embedding vector",
        min_items=320,
        max_items=320,
    )
    sample_id: Optional[str] = Field(
        default=None, description="Optional identifier for the sample"
    )

    class Config:
        json_schema_extra = {
            "example": {"embedding": [0.1] * 320, "sample_id": "sample_001"}
        }


class PredictionResult(BaseModel):
    """Model for a single prediction result."""

    sample_id: Optional[str] = None
    predicted_class: str = Field(..., description="The predicted syndrome class")
    confidence: float = Field(
        ..., description="Confidence score of the prediction (0-1)", ge=0.0, le=1.0
    )
    probabilities: Dict[str, float] = Field(
        ..., description="Probability distribution across all classes"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sample_id": "sample_001",
                "predicted_class": "300000082",
                "confidence": 0.95,
                "probabilities": {
                    "100180860": 0.01,
                    "100192430": 0.02,
                    "300000082": 0.95,
                    "300000080": 0.02,
                },
            }
        }


class BatchPredictionRequest(BaseModel):
    """Model for batch prediction requests."""

    samples: List[EmbeddingInput] = Field(
        ..., description="List of embeddings to predict"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "samples": [
                    {"embedding": [0.1] * 320, "sample_id": "sample_001"},
                    {"embedding": [0.2] * 320, "sample_id": "sample_002"},
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Model for batch prediction responses."""

    predictions: List[PredictionResult]
    total: int = Field(..., description="Total number of predictions")
    processing_time_ms: float = Field(
        ..., description="Total processing time in milliseconds"
    )


class HealthCheckResponse(BaseModel):
    """Model for health check response."""

    status: str = Field(..., description="Health status of the API")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata about the loaded model"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_metadata": {
                    "model_type": "knn",
                    "distance_metric": "cosine",
                    "k": 14,
                    "n_features": 320,
                    "n_classes": 10,
                },
            }
        }


class ErrorResponse(BaseModel):
    """Model for error responses."""

    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(default=None, description="Additional error details")
    timestamp: Optional[str] = Field(
        default=None, description="Timestamp of when the error occurred"
    )
