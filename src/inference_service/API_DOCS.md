# Inference Service

The inference service provides a REST API for making predictions using the trained KNN genetic syndrome classifier.

## Overview

The inference service consists of:

- **Model Manager** (`model_manager.py`): Handles model loading and inference
- **Pydantic Models** (`models.py`): Request/response data validation
- **FastAPI Application** (`app.py`): REST API endpoints

## Features

- Single prediction endpoint
- Batch prediction endpoint
- Health check endpoint with model status
- Model information endpoint
- Automatic model loading on startup
- Comprehensive request/response validation
- CORS support for cross-origin requests
- Interactive API documentation (Swagger UI, ReDoc)

## API Endpoints

### 1. Health Check
**Endpoint:** `GET /health`

Verify API and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_metadata": {
    "model_type": "knn",
    "distance_metric": "cosine",
    "k": 14,
    "n_features": 320,
    "n_classes": 10
  }
}
```

### 2. Single Prediction
**Endpoint:** `POST /predict`

Make a prediction for a single embedding.

**Request Body:**
```json
{
  "embedding": [0.1, 0.2, ...],  // 320-dimensional vector
  "sample_id": "sample_001"       // Optional
}
```

**Response:**
```json
{
  "sample_id": "sample_001",
  "predicted_class": "300000082",
  "confidence": 0.95,
  "probabilities": {
    "100180860": 0.01,
    "100192430": 0.02,
    "300000082": 0.95,
    "300000080": 0.02,
    ...
  }
}
```

### 3. Batch Prediction
**Endpoint:** `POST /predict_batch`

Make predictions for multiple embeddings.

**Request Body:**
```json
{
  "samples": [
    {
      "embedding": [0.1, 0.2, ...],
      "sample_id": "sample_001"
    },
    {
      "embedding": [0.3, 0.4, ...],
      "sample_id": "sample_002"
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "sample_id": "sample_001",
      "predicted_class": "300000082",
      "confidence": 0.95,
      "probabilities": {...}
    },
    {
      "sample_id": "sample_002",
      "predicted_class": "300000080",
      "confidence": 0.87,
      "probabilities": {...}
    }
  ],
  "total": 2,
  "processing_time_ms": 45.23
}
```

### 4. Model Information
**Endpoint:** `GET /model/info`

Get detailed model metadata and specifications.

**Response:**
```json
{
  "model_loaded": true,
  "metadata": {
    "model_type": "knn",
    "distance_metric": "cosine",
    "k": 14,
    "normalization": true,
    "f1_macro": 0.92,
    "auc_ovr": 0.96
  },
  "expected_embedding_dim": 320
}
```

## API Documentation

Once the server is running, you can access interactive documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Model Requirements

- **Input:** 320-dimensional embedding vector (numpy array)
- **Output:** Predicted genetic syndrome class with confidence and per-class probabilities
- **Metric:** Cosine distance with L2 normalization
- **Algorithm:** K-Nearest Neighbors (k=14)
- **Classes:** 10 genetic syndrome categories

## Sample Data

Sample input files have been generated in `artifacts/sample_inputs/`:

- `sample_1.json` through `sample_5.json`: Individual sample embeddings
- `batch_input.json`: Multiple samples for batch prediction

## Error Handling

The API returns appropriate HTTP status codes:

- **200 OK:** Successful prediction
- **400 Bad Request:** Invalid input (wrong embedding dimension, missing fields)
- **503 Service Unavailable:** Model not loaded
- **500 Internal Server Error:** Unexpected server error

All errors return an `ErrorResponse` with:
```json
{
  "error": "Error message",
  "details": "Additional details",
  "timestamp": "2024-02-24T10:30:45.123456"
}
```

## Architecture

```
src/inference_service/
├── __init__.py           # Package exports
├── app.py                # FastAPI application and endpoints
├── model_manager.py      # Model loading and inference logic
└── models.py             # Pydantic request/response models
```

## Dependencies

- FastAPI: Web framework
- Uvicorn: ASGI server
- Pydantic: Data validation
- scikit-learn: ML models
- numpy: Numerical computing
- joblib: Model serialization

All dependencies are listed in `requirements.txt`.
