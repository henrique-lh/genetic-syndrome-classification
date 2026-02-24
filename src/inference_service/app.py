from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from datetime import datetime

from .models import (
    EmbeddingInput,
    PredictionResult,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheckResponse,
    ErrorResponse,
)
from .model_manager import ModelManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Genetic Syndrome Classification API",
    description="API for classifying genetic syndromes using KNN embeddings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    logger.info("Application startup - Loading model...")
    success = model_manager.load_model()
    if success:
        logger.info("Model loaded successfully on startup")
    else:
        logger.warning("Failed to load model on startup")


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint to verify API and model status.

    Returns:
        HealthCheckResponse: Status and model information
    """
    try:
        metadata = model_manager.get_metadata() if model_manager.model_loaded else None
        return HealthCheckResponse(
            status="healthy",
            model_loaded=model_manager.model_loaded,
            model_metadata=metadata,
        )
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return HealthCheckResponse(
            status="error", model_loaded=False, model_metadata=None
        )


@app.post("/predict", response_model=PredictionResult)
async def predict(input_data: EmbeddingInput):
    """
    Single prediction endpoint.

    Args:
        input_data (EmbeddingInput): Embedding to predict

    Returns:
        PredictionResult: Prediction with confidence and probabilities
    """
    if not model_manager.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. API is not ready for predictions.",
        )

    try:
        if len(input_data.embedding) != model_manager.get_expected_embedding_dim():
            raise HTTPException(
                status_code=400,
                detail=f"Embedding dimension mismatch. Expected {model_manager.get_expected_embedding_dim()}, "
                f"got {len(input_data.embedding)}",
            )

        predicted_class, confidence, probabilities = model_manager.predict(
            input_data.embedding
        )

        return PredictionResult(
            sample_id=input_data.sample_id,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}"
        )


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction endpoint.

    Args:
        request (BatchPredictionRequest): Batch of embeddings to predict

    Returns:
        BatchPredictionResponse: List of predictions with timing
    """
    if not model_manager.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. API is not ready for predictions.",
        )

    if not request.samples:
        raise HTTPException(
            status_code=400, detail="No samples provided in the request"
        )

    try:
        expected_dim = model_manager.get_expected_embedding_dim()
        for sample in request.samples:
            if len(sample.embedding) != expected_dim:
                raise HTTPException(
                    status_code=400,
                    detail=f"Embedding dimension mismatch. Expected {expected_dim}, "
                    f"got {len(sample.embedding)} for sample {sample.sample_id}",
                )

        start_time = time.time()

        embeddings = [sample.embedding for sample in request.samples]

        predictions_data = model_manager.batch_predict(embeddings)

        predictions = []
        for sample, (predicted_class, confidence, probabilities) in zip(
            request.samples, predictions_data
        ):
            predictions.append(
                PredictionResult(
                    sample_id=sample.sample_id,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    probabilities=probabilities,
                )
            )

        processing_time_ms = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error during batch prediction: {str(e)}"
        )


@app.get("/model/info", response_model=dict)
async def get_model_info():
    """
    Get detailed information about the loaded model.

    Returns:
        dict: Model metadata and specifications
    """
    if not model_manager.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        metadata = model_manager.get_metadata()
        return {
            "model_loaded": True,
            "metadata": metadata,
            "expected_embedding_dim": model_manager.get_expected_embedding_dim(),
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail, timestamp=datetime.now().isoformat()
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc),
            timestamp=datetime.now().isoformat(),
        ).model_dump(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
