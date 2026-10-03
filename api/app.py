from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    TextAnalysisRequest,
    TextAnalysisResponse,
    HealthCheckResponse
)
from src.pipeline import ABSAPipeline

# Global pipeline instance
pipeline_instance: ABSAPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance
    ate_path = Path("artifacts/crf_ate_model.joblib")
    asc_path = Path("artifacts/logreg_asc_pipeline.joblib")

    if ate_path.exists() and asc_path.exists():
        pipeline_instance = ABSAPipeline(
            ate_model_path=ate_path,
            asc_pipeline_path=asc_path
        )
        print("[INFO] Model artifacts loaded into memory successfully.")
    else:
        print("[WARNING] Model artifacts not found. API running in degraded state.")
    
    yield
    
    # Teardown logic
    pipeline_instance = None
    print("[INFO] Model resources cleared.")


app = FastAPI(
    title="Aspect-Based Sentiment Analysis (ABSA) API",
    description="Linguistically-grounded ABSA engine utilizing CRF sequence labeling and dependency parsing.",
    version="1.0.0",
    lifespan=lifespan
)

# Standard CORS middleware for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthCheckResponse, tags=["Monitoring"])
def health_check():
    return HealthCheckResponse(
        status="healthy",
        models_loaded=pipeline_instance is not None,
        version="1.0.0"
    )


@app.post(
    "/analyze",
    response_model=TextAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
def analyze_aspect_sentiment(payload: TextAnalysisRequest):
    if pipeline_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ABSA Pipeline artifacts are unavailable. Train models before requesting inference."
        )

    try:
        results = pipeline_instance.predict(payload.text)
        return TextAnalysisResponse(**results)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}"
        )