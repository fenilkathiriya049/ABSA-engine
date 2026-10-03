from fastapi import FastAPI, HTTPException
from pathlib import Path
from api.schemas import AnalyzeRequest, AnalyzeResponse
from src.pipeline import ABSAPipeline

app = FastAPI(
    title="Aspect-Based Sentiment Analysis API",
    version="1.0.0"
)

ATE_MODEL_PATH = Path("artifacts/crf_ate_model.joblib")
ASC_MODEL_PATH = Path("artifacts/logreg_asc_pipeline.joblib")

# Lazy loading / singleton
pipeline = None

@app.on_event("startup")
def load_models():
    global pipeline
    if ATE_MODEL_PATH.exists() and ASC_MODEL_PATH.exists():
        pipeline = ABSAPipeline(ATE_MODEL_PATH, ASC_MODEL_PATH)
    else:
        print("[WARNING] Artifacts not found. Run training scripts before making inference requests.")

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(request: AnalyzeRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models are not loaded or initialized.")
    results = pipeline.predict(request.text)
    return AnalyzeResponse(text=request.text, aspects=results)

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": pipeline is not None}