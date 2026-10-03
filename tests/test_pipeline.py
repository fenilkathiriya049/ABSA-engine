import pytest
from pathlib import Path
from src.pipeline import ABSAPipeline

@pytest.fixture(scope="module")
def pipeline():
    ate_path = Path("artifacts/crf_ate_model.joblib")
    asc_path = Path("artifacts/logreg_asc_pipeline.joblib")
    if not (ate_path.exists() and asc_path.exists()):
        pytest.skip("Model artifacts not found. Run training before tests.")
    return ABSAPipeline(ate_model_path=ate_path, asc_pipeline_path=asc_path)


def test_pipeline_inference_structure(pipeline):
    text = "The crust is thin and crunchy, but the staff is aloof."
    result = pipeline.predict(text)

    assert "text" in result
    assert "aspects" in result
    assert isinstance(result["aspects"], list)
    
    if len(result["aspects"]) > 0:
        first = result["aspects"][0]
        assert "term" in first
        assert "sentiment" in first
        assert "confidence" in first
        assert "span" in first
        assert first["sentiment"] in ["positive", "negative", "neutral"]


def test_pipeline_no_aspects_found(pipeline):
    text = "Yesterday morning at eight o'clock."
    result = pipeline.predict(text)
    assert result["text"] == text
    assert isinstance(result["aspects"], list)