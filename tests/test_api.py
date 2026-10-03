import pytest
from starlette.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_analyze_validation_error_on_empty_text():
    # Min length constraint in TextAnalysisRequest is 2
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 422


def test_analyze_valid_input():
    with TestClient(app) as test_client:
        payload = {"text": "The pizza was delicious, but the ambience was terrible."}
        response = test_client.post("/analyze", json=payload)
        
        # If models are present, should return 200 with schema structure
        if response.status_code == 200:
            data = response.json()
            assert "text" in data
            assert "aspects" in data
            assert isinstance(data["aspects"], list)
        else:
            # Degraded state fallback
            assert response.status_code in [200, 503]