"""FastAPI smoke tests."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture()
def client():
    return TestClient(app)


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_models_endpoint(client: TestClient):
    r = client.get("/models")
    if r.status_code == 503:
        pytest.skip("models not trained")
    assert r.status_code == 200
    data = r.json()
    assert "metrics" in data


def test_predict_endpoint(client: TestClient):
    payload = {"timestamp": "2025-09-01T14:00:00", "horizon": "1hr", "model": "naive"}
    r = client.post("/predict", json=payload)
    if r.status_code == 503:
        pytest.skip("models missing")
    assert r.status_code == 200
    out = r.json()
    assert "sales_forecast" in out
