"""
test_api.py — Tests de l'API FastAPI (endpoints /health, /predict, /model-info)
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api.main as api_module
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.zeros((1, 14))

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])

    api_module.preprocessor = mock_preprocessor
    api_module.model = mock_model
    api_module._model_loaded = True

    return TestClient(api_module.app)


VALID_PAYLOAD = {
    "machine_type":             "CNC",
    "operating_mode":           "normal",
    "vibration_rms":            0.9,
    "temperature_motor":        42.0,
    "current_phase_avg":        8.0,
    "pressure_level":           50.0,
    "rpm":                      850.0,
    "hours_since_maintenance":  50.0,
    "ambient_temp":             13.0,
}


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data


def test_predict_returns_200(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_structure(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    assert "prediction"     in data
    assert "probability"    in data
    assert "risk_level"     in data
    assert "recommendation" in data


def test_predict_binary_output(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    assert data["prediction"] in [0, 1]


def test_predict_probability_range(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0


def test_model_info_returns_metrics(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "metrics" in data
    assert "ROC-AUC" in data["metrics"]


def test_predict_invalid_machine_type(client):
    payload = {**VALID_PAYLOAD, "machine_type": "INVALID_TYPE"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_out_of_range_temperature(client):
    payload = {**VALID_PAYLOAD, "temperature_motor": 999.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422