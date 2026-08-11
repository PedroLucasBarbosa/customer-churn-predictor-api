"""
API tests using FastAPI's TestClient (built on httpx).

Run:
    pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 427.5,
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "model_type" in body
    assert "metrics" in body


def test_predict_high_risk_customer():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["churn_prediction"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in ("Low", "Medium", "High")


def test_predict_low_risk_customer():
    low_risk_payload = dict(VALID_PAYLOAD)
    low_risk_payload.update(
        {
            "tenure": 60,
            "Contract": "Two year",
            "InternetService": "DSL",
            "TechSupport": "Yes",
            "OnlineSecurity": "Yes",
            "PaymentMethod": "Credit card (automatic)",
        }
    )
    response = client.post("/predict", json=low_risk_payload)
    assert response.status_code == 200
    body = response.json()
    # Not a hard guarantee given model randomness, but should generally trend low
    assert body["churn_probability"] < 0.9


def test_predict_invalid_payload_returns_422():
    invalid_payload = dict(VALID_PAYLOAD)
    invalid_payload["Contract"] = "Not-a-real-contract-type"
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_missing_field_returns_422():
    incomplete_payload = dict(VALID_PAYLOAD)
    del incomplete_payload["tenure"]
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422
