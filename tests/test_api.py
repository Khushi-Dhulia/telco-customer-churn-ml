import pytest
from fastapi.testclient import TestClient

from src.api import app


SAMPLE_CUSTOMER = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": "29.85",
}


def test_health():
    try:
        client = TestClient(app)
        response = client.get("/health")
    except RuntimeError:
        pytest.skip("Train the model before running API tests.")
    assert response.status_code == 200


def test_prediction():
    try:
        client = TestClient(app)
        response = client.post("/predict", json=SAMPLE_CUSTOMER)
    except RuntimeError:
        pytest.skip("Train the model before running API tests.")
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert 0 <= body["churn_probability"] <= 1
