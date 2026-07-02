import os
import json
import pytest
from pathlib import Path
from app import app
from mlProject.components.security import create_token, decode_token, USER_DB
from mlProject.components.observability import APILogger, ObservabilityCollector
from mlProject.components.retraining import RetrainingEngine

@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["DEBUG"] = False
    with app.test_client() as client:
        yield client

def test_jwt_generation_and_decoding():
    token = create_token("admin", "Admin")
    assert token is not None
    payload = decode_token(token)
    assert isinstance(payload, dict)
    assert payload["sub"] == "admin"
    assert payload["role"] == "Admin"

def test_login_endpoint(client):
    response = client.post("/auth/login", json={
        "username": "admin",
        "password": "admin_password"
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "token" in data
    assert data["role"] == "Admin"
    assert data["username"] == "admin"

def test_login_invalid_credentials(client):
    response = client.post("/auth/login", json={
        "username": "admin",
        "password": "wrong_password"
    })
    assert response.status_code == 401

def test_rbac_restrictions(client):
    # Viewer role token
    viewer_token = create_token("viewer", "Viewer")
    headers = {"Authorization": f"Bearer {viewer_token}"}

    # Viewer should be able to access read-only health and analytics endpoints
    res_health = client.get("/observability/health", headers=headers)
    assert res_health.status_code == 200

    res_analytics = client.get("/api/analytics", headers=headers)
    assert res_analytics.status_code == 200

    # Viewer should NOT be able to trigger retraining (forbidden - 403)
    res_retrain = client.post("/retrain/trigger", json={"reason": "Test trigger"}, headers=headers)
    assert res_retrain.status_code == 403

    # Viewer should NOT be able to promote model
    res_promote = client.post("/registry/promote", json={"version_id": "v1"}, headers=headers)
    assert res_promote.status_code == 403

def test_api_analytics_logger():
    api_logger = APILogger(db_path="artifacts/test_predictions.db")
    api_logger.log_request("/predict", "POST", 200, 15.5, "127.0.0.1")
    analytics = api_logger.get_analytics()
    assert analytics["total_requests"] >= 1
    assert analytics["avg_latency_ms"] > 0
    # Clean up test DB if needed
    if os.path.exists("artifacts/test_predictions.db"):
        try:
            os.remove("artifacts/test_predictions.db")
        except Exception:
            pass

def test_system_observability():
    collector = ObservabilityCollector(db_path="artifacts/test_predictions.db")
    health = collector.get_system_health()
    assert "status" in health
    assert "cpu_usage_pct" in health
    assert "ram_usage_pct" in health
