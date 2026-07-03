"""
tests/test_train_endpoint.py

Unit + integration tests for the /train endpoint security fix.
Run with:  pytest tests/test_train_endpoint.py -v
"""

import os
import threading
from collections import deque
import pytest

# Set the env vars BEFORE importing app so the app picks them up
os.environ["TRAIN_SECRET"] = "test-secret-abc123"
os.environ["FLASK_ENV"] = "testing"

# Patch subprocess.run so we never actually run main.py during tests
import subprocess
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def reset_training_state():
    """Reset global training state between every test."""
    import app as application

    application.is_training = False
    application.training_log.clear()
    # Release the lock if it was acquired by a previous test
    if application._training_lock.locked():
        try:
            application._training_lock.release()
        except RuntimeError:
            pass
    yield


@pytest.fixture
def client():
    import app as application

    application.app.config["TESTING"] = True
    with application.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# 1. No token → 401
# ---------------------------------------------------------------------------
def test_train_no_token_returns_401(client):
    response = client.get("/train")
    assert (
        response.status_code == 401
    ), "Unauthenticated request must be rejected with 401"


# ---------------------------------------------------------------------------
# 2. Wrong token → 401
# ---------------------------------------------------------------------------
def test_train_wrong_token_returns_401(client):
    response = client.get("/train?token=wrongtoken")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# 3. Correct token via query param → 200 and training starts
# ---------------------------------------------------------------------------
def test_train_correct_token_query_param(client):
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "done"
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result):
        response = client.get("/train?token=test-secret-abc123")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# 4. Correct token via header → 200
# ---------------------------------------------------------------------------
def test_train_correct_token_via_header(client):
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "done"
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result):
        response = client.get("/train", headers={"X-Train-Token": "test-secret-abc123"})
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# 5. Double-trigger while training is running → returns "already in progress"
# ---------------------------------------------------------------------------
def test_train_locked_returns_already_in_progress(client):
    import app as application

    # Simulate a training run already in progress by holding the lock
    application._training_lock.acquire(blocking=False)
    application.is_training = True

    try:
        response = client.get("/train?token=test-secret-abc123")
        assert response.status_code == 200
        data = response.data.decode()
        assert "already in progress" in data.lower()
    finally:
        application._training_lock.release()
        application.is_training = False


# ---------------------------------------------------------------------------
# 6. /train/status — no token → 401
# ---------------------------------------------------------------------------
def test_train_status_no_token_returns_401(client):
    response = client.get("/train/status")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# 7. /train/status — correct token → JSON with is_training key
# ---------------------------------------------------------------------------
def test_train_status_correct_token(client):
    response = client.get("/train/status?token=test-secret-abc123")
    assert response.status_code == 200
    data = response.get_json()
    assert "is_training" in data
    assert "log" in data


# ---------------------------------------------------------------------------
# 8. No TRAIN_SECRET set at all → fail-closed (401)
# ---------------------------------------------------------------------------
def test_train_no_env_var_fails_closed(client):
    original = os.environ.pop("TRAIN_SECRET", None)
    try:
        response = client.get("/train?token=anything")
        assert (
            response.status_code == 401
        ), "When TRAIN_SECRET is unset the endpoint must be fully disabled"
    finally:
        if original:
            os.environ["TRAIN_SECRET"] = original


# ---------------------------------------------------------------------------
# 9. Predict route is unaffected and still works without a token
# ---------------------------------------------------------------------------
def test_predict_route_works_without_token(client):
    response = client.get("/predict")
    assert response.status_code == 200


def test_predict_route_validation_error(client):
    response = client.post("/predict", data={
        "fixed_acidity": -1.0,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    })
    assert response.status_code == 400
    assert "validation error" in response.data.decode().lower()

