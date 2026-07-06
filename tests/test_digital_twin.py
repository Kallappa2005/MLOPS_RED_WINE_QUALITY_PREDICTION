import json
import shutil
import joblib
import pytest
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

from mlProject.components.digital_twin import DigitalTwin, DigitalTwinError
from mlProject.components.data_transformation import NUMERIC_FEATURES
from mlProject.utils.model_registry import load_registry, save_registry


TEST_ROOT = Path("artifacts/test_digital_twin")


def _sample_row():
    return {feature: 1.0 for feature in NUMERIC_FEATURES}


@pytest.fixture
def twin_env():
    """Sets up an isolated registry + trained dummy model for twin tests."""
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    model_dir = TEST_ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    X = np.random.rand(20, len(NUMERIC_FEATURES))
    y = np.random.rand(20)
    model = LinearRegression().fit(X, y)
    model_path = model_dir / "v1_model.joblib"
    joblib.dump(model, model_path)

    registry_path = TEST_ROOT / "model_registry.json"
    registry = {
        "production": "v1",
        "staging": None,
        "versions": [
            {
                "id": "v1",
                "path": str(model_path),
                "metrics": {"rmse": 0.6, "r2": 0.4},
                "params": {},
                "date": "2026-01-01T00:00:00",
                "status": "production",
            }
        ],
    }
    save_registry(registry_path, registry)

    db_path = TEST_ROOT / "predictions.db"
    twins_dir = TEST_ROOT / "digital_twins"

    yield {
        "twin": DigitalTwin(db_path=str(db_path), twins_dir=str(twins_dir), registry_path=str(registry_path)),
        "registry_path": registry_path,
        "model_path": model_path,
    }

    shutil.rmtree(TEST_ROOT, ignore_errors=True)


def test_create_twin_from_production(twin_env):
    dt = twin_env["twin"]
    twin = dt.create_twin("staging-mirror")
    assert twin["source_version_id"] == "v1"
    assert twin["status"] == "active"
    assert Path(twin["model_path"]).exists()


def test_create_twin_unknown_version_raises(twin_env):
    dt = twin_env["twin"]
    with pytest.raises(DigitalTwinError):
        dt.create_twin("bad-twin", source_version_id="does-not-exist")


def test_list_and_get_twin(twin_env):
    dt = twin_env["twin"]
    created = dt.create_twin("mirror-1")
    twins = dt.list_twins()
    assert any(t["id"] == created["id"] for t in twins)
    fetched = dt.get_twin(created["id"])
    assert fetched["id"] == created["id"]


def test_delete_twin_removes_artifact(twin_env):
    dt = twin_env["twin"]
    twin = dt.create_twin("throwaway")
    model_path = Path(twin["model_path"])
    assert model_path.exists()
    dt.delete_twin(twin["id"])
    assert not model_path.exists()
    with pytest.raises(DigitalTwinError):
        dt.get_twin(twin["id"])


def test_run_simulation_does_not_touch_production_model(twin_env):
    dt = twin_env["twin"]
    twin = dt.create_twin("sim-twin")
    original_bytes = Path(twin_env["model_path"]).read_bytes()

    result = dt.run_simulation(twin["id"], "baseline-scenario", [_sample_row(), _sample_row()])
    assert result["row_count"] == 2
    assert "mean_prediction" in result

    # production model file must be untouched by simulating against the twin
    assert Path(twin_env["model_path"]).read_bytes() == original_bytes


def test_run_simulation_missing_feature_raises(twin_env):
    dt = twin_env["twin"]
    twin = dt.create_twin("bad-input-twin")
    incomplete_row = _sample_row()
    incomplete_row.pop(NUMERIC_FEATURES[0])
    with pytest.raises(DigitalTwinError):
        dt.run_simulation(twin["id"], "incomplete-scenario", [incomplete_row])


def test_simulation_history_is_recorded(twin_env):
    dt = twin_env["twin"]
    twin = dt.create_twin("history-twin")
    dt.run_simulation(twin["id"], "scenario-a", [_sample_row()])
    history = dt.get_simulation_history(twin["id"])
    assert len(history) == 1
    assert history[0]["scenario_name"] == "scenario-a"


def test_sync_state_updates_source_version(twin_env):
    dt = twin_env["twin"]
    twin = dt.create_twin("sync-twin")

    # Register a new production version
    registry = load_registry(twin_env["registry_path"])
    new_model_path = TEST_ROOT / "models" / "v2_model.joblib"
    joblib.dump(LinearRegression().fit(np.random.rand(10, len(NUMERIC_FEATURES)), np.random.rand(10)), new_model_path)
    registry["production"] = "v2"
    registry["versions"].append({
        "id": "v2",
        "path": str(new_model_path),
        "metrics": {"rmse": 0.5, "r2": 0.5},
        "params": {},
        "date": "2026-02-01T00:00:00",
        "status": "production",
    })
    save_registry(twin_env["registry_path"], registry)

    synced = dt.sync_state(twin["id"])
    assert synced["source_version_id"] == "v2"


def test_detect_twin_drift_when_out_of_sync(twin_env):
    dt = twin_env["twin"]
    twin = dt.create_twin("drift-twin")

    registry = load_registry(twin_env["registry_path"])
    new_model_path = TEST_ROOT / "models" / "v3_model.joblib"
    joblib.dump(LinearRegression().fit(np.random.rand(10, len(NUMERIC_FEATURES)), np.random.rand(10)), new_model_path)
    registry["production"] = "v3"
    registry["versions"].append({
        "id": "v3",
        "path": str(new_model_path),
        "metrics": {},
        "params": {},
        "date": "2026-03-01T00:00:00",
        "status": "production",
    })
    save_registry(twin_env["registry_path"], registry)

    report = dt.detect_twin_drift(twin["id"])
    assert report["drift_detected"] is True
    assert report["details"]["current_production_version"] == "v3"


def test_detect_twin_drift_when_in_sync(twin_env):
    dt = twin_env["twin"]
    twin = dt.create_twin("in-sync-twin")
    report = dt.detect_twin_drift(twin["id"])
    assert report["drift_detected"] is False
