import shutil
import joblib
import pytest
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

from mlProject.components.autonomous_command_grid import AutonomousCommandGrid, CommandGridError
from mlProject.components.data_transformation import NUMERIC_FEATURES
from mlProject.utils.model_registry import load_registry, save_registry


TEST_ROOT = Path("artifacts/test_command_grid")


@pytest.fixture
def grid_env():
    """Sets up an isolated registry with two model versions for fleet tests."""
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    model_dir = TEST_ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    def _make_model(path, r2, rmse):
        X = np.random.rand(20, len(NUMERIC_FEATURES))
        y = np.random.rand(20)
        joblib.dump(LinearRegression().fit(X, y), path)
        return {"r2": r2, "rmse": rmse}

    v1_path = model_dir / "v1_model.joblib"
    v2_path = model_dir / "v2_model.joblib"
    v1_metrics = _make_model(v1_path, r2=0.40, rmse=0.60)
    v2_metrics = _make_model(v2_path, r2=0.55, rmse=0.50)

    registry_path = TEST_ROOT / "model_registry.json"
    registry = {
        "production": "v1",
        "staging": "v2",
        "versions": [
            {"id": "v1", "path": str(v1_path), "metrics": v1_metrics, "params": {}, "date": "2026-01-01T00:00:00", "status": "production"},
            {"id": "v2", "path": str(v2_path), "metrics": v2_metrics, "params": {}, "date": "2026-02-01T00:00:00", "status": "staging"},
        ],
    }
    save_registry(registry_path, registry)

    db_path = TEST_ROOT / "predictions.db"

    yield {
        "grid": AutonomousCommandGrid(db_path=str(db_path), registry_path=str(registry_path)),
        "registry_path": registry_path,
    }

    shutil.rmtree(TEST_ROOT, ignore_errors=True)


def test_register_and_list_node(grid_env):
    grid = grid_env["grid"]
    node = grid.register_node("wine-prod-node", "v1")
    assert node["target_ref"] == "v1"
    assert node["status"] == "active"
    nodes = grid.list_nodes()
    assert any(n["id"] == node["id"] for n in nodes)


def test_register_node_unknown_version_raises(grid_env):
    grid = grid_env["grid"]
    with pytest.raises(CommandGridError):
        grid.register_node("bad-node", "does-not-exist")


def test_deregister_node(grid_env):
    grid = grid_env["grid"]
    node = grid.register_node("temp-node", "v1")
    grid.deregister_node(node["id"])
    with pytest.raises(CommandGridError):
        grid.get_node(node["id"])


def test_execute_command_promote(grid_env):
    grid = grid_env["grid"]
    node = grid.register_node("staging-node", "v2")
    results = grid.execute_command([node["id"]], "promote")
    assert results[0]["status"] == "success"
    registry = load_registry(grid_env["registry_path"])
    assert registry["production"] == "v2"


def test_execute_command_health_check(grid_env):
    grid = grid_env["grid"]
    node = grid.register_node("health-node", "v1")
    results = grid.execute_command([node["id"]], "health_check")
    assert results[0]["status"] == "success"
    assert results[0]["detail"]["model_artifact_present"] is True


def test_execute_command_unsupported_raises(grid_env):
    grid = grid_env["grid"]
    node = grid.register_node("bad-command-node", "v1")
    with pytest.raises(CommandGridError):
        grid.execute_command([node["id"]], "self_destruct")


def test_execute_command_unknown_node_reports_error_not_raise(grid_env):
    grid = grid_env["grid"]
    results = grid.execute_command([9999], "health_check")
    assert results[0]["status"] == "error"


def test_command_history_recorded(grid_env):
    grid = grid_env["grid"]
    node = grid.register_node("history-node", "v1")
    grid.execute_command([node["id"]], "health_check")
    history = grid.get_command_history(node_id=node["id"])
    assert len(history) == 1
    assert history[0]["command"] == "health_check"


def test_fleet_health_aggregates_nodes(grid_env):
    grid = grid_env["grid"]
    grid.register_node("node-a", "v1")
    grid.register_node("node-b", "v2")
    health = grid.get_fleet_health()
    assert health["fleet_size"] == 2
    assert len(health["nodes"]) == 2
    assert "system" in health


def test_report_incident_low_severity_no_auto_remediation(grid_env):
    grid = grid_env["grid"]
    node = grid.register_node("incident-node", "v1")
    incident = grid.report_incident("Elevated latency", "low", [node["id"]], "Latency spike observed")
    assert incident["status"] == "open"
    assert incident["auto_remediated"] is False


def test_report_critical_incident_triggers_retrain(grid_env):
    grid = grid_env["grid"]
    node = grid.register_node("critical-node", "v1")
    incident = grid.report_incident("Model serving errors spiking", "critical", [node["id"]], "5xx errors across fleet")
    assert incident["severity"] == "critical"
    assert incident["auto_remediated"] is True


def test_resolve_incident(grid_env):
    grid = grid_env["grid"]
    node = grid.register_node("resolve-node", "v1")
    incident = grid.report_incident("Minor blip", "low", [node["id"]])
    resolved = grid.resolve_incident(incident["id"], "False alarm, no action needed")
    assert resolved["status"] == "resolved"
    assert resolved["resolution_notes"] == "False alarm, no action needed"


def test_list_incidents_filtered_by_status(grid_env):
    grid = grid_env["grid"]
    node = grid.register_node("filter-node", "v1")
    incident = grid.report_incident("Filterable incident", "medium", [node["id"]])
    grid.resolve_incident(incident["id"])
    open_incidents = grid.list_incidents(status="open")
    resolved_incidents = grid.list_incidents(status="resolved")
    assert all(i["status"] == "open" for i in open_incidents)
    assert any(i["id"] == incident["id"] for i in resolved_incidents)


def test_optimize_allocation_recommends_best_scoring_version(grid_env):
    grid = grid_env["grid"]
    result = grid.optimize_allocation()
    # v2 has higher r2 and lower rmse, should score higher than v1
    assert result["recommended_version"] == "v2"
    assert len(result["ranked"]) == 2