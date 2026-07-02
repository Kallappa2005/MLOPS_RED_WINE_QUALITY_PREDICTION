import os
import json
import pytest
import pandas as pd
from pathlib import Path
from app import app
from mlProject.components.ai_assistant import PredictionAssistant
from mlProject.components.data_quality import QualityValidator
from mlProject.components.feature_store import EnterpriseFeatureStore
from mlProject.components.alerting_framework import AlertEngine
from mlProject.components.drift_intelligence import AdvancedDriftEngine
from mlProject.components.governance import ModelGovernanceEngine
from mlProject.components.hyperparameter_optimizer import HyperparameterOptimizer
from mlProject.components.federated_learning import FederatedCoordinator
from mlProject.components.security import create_token

@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["DEBUG"] = False
    with app.test_client() as client:
        yield client

@pytest.fixture
def admin_headers():
    token = create_token("admin", "Admin")
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def viewer_headers():
    token = create_token("viewer", "Viewer")
    return {"Authorization": f"Bearer {token}"}

# Phase 11: AI-Powered Prediction Assistant
def test_ai_assistant_component():
    assistant = PredictionAssistant()
    features = {
        "alcohol": 12.0,
        "volatile acidity": 0.3
    }
    explanation = assistant.explain_prediction_nl(features, 6.8)
    assert "quality" in explanation.lower()
    assert "alcohol" in explanation
    
    recs = assistant.generate_recommendations(features, 5.5)
    assert len(recs) > 0
    assert any(r.get("feature") == "sulphates" for r in recs if isinstance(r, dict))

def test_ai_assistant_endpoint(client, viewer_headers):
    response = client.post("/assistant/chat", json={
        "features": {
            "alcohol": 12.0,
            "volatile acidity": 0.3
        },
        "prediction": 6.8
    }, headers=viewer_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "explanation" in data
    assert "recommendations" in data

# Phase 12: Data Quality Intelligence Platform
def test_data_quality_component():
    validator = QualityValidator()
    df = pd.DataFrame({
        "alcohol": [10.0, 10.2, 10.1, 10.3, 15.0], # 15.0 is outlier/high
        "volatile acidity": [0.5, 0.52, 0.51, 0.49, 0.5],
        "pH": [3.3, 3.31, 3.29, 3.32, 3.3]
    })
    report = validator.analyze_dataset_quality(df)
    assert "quality_score" in report
    assert "missing_count" in report
    assert "anomalies" in report

def test_data_quality_endpoint(client, viewer_headers):
    response = client.get("/data-quality/check", headers=viewer_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "quality_score" in data

# Phase 13: Feature Store Management System
def test_feature_store_component():
    db_path = "artifacts/test_predictions_feature_store.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    try:
        store = EnterpriseFeatureStore(db_path=db_path)
        catalog = store.get_feature_catalog()
        assert len(catalog) >= 4
        
        success = store.register_feature("citric acid", "v1", "Citric acid level", "float", 0.27)
        assert success is True
        catalog_updated = store.get_feature_catalog()
        assert any(f["feature_name"] == "citric acid" for f in catalog_updated)
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

def test_feature_store_endpoints(client, admin_headers, viewer_headers):
    # Retrieve
    res_get = client.get("/feature-store/serve", headers=viewer_headers)
    assert res_get.status_code == 200
    assert isinstance(res_get.get_json(), list)
    
    # Register
    res_post = client.post("/feature-store/register", json={
        "name": "fixed acidity",
        "version": "v1",
        "description": "Fixed acid amount",
        "data_type": "float",
        "mean_val": 8.3
    }, headers=admin_headers)
    assert res_post.status_code == 200
    assert "registered successfully" in res_post.get_json()["message"]

# Phase 14: Enterprise Alerting Framework
def test_alerting_component():
    db_path = "artifacts/test_predictions_alerting.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    try:
        engine = AlertEngine(db_path=db_path)
        active = engine.get_active_alerts()
        assert len(active) == 0
        
        triggered = engine.trigger_alert("CPU_Usage", 92.5, 80.0, "CRITICAL", "CPU limit exceeded")
        assert triggered is True
        
        active_updated = engine.get_active_alerts()
        assert len(active_updated) == 1
        alert_id = active_updated[0]["id"]
        
        resolved = engine.resolve_alert(alert_id)
        assert resolved is True
        assert len(engine.get_active_alerts()) == 0
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

def test_alerting_endpoints(client, admin_headers, viewer_headers):
    # GET alerts
    res_get = client.get("/alerts/active", headers=viewer_headers)
    assert res_get.status_code == 200
    
    # Trigger alert
    res_trigger = client.post("/alerts/trigger", json={
        "metric": "MemoryLeak",
        "value": 0.95,
        "threshold": 0.85,
        "severity": "WARNING",
        "message": "Memory threshold warnings"
    }, headers=admin_headers)
    assert res_trigger.status_code == 200
    
    # Get alerts to resolve
    res_get2 = client.get("/alerts/active", headers=viewer_headers)
    active_alerts = res_get2.get_json()
    assert len(active_alerts) > 0
    alert_id = active_alerts[0]["id"]
    
    # Resolve alert
    res_resolve = client.post(f"/alerts/resolve/{alert_id}", headers=admin_headers)
    assert res_resolve.status_code == 200

# Phase 15: Advanced Drift Intelligence Platform
def test_drift_intelligence_component():
    engine = AdvancedDriftEngine()
    # Mock some recent prediction samples
    samples = [
        {"alcohol": 12.5, "volatile acidity": 0.3, "pH": 3.2, "sulphates": 0.6, "prediction": 5.8},
        {"alcohol": 12.6, "volatile acidity": 0.31, "pH": 3.19, "sulphates": 0.61, "prediction": 5.9},
        {"alcohol": 12.4, "volatile acidity": 0.29, "pH": 3.21, "sulphates": 0.59, "prediction": 5.7},
        {"alcohol": 12.5, "volatile acidity": 0.3, "pH": 3.2, "sulphates": 0.6, "prediction": 5.8},
        {"alcohol": 12.5, "volatile acidity": 0.3, "pH": 3.2, "sulphates": 0.6, "prediction": 5.8}
    ]
    f_drift = engine.detect_feature_drift(samples)
    assert "drift_detected" in f_drift
    assert "drift_ratios" in f_drift
    
    c_drift = engine.detect_concept_drift(samples)
    assert "concept_drift_detected" in c_drift

def test_drift_intelligence_endpoint(client, viewer_headers):
    response = client.get("/drift/advanced", headers=viewer_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "feature_drift" in data
    assert "concept_drift" in data

# Phase 16 & 19: Model Governance Platform & Explainable MLOps Governance
def test_governance_component():
    gov_path = "artifacts/test_governance_manifest.json"
    if os.path.exists(gov_path):
        os.remove(gov_path)
    try:
        engine = ModelGovernanceEngine(governance_path=gov_path)
        req = engine.request_promotion_approval("v99", "EngineerBob")
        assert req["version_id"] == "v99"
        assert req["status"] == "PENDING_APPROVAL"
        
        manifest = engine.load_manifest()
        assert len(manifest["approvals"]) == 1
        
        approved = engine.approve_promotion("v99", "AdminAlice")
        assert approved is True
        
        manifest_updated = engine.load_manifest()
        assert manifest_updated["approvals"][0]["status"] == "APPROVED"
        assert len(manifest_updated["audit_history"]) == 1
        
        comp = engine.run_compliance_check("v99", 0.55, 0.42)
        assert comp["status"] == "COMPLIANT"
    finally:
        if os.path.exists(gov_path):
            os.remove(gov_path)

def test_governance_endpoints(client, admin_headers, viewer_headers):
    # Get manifest
    res_get = client.get("/governance/manifest", headers=viewer_headers)
    assert res_get.status_code == 200
    
    # Request promotion
    res_req = client.post("/governance/request-promotion", json={
        "version_id": "v101",
        "requested_by": "EngineerTest"
    }, headers=admin_headers)
    assert res_req.status_code == 200
    
    # Compliance check
    res_comp = client.post("/governance/compliance-check", json={
        "version_id": "v101",
        "rmse": 0.60,
        "r2": 0.40
    }, headers=admin_headers)
    assert res_comp.status_code == 200
    assert res_comp.get_json()["status"] == "COMPLIANT"
    
    # Approve
    res_app = client.post("/governance/approve", json={
        "version_id": "v101",
        "approved_by": "AdminTest"
    }, headers=admin_headers)
    assert res_app.status_code == 200 or res_app.status_code == 400

# Phase 17: Automated Hyperparameter Optimization
def test_hyperparameter_optimizer_component():
    opt_path = "artifacts/test_optimization_history.json"
    if os.path.exists(opt_path):
        os.remove(opt_path)
    try:
        opt = HyperparameterOptimizer(history_path=opt_path)
        sweep = opt.run_optimization_sweep("ElasticNet", iterations=3)
        assert sweep["model_type"] == "ElasticNet"
        assert len(sweep["trials"]) == 3
        assert sweep["best_r2"] > 0
        
        history = opt.get_sweep_history()
        assert len(history) == 1
    finally:
        if os.path.exists(opt_path):
            os.remove(opt_path)

def test_hyperparameter_optimizer_endpoint(client, admin_headers):
    response = client.post("/optimization/run", json={
        "model_type": "RandomForest",
        "strategy": "random",
        "iterations": 4
    }, headers=admin_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["trials"]) == 4
    assert "best_params" in data

# Phase 18: Federated Learning Framework
def test_federated_learning_component():
    coord = FederatedCoordinator()
    report = coord.collect_and_aggregate()
    assert report["nodes_participated"] == 3
    assert "aggregated_parameters" in report
    assert "weights" in report["aggregated_parameters"]
    assert "intercept" in report["aggregated_parameters"]

def test_federated_learning_endpoint(client, admin_headers):
    response = client.post("/federated/aggregate", headers=admin_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "aggregated_parameters" in data
    assert data["nodes_participated"] == 3
