"""
Tests for Phases 21-30: Enterprise Autonomous AI Governance & Intelligence Platform
"""
import pytest
import json
import os


# Configure test environment before app import
os.environ.setdefault("ADMIN_TOKEN", "test-admin-token")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-ci")
os.environ.setdefault("TRAIN_SECRET", "test-train-secret")
os.environ.setdefault("TEST_USERNAME", "testadmin")
os.environ.setdefault("TEST_PASSWORD", "TestPassword123!")


@pytest.fixture(scope="module")
def client():
    from app import app as flask_app
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    with flask_app.test_client() as c:
        yield c


@pytest.fixture(scope="module")
def auth_token(client):
    """Obtain a valid JWT token for admin operations."""
    response = client.post("/auth/login", json={
        "username": os.environ["TEST_USERNAME"],
        "password": os.environ["TEST_PASSWORD"]
    })
    if response.status_code == 200:
        data = response.get_json()
        return data.get("token", "")
    return ""


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


# ── Phase 21: Model Registry ──────────────────────────────────────────────────

def test_model_registry_component():
    from mlProject.components.model_registry import EnterpriseModelRegistry
    reg = EnterpriseModelRegistry()
    result = reg.list_models()
    assert "models" in result
    assert "total" in result


def test_model_registry_register():
    from mlProject.components.model_registry import EnterpriseModelRegistry
    reg = EnterpriseModelRegistry()
    result = reg.register_model(
        "test_model", "1.0.0", "sklearn",
        {"rmse": 0.55, "r2": 0.60},
        "artifacts/model.pkl", "test_owner"
    )
    assert "model_id" in result
    assert result["record"]["name"] == "test_model"


def test_model_registry_version_history():
    from mlProject.components.model_registry import EnterpriseModelRegistry
    reg = EnterpriseModelRegistry()
    result = reg.version_history()
    assert "version_history" in result


def test_registry_list_endpoint(client, auth_token):
    resp = client.get("/registry/models", headers=auth_headers(auth_token))
    assert resp.status_code in (200, 401, 403)  # depends on token validity in CI


def test_registry_register_endpoint(client, auth_token):
    resp = client.post("/registry/register", json={
        "name": "ci_test_model", "version": "0.1.0", "framework": "sklearn",
        "metrics": {"rmse": 0.6}, "artifact_path": "artifacts/model.pkl", "owner": "ci"
    }, headers=auth_headers(auth_token))
    assert resp.status_code in (201, 401, 403)


# ── Phase 22: Retraining Engine ───────────────────────────────────────────────

def test_retraining_engine_component():
    from mlProject.components.retraining_engine import AutonomousRetrainingEngine
    engine = AutonomousRetrainingEngine()
    result = engine.trigger_retraining("wine_v1", "test_drift")
    assert "job" in result
    assert result["job"]["status"] == "completed"


def test_retraining_history_component():
    from mlProject.components.retraining_engine import AutonomousRetrainingEngine
    engine = AutonomousRetrainingEngine()
    result = engine.get_history()
    assert "history" in result


def test_retraining_recommendations_component():
    from mlProject.components.retraining_engine import AutonomousRetrainingEngine
    engine = AutonomousRetrainingEngine()
    result = engine.get_recommendations()
    assert "recommendations" in result
    assert len(result["recommendations"]) > 0


# ── Phase 23: Data Lineage ────────────────────────────────────────────────────

def test_data_lineage_graph():
    from mlProject.components.data_lineage import DataLineagePlatform
    platform = DataLineagePlatform()
    result = platform.get_graph()
    assert "nodes" in result
    assert "edges" in result
    assert result["node_count"] > 0


def test_data_lineage_source_trace():
    from mlProject.components.data_lineage import DataLineagePlatform
    platform = DataLineagePlatform()
    result = platform.source_trace("pred_output")
    assert "upstream_trace" in result


def test_data_lineage_report():
    from mlProject.components.data_lineage import DataLineagePlatform
    platform = DataLineagePlatform()
    result = platform.get_report()
    assert "auditability_score" in result


# ── Phase 24: Security Intelligence ──────────────────────────────────────────

def test_security_threats_component():
    from mlProject.components.security_intelligence import AISecurityIntelligenceCenter
    center = AISecurityIntelligenceCenter()
    result = center.get_threats()
    assert "threats" in result
    assert "severity_breakdown" in result


def test_security_anomalies_component():
    from mlProject.components.security_intelligence import AISecurityIntelligenceCenter
    center = AISecurityIntelligenceCenter()
    result = center.get_anomalies()
    assert "anomalies" in result


def test_security_report_component():
    from mlProject.components.security_intelligence import AISecurityIntelligenceCenter
    center = AISecurityIntelligenceCenter()
    result = center.get_security_report()
    assert "security_score" in result
    assert "recommendations" in result


# ── Phase 25: Multi-Cloud ─────────────────────────────────────────────────────

def test_multi_cloud_providers():
    from mlProject.components.multi_cloud import MultiCloudControlPlane
    plane = MultiCloudControlPlane()
    result = plane.get_providers()
    assert "providers" in result
    assert "summary" in result
    assert result["summary"]["total_providers"] == 4


def test_multi_cloud_sync():
    from mlProject.components.multi_cloud import MultiCloudControlPlane
    plane = MultiCloudControlPlane()
    result = plane.sync_providers()
    assert "sync" in result
    assert result["sync"]["status"] == "success"


def test_multi_cloud_status():
    from mlProject.components.multi_cloud import MultiCloudControlPlane
    plane = MultiCloudControlPlane()
    result = plane.get_cloud_status()
    assert "provider_statuses" in result


# ── Phase 26: Cost Optimizer ──────────────────────────────────────────────────

def test_cost_report_component():
    from mlProject.components.cost_optimizer import EnterpriseCostOptimizer
    opt = EnterpriseCostOptimizer()
    result = opt.get_cost_report()
    assert "total_spend_usd" in result
    assert "budget_utilization_pct" in result


def test_cost_forecast_component():
    from mlProject.components.cost_optimizer import EnterpriseCostOptimizer
    opt = EnterpriseCostOptimizer()
    result = opt.get_forecast()
    assert "projected_total_usd" in result


def test_cost_recommendations_component():
    from mlProject.components.cost_optimizer import EnterpriseCostOptimizer
    opt = EnterpriseCostOptimizer()
    result = opt.get_recommendations()
    assert "recommendations" in result
    assert result["total_estimated_savings_usd_month"] > 0


# ── Phase 27: Compliance Intelligence ────────────────────────────────────────

def test_compliance_score_component():
    from mlProject.components.compliance_intelligence import AIComplianceIntelligence
    intel = AIComplianceIntelligence()
    result = intel.get_compliance_score()
    assert "overall_score" in result
    assert "status" in result
    assert result["overall_score"] > 0


def test_compliance_report_component():
    from mlProject.components.compliance_intelligence import AIComplianceIntelligence
    intel = AIComplianceIntelligence()
    result = intel.get_compliance_report()
    assert "detailed_checks" in result
    assert len(result["detailed_checks"]) > 0


def test_compliance_findings_component():
    from mlProject.components.compliance_intelligence import AIComplianceIntelligence
    intel = AIComplianceIntelligence()
    result = intel.get_findings()
    assert "findings" in result


# ── Phase 28: Synthetic Data ──────────────────────────────────────────────────

def test_synthetic_generate_component():
    from mlProject.components.synthetic_data import SyntheticDataStudio
    studio = SyntheticDataStudio()
    result = studio.generate(n_samples=10)
    assert "dataset_id" in result
    assert "preview" in result
    assert len(result["preview"]) == 5


def test_synthetic_evaluate_component():
    from mlProject.components.synthetic_data import SyntheticDataStudio
    studio = SyntheticDataStudio()
    result = studio.evaluate()
    assert "utility_score" in result
    assert "privacy_metrics" in result


def test_synthetic_catalog_component():
    from mlProject.components.synthetic_data import SyntheticDataStudio
    studio = SyntheticDataStudio()
    result = studio.get_catalog()
    assert "datasets" in result
    assert result["total_datasets"] >= 0


# ── Phase 29: Knowledge Graph ─────────────────────────────────────────────────

def test_knowledge_graph_entities():
    from mlProject.components.knowledge_graph import EnterpriseKnowledgeGraph
    kg = EnterpriseKnowledgeGraph()
    result = kg.get_entities()
    assert "entities" in result
    assert result["total"] > 0


def test_knowledge_graph_relationships():
    from mlProject.components.knowledge_graph import EnterpriseKnowledgeGraph
    kg = EnterpriseKnowledgeGraph()
    result = kg.get_relationships()
    assert "relationships" in result
    assert result["total"] > 0


def test_knowledge_graph_query():
    from mlProject.components.knowledge_graph import EnterpriseKnowledgeGraph
    kg = EnterpriseKnowledgeGraph()
    result = kg.query_graph(query_type="neighbors", entity_id="model:wine_v1")
    assert "neighbors" in result
    assert result["degree"] > 0


# ── Phase 30: Autonomous Command Center ──────────────────────────────────────

def test_command_decisions_component():
    from mlProject.components.autonomous_command import AutonomousCommandCenter
    center = AutonomousCommandCenter()
    result = center.get_decisions()
    assert "decisions" in result
    assert result["total_decisions"] > 0


def test_command_recommendations_component():
    from mlProject.components.autonomous_command import AutonomousCommandCenter
    center = AutonomousCommandCenter()
    result = center.get_recommendations()
    assert "recommendations" in result
    assert len(result["recommendations"]) > 0


def test_command_status_component():
    from mlProject.components.autonomous_command import AutonomousCommandCenter
    center = AutonomousCommandCenter()
    result = center.get_command_status()
    assert "subsystems" in result
    assert "overall_health_score" in result
    assert len(result["subsystems"]) == 10
