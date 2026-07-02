"""
Phase 29: Enterprise AI Knowledge Graph
Connects models, datasets, features, experiments, alerts, and governance records.
"""
import json
from datetime import datetime
from pathlib import Path

GRAPH_FILE = Path("artifacts/knowledge_graph.json")


class EnterpriseKnowledgeGraph:
    """Knowledge graph connecting all AI assets and their relationships."""

    def __init__(self):
        self._ensure_graph()

    def _ensure_graph(self):
        GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not GRAPH_FILE.exists():
            graph = {
                "entities": [
                    {"id": "model:wine_v1", "type": "Model", "name": "ElasticNet Wine Quality v1",
                     "attributes": {"rmse": 0.58, "r2": 0.55, "stage": "production"}},
                    {"id": "model:wine_v2", "type": "Model", "name": "RandomForest Wine Quality v2",
                     "attributes": {"rmse": 0.49, "r2": 0.68, "stage": "staging"}},
                    {"id": "dataset:winequality_raw", "type": "Dataset", "name": "winequality-red.csv",
                     "attributes": {"rows": 1599, "features": 11, "source": "UCI ML Repository"}},
                    {"id": "dataset:winequality_clean", "type": "Dataset", "name": "cleaned_dataset",
                     "attributes": {"rows": 1599, "missing_pct": 0.0}},
                    {"id": "feature:alcohol", "type": "Feature", "name": "alcohol",
                     "attributes": {"type": "continuous", "importance": 0.42}},
                    {"id": "feature:volatile_acidity", "type": "Feature", "name": "volatile_acidity",
                     "attributes": {"type": "continuous", "importance": 0.31}},
                    {"id": "experiment:exp_001", "type": "Experiment", "name": "ElasticNet Alpha Search",
                     "attributes": {"status": "completed", "runs": 12}},
                    {"id": "governance:gv_001", "type": "GovernanceRecord", "name": "Model Promotion Gate",
                     "attributes": {"status": "APPROVED", "reviewer": "admin"}},
                    {"id": "alert:alt_drift_001", "type": "Alert", "name": "Volatile Acidity Drift Alert",
                     "attributes": {"severity": "MEDIUM", "status": "active"}}
                ],
                "relationships": [
                    {"from": "model:wine_v1", "to": "dataset:winequality_clean",
                     "relation": "TRAINED_ON", "weight": 1.0},
                    {"from": "model:wine_v2", "to": "dataset:winequality_clean",
                     "relation": "TRAINED_ON", "weight": 1.0},
                    {"from": "dataset:winequality_raw", "to": "dataset:winequality_clean",
                     "relation": "TRANSFORMED_TO", "weight": 1.0},
                    {"from": "dataset:winequality_clean", "to": "feature:alcohol",
                     "relation": "CONTAINS_FEATURE", "weight": 0.42},
                    {"from": "dataset:winequality_clean", "to": "feature:volatile_acidity",
                     "relation": "CONTAINS_FEATURE", "weight": 0.31},
                    {"from": "model:wine_v1", "to": "experiment:exp_001",
                     "relation": "PRODUCED_BY", "weight": 1.0},
                    {"from": "model:wine_v1", "to": "governance:gv_001",
                     "relation": "GOVERNED_BY", "weight": 1.0},
                    {"from": "alert:alt_drift_001", "to": "feature:volatile_acidity",
                     "relation": "TRIGGERED_BY", "weight": 0.85}
                ]
            }
            GRAPH_FILE.write_text(json.dumps(graph, indent=2))

    def _load(self):
        return json.loads(GRAPH_FILE.read_text())

    def get_entities(self, entity_type: str = None) -> dict:
        data = self._load()
        entities = data["entities"]
        if entity_type:
            entities = [e for e in entities if e["type"] == entity_type]
        type_counts = {}
        for e in data["entities"]:
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
        return {
            "entities": entities,
            "total": len(entities),
            "type_distribution": type_counts
        }

    def get_relationships(self, entity_id: str = None) -> dict:
        data = self._load()
        rels = data["relationships"]
        if entity_id:
            rels = [r for r in rels if r["from"] == entity_id or r["to"] == entity_id]
        return {
            "relationships": rels,
            "total": len(rels)
        }

    def query_graph(self, query_type: str = "neighbors", entity_id: str = None) -> dict:
        data = self._load()
        if query_type == "neighbors" and entity_id:
            neighbors = []
            for r in data["relationships"]:
                if r["from"] == entity_id:
                    neighbors.append({"direction": "outbound", "relation": r["relation"], "entity": r["to"]})
                elif r["to"] == entity_id:
                    neighbors.append({"direction": "inbound", "relation": r["relation"], "entity": r["from"]})
            return {
                "entity_id": entity_id,
                "query_type": query_type,
                "neighbors": neighbors,
                "degree": len(neighbors)
            }
        # Default: return summary
        return {
            "query_type": query_type,
            "entity_count": len(data["entities"]),
            "relationship_count": len(data["relationships"]),
            "entity_types": list(set(e["type"] for e in data["entities"]))
        }
