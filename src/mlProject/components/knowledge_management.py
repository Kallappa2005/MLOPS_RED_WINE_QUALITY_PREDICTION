import os
import json
from datetime import datetime

class KnowledgeRepository:
    def __init__(self, storage_path="artifacts/knowledge_repo.json"):
        self.storage_path = storage_path
        self._init_repo()

    def _init_repo(self):
        if not os.path.exists(self.storage_path):
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            initial_data = {
                "documents": [
                    {"id": "doc_1", "title": "Red Wine Volatiles Overview", "category": "Domain", "content": "Volatile acidity represents acetic acid. High levels lead to sour taste."},
                    {"id": "doc_2", "title": "SHAP Explanation Guidelines", "category": "ML", "content": "Alcohol has positive correlation; volatile acidity has negative impact."}
                ],
                "nodes": [
                    {"id": "dataset_red_wine", "label": "Red Wine Dataset", "type": "Dataset"},
                    {"id": "exp_xgboost", "label": "XGBoost Run", "type": "Experiment"},
                    {"id": "metric_r2", "label": "R2 Score: 0.51", "type": "Metric"},
                    {"id": "feature_alcohol", "label": "Alcohol Feature", "type": "Feature"}
                ],
                "edges": [
                    {"source": "dataset_red_wine", "target": "exp_xgboost", "label": "trained_on"},
                    {"source": "exp_xgboost", "target": "metric_r2", "label": "resulted_in"},
                    {"source": "exp_xgboost", "target": "feature_alcohol", "label": "uses_feature"}
                ]
            }
            with open(self.storage_path, "w") as f:
                json.dump(initial_data, f, indent=2)

    def get_knowledge_graph(self) -> dict:
        try:
            with open(self.storage_path, "r") as f:
                return json.load(f)
        except Exception:
            return {"documents": [], "nodes": [], "edges": []}

    def add_document(self, title: str, category: str, content: str) -> bool:
        try:
            repo = self.get_knowledge_graph()
            doc_id = f"doc_{len(repo.get('documents', [])) + 1}"
            repo["documents"].append({
                "id": doc_id,
                "title": title,
                "category": category,
                "content": content,
                "created_at": datetime.utcnow().isoformat()
            })
            with open(self.storage_path, "w") as f:
                json.dump(repo, f, indent=2)
            return True
        except Exception:
            return False
