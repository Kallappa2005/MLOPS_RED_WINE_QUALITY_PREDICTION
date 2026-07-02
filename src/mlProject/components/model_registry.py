"""
Phase 21: Enterprise Model Registry Hub
Centralized model registry with version lineage, metadata tracking,
deployment history, and rollback capabilities.
"""
import json
import os
import time
import uuid
from pathlib import Path
from datetime import datetime

REGISTRY_FILE = Path("artifacts/enterprise_model_registry.json")


class EnterpriseModelRegistry:
    """Centralized model registry with full version lineage and rollback support."""

    def __init__(self):
        self._ensure_registry()

    def _ensure_registry(self):
        REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not REGISTRY_FILE.exists():
            REGISTRY_FILE.write_text(json.dumps({"models": [], "deployments": []}))

    def _load(self):
        return json.loads(REGISTRY_FILE.read_text())

    def _save(self, data):
        REGISTRY_FILE.write_text(json.dumps(data, indent=2))

    def list_models(self):
        data = self._load()
        return {
            "total": len(data["models"]),
            "models": data["models"],
            "status": "ok"
        }

    def register_model(self, name: str, version: str, framework: str,
                        metrics: dict, artifact_path: str, owner: str) -> dict:
        data = self._load()
        model_id = str(uuid.uuid4())[:8]
        record = {
            "id": model_id,
            "name": name,
            "version": version,
            "framework": framework,
            "metrics": metrics,
            "artifact_path": artifact_path,
            "owner": owner,
            "status": "registered",
            "stage": "staging",
            "registered_at": datetime.utcnow().isoformat() + "Z",
            "lineage": []
        }
        data["models"].append(record)
        self._save(data)
        return {"message": "Model registered successfully", "model_id": model_id, "record": record}

    def version_history(self, model_name: str = None) -> dict:
        data = self._load()
        models = data["models"]
        if model_name:
            models = [m for m in models if m["name"] == model_name]
        # Group by name
        history = {}
        for m in models:
            history.setdefault(m["name"], []).append({
                "version": m["version"],
                "stage": m["stage"],
                "registered_at": m["registered_at"],
                "metrics": m.get("metrics", {}),
                "owner": m.get("owner", "unknown")
            })
        return {"version_history": history}

    def promote_model(self, model_id: str, target_stage: str) -> dict:
        data = self._load()
        for m in data["models"]:
            if m["id"] == model_id:
                old_stage = m["stage"]
                m["stage"] = target_stage
                m["lineage"].append({
                    "event": "stage_promotion",
                    "from": old_stage,
                    "to": target_stage,
                    "at": datetime.utcnow().isoformat() + "Z"
                })
                self._save(data)
                return {"message": f"Model {model_id} promoted to {target_stage}", "record": m}
        return {"error": f"Model {model_id} not found"}

    def rollback_model(self, model_name: str) -> dict:
        data = self._load()
        candidates = sorted(
            [m for m in data["models"] if m["name"] == model_name],
            key=lambda x: x["registered_at"]
        )
        if len(candidates) < 2:
            return {"error": "Not enough versions for rollback"}
        current = candidates[-1]
        previous = candidates[-2]
        current["stage"] = "archived"
        previous["stage"] = "production"
        previous.setdefault("lineage", []).append({
            "event": "rollback",
            "from": current["version"],
            "to": previous["version"],
            "at": datetime.utcnow().isoformat() + "Z"
        })
        self._save(data)
        return {
            "message": f"Rolled back {model_name} from {current['version']} to {previous['version']}",
            "active_version": previous["version"]
        }
