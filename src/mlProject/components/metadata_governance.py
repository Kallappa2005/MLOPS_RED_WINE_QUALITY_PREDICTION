import os
import sqlite3
import json
from datetime import datetime

class MetadataRegistry:
    def __init__(self, db_path="artifacts/predictions.db"):
        self.db_path = db_path
        self._init_metadata()

    def _init_metadata(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata_registry (
                resource_id TEXT PRIMARY KEY,
                resource_type TEXT,
                tags_json TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()
        self._seed_default_metadata()

    def _seed_default_metadata(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM metadata_registry")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO metadata_registry (resource_id, resource_type, tags_json, updated_at)
                VALUES (?, ?, ?, ?)
            """, (
                "model_joblib", 
                "Model", 
                '{"environment": "production", "framework": "scikit-learn", "owner": "mldev-team", "accuracy_tier": "gold"}', 
                datetime.utcnow().isoformat()
            ))
            conn.commit()
        conn.close()

    def register_metadata(self, resource_id: str, resource_type: str, tags: dict) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO metadata_registry (resource_id, resource_type, tags_json, updated_at)
                VALUES (?, ?, ?, ?)
            """, (resource_id, resource_type, json.dumps(tags) if isinstance(tags, dict) else str(tags), datetime.utcnow().isoformat()))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def search_resources_by_tag(self, tag_key: str, tag_value: str) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM metadata_registry")
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for r in rows:
                tags = json.loads(r["tags_json"])
                if tags.get(tag_key) == tag_value:
                    results.append(dict(r))
            return results
        except Exception:
            return []
            
    def get_all_metadata(self) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM metadata_registry ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []
