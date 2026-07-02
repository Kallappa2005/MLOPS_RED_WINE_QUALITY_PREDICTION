import os
import sqlite3
import json
from datetime import datetime

class EnterpriseFeatureStore:
    def __init__(self, db_path="artifacts/predictions.db"):
        self.db_path = db_path
        self._init_store()

    def _init_store(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_store (
                feature_name TEXT PRIMARY KEY,
                version TEXT,
                description TEXT,
                data_type TEXT,
                value_mean REAL,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()
        
        # Populate basic defaults if empty
        self._seed_default_features()

    def _seed_default_features(self):
        defaults = [
            ("alcohol", "v1", "Alcohol content of red wine", "float", 10.42),
            ("volatile acidity", "v1", "Acetic acid amount in wine", "float", 0.52),
            ("pH", "v1", "Acidity scale level", "float", 3.31),
            ("sulphates", "v1", "Preservative additions", "float", 0.65)
        ]
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for name, version, desc, dtype, val in defaults:
            cursor.execute("""
                INSERT OR IGNORE INTO feature_store (feature_name, version, description, data_type, value_mean, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, version, desc, dtype, val, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

    def get_feature_catalog(self) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feature_store")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def register_feature(self, name: str, version: str, description: str, data_type: str, mean_val: float) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO feature_store (feature_name, version, description, data_type, value_mean, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, version, description, data_type, float(mean_val), datetime.utcnow().isoformat()))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
