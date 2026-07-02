import os
import sqlite3
import pandas as pd
from datetime import datetime

class DatasetRegistry:
    def __init__(self, db_path="artifacts/predictions.db"):
        self.db_path = db_path
        self._init_registry()

    def _init_registry(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_registry (
                dataset_id TEXT PRIMARY KEY,
                version TEXT,
                file_path TEXT,
                record_count INTEGER,
                schema_json TEXT,
                lineage_parent TEXT,
                registered_at TEXT
            )
        """)
        conn.commit()
        conn.close()
        self._seed_default_dataset()

    def _seed_default_dataset(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM dataset_registry")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO dataset_registry (dataset_id, version, file_path, record_count, schema_json, lineage_parent, registered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                "wine_raw_v1", 
                "v1.0", 
                "artifacts/data_ingestion/winequality-red.csv", 
                1599, 
                '{"columns": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]}',
                "Kaggle_Source", 
                datetime.utcnow().isoformat()
            ))
            conn.commit()
        conn.close()

    def register_dataset(self, dataset_id: str, version: str, file_path: str, record_count: int, schema_dict: dict, lineage_parent: str) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO dataset_registry (dataset_id, version, file_path, record_count, schema_json, lineage_parent, registered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (dataset_id, version, file_path, int(record_count), json.dumps(schema_dict) if isinstance(schema_dict, dict) else str(schema_dict), lineage_parent, datetime.utcnow().isoformat()))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def list_datasets(self) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM dataset_registry ORDER BY registered_at DESC")
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def validate_schema(self, df: pd.DataFrame, expected_cols: list) -> dict:
        missing = [col for col in expected_cols if col not in df.columns]
        types_match = True
        for col in df.columns:
            if col in expected_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    types_match = False
        return {
            "is_valid": len(missing) == 0 and types_match,
            "missing_columns": missing,
            "numeric_types_valid": types_match,
            "timestamp": datetime.utcnow().isoformat()
        }
