import os
import json
import uuid
import sqlite3
import shutil
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from mlProject import logger
from mlProject.components.data_transformation import NUMERIC_FEATURES
from mlProject.utils.model_registry import load_registry


class DigitalTwinError(Exception):
    """Raised for twin creation/lookup failures that callers should surface as 4xx."""


class DigitalTwin:
    """
    Manages digital replicas ("twins") of the production model + its config
    so changes can be simulated and tested before touching the real thing.

    A twin is a frozen copy of a registered model version plus a snapshot of
    the registry entry that produced it. Twins are stored under
    artifacts/digital_twins/<twin_id>/ and tracked in the same predictions.db
    used by the rest of the monitoring stack.
    """

    def __init__(self, db_path: str = "artifacts/predictions.db",
                 twins_dir: str = "artifacts/digital_twins",
                 registry_path: str = "artifacts/model_registry.json"):
        self.db_path = db_path
        self.twins_dir = Path(twins_dir)
        self.registry_path = Path(registry_path)
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS digital_twins (
                id TEXT PRIMARY KEY,
                name TEXT,
                source_version_id TEXT,
                model_path TEXT,
                created_at TEXT,
                last_synced_at TEXT,
                status TEXT,
                metadata TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS twin_simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                twin_id TEXT,
                scenario_name TEXT,
                input_json TEXT,
                result_json TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS twin_drift_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                twin_id TEXT,
                timestamp TEXT,
                drift_detected INTEGER,
                details TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Twin lifecycle
    # ------------------------------------------------------------------

    def create_twin(self, name: str, source_version_id: str = None) -> dict:
        """
        Create a digital twin from a production/registered model version.

        If source_version_id is omitted, the current production version is
        used. Copies the model artifact so the twin is fully isolated from
        anything happening to the live model afterwards.
        """
        registry = load_registry(self.registry_path)
        version_id = source_version_id or registry.get("production")
        if not version_id:
            raise DigitalTwinError("No source_version_id given and no production version is registered.")

        version_entry = next((v for v in registry.get("versions", []) if v.get("id") == version_id), None)
        if version_entry is None:
            raise DigitalTwinError(f"Version {version_id} not found in model registry.")

        source_model_path = Path(version_entry.get("path", ""))
        if not source_model_path.exists():
            raise DigitalTwinError(f"Model artifact for version {version_id} is missing at {source_model_path}.")

        twin_id = uuid.uuid4().hex[:12]
        twin_dir = self.twins_dir / twin_id
        twin_dir.mkdir(parents=True, exist_ok=True)
        twin_model_path = twin_dir / "model.joblib"
        shutil.copy2(source_model_path, twin_model_path)

        now = datetime.utcnow().isoformat()
        metadata = {
            "source_metrics": version_entry.get("metrics", {}),
            "source_created_at": version_entry.get("date"),
        }

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO digital_twins (id, name, source_version_id, model_path, created_at, last_synced_at, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (twin_id, name, version_id, str(twin_model_path), now, now, "active", json.dumps(metadata)))
        conn.commit()
        conn.close()

        logger.info(f"Digital twin '{name}' ({twin_id}) created from version {version_id}")
        return self.get_twin(twin_id)

    def get_twin(self, twin_id: str) -> dict:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM digital_twins WHERE id = ?", (twin_id,))
        row = cursor.fetchone()
        conn.close()
        if row is None:
            raise DigitalTwinError(f"Twin {twin_id} not found.")
        twin = dict(row)
        twin["metadata"] = json.loads(twin.get("metadata") or "{}")
        return twin

    def list_twins(self) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM digital_twins ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        twins = [dict(r) for r in rows]
        for t in twins:
            t["metadata"] = json.loads(t.get("metadata") or "{}")
        return twins

    def delete_twin(self, twin_id: str) -> bool:
        twin = self.get_twin(twin_id)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM digital_twins WHERE id = ?", (twin_id,))
        cursor.execute("DELETE FROM twin_simulations WHERE twin_id = ?", (twin_id,))
        cursor.execute("DELETE FROM twin_drift_reports WHERE twin_id = ?", (twin_id,))
        conn.commit()
        conn.close()
        twin_dir = Path(twin["model_path"]).parent
        try:
            twin_dir.resolve().relative_to(self.twins_dir.resolve())
            if twin_dir.exists():
                shutil.rmtree(twin_dir, ignore_errors=True)
        except ValueError:
            pass  # twin_dir isn't under twins_dir - don't delete unexpected paths
        logger.info(f"Digital twin {twin_id} deleted")
        return True

    def sync_state(self, twin_id: str) -> dict:
        """
        Re-point a twin at whatever is currently registered as production,
        replacing its model copy and metrics snapshot. Does not mutate
        production in any way — read-only against the registry.
        """
        twin = self.get_twin(twin_id)
        registry = load_registry(self.registry_path)
        current_prod_id = registry.get("production")
        if not current_prod_id:
            raise DigitalTwinError("No production version is currently registered.")

        version_entry = next((v for v in registry.get("versions", []) if v.get("id") == current_prod_id), None)
        if version_entry is None:
            raise DigitalTwinError(f"Production version {current_prod_id} missing from registry.")

        source_model_path = Path(version_entry.get("path", ""))
        if not source_model_path.exists():
            raise DigitalTwinError(f"Model artifact missing at {source_model_path}.")

        twin_model_path = Path(twin["model_path"])
        shutil.copy2(source_model_path, twin_model_path)

        now = datetime.utcnow().isoformat()
        metadata = twin["metadata"]
        metadata["source_metrics"] = version_entry.get("metrics", {})
        metadata["source_created_at"] = version_entry.get("date")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE digital_twins SET source_version_id = ?, last_synced_at = ?, metadata = ?
            WHERE id = ?
        """, (current_prod_id, now, json.dumps(metadata), twin_id))
        conn.commit()
        conn.close()

        logger.info(f"Digital twin {twin_id} synced to production version {current_prod_id}")
        return self.get_twin(twin_id)

    # ------------------------------------------------------------------
    # Simulation / what-if analysis
    # ------------------------------------------------------------------

    def run_simulation(self, twin_id: str, scenario_name: str, input_rows: list) -> dict:
        """
        Run a set of feature rows through the twin's frozen model. This never
        touches the live prediction pipeline or production traffic.
        """
        twin = self.get_twin(twin_id)
        model_path = Path(twin["model_path"])
        if not model_path.exists():
            raise DigitalTwinError(f"Twin {twin_id} has no model artifact on disk.")

        missing = [col for col in NUMERIC_FEATURES if not all(col in row for row in input_rows)]
        if missing:
            raise DigitalTwinError(f"Missing required features in scenario input: {missing}")

        df = pd.DataFrame(input_rows)[NUMERIC_FEATURES]
        model = joblib.load(model_path)
        predictions = model.predict(df)

        result = {
            "predictions": [round(float(p), 4) for p in np.atleast_1d(predictions)],
            "row_count": len(input_rows),
            "mean_prediction": round(float(np.mean(predictions)), 4),
        }

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO twin_simulations (twin_id, scenario_name, input_json, result_json, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (twin_id, scenario_name, json.dumps(input_rows), json.dumps(result), datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

        return result

    def get_simulation_history(self, twin_id: str, limit: int = 50) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM twin_simulations WHERE twin_id = ? ORDER BY created_at DESC LIMIT ?
        """, (twin_id, limit))
        rows = cursor.fetchall()
        conn.close()
        history = []
        for r in rows:
            entry = dict(r)
            entry["input_json"] = json.loads(entry["input_json"])
            entry["result_json"] = json.loads(entry["result_json"])
            history.append(entry)
        return history

    # ------------------------------------------------------------------
    # Pre-deployment change testing
    # ------------------------------------------------------------------

    def test_change(self, twin_id: str, candidate_model_path: str, test_rows: list) -> dict:
        """
        Compare a candidate model against the twin's current (production-mirroring)
        model on the same input rows, so a change can be evaluated before it's
        ever promoted to production.
        """
        twin = self.get_twin(twin_id)
        current_model_path = Path(twin["model_path"])
        candidate_path = Path(candidate_model_path)

        if not current_model_path.exists():
            raise DigitalTwinError(f"Twin {twin_id} has no baseline model artifact.")
        if not candidate_path.exists():
            raise DigitalTwinError(f"Candidate model not found at {candidate_path}.")

        df = pd.DataFrame(test_rows)[NUMERIC_FEATURES]
        baseline_model = joblib.load(current_model_path)
        candidate_model = joblib.load(candidate_path)

        baseline_preds = np.atleast_1d(baseline_model.predict(df))
        candidate_preds = np.atleast_1d(candidate_model.predict(df))

        diff = candidate_preds - baseline_preds
        return {
            "baseline_mean": round(float(np.mean(baseline_preds)), 4),
            "candidate_mean": round(float(np.mean(candidate_preds)), 4),
            "mean_abs_diff": round(float(np.mean(np.abs(diff))), 4),
            "max_abs_diff": round(float(np.max(np.abs(diff))), 4),
            "row_count": len(test_rows),
        }

    # ------------------------------------------------------------------
    # Drift between twin and the real (current production) system
    # ------------------------------------------------------------------

    def detect_twin_drift(self, twin_id: str) -> dict:
        """
        Flags whether the twin has fallen out of sync with the current
        production version — i.e. the twin no longer faithfully represents
        the real system and should be re-synced.
        """
        twin = self.get_twin(twin_id)
        registry = load_registry(self.registry_path)
        current_prod_id = registry.get("production")

        drift_detected = bool(current_prod_id) and twin["source_version_id"] != current_prod_id
        details = {
            "twin_source_version": twin["source_version_id"],
            "current_production_version": current_prod_id,
            "last_synced_at": twin["last_synced_at"],
        }

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO twin_drift_reports (twin_id, timestamp, drift_detected, details)
            VALUES (?, ?, ?, ?)
        """, (twin_id, datetime.utcnow().isoformat(), int(drift_detected), json.dumps(details)))
        conn.commit()
        conn.close()

        return {
            "status": "success",
            "drift_detected": drift_detected,
            "details": details,
        }
