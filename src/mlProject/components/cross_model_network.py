"""
Cross Model Intelligence Network (Phase 48)

Builds a network enabling knowledge transfer and ensemble intelligence across
diverse AI models. Discovers and catalogs knowledge signatures of trained
models, facilitates knowledge transfer between models for transfer learning,
and creates optimized model ensembles for improved predictions. Tracks model
capabilities and relationships, optimizes ensemble weights dynamically, and
provides insights into cross-model synergies.

Persistence follows the project convention of a single shared SQLite database
(artifacts/predictions.db) with dedicated tables per component.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime


class CrossModelNetwork:
    """Core engine for cross-model knowledge discovery, transfer, and ensembling."""

    def __init__(self, db_path="artifacts/predictions.db"):
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cmn_models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model_type TEXT,
                version TEXT,
                capabilities TEXT,
                metadata TEXT,
                registered_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cmn_signatures (
                model_id TEXT PRIMARY KEY,
                signature TEXT,
                discovered_at TEXT,
                FOREIGN KEY (model_id) REFERENCES cmn_models (id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cmn_transfers (
                id TEXT PRIMARY KEY,
                source_model_id TEXT,
                target_model_id TEXT,
                transfer_type TEXT,
                compatibility_score REAL,
                status TEXT,
                details TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cmn_ensembles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                member_model_ids TEXT,
                weights TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Model registration & capability tracking
    # ------------------------------------------------------------------
    def register_model(self, name, model_type="unknown", version="1.0",
                        capabilities=None, metadata=None) -> str:
        """Register a model in the network inventory. Returns the new model_id."""
        model_id = str(uuid.uuid4())
        capabilities = capabilities or []
        metadata = metadata or {}
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cmn_models (id, name, model_type, version, capabilities, metadata, registered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, name, model_type, version,
                json.dumps(capabilities), json.dumps(metadata),
                datetime.utcnow().isoformat(),
            ))
            conn.commit()
            conn.close()
        except Exception:
            return None
        return model_id

    def get_model(self, model_id) -> dict:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cmn_models WHERE id = ?", (model_id,))
            row = cursor.fetchone()
            conn.close()
            if not row:
                return None
            return self._deserialize_model(dict(row))
        except Exception:
            return None

    def list_models(self) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cmn_models ORDER BY registered_at DESC")
            rows = cursor.fetchall()
            conn.close()
            return [self._deserialize_model(dict(r)) for r in rows]
        except Exception:
            return []

    @staticmethod
    def _deserialize_model(row: dict) -> dict:
        row["capabilities"] = json.loads(row.get("capabilities") or "[]")
        row["metadata"] = json.loads(row.get("metadata") or "{}")
        return row

    # ------------------------------------------------------------------
    # Knowledge signature discovery
    # ------------------------------------------------------------------
    def discover_knowledge_signature(self, model_id) -> dict:
        """
        Build (or refresh) a knowledge signature for a registered model,
        summarizing its capabilities and performance metadata so it can be
        compared against other models for transfer/ensemble decisions.
        """
        model = self.get_model(model_id)
        if not model:
            return {"status": "error", "message": f"Model {model_id} not found"}

        metadata = model.get("metadata", {})
        signature = {
            "model_id": model_id,
            "model_type": model.get("model_type"),
            "capabilities": model.get("capabilities", []),
            "performance_score": self._extract_performance_score(metadata),
            "capability_count": len(model.get("capabilities", [])),
        }

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cmn_signatures (model_id, signature, discovered_at)
                VALUES (?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    signature = excluded.signature,
                    discovered_at = excluded.discovered_at
            """, (model_id, json.dumps(signature), datetime.utcnow().isoformat()))
            conn.commit()
            conn.close()
        except Exception:
            pass

        return {"status": "success", "signature": signature}

    def get_knowledge_signature(self, model_id) -> dict:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cmn_signatures WHERE model_id = ?", (model_id,))
            row = cursor.fetchone()
            conn.close()
            if not row:
                return None
            data = dict(row)
            data["signature"] = json.loads(data.get("signature") or "{}")
            return data
        except Exception:
            return None

    @staticmethod
    def _extract_performance_score(metadata: dict) -> float:
        """Best-effort extraction of a single comparable performance score."""
        for key in ("r2", "accuracy", "f1", "score"):
            if key in metadata:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    continue
        return 0.0

    # ------------------------------------------------------------------
    # Knowledge transfer
    # ------------------------------------------------------------------
    def facilitate_transfer(self, source_model_id, target_model_id,
                             transfer_type="knowledge_distillation", details=None) -> dict:
        """Record and score a knowledge transfer between two registered models."""
        source = self.get_model(source_model_id)
        target = self.get_model(target_model_id)
        if not source or not target:
            return {"status": "error", "message": "Source or target model not found"}

        compatibility_score = self._compute_compatibility(source, target)
        transfer_id = str(uuid.uuid4())
        status = "completed" if compatibility_score >= 0.3 else "rejected"

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cmn_transfers
                    (id, source_model_id, target_model_id, transfer_type,
                     compatibility_score, status, details, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transfer_id, source_model_id, target_model_id, transfer_type,
                compatibility_score, status, json.dumps(details or {}),
                datetime.utcnow().isoformat(),
            ))
            conn.commit()
            conn.close()
        except Exception:
            return {"status": "error", "message": "Failed to persist transfer record"}

        return {
            "status": "success",
            "transfer_id": transfer_id,
            "transfer_status": status,
            "compatibility_score": compatibility_score,
        }

    def list_transfers(self, limit=100) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM cmn_transfers ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            rows = cursor.fetchall()
            conn.close()
            results = []
            for r in rows:
                d = dict(r)
                d["details"] = json.loads(d.get("details") or "{}")
                results.append(d)
            return results
        except Exception:
            return []

    @staticmethod
    def _compute_compatibility(source: dict, target: dict) -> float:
        """
        Heuristic compatibility score in [0, 1] based on shared capabilities
        and matching model type. Real transfer-learning compatibility would
        compare architecture/embedding spaces; this network tracks a
        simplified capability-overlap signal sufficient for routing decisions.
        """
        source_caps = set(source.get("capabilities", []))
        target_caps = set(target.get("capabilities", []))
        if not source_caps or not target_caps:
            overlap_score = 0.0
        else:
            overlap_score = len(source_caps & target_caps) / len(source_caps | target_caps)

        type_bonus = 0.2 if source.get("model_type") == target.get("model_type") else 0.0
        return round(min(1.0, overlap_score + type_bonus), 4)

    # ------------------------------------------------------------------
    # Ensemble creation & optimization
    # ------------------------------------------------------------------
    def create_ensemble(self, name, member_model_ids, weights=None) -> dict:
        if not member_model_ids or len(member_model_ids) < 2:
            return {"status": "error", "message": "An ensemble requires at least 2 member models"}

        for mid in member_model_ids:
            if not self.get_model(mid):
                return {"status": "error", "message": f"Model {mid} not found"}

        if weights is None:
            equal_weight = round(1.0 / len(member_model_ids), 4)
            weights = [equal_weight] * len(member_model_ids)

        if len(weights) != len(member_model_ids):
            return {"status": "error", "message": "weights and member_model_ids must be the same length"}

        ensemble_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cmn_ensembles (id, name, member_model_ids, weights, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                ensemble_id, name, json.dumps(member_model_ids),
                json.dumps(weights), now, now,
            ))
            conn.commit()
            conn.close()
        except Exception:
            return {"status": "error", "message": "Failed to persist ensemble"}

        return {"status": "success", "ensemble_id": ensemble_id, "weights": weights}

    def get_ensemble(self, ensemble_id) -> dict:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cmn_ensembles WHERE id = ?", (ensemble_id,))
            row = cursor.fetchone()
            conn.close()
            if not row:
                return None
            return self._deserialize_ensemble(dict(row))
        except Exception:
            return None

    def list_ensembles(self) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cmn_ensembles ORDER BY created_at DESC")
            rows = cursor.fetchall()
            conn.close()
            return [self._deserialize_ensemble(dict(r)) for r in rows]
        except Exception:
            return []

    @staticmethod
    def _deserialize_ensemble(row: dict) -> dict:
        row["member_model_ids"] = json.loads(row.get("member_model_ids") or "[]")
        row["weights"] = json.loads(row.get("weights") or "[]")
        return row

    def optimize_ensemble_weights(self, ensemble_id) -> dict:
        """
        Recompute ensemble weights proportionally to each member model's
        performance score (from its metadata), so stronger models contribute
        more to the ensemble prediction.
        """
        ensemble = self.get_ensemble(ensemble_id)
        if not ensemble:
            return {"status": "error", "message": f"Ensemble {ensemble_id} not found"}

        member_ids = ensemble["member_model_ids"]
        scores = []
        for mid in member_ids:
            model = self.get_model(mid)
            score = self._extract_performance_score(model.get("metadata", {})) if model else 0.0
            scores.append(max(score, 0.0))

        total = sum(scores)
        if total <= 0:
            new_weights = [round(1.0 / len(member_ids), 4)] * len(member_ids)
        else:
            new_weights = [round(s / total, 4) for s in scores]

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE cmn_ensembles SET weights = ?, updated_at = ? WHERE id = ?
            """, (json.dumps(new_weights), datetime.utcnow().isoformat(), ensemble_id))
            conn.commit()
            conn.close()
        except Exception:
            return {"status": "error", "message": "Failed to persist optimized weights"}

        return {"status": "success", "ensemble_id": ensemble_id, "weights": new_weights}

    # ------------------------------------------------------------------
    # Synergy analysis
    # ------------------------------------------------------------------
    def analyze_synergies(self) -> list:
        """
        Pairwise synergy analysis across all registered models, ranked by
        capability complementarity (models that cover different capability
        sets score higher, since combining them adds coverage).
        """
        models = self.list_models()
        synergies = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                caps_a = set(a.get("capabilities", []))
                caps_b = set(b.get("capabilities", []))
                union = caps_a | caps_b
                shared = caps_a & caps_b
                if not union:
                    synergy_score = 0.0
                else:
                    complementarity = len(union - shared) / len(union)
                    synergy_score = round(complementarity, 4)
                synergies.append({
                    "model_a": a["id"],
                    "model_a_name": a["name"],
                    "model_b": b["id"],
                    "model_b_name": b["name"],
                    "synergy_score": synergy_score,
                })
        synergies.sort(key=lambda s: s["synergy_score"], reverse=True)
        return synergies

    # ------------------------------------------------------------------
    # Network-level summary (dashboard widget + alerting)
    # ------------------------------------------------------------------
    def get_network_summary(self) -> dict:
        models = self.list_models()
        ensembles = self.list_ensembles()
        transfers = self.list_transfers(limit=1000)
        synergies = self.analyze_synergies()

        avg_synergy = (
            round(sum(s["synergy_score"] for s in synergies) / len(synergies), 4)
            if synergies else 0.0
        )

        needing_optimization = 0
        for ens in ensembles:
            for mid in ens["member_model_ids"]:
                sig = self.get_knowledge_signature(mid)
                if sig and sig["discovered_at"] > ens["updated_at"]:
                    needing_optimization += 1
                    break

        return {
            "total_models": len(models),
            "total_transfers": len(transfers),
            "total_ensembles": len(ensembles),
            "avg_synergy_score": avg_synergy,
            "top_synergy_pair": synergies[0] if synergies else None,
            "ensembles_needing_optimization": needing_optimization,
        }
