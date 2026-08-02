import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from mlProject import logger
from mlProject.utils.model_registry import load_registry, update_registration


class CommandGridError(Exception):
    """Raised for fleet/command/incident lookup and validation failures."""


# Commands that can be dispatched to a fleet node via execute_command().
_SUPPORTED_COMMANDS = {"promote", "demote", "archive", "retrain", "health_check"}
_REGISTRY_STATUS_COMMANDS = {"promote": "production", "demote": "staging", "archive": "archived"}


class AutonomousCommandGrid:
    """
    Unified command and control layer that sits on top of the existing
    MLOps components (model registry, retraining engine, observability).

    A "fleet node" is a registered model version being tracked by the grid.
    The grid can dispatch commands across many nodes at once, aggregate
    fleet-wide health, track cross-system incidents, and recommend which
    version should be serving traffic based on registered metrics.
    """

    def __init__(self, db_path: str = "artifacts/predictions.db",
                 registry_path: str = "artifacts/model_registry.json"):
        self.db_path = db_path
        self.registry_path = Path(registry_path)
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fleet_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                node_type TEXT,
                target_ref TEXT,
                status TEXT,
                registered_at TEXT,
                last_health_check TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS command_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id INTEGER,
                command TEXT,
                params TEXT,
                status TEXT,
                result TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                severity TEXT,
                affected_nodes TEXT,
                description TEXT,
                status TEXT,
                auto_remediated INTEGER,
                created_at TEXT,
                resolved_at TEXT,
                resolution_notes TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Fleet node management
    # ------------------------------------------------------------------

    def register_node(self, name: str, target_ref: str, node_type: str = "model_version") -> dict:
        """Add a model version (or other tracked target) to the fleet."""
        if node_type == "model_version":
            registry = load_registry(self.registry_path)
            if not any(v.get("id") == target_ref for v in registry.get("versions", [])):
                raise CommandGridError(f"Model version {target_ref} not found in registry.")

        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO fleet_nodes (name, node_type, target_ref, status, registered_at, last_health_check)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, node_type, target_ref, "active", now, None))
        node_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logger.info(f"Fleet node '{name}' registered (id={node_id}, target={target_ref})")
        return self.get_node(node_id)

    def get_node(self, node_id: int) -> dict:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM fleet_nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        conn.close()
        if row is None:
            raise CommandGridError(f"Fleet node {node_id} not found.")
        return dict(row)

    def list_nodes(self) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM fleet_nodes ORDER BY registered_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def deregister_node(self, node_id: int) -> bool:
        self.get_node(node_id)  # raises if missing
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM fleet_nodes WHERE id = ?", (node_id,))
        conn.commit()
        conn.close()
        logger.info(f"Fleet node {node_id} deregistered")
        return True

    # ------------------------------------------------------------------
    # Fleet-wide command execution
    # ------------------------------------------------------------------

    def execute_command(self, node_ids: list, command: str, params: dict = None) -> list:
        """
        Dispatch a single command across many fleet nodes at once and
        return a per-node result list. Unknown nodes fail individually
        rather than aborting the whole batch.
        """
        if command not in _SUPPORTED_COMMANDS:
            raise CommandGridError(f"Unsupported command '{command}'. Supported: {sorted(_SUPPORTED_COMMANDS)}")

        params = params or {}
        results = []
        for node_id in node_ids:
            result = self._execute_single(node_id, command, params)
            results.append(result)
        return results

    def _execute_single(self, node_id: int, command: str, params: dict) -> dict:
        status = "success"
        detail = {}
        try:
            node = self.get_node(node_id)

            if command in _REGISTRY_STATUS_COMMANDS:
                if node["node_type"] != "model_version":
                    raise CommandGridError(f"Command '{command}' only applies to model_version nodes.")
                target_status = _REGISTRY_STATUS_COMMANDS[command]
                ok = update_registration(
                    registry_path=self.registry_path,
                    version_id=node["target_ref"],
                    status=target_status,
                )
                if not ok:
                    raise CommandGridError(f"Version {node['target_ref']} not found in registry.")
                detail = {"new_status": target_status}

            elif command == "retrain":
                from mlProject.components.retraining import RetrainingEngine
                reason = params.get("reason", f"Autonomous command grid retrain for node {node_id}")
                triggered = RetrainingEngine(db_path=self.db_path).trigger_retraining(reason=reason)
                if not triggered:
                    status = "rejected"
                    detail = {"message": "Retraining already in progress"}
                else:
                    detail = {"message": "Retraining triggered"}

            elif command == "health_check":
                registry = load_registry(self.registry_path)
                version_entry = next((v for v in registry.get("versions", []) if v.get("id") == node["target_ref"]), None)
                model_exists = bool(version_entry) and Path(version_entry.get("path", "")).exists()
                detail = {"model_artifact_present": model_exists, "registry_status": version_entry.get("status") if version_entry else "unknown"}
                self._update_last_health_check(node_id)

        except CommandGridError as e:
            status = "error"
            detail = {"error": str(e)}
        except Exception as e:
            logger.error(f"Command '{command}' failed for node {node_id}: {e}")
            status = "error"
            detail = {"error": str(e)}

        self._log_command(node_id, command, params, status, detail)
        return {"node_id": node_id, "command": command, "status": status, "detail": detail}

    def _update_last_health_check(self, node_id: int):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE fleet_nodes SET last_health_check = ? WHERE id = ?",
                       (datetime.utcnow().isoformat(), node_id))
        conn.commit()
        conn.close()

    def _log_command(self, node_id: int, command: str, params: dict, status: str, result: dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO command_executions (node_id, command, params, status, result, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (node_id, command, json.dumps(params), status, json.dumps(result), datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

    def get_command_history(self, node_id: int = None, limit: int = 100) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if node_id is not None:
            cursor.execute("""
                SELECT * FROM command_executions WHERE node_id = ? ORDER BY created_at DESC LIMIT ?
            """, (node_id, limit))
        else:
            cursor.execute("SELECT * FROM command_executions ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        history = []
        for r in rows:
            entry = dict(r)
            entry["params"] = json.loads(entry["params"] or "{}")
            entry["result"] = json.loads(entry["result"] or "{}")
            history.append(entry)
        return history

    # ------------------------------------------------------------------
    # Fleet health
    # ------------------------------------------------------------------

    def get_fleet_health(self) -> dict:
        """Aggregate shared system health with the status of every fleet node."""
        from mlProject.components.observability import ObservabilityCollector
        system_health = ObservabilityCollector(db_path=self.db_path).get_system_health()

        registry = load_registry(self.registry_path)
        nodes = self.list_nodes()
        node_health = []
        for node in nodes:
            version_entry = next((v for v in registry.get("versions", []) if v.get("id") == node["target_ref"]), None)
            node_health.append({
                "node_id": node["id"],
                "name": node["name"],
                "target_ref": node["target_ref"],
                "model_artifact_present": bool(version_entry) and Path(version_entry.get("path", "")).exists() if version_entry else False,
                "registry_status": version_entry.get("status") if version_entry else "unknown",
                "last_health_check": node["last_health_check"],
            })

        return {
            "system": system_health,
            "fleet_size": len(nodes),
            "nodes": node_health,
        }

    # ------------------------------------------------------------------
    # Incident handling
    # ------------------------------------------------------------------

    def report_incident(self, title: str, severity: str, affected_node_ids: list, description: str = "") -> dict:
        """
        Record an incident affecting one or more fleet nodes. Critical
        incidents automatically trigger a retraining run as a first line
        of remediation.
        """
        valid_severities = {"low", "medium", "high", "critical"}
        if severity not in valid_severities:
            raise CommandGridError(f"severity must be one of {sorted(valid_severities)}")

        auto_remediated = False
        if severity == "critical" and affected_node_ids:
            results = self.execute_command(affected_node_ids, "retrain", {"reason": f"Auto-remediation for incident: {title}"})
            auto_remediated = any(r["status"] == "success" for r in results)

        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO incidents (title, severity, affected_nodes, description, status, auto_remediated, created_at, resolved_at, resolution_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (title, severity, json.dumps(affected_node_ids), description, "open", int(auto_remediated), now, None, None))
        incident_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.warning(f"Incident reported: {title} (severity={severity}, auto_remediated={auto_remediated})")
        return self.get_incident(incident_id)

    def get_incident(self, incident_id: int) -> dict:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,))
        row = cursor.fetchone()
        conn.close()
        if row is None:
            raise CommandGridError(f"Incident {incident_id} not found.")
        incident = dict(row)
        incident["affected_nodes"] = json.loads(incident["affected_nodes"] or "[]")
        incident["auto_remediated"] = bool(incident["auto_remediated"])
        return incident

    def list_incidents(self, status: str = None) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if status:
            cursor.execute("SELECT * FROM incidents WHERE status = ? ORDER BY created_at DESC", (status,))
        else:
            cursor.execute("SELECT * FROM incidents ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        incidents = []
        for r in rows:
            incident = dict(r)
            incident["affected_nodes"] = json.loads(incident["affected_nodes"] or "[]")
            incident["auto_remediated"] = bool(incident["auto_remediated"])
            incidents.append(incident)
        return incidents

    def resolve_incident(self, incident_id: int, resolution_notes: str = "") -> dict:
        self.get_incident(incident_id)  # raises if missing
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE incidents SET status = ?, resolved_at = ?, resolution_notes = ? WHERE id = ?
        """, ("resolved", datetime.utcnow().isoformat(), resolution_notes, incident_id))
        conn.commit()
        conn.close()
        return self.get_incident(incident_id)

    # ------------------------------------------------------------------
    # Resource allocation optimization
    # ------------------------------------------------------------------

    def optimize_allocation(self) -> dict:
        """
        Rank registered, non-archived model versions by their recorded
        metrics (higher r2 / lower rmse is better) and recommend which
        one the fleet should route production traffic to.
        """
        registry = load_registry(self.registry_path)
        candidates = [v for v in registry.get("versions", []) if v.get("status") != "archived"]
        if not candidates:
            return {"recommended_version": None, "ranked": [], "message": "No eligible model versions to optimize over."}

        def score(entry):
            metrics = entry.get("metrics", {})
            r2 = metrics.get("r2", 0.0) or 0.0
            rmse = metrics.get("rmse", float("inf"))
            rmse = rmse if rmse not in (None, 0) else float("inf")
            # Simple composite: reward higher r2, penalize higher rmse.
            return r2 - (0.1 * rmse if rmse != float("inf") else 0)

        ranked = sorted(candidates, key=score, reverse=True)
        ranked_summary = [
            {"version_id": v["id"], "status": v.get("status"), "metrics": v.get("metrics", {}), "score": round(score(v), 4)}
            for v in ranked
        ]

        return {
            "recommended_version": ranked_summary[0]["version_id"],
            "ranked": ranked_summary,
            "current_production": registry.get("production"),
        }