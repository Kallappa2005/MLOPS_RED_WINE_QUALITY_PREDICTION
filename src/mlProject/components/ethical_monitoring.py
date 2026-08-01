import os
import json
import sqlite3
from datetime import datetime
from typing import Optional

from mlProject import logger

DEFAULT_DB_PATH = "artifacts/ethics.db"


class EthicalAIMonitor:
    """
    Continuous monitoring for ethical compliance and bias detection in
    production AI systems. Tracks demographic-parity bias across protected
    attributes, logs compliance violations, and manages ethical monitoring
    alerts with mitigation recommendations.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bias_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                protected_attribute TEXT,
                disparity REAL,
                threshold REAL,
                bias_detected INTEGER,
                group_rates TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compliance_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                violation_type TEXT,
                severity TEXT,
                description TEXT,
                status TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ethics_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                status TEXT,
                resolved_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Bias detection
    # ------------------------------------------------------------------
    def detect_bias(self, records: list, protected_attribute: str, threshold: float = 0.1) -> dict:
        """
        Compute demographic parity difference across groups of a protected
        attribute. Each record must contain the protected attribute value
        and a binary/numeric 'prediction' field.
        """
        if not records:
            return {"status": "error", "message": "No records provided"}

        groups: dict = {}
        for rec in records:
            if protected_attribute not in rec or "prediction" not in rec:
                return {
                    "status": "error",
                    "message": f"Each record must include '{protected_attribute}' and 'prediction'"
                }
            group_val = rec[protected_attribute]
            groups.setdefault(group_val, []).append(float(rec["prediction"]))

        if len(groups) < 2:
            return {
                "status": "insufficient_groups",
                "message": "Need at least two distinct groups to compute bias metrics"
            }

        group_rates = {str(group): sum(vals) / len(vals) for group, vals in groups.items()}
        disparity = max(group_rates.values()) - min(group_rates.values())
        bias_detected = disparity > threshold

        self._log_bias_check(protected_attribute, disparity, threshold, bias_detected, group_rates)

        if bias_detected:
            self.trigger_alert(
                alert_type="bias_detected",
                severity="high" if disparity > (threshold * 2) else "medium",
                message=f"Demographic parity disparity of {disparity:.3f} detected across '{protected_attribute}'"
            )

        return {
            "status": "success",
            "protected_attribute": protected_attribute,
            "group_rates": group_rates,
            "disparity": disparity,
            "threshold": threshold,
            "bias_detected": bias_detected,
            "recommendations": self._mitigation_recommendations(disparity, threshold)
        }

    def _mitigation_recommendations(self, disparity: float, threshold: float) -> list:
        if disparity <= threshold:
            return ["No mitigation required — disparity within acceptable threshold."]
        recs = [
            "Review training data for representation imbalance across the affected groups.",
            "Consider reweighting or resampling underrepresented groups before retraining.",
        ]
        if disparity > threshold * 2:
            recs.append(
                "Disparity is significant — recommend pausing automated decisions "
                "for this attribute pending review."
            )
        recs.append("Re-run detect_bias() after mitigation to confirm disparity has decreased.")
        return recs

    def _log_bias_check(self, protected_attribute, disparity, threshold, bias_detected, group_rates):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO bias_checks (
                    timestamp, protected_attribute, disparity, threshold, bias_detected, group_rates
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                protected_attribute,
                disparity,
                threshold,
                int(bias_detected),
                json.dumps(group_rates)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log bias check: {e}")

    def get_bias_history(self, limit: int = 100) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM bias_checks ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"Failed to fetch bias history: {e}")
            return []

    # ------------------------------------------------------------------
    # Compliance violations
    # ------------------------------------------------------------------
    def log_violation(self, violation_type: str, severity: str, description: str = "") -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO compliance_violations (timestamp, violation_type, severity, description, status)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), violation_type, severity, description, "open"))
        conn.commit()
        violation_id = cursor.lastrowid
        conn.close()
        assert violation_id is not None

        if severity in ("high", "critical"):
            self.trigger_alert(
                alert_type="compliance_violation",
                severity=severity,
                message=f"{severity.upper()} violation logged: {violation_type} — {description}"
            )
        return violation_id

    def get_compliance_report(self, limit: int = 100) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM compliance_violations ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"Failed to fetch compliance report: {e}")
            return []

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------
    def trigger_alert(self, alert_type: str, severity: str, message: str) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO ethics_alerts (timestamp, alert_type, severity, message, status, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), alert_type, severity, message, "open", None))
        conn.commit()
        alert_id = cursor.lastrowid
        conn.close()
        assert alert_id is not None
        logger.warning(f"Ethics alert triggered [{severity}] {alert_type}: {message}")
        return alert_id

    def get_alerts(self, status: Optional[str] = "open", limit: int = 100) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    "SELECT * FROM ethics_alerts WHERE status = ? ORDER BY timestamp DESC LIMIT ?",
                    (status, limit)
                )
            else:
                cursor.execute("SELECT * FROM ethics_alerts ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"Failed to fetch alerts: {e}")
            return []

    def resolve_alert(self, alert_id: int) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE ethics_alerts SET status = ?, resolved_at = ? WHERE id = ? AND status = 'open'",
            ("resolved", datetime.utcnow().isoformat(), alert_id)
        )
        conn.commit()
        updated = cursor.rowcount > 0
        conn.close()
        return updated

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    def get_dashboard_summary(self) -> dict:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM bias_checks")
            total_bias_checks = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM bias_checks WHERE bias_detected = 1")
            bias_flagged = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM compliance_violations WHERE status = 'open'")
            open_violations = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM ethics_alerts WHERE status = 'open'")
            active_alerts = cursor.fetchone()[0]
            conn.close()
            return {
                "total_bias_checks": total_bias_checks,
                "bias_flagged_checks": bias_flagged,
                "open_violations": open_violations,
                "active_alerts": active_alerts
            }
        except Exception as e:
            logger.error(f"Failed to build dashboard summary: {e}")
            return {
                "total_bias_checks": 0,
                "bias_flagged_checks": 0,
                "open_violations": 0,
                "active_alerts": 0
            }
