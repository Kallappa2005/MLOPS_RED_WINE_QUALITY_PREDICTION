import os
import sqlite3
from datetime import datetime

class IncidentResponseEngine:
    def __init__(self, db_path="artifacts/predictions.db"):
        self.db_path = db_path
        self._init_incidents()

    def _init_incidents(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_type TEXT,
                status TEXT,
                severity TEXT,
                root_cause TEXT,
                mitigation TEXT,
                opened_at TEXT,
                resolved_at TEXT
            )
        """)
        conn.commit()
        conn.close()
        self._seed_default_incident()

    def _seed_default_incident(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM system_incidents")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO system_incidents (incident_type, status, severity, root_cause, mitigation, opened_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ("LatencySpike", "RESOLVED", "WARNING", "API Gateway congestion", "Optimized database caching layer", datetime.utcnow().isoformat(), datetime.utcnow().isoformat()))
            conn.commit()
        conn.close()

    def open_incident(self, incident_type: str, severity: str, root_cause: str) -> int:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_incidents (incident_type, status, severity, root_cause, mitigation, opened_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (incident_type, "OPEN", severity, root_cause, "Pending investigation", datetime.utcnow().isoformat(), None))
            inc_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return inc_id
        except Exception:
            return -1

    def resolve_incident(self, incident_id: int, mitigation: str) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE system_incidents 
                SET status = 'RESOLVED', mitigation = ?, resolved_at = ?
                WHERE id = ?
            """, (mitigation, datetime.utcnow().isoformat(), int(incident_id)))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def get_incidents(self, status=None) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if status:
                cursor.execute("SELECT * FROM system_incidents WHERE status = ? ORDER BY opened_at DESC", (status,))
            else:
                cursor.execute("SELECT * FROM system_incidents ORDER BY opened_at DESC")
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def analyze_root_cause(self, error_msg: str) -> str:
        msg = error_msg.lower()
        if "timeout" in msg or "latency" in msg:
            return "Connection timeout: High request volume or database concurrency lock."
        elif "value" in msg or "type" in msg:
            return "Data Validation: Incoming prediction features schema mismatch."
        elif "drift" in msg:
            return "Statistical Drift: Incoming feature distribution deviated from reference parameters."
        return "Unknown anomaly: Pending developer logs audit."
