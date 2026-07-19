import os
import sqlite3
from datetime import datetime

class AlertEngine:
    def __init__(self, db_path="artifacts/predictions.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metric_name TEXT,
                metric_value REAL,
                threshold REAL,
                severity TEXT,
                status TEXT,
                message TEXT
            )
        """)
        conn.commit()
        conn.close()

    def trigger_alert(self, metric_name: str, metric_value: float, threshold: float, severity: str, message: str) -> bool:
        """
        Register a system incident alert and trigger mock email alert.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_alerts (timestamp, metric_name, metric_value, threshold, severity, status, message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (datetime.utcnow().isoformat(), metric_name, float(metric_value), float(threshold), severity, "ACTIVE", message))
            conn.commit()
            conn.close()
            
            # Simulated alerting log
            print(f"[ALERT NOTIFICATION EMAIL] TO: ops@redwineiq.io | SUBJECT: ALERT [{severity}] {metric_name} | MSG: {message}")
            return True
        except Exception:
            return False

    def get_active_alerts(self, limit=100) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM system_alerts WHERE status = 'ACTIVE' ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []
            
    def resolve_alert(self, alert_id: int) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE system_alerts SET status = 'RESOLVED' WHERE id = ?", (alert_id,))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
