import os
import sqlite3
import time
from datetime import datetime, timedelta
from mlProject import logger

class APILogger:
    def __init__(self, db_path="artifacts/predictions.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                endpoint TEXT,
                method TEXT,
                status_code INTEGER,
                latency_ms REAL,
                ip TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_request(self, endpoint, method, status_code, latency_ms, ip):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO api_logs (timestamp, endpoint, method, status_code, latency_ms, ip)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (datetime.utcnow().isoformat(), endpoint, method, int(status_code), float(latency_ms), ip))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_analytics(self, hours=24) -> dict:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    AVG(latency_ms) as avg_latency,
                    SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as errors
                FROM api_logs
                WHERE timestamp >= ?
            """, (cutoff,))
            summary = dict(cursor.fetchone())
            
            cursor.execute("""
                SELECT endpoint, COUNT(*) as count, AVG(latency_ms) as latency
                FROM api_logs
                WHERE timestamp >= ?
                GROUP BY endpoint
            """, (cutoff,))
            breakdown = [dict(r) for r in cursor.fetchall()]
            
            conn.close()
            
            total = summary["total_requests"] or 0
            errs = summary["errors"] or 0
            err_rate = (errs / total * 100) if total > 0 else 0.0
            
            return {
                "total_requests": total,
                "avg_latency_ms": round(summary["avg_latency"] or 0.0, 2),
                "error_rate_pct": round(err_rate, 2),
                "endpoints": breakdown
            }
        except Exception:
            return {"total_requests": 0, "avg_latency_ms": 0.0, "error_rate_pct": 0.0, "endpoints": []}


class ObservabilityCollector:
    def __init__(self, db_path="artifacts/predictions.db"):
        self.db_path = db_path
        self.api_logger = APILogger(db_path)

    def get_system_health(self) -> dict:
        cpu_usage = 15.4
        ram_usage = 42.1
        
        try:
            import psutil
            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent
        except ImportError:
            pass
            
        api_stats = self.api_logger.get_analytics(hours=1)
        
        status = "healthy"
        alerts = []
        
        if api_stats["error_rate_pct"] > 5.0:
            status = "unhealthy"
            alerts.append(f"High API Error Rate: {api_stats['error_rate_pct']}%")
            
        from mlProject.components.monitoring import DriftDetector
        try:
            detector = DriftDetector(db_path=self.db_path)
            report = detector.detect_drift(min_predictions=5)
            if report.get("status") == "success" and report.get("drift_detected", False):
                alerts.append(f"Model Data Drift Detected! Ratio: {report.get('drifted_features_ratio', 0)*100:.1f}%")
        except Exception:
            pass

        return {
            "status": status,
            "cpu_usage_pct": cpu_usage,
            "ram_usage_pct": ram_usage,
            "api_requests_last_hour": api_stats["total_requests"],
            "avg_latency_last_hour_ms": api_stats["avg_latency_ms"],
            "error_rate_last_hour_pct": api_stats["error_rate_pct"],
            "alerts": alerts
        }
