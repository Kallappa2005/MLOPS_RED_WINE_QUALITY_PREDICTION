import os
import json
import sqlite3
from datetime import datetime
from typing import Optional

from mlProject import logger

DEFAULT_DB_PATH = "artifacts/decision_intelligence.db"


class DecisionIntelligenceEngine:
    """
    Augments business decisions with AI-powered insights. Analyzes a decision
    context against a set of candidate options, generates ranked
    recommendations with confidence scores and explainable reasoning, and
    learns from logged outcomes to track recommendation performance over time.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                context TEXT,
                options TEXT,
                recommendation TEXT,
                confidence REAL,
                reasoning TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decision_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id INTEGER,
                timestamp TEXT,
                chosen_option TEXT,
                actual_outcome REAL,
                matched_recommendation INTEGER,
                FOREIGN KEY (decision_id) REFERENCES decisions (id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decision_alerts (
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
    # Decision analysis
    # ------------------------------------------------------------------
    def analyze_decision(self, context: dict, options: list) -> dict:
        """
        Score each candidate option against the weighted factors in context,
        rank them, and return a recommendation with a confidence score and
        explainable reasoning.

        `context` maps factor name -> weight (float).
        Each entry in `options` is a dict with a 'name' key and factor
        values matching the context's factor names.
        """
        if not context:
            return {"status": "error", "message": "context (factor weights) is required"}
        if not options or len(options) < 2:
            return {"status": "error", "message": "At least two options are required"}

        for opt in options:
            if "name" not in opt:
                return {"status": "error", "message": "Each option must include a 'name'"}
            missing = [factor for factor in context if factor not in opt]
            if missing:
                return {
                    "status": "error",
                    "message": f"Option '{opt.get('name')}' is missing factor(s): {missing}"
                }

        scored = []
        for opt in options:
            score = sum(float(opt[factor]) * float(weight) for factor, weight in context.items())
            scored.append({"name": opt["name"], "score": score})

        scored.sort(key=lambda o: o["score"], reverse=True)
        scores = [o["score"] for o in scored]
        top_score = scores[0]
        runner_up = scores[1] if len(scores) > 1 else scores[0]
        spread = top_score - runner_up
        max_possible_spread = max(abs(top_score), abs(runner_up), 1e-9)
        confidence = max(0.0, min(1.0, spread / max_possible_spread))

        reasoning = self._build_reasoning(context, options, scored[0]["name"])

        decision_id = self._log_decision(context, options, scored, confidence, reasoning)

        if confidence < 0.15:
            self.trigger_alert(
                alert_type="low_confidence_decision",
                severity="medium",
                message=(
                    f"Decision {decision_id} recommended '{scored[0]['name']}' "
                    f"with low confidence ({confidence:.2f})"
                )
            )

        return {
            "status": "success",
            "decision_id": decision_id,
            "recommendation": scored[0]["name"],
            "confidence": round(confidence, 4),
            "ranked_options": scored,
            "reasoning": reasoning
        }

    def _build_reasoning(self, context: dict, options: list, winner_name: str) -> str:
        winner = next(opt for opt in options if opt["name"] == winner_name)
        contributions = {
            factor: float(winner[factor]) * float(weight)
            for factor, weight in context.items()
        }
        top_factor = max(contributions, key=lambda k: contributions[k])
        return (
            f"'{winner_name}' was recommended primarily due to its strength in "
            f"'{top_factor}' (weighted contribution: {contributions[top_factor]:.3f})."
        )

    def _log_decision(self, context, options, scored, confidence, reasoning) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO decisions (timestamp, context, options, recommendation, confidence, reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            json.dumps(context),
            json.dumps(options),
            json.dumps(scored),
            confidence,
            reasoning
        ))
        conn.commit()
        decision_id = cursor.lastrowid
        conn.close()
        assert decision_id is not None
        return decision_id

    def get_decision_history(self, limit: int = 100) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"Failed to fetch decision history: {e}")
            return []

    # ------------------------------------------------------------------
    # Outcome logging / learning
    # ------------------------------------------------------------------
    def log_outcome(self, decision_id: int, chosen_option: str, actual_outcome: float) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT recommendation FROM decisions WHERE id = ?", (decision_id,))
        row = cursor.fetchone()
        matched = False
        if row:
            recommendation = json.loads(row[0])
            if recommendation and recommendation[0]["name"] == chosen_option:
                matched = True

        cursor.execute("""
            INSERT INTO decision_outcomes (
                decision_id, timestamp, chosen_option, actual_outcome, matched_recommendation
            )
            VALUES (?, ?, ?, ?, ?)
        """, (decision_id, datetime.utcnow().isoformat(), chosen_option, actual_outcome, int(matched)))
        conn.commit()
        outcome_id = cursor.lastrowid
        conn.close()
        assert outcome_id is not None
        return outcome_id

    def get_performance_metrics(self) -> dict:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM decision_outcomes")
            total_outcomes = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM decision_outcomes WHERE matched_recommendation = 1")
            matched_outcomes = cursor.fetchone()[0]
            conn.close()
            match_rate = (matched_outcomes / total_outcomes) if total_outcomes else 0.0
            return {
                "total_outcomes_logged": total_outcomes,
                "matched_recommendation_count": matched_outcomes,
                "recommendation_match_rate": round(match_rate, 4)
            }
        except Exception as e:
            logger.error(f"Failed to compute performance metrics: {e}")
            return {
                "total_outcomes_logged": 0,
                "matched_recommendation_count": 0,
                "recommendation_match_rate": 0.0
            }

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------
    def trigger_alert(self, alert_type: str, severity: str, message: str) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO decision_alerts (timestamp, alert_type, severity, message, status, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), alert_type, severity, message, "open", None))
        conn.commit()
        alert_id = cursor.lastrowid
        conn.close()
        assert alert_id is not None
        logger.warning(f"Decision intelligence alert triggered [{severity}] {alert_type}: {message}")
        return alert_id

    def get_alerts(self, status: Optional[str] = "open", limit: int = 100) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    "SELECT * FROM decision_alerts WHERE status = ? ORDER BY timestamp DESC LIMIT ?",
                    (status, limit)
                )
            else:
                cursor.execute("SELECT * FROM decision_alerts ORDER BY timestamp DESC LIMIT ?", (limit,))
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
            "UPDATE decision_alerts SET status = ?, resolved_at = ? WHERE id = ? AND status = 'open'",
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
            cursor.execute("SELECT COUNT(*) FROM decisions")
            total_decisions = cursor.fetchone()[0]
            cursor.execute("SELECT AVG(confidence) FROM decisions")
            avg_confidence = cursor.fetchone()[0] or 0.0
            cursor.execute("SELECT COUNT(*) FROM decision_outcomes")
            total_outcomes = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM decision_alerts WHERE status = 'open'")
            active_alerts = cursor.fetchone()[0]
            conn.close()
            return {
                "total_decisions": total_decisions,
                "average_confidence": round(avg_confidence, 4),
                "total_outcomes_logged": total_outcomes,
                "active_alerts": active_alerts
            }
        except Exception as e:
            logger.error(f"Failed to build dashboard summary: {e}")
            return {
                "total_decisions": 0,
                "average_confidence": 0.0,
                "total_outcomes_logged": 0,
                "active_alerts": 0
            }
