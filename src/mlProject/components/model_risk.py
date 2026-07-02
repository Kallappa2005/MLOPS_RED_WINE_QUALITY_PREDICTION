import numpy as np
import pandas as pd
from datetime import datetime

class ModelRiskScoringEngine:
    def __init__(self):
        pass

    def evaluate_model_risk(self, performance_metrics: dict, quality_metrics: dict) -> dict:
        """
        Evaluate overall risk score (0-100, where higher represents lower risk/higher health status).
        Or risk index (0.0 to 1.0, where lower is better).
        Let's compute:
        Risk Score = (rmse * 40) + (drift_ratio * 40) + (missing_rate * 20)
        We will return a status classification: LOW, MEDIUM, HIGH risk.
        """
        rmse = performance_metrics.get("rmse", 0.60)
        drift = performance_metrics.get("drift_ratio", 0.0)
        missing_rate = quality_metrics.get("missing_rate", 0.0)

        # Heuristic score between 0.0 and 1.0 (higher = riskier)
        risk_index = (rmse * 0.5) + (drift * 0.3) + (missing_rate * 0.2)
        risk_index = min(1.0, max(0.0, float(risk_index)))
        
        status = "LOW" if risk_index < 0.35 else "MEDIUM" if risk_index < 0.65 else "HIGH"
        
        return {
            "risk_index": round(risk_index, 3),
            "status": status,
            "metrics": {
                "performance_risk": round(rmse * 0.5, 3),
                "drift_risk": round(drift * 0.3, 3),
                "data_quality_risk": round(missing_rate * 0.2, 3)
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    def detect_bias(self, df: pd.DataFrame, target_col="quality") -> dict:
        """
        Calculates prediction/target quality fairness checks across high/low alcohol categories.
        """
        if df.empty or "alcohol" not in df.columns or target_col not in df.columns:
            return {"status": "INSUFFICIENT_DATA", "disparity": 0.0}

        # Splitting group by median alcohol content
        median_alc = df["alcohol"].median()
        high_alc_group = df[df["alcohol"] >= median_alc]
        low_alc_group = df[df["alcohol"] < median_alc]
        
        mean_high = float(high_alc_group[target_col].mean()) if not high_alc_group.empty else 0.0
        mean_low = float(low_alc_group[target_col].mean()) if not low_alc_group.empty else 0.0
        disparity = abs(mean_high - mean_low)
        
        passed = disparity < 0.15 # fairness threshold
        
        return {
            "status": "PASSED" if passed else "FAILED",
            "disparity": round(disparity, 4),
            "group_means": {
                "high_alcohol": round(mean_high, 4),
                "low_alcohol": round(mean_low, 4)
            },
            "threshold": 0.15,
            "timestamp": datetime.utcnow().isoformat()
        }
