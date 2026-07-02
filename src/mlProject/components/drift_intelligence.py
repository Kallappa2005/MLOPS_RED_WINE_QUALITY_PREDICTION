import numpy as np
import pandas as pd
from datetime import datetime

class AdvancedDriftEngine:
    def __init__(self, baseline_df=None):
        # Fallback default values
        self.baseline_means = {
            "alcohol": 10.4,
            "volatile acidity": 0.52,
            "pH": 3.31,
            "sulphates": 0.65
        }

    def detect_feature_drift(self, prediction_samples: list) -> dict:
        """
        Check feature drift between baseline means and prediction samples averages.
        """
        if not prediction_samples:
            return {"drift_detected": False, "details": "No incoming predictions logged."}
            
        # Compute mean of samples
        df = pd.DataFrame(prediction_samples)
        drifted_features = []
        drift_ratios = {}
        
        for feature, base_val in self.baseline_means.items():
            col = feature
            if feature not in df.columns:
                col = feature.replace(" ", "_")
                
            if col in df.columns:
                sample_mean = float(df[col].mean())
                # If mean deviates by more than 15%, flag feature drift
                deviation = abs(sample_mean - base_val) / base_val
                drift_ratios[feature] = round(deviation, 3)
                if deviation >= 0.15:
                    drifted_features.append(feature)
                    
        ratio = len(drifted_features) / len(self.baseline_means) if self.baseline_means else 0
        return {
            "drift_detected": ratio >= 0.25,
            "drifted_features": drifted_features,
            "drift_ratios": drift_ratios,
            "drift_ratio_score": ratio,
            "timestamp": datetime.utcnow().isoformat()
        }

    def detect_concept_drift(self, performance_history: list) -> dict:
        """
        Detect concept drift based on model prediction performance over time.
        """
        # Checks if average quality predictions are continuously declining below standard bounds
        if len(performance_history) < 5:
            return {"concept_drift_detected": False, "message": "Insufficient data to test concept drift."}
            
        recent_scores = [p.get("prediction", 5.6) for p in performance_history[-10:]]
        mean_score = np.mean(recent_scores)
        
        # Standard wine quality prediction average is 5.6. If it slips below 5.2, signal concept drift
        drift_detected = mean_score < 5.2
        return {
            "concept_drift_detected": drift_detected,
            "current_performance_index": round(float(mean_score), 3),
            "message": "Concept drift detected: model outputs are declining!" if drift_detected else "Concept boundaries stable."
        }
