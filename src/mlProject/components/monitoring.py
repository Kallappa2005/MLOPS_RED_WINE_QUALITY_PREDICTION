import os
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp
from datetime import datetime
from mlProject import logger
from mlProject.components.data_transformation import NUMERIC_FEATURES

class PredictionLogger:
    def __init__(self, db_path: str = "artifacts/predictions.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        columns_sql = ", ".join([f"`{col.replace(' ', '_')}` REAL" for col in NUMERIC_FEATURES])
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                {columns_sql},
                prediction REAL
            )
        """)
        conn.commit()
        conn.close()

    def log_prediction(self, features_dict: dict, prediction_val: float):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cols = ["timestamp", "prediction"] + [col.replace(' ', '_') for col in NUMERIC_FEATURES]
            vals = [datetime.utcnow().isoformat(), float(prediction_val)] + [float(features_dict[col]) for col in NUMERIC_FEATURES]
            
            placeholders = ", ".join(["?" for _ in vals])
            columns_str = ", ".join([f"`{c}`" for c in cols])
            
            cursor.execute(f"INSERT INTO predictions ({columns_str}) VALUES ({placeholders})", vals)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log prediction to DB: {e}")

    def get_logged_predictions(self, limit: int = 1000) -> pd.DataFrame:
        if not os.path.exists(self.db_path):
            return pd.DataFrame()
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(f"SELECT * FROM predictions ORDER BY timestamp DESC LIMIT {limit}", conn)
            conn.close()
            rename_dict = {col.replace(' ', '_'): col for col in NUMERIC_FEATURES}
            df = df.rename(columns=rename_dict)
            return df
        except Exception as e:
            logger.error(f"Failed to query logged predictions: {e}")
            return pd.DataFrame()


class DriftDetector:
    def __init__(self, reference_data_path: str = "artifacts/data_transformation/train.csv", db_path: str = "artifacts/predictions.db"):
        self.reference_data_path = reference_data_path
        self.db_path = db_path

    def detect_drift(self, min_predictions: int = 10) -> dict:
        """
        Detect feature and prediction distribution drift using KS test.
        """
        if not os.path.exists(self.reference_data_path):
            return {"status": "error", "message": "Reference data not found. Run training first."}
            
        ref_df = pd.read_csv(self.reference_data_path)
        
        logger_db = PredictionLogger(self.db_path)
        pred_df = logger_db.get_logged_predictions(limit=1000)
        
        if pred_df.empty or len(pred_df) < min_predictions:
            return {
                "status": "insufficient_data",
                "message": f"Need at least {min_predictions} predictions to calculate drift (Currently have {len(pred_df)})",
                "prediction_count": len(pred_df)
            }
            
        drift_report = {}
        drift_detected = False
        drift_features_count = 0
        
        for col in NUMERIC_FEATURES:
            if col in ref_df.columns and col in pred_df.columns:
                ref_dist = ref_df[col].dropna().values
                pred_dist = pred_df[col].dropna().values
                
                stat, p_value = ks_2samp(ref_dist, pred_dist)
                has_drift = bool(p_value < 0.05)
                if has_drift:
                    drift_features_count += 1
                    drift_detected = True
                    
                drift_report[col] = {
                    "ks_statistic": float(stat),
                    "p_value": float(p_value),
                    "drift_detected": has_drift,
                    "ref_mean": float(np.mean(ref_dist)),
                    "pred_mean": float(np.mean(pred_dist))
                }
                
        return {
            "status": "success",
            "drift_detected": drift_detected,
            "drifted_features_ratio": float(drift_features_count / len(NUMERIC_FEATURES)),
            "total_predictions": len(pred_df),
            "metrics": drift_report
        }
