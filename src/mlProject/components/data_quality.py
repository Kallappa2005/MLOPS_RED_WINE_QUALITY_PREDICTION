import numpy as np
import pandas as pd
import json

class QualityValidator:
    def __init__(self, baseline_csv_path="artifacts/data_ingestion/winequality-red.csv"):
        self.baseline_path = baseline_csv_path

    def analyze_dataset_quality(self, incoming_df: pd.DataFrame) -> dict:
        """
        Scan dataset for missing values, range bounds, and compute a data quality score (0-100).
        """
        total_cells = incoming_df.size
        if total_cells == 0:
            return {"quality_score": 0, "missing_cells": 0, "anomalies": []}

        # Check missing values
        missing_count = int(incoming_df.isnull().sum().sum())
        missing_rate = missing_count / total_cells

        # Check anomaly outliers (using IQR on alcohol and acidity)
        anomalies = []
        for col in ["alcohol", "volatile acidity", "pH"]:
            col_name = col
            if col not in incoming_df.columns:
                # Fallback to underscores
                col_name = col.replace(" ", "_")
                
            if col_name in incoming_df.columns:
                series = incoming_df[col_name].dropna()
                if len(series) > 4:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = series[(series < lower_bound) | (series > upper_bound)]
                    if not outliers.empty:
                        anomalies.append({
                            "feature": col,
                            "outliers_count": len(outliers),
                            "min_outlier": float(outliers.min()),
                            "max_outlier": float(outliers.max())
                        })

        # Score calculation
        quality_score = 100.0 - (missing_rate * 100) - (len(anomalies) * 5)
        quality_score = max(0.0, min(100.0, quality_score))

        return {
            "quality_score": round(quality_score, 1),
            "missing_count": missing_count,
            "anomalies": anomalies,
            "total_records": len(incoming_df),
            "timestamp": pd.Timestamp.now().isoformat()
        }
